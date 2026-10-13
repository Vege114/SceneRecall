import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, DEFAULT_EXECUTION, type Asset, type Estimate, type ExecutionOptions, type Profile, type Settings as SettingsType } from '../src/types';
import JobModal from '../src/components/JobModal';
import Jobs from '../src/components/Jobs';
import Settings from '../src/components/Settings';
import CharacterProfiles from '../src/components/CharacterProfiles';
import { DeepenSceneButton } from '../src/components/SceneAnalysis';

const asset: Asset = { id: 'one', work_id: 'work', title: '测试影片', kind: 'movie', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const model: Profile = { id: 'api', name: '测试 API', provider_type: 'openai_compatible', base_url: 'https://example.invalid/v1', model: 'test-model', capabilities: ['preparation', 'vision', 'subtitle'], secret_mode: 'none', max_concurrency: 4 };
const estimate: Estimate = { duration_ms: 12000, estimated_frames: 8, estimated_requests: 2, cost_estimate: null, warnings: [] };
const mixed: ExecutionOptions = { transcript: 'serial', preparation: 'parallel', subtitle: 'serial', max_concurrency: 5 };
function context(execution: ExecutionOptions | undefined = DEFAULT_EXECUTION): AppContextValue {
  return { bootstrap: { library_path: '/tmp/execution-ui-tests', settings: { defaults: { ...DEFAULTS, execution }, bindings: { preparation: model.id, vision: model.id, subtitle: model.id } }, provider_profiles: [model], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' }, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function mount(ui: ReactNode, app = context()) { return render(<AppContext.Provider value={app}>{ui}</AppContext.Provider>); }
function mockFetch(handler?: (url: string, options?: RequestInit) => unknown) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const value = handler ? handler(url, options) : url === '/api/keyframe-plans' ? { id: 'plan', status: 'completed', frames: [], frame_count: 8 } : url.endsWith('/estimate') ? estimate : { id: 'job' };
    return new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
function body(fetch: ReturnType<typeof mockFetch>, path: string) { return JSON.parse(fetch.mock.calls.filter(([url]) => url === path).at(-1)![1]!.body as string); }
async function runEstimate(user: ReturnType<typeof userEvent.setup>, name = '估算用量', startName = '开始分析') {
  await user.click(screen.getByRole('button', { name }));
  await waitFor(() => expect((screen.getByRole('button', { name: startName }) as HTMLButtonElement).disabled).toBe(false));
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Stage execution selection', () => {
  it('uses parallel defaults for new tasks, shows connection limits, and invalidates estimates for mode and width changes', async () => {
    const app = context(); delete app.bootstrap.settings.defaults.execution;
    const fetch = mockFetch(); const user = userEvent.setup(); const close = vi.fn();
    const view = mount(<JobModal asset={asset} onClose={close} />, app);
    expect((screen.getByLabelText('台词整理执行方式') as HTMLSelectElement).value).toBe('parallel');
    expect((screen.getByLabelText('视觉分块执行方式') as HTMLSelectElement).value).toBe('parallel');
    expect(screen.queryByLabelText('字幕识别执行方式')).toBeNull();
    expect(screen.getAllByText('连接并发上限 4；本次最多同时 3 个请求')).toHaveLength(2);
    expect(screen.getByText(/共享开始时的此前记忆/)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
    await screen.findByRole('button', { name: '重新扫描关键帧' });
    await runEstimate(user);
    expect(body(fetch, '/api/jobs/estimate').execution).toEqual(DEFAULT_EXECUTION);
    await user.selectOptions(screen.getByLabelText('台词整理执行方式'), 'serial');
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    await runEstimate(user);
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), '5');
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText('连接并发上限 4；本次最多同时 4 个请求')).toBeTruthy();
    await runEstimate(user);
    const updated = { ...app, bootstrap: { ...app.bootstrap, provider_profiles: [{ ...model, max_concurrency: 2 }] } };
    view.rerender(<AppContext.Provider value={updated}><JobModal asset={asset} onClose={close} /></AppContext.Provider>);
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText('连接并发上限 2；本次最多同时 2 个请求')).toBeTruthy();
    await runEstimate(user); await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    expect(body(fetch, '/api/jobs').execution).toEqual({ ...DEFAULT_EXECUTION, transcript: 'serial', max_concurrency: 5 });
    expect(body(fetch, '/api/jobs').execution).toEqual(body(fetch, '/api/jobs/estimate').execution);
    expect(fetch.mock.calls.some(([url]) => url === '/api/settings')).toBe(false);
  });

  it('shows only OCR execution for subtitle-only batches and snapshots choices on every item', async () => {
    const selected = [asset, { ...asset, id: 'two' }].map((entry) => ({ ...entry, subtitle_mode: 'embedded' as const }));
    const fetch = mockFetch((url) => url.endsWith('/estimate') ? { ...estimate, items: selected.map((item) => ({ ...estimate, asset_id: item.id })) } : {});
    const user = userEvent.setup(); mount(<JobModal assets={selected} onClose={vi.fn()} />);
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    expect(screen.queryByLabelText('台词整理执行方式')).toBeNull();
    expect(screen.queryByLabelText('视觉分块执行方式')).toBeNull();
    await user.selectOptions(screen.getByLabelText('字幕识别执行方式'), 'serial');
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).disabled).toBe(true);
    await runEstimate(user); await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/jobs/batch')).toBe(true));
    const items = body(fetch, '/api/jobs/batch').items;
    expect(items).toEqual(body(fetch, '/api/jobs/batch/estimate').items);
    expect(items.map((item: { execution: ExecutionOptions; stages: string[] }) => ({ execution: item.execution, stages: item.stages }))).toEqual(selected.map(() => ({ execution: { ...DEFAULT_EXECUTION, subtitle: 'serial' }, stages: ['subtitle'] })));
  });

  it('rejects concurrency outside 1–8 before estimating', async () => {
    const fetch = mockFetch(); const user = userEvent.setup(); mount(<JobModal asset={{ ...asset, subtitle_mode: 'embedded' }} onClose={vi.fn()} />);
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), '9');
    await user.click(screen.getByRole('button', { name: '估算用量' }));
    expect(fetch).not.toHaveBeenCalled();
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).validity.rangeOverflow).toBe(true);
  });

  it('persists defaults and applies them when a new modal opens', async () => {
    let saved: SettingsType | undefined;
    const fetch = mockFetch((url, options) => { if (url === '/api/settings') { saved = JSON.parse(options!.body as string); return saved; } return {}; });
    const user = userEvent.setup(); const app = context(); const settingsView = mount(<Settings />, app);
    await user.click(screen.getByRole('button', { name: '分析默认值' }));
    await user.selectOptions(screen.getByLabelText('台词整理执行方式'), mixed.transcript);
    await user.selectOptions(screen.getByLabelText('字幕识别执行方式'), mixed.subtitle);
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), String(mixed.max_concurrency));
    await user.click(screen.getByRole('button', { name: '保存默认值' }));
    await waitFor(() => expect(app.refresh).toHaveBeenCalledOnce());
    expect(body(fetch, '/api/settings').defaults.execution).toEqual(mixed);
    settingsView.unmount();
    mount(<JobModal asset={asset} onClose={vi.fn()} />, { ...app, bootstrap: { ...app.bootstrap, settings: saved! } });
    expect((screen.getByLabelText('台词整理执行方式') as HTMLSelectElement).value).toBe('serial');
    expect((screen.getByLabelText('视觉分块执行方式') as HTMLSelectElement).value).toBe('parallel');
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).value).toBe('5');
  });

  it('keeps reanalysis estimates current and passes the same execution choices into creation', async () => {
    const library = { scope_id: 'work', revision: 7, profiles: [], assets: [asset] };
    const fetch = mockFetch((url) => url === '/api/keyframe-plans' ? { id: 'plan', status: 'completed', frames: [] } : url.endsWith('/estimate') ? estimate : url.endsWith('/reanalyze') ? { jobs: [{ id: 'job' }], revision: 7 } : library);
    const user = userEvent.setup(); mount(<CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={vi.fn().mockResolvedValue(undefined)} />, context(mixed));
    await user.click(await screen.findByRole('button', { name: '按新版档案重新识别' }));
    expect(screen.queryByLabelText('字幕识别执行方式')).toBeNull();
    await runEstimate(user, '估算重新识别用量', '开始重新识别');
    expect(body(fetch, '/api/jobs/estimate').execution).toEqual(mixed);
    await user.selectOptions(screen.getByLabelText('视觉分块执行方式'), 'serial');
    expect((screen.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement).disabled).toBe(true);
    await runEstimate(user, '估算重新识别用量', '开始重新识别');
    await user.click(screen.getByRole('button', { name: '开始重新识别' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(body(fetch, '/api/assets/one/characters/reanalyze').execution).toEqual(body(fetch, '/api/jobs/estimate').execution);
    expect(body(fetch, '/api/assets/one/characters/reanalyze').execution).toEqual({ ...mixed, preparation: 'serial' });
  });

  it('explains dependent deep analysis as serial without offering unrelated execution controls', async () => {
    const user = userEvent.setup(); mount(<DeepenSceneButton assetId={asset.id} chunkId="chunk" title="场景" />);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    expect(screen.getByText('深入理解与连续理解更新依赖前序记忆，始终串行。')).toBeTruthy();
    expect(screen.queryByLabelText('并发请求上限')).toBeNull();
    expect(screen.queryByLabelText('台词整理执行方式')).toBeNull();
  });

  it('shows persisted modes and old serial fallback while retrying with the saved settings', async () => {
    const fetch = mockFetch(); const user = userEvent.setup(); const app = context();
    const base = { asset_id: asset.id, type: 'analysis', status: 'partial', progress: 0.5, completed: 1, total: 2, created_at: asset.created_at, analysis_strategy: 'scene-chunks-v1', stages: ['vision' as const, 'subtitle' as const] };
    app.jobs = [{ ...base, id: 'saved', execution: mixed, stage_metrics: { preparation: { request_count: 3, wall_elapsed_ms: 400, elapsed_ms: 1000 } } }, { ...base, id: 'old' }];
    mount(<Jobs />, app);
    const snapshots = screen.getAllByLabelText('任务创建时的执行方式');
    expect(within(snapshots[0]).getByText('台词整理：串行 · 视觉分块：并行 · 字幕识别：串行 · 并发请求上限 5')).toBeTruthy();
    expect(within(snapshots[1]).getByText('台词整理：串行 · 视觉分块：串行 · 字幕识别：串行')).toBeTruthy();
    expect(within(snapshots[1]).getByText('历史任务未设置执行方式，沿用串行。')).toBeTruthy();
    expect(within(screen.getByLabelText('分阶段用量')).getByText('阶段耗时 0.4 秒')).toBeTruthy();
    expect(within(screen.getByLabelText('分阶段用量')).getByText('请求累计 1.0 秒')).toBeTruthy();
    await user.click(screen.getAllByRole('button', { name: '重试' })[0]);
    expect(within(screen.getByRole('dialog')).queryByLabelText('并发请求上限')).toBeNull();
    await user.click(screen.getByRole('button', { name: '继续任务' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(body(fetch, '/api/jobs/saved/retry')).toEqual({ max_requests: 1000, max_cost: null });
  });
});
