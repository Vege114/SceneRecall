import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, DEFAULT_EXECUTION, type Asset, type Estimate, type ExecutionOptions, type Profile } from '../src/types';
import JobModal from '../src/components/JobModal';
import Jobs from '../src/components/Jobs';
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
    return value instanceof Response ? value : new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } });
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
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).value).toBe('16');
    expect(screen.getAllByText('连接并发上限 4；本次最多同时 4 个请求')).toHaveLength(2);
    expect(screen.getByText(/各集台词先独立并行整理，再按集理解画面/)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
    await screen.findByRole('button', { name: '重新扫描关键帧' });
    await runEstimate(user);
    expect(body(fetch, '/api/jobs/estimate').execution).toEqual(DEFAULT_EXECUTION);
    await user.selectOptions(screen.getByLabelText('台词整理执行方式'), 'serial');
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.queryByText('重新估算')).toBeNull();
    await runEstimate(user);
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), '5');
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.queryByText('重新估算')).toBeNull();
    expect(screen.getByText('连接并发上限 4；本次最多同时 4 个请求')).toBeTruthy();
    await runEstimate(user);
    const updated = { ...app, bootstrap: { ...app.bootstrap, provider_profiles: [{ ...model, max_concurrency: 2 }] } };
    view.rerender(<AppContext.Provider value={updated}><JobModal asset={asset} onClose={close} /></AppContext.Provider>);
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.queryByText('重新估算')).toBeNull();
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

  it.each(['0', '0.5', '9007199254740992'])('rejects invalid browser concurrency %s before estimating', async (value) => {
    const fetch = mockFetch(); const user = userEvent.setup(); mount(<JobModal asset={{ ...asset, subtitle_mode: 'embedded' }} onClose={vi.fn()} />);
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), value);
    await user.click(screen.getByRole('button', { name: '估算用量' }));
    expect(fetch).not.toHaveBeenCalled();
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).validity.customError).toBe(true);
  });

  it.each([32, 64])('accepts user-selected task concurrency %i and shows the lower connection ceiling', async (value) => {
    const app = context(); app.bootstrap.provider_profiles = [{ ...model, max_concurrency: 32 }];
    const fetch = mockFetch(); const user = userEvent.setup(); mount(<JobModal asset={{ ...asset, subtitle_mode: 'embedded' }} onClose={vi.fn()} />, app);
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    const input = screen.getByLabelText('并发请求上限') as HTMLInputElement;
    expect(input.hasAttribute('max')).toBe(false);
    await user.clear(input); await user.type(input, String(value));
    expect(input.validity.valid).toBe(true);
    expect(screen.getByText('连接并发上限 32；本次最多同时 32 个请求')).toBeTruthy();
    await runEstimate(user); await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/jobs')).toBe(true));
    expect(body(fetch, '/api/jobs').execution.max_concurrency).toBe(value);
    expect(body(fetch, '/api/jobs/estimate').execution.max_concurrency).toBe(value);
  });

  it('preserves saved task defaults and an unchecked preference affects only this task without invalidating the estimate', async () => {
    const old = { ...mixed, max_concurrency: 3 };
    const fetch = mockFetch(); const user = userEvent.setup(); mount(<JobModal asset={{ ...asset, subtitle_mode: 'embedded' }} onClose={vi.fn()} />, context(old));
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).value).toBe('3');
    const checkbox = screen.getByRole('checkbox', { name: '设为默认值' }) as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    await user.selectOptions(screen.getByLabelText('字幕识别执行方式'), 'parallel');
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), '12');
    await runEstimate(user);
    await user.click(checkbox);
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(false);
    await user.click(checkbox);
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(false);
    await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/jobs')).toBe(true));
    expect(body(fetch, '/api/jobs').execution.max_concurrency).toBe(12);
    expect(fetch.mock.calls.some(([url]) => url === '/api/settings')).toBe(false);
  });

  it.each(['single', 'batch'])('saves only the active execution defaults before creating a checked %s task and the next modal inherits them', async (kind) => {
    const app = context({ ...DEFAULT_EXECUTION, subtitle: 'serial', max_concurrency: 3 });
    app.bootstrap.settings.defaults = { ...app.bootstrap.settings.defaults, max_requests: 37, max_cost: 0.25, subtitle_fps: 1 };
    let saved = app.bootstrap.settings;
    const selected = kind === 'batch' ? [asset, { ...asset, id: 'two' }] : [asset];
    const fetch = mockFetch((url, options) => {
      if (url === '/api/settings') { const patch = JSON.parse(options!.body as string); saved = { ...saved, defaults: { ...saved.defaults, execution: { ...saved.defaults.execution!, ...patch.defaults.execution } } }; return saved; }
      if (url === '/api/keyframe-plans') return { id: `plan-${JSON.parse(options!.body as string).asset_id}`, status: 'completed', frames: [] };
      if (url.endsWith('/estimate')) return { ...estimate, items: selected.map((item) => ({ ...estimate, asset_id: item.id })) };
      return {};
    });
    const original = structuredClone(saved);
    vi.mocked(app.refresh).mockImplementation(async () => { app.bootstrap = { ...app.bootstrap, settings: saved }; });
    const user = userEvent.setup(); const close = vi.fn(); const view = mount(<JobModal assets={selected} onClose={close} />, app);
    await user.selectOptions(screen.getByLabelText('台词整理执行方式'), 'serial');
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), '12');
    await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
    await screen.findByRole('button', { name: '重新扫描关键帧' });
    await runEstimate(user);
    await user.click(screen.getByRole('checkbox', { name: '设为默认值' }));
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(false);
    await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    const taskPath = kind === 'batch' ? '/api/jobs/batch' : '/api/jobs';
    expect(body(fetch, '/api/settings')).toEqual({ defaults: { execution: { transcript: 'serial', preparation: 'parallel', max_concurrency: 12 } } });
    expect(fetch.mock.calls.findIndex(([url]) => url === '/api/settings')).toBeLessThan(fetch.mock.calls.findIndex(([url]) => url === taskPath));
    expect(saved).toEqual({ ...original, defaults: { ...original.defaults, execution: { ...original.defaults.execution!, transcript: 'serial', preparation: 'parallel', max_concurrency: 12 } } });
    const created = kind === 'batch' ? body(fetch, taskPath).items : [body(fetch, taskPath)];
    expect(created.map((item: { execution: ExecutionOptions }) => item.execution.max_concurrency)).toEqual(selected.map(() => 12));
    view.unmount();
    mount(<JobModal asset={asset} onClose={vi.fn()} />, app);
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).value).toBe('12');
    expect((screen.getByLabelText('台词整理执行方式') as HTMLSelectElement).value).toBe('serial');
    expect((screen.getByRole('checkbox', { name: '设为默认值' }) as HTMLInputElement).checked).toBe(false);
  });

  it('stops before task creation if saving checked defaults fails and allows an unchecked retry', async () => {
    const fetch = mockFetch((url) => url === '/api/settings' ? new Response(JSON.stringify({ detail: '设置写入失败' }), { status: 503 }) : url.endsWith('/estimate') ? estimate : {});
    const user = userEvent.setup(); mount(<JobModal asset={{ ...asset, subtitle_mode: 'embedded' }} onClose={vi.fn()} />);
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    await runEstimate(user);
    await user.click(screen.getByRole('checkbox', { name: '设为默认值' }));
    await user.click(screen.getByRole('button', { name: '开始分析' }));
    await screen.findByText(/保存默认执行方式失败：设置写入失败。任务尚未创建/);
    expect(fetch.mock.calls.some(([url]) => url === '/api/jobs')).toBe(false);
    await user.click(screen.getByRole('checkbox', { name: '设为默认值' }));
    await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/jobs')).toBe(true));
    expect(fetch.mock.calls.filter(([url]) => url === '/api/settings')).toHaveLength(1);
  });

  it('refreshes saved defaults and reports the partial success when task creation fails after saving', async () => {
    const fetch = mockFetch((url) => url === '/api/jobs' ? new Response(JSON.stringify({ detail: '任务队列不可用' }), { status: 503 }) : url.endsWith('/estimate') ? estimate : {});
    const app = context(); const user = userEvent.setup(); mount(<JobModal asset={{ ...asset, subtitle_mode: 'embedded' }} onClose={vi.fn()} />, app);
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    await runEstimate(user);
    await user.click(screen.getByRole('checkbox', { name: '设为默认值' }));
    await user.click(screen.getByRole('button', { name: '开始分析' }));
    await screen.findByText(/默认执行方式已保存，但任务创建失败：任务队列不可用/);
    expect(app.refresh).toHaveBeenCalledOnce();
    expect(body(fetch, '/api/settings')).toEqual({ defaults: { execution: { subtitle: 'parallel', max_concurrency: 16 } } });
  });

  it('never duplicates a successfully created task when refreshing the queue fails', async () => {
    const fetch = mockFetch(); const app = context(); vi.mocked(app.refresh).mockRejectedValue(new Error('刷新失败'));
    const user = userEvent.setup(); const close = vi.fn(); mount(<JobModal asset={{ ...asset, subtitle_mode: 'embedded' }} onClose={close} />, app);
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    await runEstimate(user); await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    await user.click(screen.getByRole('button', { name: '开始分析' }));
    expect(fetch.mock.calls.filter(([url]) => url === '/api/jobs')).toHaveLength(1);
    expect(app.notify).toHaveBeenCalledWith(expect.stringContaining('任务已创建，但列表刷新失败'), 'error');
  });

  it('keeps an invalid ceiling correctable when the last parallel stage is changed to serial', async () => {
    const fetch = mockFetch(); const user = userEvent.setup(); mount(<JobModal asset={{ ...asset, subtitle_mode: 'embedded' }} onClose={vi.fn()} />);
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    const input = screen.getByLabelText('并发请求上限') as HTMLInputElement;
    await user.clear(input); await user.type(input, '0');
    await user.selectOptions(screen.getByLabelText('字幕识别执行方式'), 'serial');
    expect(input.disabled).toBe(false);
    await user.click(screen.getByRole('button', { name: '估算用量' }));
    expect(fetch).not.toHaveBeenCalled();
    await user.clear(input); await user.type(input, '3');
    expect(input.disabled).toBe(true);
    await runEstimate(user);
    expect(body(fetch, '/api/jobs/estimate').execution.max_concurrency).toBe(3);
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
    expect((screen.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.queryByText('重新估算')).toBeNull();
    await runEstimate(user, '估算重新识别用量', '开始重新识别');
    await user.click(screen.getByRole('button', { name: '开始重新识别' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(body(fetch, '/api/assets/one/characters/reanalyze').execution).toEqual(body(fetch, '/api/jobs/estimate').execution);
    expect(body(fetch, '/api/assets/one/characters/reanalyze').execution).toEqual({ ...mixed, preparation: 'serial' });
  });

  it('saves checked scene reanalysis defaults before the series tasks and reopens with those defaults', async () => {
    const second = { ...asset, id: 'two' };
    const library = { scope_id: 'work', revision: 7, profiles: [], assets: [asset, second] };
    const app = context(mixed); let saved = app.bootstrap.settings;
    const fetch = mockFetch((url, options) => {
      if (url === '/api/settings') { const patch = JSON.parse(options!.body as string); saved = { ...saved, defaults: { ...saved.defaults, execution: { ...saved.defaults.execution!, ...patch.defaults.execution } } }; return saved; }
      if (url === '/api/keyframe-plans') return { id: `plan-${JSON.parse(options!.body as string).asset_id}`, status: 'completed', frames: [] };
      if (url.endsWith('/estimate')) return estimate;
      if (url.endsWith('/reanalyze')) return { jobs: [{ id: 'job-one' }, { id: 'job-two' }], revision: 7 };
      return library;
    });
    vi.mocked(app.refresh).mockImplementation(async () => { app.bootstrap = { ...app.bootstrap, settings: saved }; });
    const user = userEvent.setup(); mount(<CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={vi.fn().mockResolvedValue(undefined)} />, app);
    await user.click(await screen.findByRole('button', { name: '按新版档案重新识别' }));
    await user.selectOptions(screen.getByLabelText('重新识别范围'), 'series');
    await user.selectOptions(screen.getByLabelText('台词整理执行方式'), 'parallel');
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), '12');
    await runEstimate(user, '估算重新识别用量', '开始重新识别');
    await user.click(screen.getByRole('checkbox', { name: '设为默认值' }));
    expect((screen.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement).disabled).toBe(false);
    await user.click(screen.getByRole('button', { name: '开始重新识别' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(body(fetch, '/api/settings')).toEqual({ defaults: { execution: { transcript: 'parallel', preparation: 'parallel', max_concurrency: 12 } } });
    expect(saved.defaults.execution).toEqual({ ...mixed, transcript: 'parallel', max_concurrency: 12 });
    expect(fetch.mock.calls.findIndex(([url]) => url === '/api/settings')).toBeLessThan(fetch.mock.calls.findIndex(([url]) => url.endsWith('/reanalyze')));
    expect(body(fetch, '/api/assets/one/characters/reanalyze')).toMatchObject({ scope: 'series', execution: { ...mixed, transcript: 'parallel', max_concurrency: 12 } });
    await user.click(screen.getByRole('button', { name: '按新版档案重新识别' }));
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).value).toBe('12');
    expect((screen.getByRole('checkbox', { name: '设为默认值' }) as HTMLInputElement).checked).toBe(false);
  });

  it('does not create reanalysis tasks when saving checked defaults fails', async () => {
    const library = { scope_id: 'work', revision: 7, profiles: [], assets: [asset] };
    const fetch = mockFetch((url) => url === '/api/settings' ? new Response(JSON.stringify({ detail: '设置写入失败' }), { status: 503 }) : url === '/api/keyframe-plans' ? { id: 'plan', status: 'completed', frames: [] } : url.endsWith('/estimate') ? estimate : library);
    const user = userEvent.setup(); mount(<CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={vi.fn().mockResolvedValue(undefined)} />);
    await user.click(await screen.findByRole('button', { name: '按新版档案重新识别' }));
    await runEstimate(user, '估算重新识别用量', '开始重新识别');
    await user.click(screen.getByRole('checkbox', { name: '设为默认值' }));
    await user.click(screen.getByRole('button', { name: '开始重新识别' }));
    await screen.findByText(/保存默认执行方式失败：设置写入失败。任务尚未创建/);
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/reanalyze'))).toBe(false);
  });

  it('explains dependent deep analysis as serial without offering unrelated execution controls', async () => {
    const user = userEvent.setup(); mount(<DeepenSceneButton assetId={asset.id} chunkId="chunk" title="场景" />);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    expect(screen.getByText(/深入理解与连续理解更新依赖前序记忆，始终串行。/)).toBeTruthy();
    expect(screen.queryByLabelText('并发请求上限')).toBeNull();
    expect(screen.queryByLabelText('台词整理执行方式')).toBeNull();
    expect(screen.queryByRole('checkbox', { name: '设为默认值' })).toBeNull();
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
