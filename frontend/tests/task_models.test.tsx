import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type CodexModelCatalog, type Estimate, type Profile } from '../src/types';
import JobModal from '../src/components/JobModal';
import CharacterProfiles from '../src/components/CharacterProfiles';
import { DeepenSceneButton } from '../src/components/SceneAnalysis';
import Jobs from '../src/components/Jobs';

const asset: Asset = { id: 'one', work_id: 'work', title: '测试影片', kind: 'movie', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const codex: Profile = { id: 'codex', name: '我的 Codex', provider_type: 'codex_cli', base_url: '', model: 'test-deep', capabilities: ['preparation', 'vision', 'subtitle'], secret_mode: 'none' };
const apiProfile: Profile = { ...codex, id: 'api', name: '我的 API', provider_type: 'openai_compatible', model: 'api-model', base_url: 'https://example.invalid/v1' };
const catalog: CodexModelCatalog = { models: [{ id: 'test-deep', name: '测试深入模型', reasoning_efforts: ['medium', 'high', 'xhigh'], default_reasoning_effort: 'medium' }, { id: 'test-fast', name: '测试快速模型', reasoning_efforts: ['low', 'medium'], default_reasoning_effort: 'low' }], default_model: 'test-deep', message: '已读取本机 Codex 模型目录缓存。' };
const estimate: Estimate = { duration_ms: 12000, estimated_frames: 8, estimated_requests: 2, cost_estimate: null, warnings: [] };
function context(profiles = [codex]): AppContextValue {
  return { bootstrap: { library_path: '/tmp/task-model-ui-tests', settings: { defaults: DEFAULTS, bindings: { preparation: profiles[0].id, vision: profiles[0].id, subtitle: profiles[0].id } }, provider_profiles: profiles, stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' }, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function mount(ui: ReactNode, app = context()) { return render(<AppContext.Provider value={app}>{ui}</AppContext.Provider>); }
function mockFetch(handler?: (url: string, options?: RequestInit) => unknown) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const value = handler ? handler(url, options) : url === '/api/codex/models' ? catalog : url === '/api/keyframe-plans' ? { id: 'plan', status: 'completed', frames: [], frame_count: 8 } : url.endsWith('/estimate') ? estimate : { id: 'job' };
    return value instanceof Response ? value : new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
function body(fetch: ReturnType<typeof mockFetch>, path: string) {
  return JSON.parse(fetch.mock.calls.filter((entry) => entry[0] === path).at(-1)![1]!.body as string);
}
async function scan(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
  await screen.findByRole('button', { name: '重新扫描关键帧' });
}
async function estimateJob(user: ReturnType<typeof userEvent.setup>, startName = '开始分析') {
  await user.click(screen.getByRole('button', { name: '估算用量' }));
  await waitFor(() => expect((screen.getByRole('button', { name: startName }) as HTMLButtonElement).disabled).toBe(false));
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Task model selection by source', () => {
  it('loads supported efforts, separates stages and invalidates the estimate when either selection changes', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(); const app = context(); const onClose = vi.fn();
    mount(<JobModal asset={asset} onClose={onClose} />, app);
    await screen.findByText(catalog.message!);
    expect(screen.getByText(/模型列表来自本机 Codex 缓存/)).toBeTruthy();
    await user.selectOptions(screen.getByLabelText('分块与简要索引模型'), 'model:test-fast');
    await user.selectOptions(screen.getByLabelText('分块与简要索引思考等级'), 'low');
    await user.selectOptions(screen.getByLabelText('画面理解模型'), 'model:test-deep');
    await user.selectOptions(screen.getByLabelText('画面理解思考等级'), 'xhigh');
    await scan(user); await estimateJob(user);
    const firstOverrides = body(fetch, '/api/jobs/estimate').model_overrides;
    expect(firstOverrides).toEqual({ preparation: { provider_type: 'codex_cli', profile_id: codex.id, model: 'test-fast', reasoning_effort: 'low' }, vision: { provider_type: 'codex_cli', profile_id: codex.id, model: 'test-deep', reasoning_effort: 'xhigh' } });
    await user.selectOptions(screen.getByLabelText('画面理解模型'), 'model:test-fast');
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByLabelText('画面理解思考等级') as HTMLSelectElement).value).toBe('');
    expect(screen.getByText('test-fast · 仅发送内容关键帧')).toBeTruthy();
    expect(screen.getByText(/台词整理 → 视觉分块（test-fast）/)).toBeTruthy();
    expect(within(screen.getByLabelText('画面理解思考等级')).queryByRole('option', { name: /xhigh/ })).toBeNull();
    await estimateJob(user);
    await user.selectOptions(screen.getByLabelText('画面理解思考等级'), 'medium');
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    await estimateJob(user); await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(body(fetch, '/api/jobs').model_overrides).toEqual(body(fetch, '/api/jobs/estimate').model_overrides);
    expect(body(fetch, '/api/jobs').model_overrides.vision).toEqual({ provider_type: 'codex_cli', profile_id: codex.id, model: 'test-fast', reasoning_effort: 'medium' });
    expect(app.bootstrap.provider_profiles[0].model).toBe('test-deep');
    expect(fetch.mock.calls.some(([url]) => url === '/api/settings' || url === '/api/profiles')).toBe(false);
  });

  it('does not override untouched Codex connections and explicitly supports the built-in default', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(); mount(<JobModal asset={asset} onClose={vi.fn()} />);
    await screen.findByText(catalog.message!); await scan(user); await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate')).not.toHaveProperty('model_overrides');
    await user.selectOptions(screen.getByLabelText('画面理解模型'), 'model:');
    expect(screen.getByText('Codex 内置默认 · 仅发送内容关键帧')).toBeTruthy();
    await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate').model_overrides).toEqual({ vision: { provider_type: 'codex_cli', profile_id: codex.id, model: '', reasoning_effort: null } });
    await user.selectOptions(screen.getByLabelText('画面理解模型'), 'inherit');
    await user.selectOptions(screen.getByLabelText('画面理解思考等级'), 'high');
    await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate').model_overrides).toEqual({ vision: { provider_type: 'codex_cli', profile_id: codex.id, reasoning_effort: 'high' } });
    await user.selectOptions(screen.getByLabelText('画面理解思考等级'), '');
    await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate')).not.toHaveProperty('model_overrides');
  });

  it('keeps API routes unchanged without fetching a Codex catalog or showing selection controls', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(); mount(<JobModal asset={asset} onClose={vi.fn()} />, context([apiProfile]));
    expect(screen.queryByLabelText('画面理解模型')).toBeNull();
    expect(screen.queryByLabelText('画面理解思考等级')).toBeNull();
    expect(screen.getAllByText('API · 我的 API')).toHaveLength(2);
    expect(fetch).not.toHaveBeenCalled();
    await scan(user); await estimateJob(user); await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/jobs')).toBe(true));
    expect(body(fetch, '/api/jobs')).not.toHaveProperty('model_overrides');
    expect(fetch.mock.calls.some(([url]) => url === '/api/codex/models')).toBe(false);
  });

  it('shows a usable default path on catalog failure and can retry', async () => {
    let catalogCalls = 0;
    const user = userEvent.setup(); const fetch = mockFetch((url) => url === '/api/codex/models' ? ++catalogCalls === 1 ? new Response(JSON.stringify({ detail: 'Codex 缓存不存在' }), { status: 503 }) : catalog : url === '/api/keyframe-plans' ? { id: 'plan', status: 'completed', frames: [] } : estimate);
    mount(<JobModal asset={asset} onClose={vi.fn()} />);
    await screen.findByText(/无法读取 Codex 模型目录：Codex 缓存不存在/);
    expect((screen.getByLabelText(/^画面理解思考等级/) as HTMLSelectElement).disabled).toBe(true);
    await user.selectOptions(screen.getByLabelText('画面理解模型'), 'model:');
    await scan(user); await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate').model_overrides.vision.model).toBe('');
    await user.click(screen.getByRole('button', { name: '重试读取模型' }));
    await screen.findByText(catalog.message!);
    expect((screen.getByLabelText('画面理解思考等级') as HTMLSelectElement).disabled).toBe(false);
    expect(catalogCalls).toBe(2);
  });

  it('applies per-stage overrides only to enabled stages and relevant batch items', async () => {
    const user = userEvent.setup(); const embedded = { ...asset, id: 'embedded', subtitle_mode: 'embedded' as const };
    const fetch = mockFetch((url, options) => url === '/api/codex/models' ? catalog : url === '/api/keyframe-plans' ? { id: `plan-${JSON.parse(options!.body as string).asset_id}`, status: 'completed', frames: [] } : url.endsWith('/estimate') ? { ...estimate, items: [asset, embedded].map((entry) => ({ ...estimate, asset_id: entry.id })) } : {});
    mount(<JobModal assets={[asset, embedded]} onClose={vi.fn()} />);
    await screen.findByText(catalog.message!);
    await user.selectOptions(screen.getByLabelText('画面字幕 OCR模型'), 'model:test-fast');
    await user.selectOptions(screen.getByLabelText('画面字幕 OCR思考等级'), 'low');
    await scan(user); await estimateJob(user);
    const items = body(fetch, '/api/jobs/batch/estimate').items;
    expect(items[0]).not.toHaveProperty('model_overrides');
    expect(items[1].model_overrides).toEqual({ subtitle: { provider_type: 'codex_cli', profile_id: codex.id, model: 'test-fast', reasoning_effort: 'low' } });
    await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/jobs/batch')).toBe(true));
    expect(body(fetch, '/api/jobs/batch').items).toEqual(items);
  });

  it('reloads an empty catalog and clears an effort that a refreshed catalog no longer supports', async () => {
    let catalogCalls = 0;
    const user = userEvent.setup();
    const fetch = mockFetch((url) => url === '/api/codex/models' ? ++catalogCalls === 1 ? { models: [], message: '尚未生成本机缓存' } : catalogCalls === 2 ? catalog : catalogCalls === 3 ? { models: [] } : { ...catalog, models: [{ ...catalog.models[0], reasoning_efforts: ['medium'] }] } : url === '/api/keyframe-plans' ? { id: 'plan', status: 'completed', frames: [] } : estimate);
    mount(<JobModal asset={asset} onClose={vi.fn()} />);
    await screen.findByText('当前目录没有可选模型，可沿用连接或使用 Codex 内置默认。');
    await user.click(screen.getByRole('button', { name: '重新读取模型' }));
    await screen.findByText(catalog.message!);
    await user.selectOptions(screen.getByLabelText('画面理解思考等级'), 'high');
    await scan(user); await estimateJob(user);
    await user.click(screen.getByRole('button', { name: '重新读取模型' }));
    await screen.findByText('当前目录没有可选模型，可沿用连接或使用 Codex 内置默认。');
    const unavailable = screen.getByLabelText(/^画面理解思考等级/) as HTMLSelectElement;
    expect(unavailable.value).toBe('high');
    expect(unavailable.disabled).toBe(true);
    expect(within(unavailable).getByRole('option', { name: /已选等级，目录暂不可验证/ }).textContent).toContain('high');
    await user.click(screen.getByRole('button', { name: '重新读取模型' }));
    await waitFor(() => expect((screen.getByLabelText('画面理解思考等级') as HTMLSelectElement).value).toBe(''));
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate').model_overrides.vision.reasoning_effort).toBeNull();
    expect(catalogCalls).toBe(4);
  });

  it('removes deselected stage overrides and keeps only subtitle choices in the request', async () => {
    const user = userEvent.setup(); const fetch = mockFetch();
    mount(<JobModal asset={{ ...asset, subtitle_mode: 'embedded' }} onClose={vi.fn()} />);
    await screen.findByText(catalog.message!);
    await user.selectOptions(screen.getByLabelText('分块与简要索引模型'), 'model:test-deep');
    await user.selectOptions(screen.getByLabelText('画面理解思考等级'), 'high');
    await user.selectOptions(screen.getByLabelText('画面字幕 OCR模型'), 'model:test-fast');
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    expect(screen.queryByLabelText('画面理解模型')).toBeNull();
    await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate')).toMatchObject({ stages: ['subtitle'], model_overrides: { subtitle: { provider_type: 'codex_cli', profile_id: codex.id, model: 'test-fast', reasoning_effort: null } } });
    expect(Object.keys(body(fetch, '/api/jobs/estimate').model_overrides)).toEqual(['subtitle']);
  });

  it('invalidates an untouched estimate when a connection model changes and does not guess default model efforts', async () => {
    const user = userEvent.setup(); const app = context();
    const fetch = mockFetch((url) => url === '/api/codex/models' ? { ...catalog, default_model: null } : url === '/api/keyframe-plans' ? { id: 'plan', status: 'completed', frames: [] } : estimate);
    const view = mount(<JobModal asset={asset} onClose={vi.fn()} />, app);
    await screen.findByText(catalog.message!); await scan(user); await estimateJob(user);
    const updated = { ...app, bootstrap: { ...app.bootstrap, provider_profiles: [{ ...codex, model: '' }] } };
    view.rerender(<AppContext.Provider value={updated}><JobModal asset={asset} onClose={vi.fn()} /></AppContext.Provider>);
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByLabelText(/^画面理解思考等级/) as HTMLSelectElement).disabled).toBe(true);
    await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate')).not.toHaveProperty('model_overrides');
  });

  it('uses each stage bound connection and drops overrides when a binding changes to API', async () => {
    const secondCodex = { ...codex, id: 'codex-two', name: '分块连接', model: 'test-fast' };
    const app = context([codex, secondCodex, apiProfile]); app.bootstrap.settings.bindings.preparation = secondCodex.id;
    const user = userEvent.setup(); const fetch = mockFetch(); const view = mount(<JobModal asset={asset} onClose={vi.fn()} />, app);
    await screen.findByText(catalog.message!);
    expect(within(screen.getByLabelText('分块与简要索引模型设置')).getByText('Codex CLI · ChatGPT 订阅 · 分块连接')).toBeTruthy();
    await user.selectOptions(screen.getByLabelText('分块与简要索引思考等级'), 'low');
    await user.selectOptions(screen.getByLabelText('画面理解思考等级'), 'high');
    await scan(user); await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate').model_overrides.preparation.profile_id).toBe(secondCodex.id);
    const updated = { ...app, bootstrap: { ...app.bootstrap, settings: { ...app.bootstrap.settings, bindings: { ...app.bootstrap.settings.bindings, vision: apiProfile.id } } } };
    view.rerender(<AppContext.Provider value={updated}><JobModal asset={asset} onClose={vi.fn()} /></AppContext.Provider>);
    expect(screen.queryByLabelText('画面理解思考等级')).toBeNull();
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    await estimateJob(user);
    expect(body(fetch, '/api/jobs/estimate').model_overrides).toEqual({ preparation: { provider_type: 'codex_cli', profile_id: secondCodex.id, reasoning_effort: 'low' } });
  });
});

describe('All analysis entry points keep task selections', () => {
  it('passes the chosen model and effort through scene deepening estimate and start', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(); mount(<DeepenSceneButton assetId={asset.id} chunkId="chunk" title="测试片段" />);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    await screen.findByText(catalog.message!);
    expect(screen.queryByLabelText('分块与简要索引模型')).toBeNull();
    await user.selectOptions(screen.getByLabelText('画面理解模型'), 'model:test-fast');
    expect(screen.getByText(/模型：test-fast。/)).toBeTruthy();
    await estimateJob(user, '开始深入分析');
    await user.selectOptions(screen.getByLabelText('画面理解思考等级'), 'low');
    expect((screen.getByRole('button', { name: '开始深入分析' }) as HTMLButtonElement).disabled).toBe(true);
    await estimateJob(user, '开始深入分析');
    await user.click(screen.getByRole('button', { name: '开始深入分析' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(body(fetch, '/api/assets/one/chunks/chunk/deepen')).toEqual(body(fetch, '/api/assets/one/chunks/chunk/deepen/estimate'));
    expect(body(fetch, '/api/assets/one/chunks/chunk/deepen').model_overrides).toEqual({ vision: { provider_type: 'codex_cli', profile_id: codex.id, model: 'test-fast', reasoning_effort: 'low' } });
  });

  it('applies Codex preparation choices and budget rules to character reanalysis with API vision', async () => {
    const user = userEvent.setup();
    const library = { scope_id: 'work', revision: 7, profiles: [{ id: 'person', name: '小夏', aliases: [], description: '', notes: '', appearances: [], appearance_count: 0 }], assets: [asset] };
    const fetch = mockFetch((url) => url === '/api/codex/models' ? catalog : url === '/api/keyframe-plans' ? { id: 'plan', status: 'completed', frames: [] } : url.endsWith('/estimate') ? estimate : url.endsWith('/reanalyze') ? { jobs: [{ id: 'job' }], revision: 7 } : library);
    const app = context([codex, apiProfile]); app.bootstrap.settings.bindings.vision = apiProfile.id; app.bootstrap.settings.defaults = { ...DEFAULTS, max_cost: 10 };
    mount(<CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={vi.fn().mockResolvedValue(undefined)} />, app);
    await user.click(await screen.findByRole('button', { name: '按新版档案重新识别' }));
    await screen.findByText(catalog.message!);
    expect(screen.queryByLabelText('画面理解模型')).toBeNull();
    expect((screen.getByLabelText(/^每个视频费用上限/) as HTMLInputElement).disabled).toBe(true);
    await user.selectOptions(screen.getByLabelText('分块与简要索引模型'), 'model:test-fast');
    await user.selectOptions(screen.getByLabelText('分块与简要索引思考等级'), 'medium');
    await user.click(screen.getByRole('button', { name: '估算重新识别用量' }));
    await waitFor(() => expect((screen.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement).disabled).toBe(false));
    await user.selectOptions(screen.getByLabelText('分块与简要索引思考等级'), 'low');
    expect((screen.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: '估算重新识别用量' }));
    await waitFor(() => expect((screen.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement).disabled).toBe(false));
    await user.click(screen.getByRole('button', { name: '开始重新识别' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(body(fetch, '/api/assets/one/characters/reanalyze')).toMatchObject({ max_cost: null, expected_revision: 7, model_overrides: body(fetch, '/api/jobs/estimate').model_overrides });
    expect(body(fetch, '/api/jobs/estimate')).toMatchObject({ analysis_strategy: 'scene-chunks-v1', max_cost: null, model_overrides: { preparation: { provider_type: 'codex_cli', profile_id: codex.id, model: 'test-fast', reasoning_effort: 'low' } } });
  });

  it('displays the saved selection and retries without allowing a model change', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(); const app = context();
    app.jobs = [{ id: 'saved', asset_id: asset.id, type: 'analysis', status: 'partial', progress: 0.5, completed: 1, total: 2, created_at: asset.created_at, task_models: { vision: { provider_type: 'codex_cli', profile_id: codex.id, model: 'test-deep', reasoning_effort: 'high' } } }];
    mount(<Jobs />, app);
    await user.click(screen.getByText('任务创建时的模型与思考等级'));
    expect(screen.getByText('深入理解 · Codex CLI · test-deep · 思考等级：high')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: '重试' }));
    expect(screen.getByText('继续使用各任务创建时的模型与思考等级。')).toBeTruthy();
    expect(within(screen.getByRole('dialog')).queryByRole('combobox')).toBeNull();
    await user.click(screen.getByRole('button', { name: '继续任务' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(body(fetch, '/api/jobs/saved/retry')).toEqual({ max_requests: 1000, max_cost: null });
    expect(fetch.mock.calls.some(([url]) => url === '/api/codex/models')).toBe(false);
  });

  it('respects a saved content-keyframe reanalysis strategy without overriding the unused preparation stage', async () => {
    const user = userEvent.setup();
    const library = { scope_id: 'work', revision: 7, profiles: [{ id: 'person', name: '小夏', aliases: [], description: '', notes: '', appearances: [], appearance_count: 0 }], assets: [asset] };
    const fetch = mockFetch((url) => url === '/api/keyframe-plans' ? { id: 'plan', status: 'completed', frames: [] } : url.endsWith('/estimate') ? estimate : url.endsWith('/reanalyze') ? { jobs: [{ id: 'job' }], revision: 7 } : library);
    const app = context([codex, apiProfile]);
    app.bootstrap.settings.bindings.vision = apiProfile.id;
    app.bootstrap.settings.defaults = { ...DEFAULTS, analysis_strategy: 'content-keyframes-v1', max_cost: 10 };
    mount(<CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={vi.fn().mockResolvedValue(undefined)} />, app);
    await user.click(await screen.findByRole('button', { name: '按新版档案重新识别' }));
    expect(screen.queryByLabelText('分块与简要索引模型设置')).toBeNull();
    expect((screen.getByLabelText(/^每个视频费用上限/) as HTMLInputElement).disabled).toBe(false);
    await user.click(screen.getByRole('button', { name: '估算重新识别用量' }));
    await waitFor(() => expect((screen.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement).disabled).toBe(false));
    await user.click(screen.getByRole('button', { name: '开始重新识别' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(body(fetch, '/api/jobs/estimate')).toMatchObject({ analysis_strategy: 'content-keyframes-v1', max_cost: 10 });
    expect(body(fetch, '/api/jobs/estimate')).not.toHaveProperty('model_overrides');
    expect(body(fetch, '/api/assets/one/characters/reanalyze')).toMatchObject({ max_cost: 10 });
    expect(body(fetch, '/api/assets/one/characters/reanalyze')).not.toHaveProperty('model_overrides');
    expect(fetch.mock.calls.some(([url]) => url === '/api/codex/models')).toBe(false);
  });
});
