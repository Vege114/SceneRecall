import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type Bootstrap, type Job, type Profile } from '../src/types';
import ImportModal from '../src/components/ImportModal';
import JobModal from '../src/components/JobModal';
import SearchPage from '../src/components/Search';
import Settings from '../src/components/Settings';
import Jobs from '../src/components/Jobs';
import App from '../src/App';

const asset: Asset = { id: 'synthetic-asset', work_id: 'synthetic-work', title: '合成测试片段', kind: 'movie', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const profile: Profile = { id: 'vision-test', name: '测试视觉接口', base_url: 'http://127.0.0.1:9000/v1', model: 'local-test-model', capabilities: ['vision'], secret_mode: 'session', has_key: true };
const bootstrap: Bootstrap = { library_path: '/tmp/scenerecall-ui-tests', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: { assets: 1, records: 1, favorites: 0 }, tools: { ffmpeg: true, ffprobe: true }, version: '0.1.0' };

function appContext(overrides: Partial<AppContextValue> = {}): AppContextValue {
  return { bootstrap, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn(), ...overrides };
}
function renderApp(ui: ReactNode, context: AppContextValue = appContext()) {
  return render(<AppContext.Provider value={context}>{ui}</AppContext.Provider>);
}
function mockFetch(handler: (url: string, options?: RequestInit) => unknown | Promise<unknown>) {
  const mock = vi.fn(async (url: string, options?: RequestInit) => new Response(JSON.stringify(await handler(url, options)), { status: 200, headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', mock);
  return mock;
}
afterEach(() => {
  cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals();
  window.history.replaceState({}, '', '/');
});

describe('Import and model prerequisites', () => {
  it('requires a subtitle path before sending an external-subtitle import', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => asset);
    const onClose = vi.fn();
    renderApp(<ImportModal onClose={onClose} />);
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/synthetic.mp4');
    await user.click(screen.getByRole('button', { name: /外挂字幕文件/ }));
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    expect(fetch).not.toHaveBeenCalled();
    expect((screen.getByLabelText(/字幕文件绝对路径/) as HTMLInputElement).validity.valueMissing).toBe(true);
    await user.type(screen.getByLabelText(/字幕文件绝对路径/), '/tmp/synthetic.srt');
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({ video_path: '/tmp/synthetic.mp4', subtitle_path: '/tmp/synthetic.srt', subtitle_mode: 'external', series: '', title: 'synthetic' });
    expect(window.location.hash).toBe('#/asset/synthetic-asset');
  });

  it('blocks embedded import until a subtitle model has been assigned', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => asset);
    renderApp(<ImportModal onClose={vi.fn()} />);
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/synthetic.mp4');
    await user.click(screen.getByRole('button', { name: /画面字幕 OCR/ }));
    expect((screen.getByRole('button', { name: /加入资料库/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/中绑定字幕视觉模型/)).toBeTruthy();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('keeps an existing API key input empty and omits it on unchanged profile updates', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => profile);
    renderApp(<Settings />, appContext({ bootstrap: { ...bootstrap, provider_profiles: [profile] } }));
    await user.click(screen.getByRole('button', { name: /编辑 测试视觉接口/ }));
    const dialog = screen.getByRole('dialog');
    const keyInput = within(dialog).getByLabelText(/API Key/) as HTMLInputElement;
    expect(keyInput.type).toBe('password');
    expect(keyInput.value).toBe('');
    expect(window.localStorage.length).toBe(0);
    await user.click(within(dialog).getByRole('button', { name: /保存连接/ }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    const payload = JSON.parse(fetch.mock.calls[0][1]!.body as string);
    expect(payload.id).toBe(profile.id);
    expect(payload.provider_type).toBe('openai_compatible');
    expect(payload).not.toHaveProperty('api_key');
    expect(payload.capabilities).toEqual(['vision']);
  });
});

describe('Codex CLI connections', () => {
  const codex: Profile = { id: 'codex-test', name: '我的 Codex', provider_type: 'codex_cli', base_url: '', model: '', capabilities: ['vision', 'subtitle', 'query', 'decision', 'answer'], secret_mode: 'none', env_var: null, input_price_per_million: null, output_price_per_million: null };
  const ready = { installed: true, authenticated: true, compatible: true, auth_method: 'chatgpt', message: '已通过 ChatGPT 登录。', version: 'codex-cli test' };

  it.each([
    [{ ...ready, installed: false, authenticated: false, compatible: false, message: '请安装 Codex CLI。' }, '未检测到 Codex CLI'],
    [{ ...ready, authenticated: false, auth_method: null, message: '请执行 codex login。' }, '需要登录 ChatGPT'],
    [{ ...ready, compatible: false, message: '当前版本不支持所需功能，请升级。' }, 'Codex CLI 需要升级'],
  ])('shows installation, login and compatibility prerequisites: %s', async (status, title) => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => status);
    renderApp(<Settings />, appContext({ bootstrap: { ...bootstrap, provider_profiles: [codex] } }));
    await user.click(screen.getByRole('button', { name: '编辑 我的 Codex' }));
    const dialog = screen.getByRole('dialog');
    await within(dialog).findByText(title);
    expect(within(dialog).getByText(status.message)).toBeTruthy();
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/codex/status']);
    expect((within(dialog).getByLabelText(/接入方式/) as HTMLSelectElement).value).toBe('codex_cli');
    expect(within(dialog).queryByLabelText(/API Key/)).toBeNull();
    expect(within(dialog).getByText(/同一系统用户安装 Codex CLI/)).toBeTruthy();
  });

  it('rechecks status after an error without creating a connection', async () => {
    const user = userEvent.setup();
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: '无法检查 CLI 状态' }), { status: 503 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(ready), { status: 200 }));
    vi.stubGlobal('fetch', fetch);
    renderApp(<Settings />);
    await user.click(screen.getByRole('button', { name: '添加连接' }));
    await user.selectOptions(screen.getByLabelText('接入方式'), 'codex_cli');
    await screen.findByText('无法检查 CLI 状态');
    await user.click(screen.getByRole('button', { name: '重新检查' }));
    await screen.findByText('已就绪');
    expect(screen.queryByText('无法检查 CLI 状态')).toBeNull();
    expect(screen.getByText(ready.version)).toBeTruthy();
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/codex/status', '/api/codex/status']);
  });

  it('saves a subscription connection with the default model and holds the saving state until completion', async () => {
    const user = userEvent.setup();
    let finishSave!: (value: unknown) => void;
    const pendingSave = new Promise((resolve) => { finishSave = resolve; });
    const fetch = mockFetch((url) => url === '/api/codex/status' ? ready : pendingSave);
    const context = appContext();
    renderApp(<Settings />, context);
    await user.click(screen.getByRole('button', { name: '添加连接' }));
    await user.selectOptions(screen.getByLabelText('接入方式'), 'codex_cli');
    await screen.findByText('已就绪');
    const dialog = screen.getByRole('dialog');
    await user.type(within(dialog).getByLabelText('连接名称'), '我的订阅连接');
    expect((within(dialog).getByLabelText(/模型 ID/) as HTMLInputElement).required).toBe(false);
    expect(within(dialog).getByText('留空使用 Codex CLI 内置默认模型；自选模型请填写 ID。')).toBeTruthy();
    expect(within(dialog).queryByLabelText(/Base URL/)).toBeNull();
    expect(within(dialog).queryByLabelText(/凭据保存方式/)).toBeNull();
    expect(within(dialog).queryByLabelText(/输入单价/)).toBeNull();
    expect((within(dialog).getByLabelText('语义向量') as HTMLInputElement).disabled).toBe(true);
    await user.click(within(dialog).getByRole('button', { name: '保存连接' }));
    expect((within(dialog).getByRole('button', { name: '保存中' }) as HTMLButtonElement).disabled).toBe(true);
    expect(context.refresh).not.toHaveBeenCalled();
    const saved = JSON.parse(fetch.mock.calls.find((call) => call[0] === '/api/profiles')![1]!.body as string);
    expect(saved).toMatchObject({ name: '我的订阅连接', provider_type: 'codex_cli', base_url: '', model: '', capabilities: ['vision'], secret_mode: 'none', env_var: null, input_price_per_million: null, output_price_per_million: null });
    expect(saved).not.toHaveProperty('api_key');
    finishSave(codex);
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(context.refresh).toHaveBeenCalledOnce();
    expect(context.notify).toHaveBeenCalledWith('模型连接已保存');
  });

  it('clears API credentials, pricing and embedding when converting an existing profile', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url) => url === '/api/codex/status' ? ready : codex);
    const existing: Profile = { ...profile, capabilities: ['vision', 'embedding', 'query'], secret_mode: 'env', env_var: 'OLD_API_KEY', input_price_per_million: 2, output_price_per_million: 8 };
    renderApp(<Settings />, appContext({ bootstrap: { ...bootstrap, provider_profiles: [existing] } }));
    await user.click(screen.getByRole('button', { name: `编辑 ${profile.name}` }));
    await user.selectOptions(screen.getByLabelText('凭据保存方式'), 'session');
    await user.type(screen.getByLabelText(/API Key/), 'synthetic-api-key');
    await user.selectOptions(screen.getByLabelText('接入方式'), 'codex_cli');
    await screen.findByText('已就绪');
    expect((screen.getByLabelText('语义向量') as HTMLInputElement).checked).toBe(false);
    expect((screen.getByLabelText(/模型 ID/) as HTMLInputElement).value).toBe('');
    await user.type(screen.getByLabelText(/模型 ID/), 'custom-codex-model');
    await user.click(screen.getByRole('button', { name: '保存连接' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    const saved = JSON.parse(fetch.mock.calls.find((call) => call[0] === '/api/profiles')![1]!.body as string);
    expect(saved).toMatchObject({ id: profile.id, provider_type: 'codex_cli', base_url: '', model: 'custom-codex-model', capabilities: ['vision', 'query'], secret_mode: 'none', env_var: null, input_price_per_million: null, output_price_per_million: null });
    expect(saved).not.toHaveProperty('api_key');
  });

  it('restores required API fields without retaining credentials when switching back', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => ready);
    renderApp(<Settings />);
    await user.click(screen.getByRole('button', { name: '添加连接' }));
    await user.type(screen.getByLabelText('连接名称'), '切换测试');
    await user.type(screen.getByLabelText(/Base URL/), 'https://api.example.com/v1');
    await user.type(screen.getByLabelText(/模型 ID/), 'api-model');
    await user.type(screen.getByLabelText(/API Key/), 'synthetic-key');
    await user.selectOptions(screen.getByLabelText('接入方式'), 'codex_cli');
    await screen.findByText('已就绪');
    await user.selectOptions(screen.getByLabelText('接入方式'), 'openai_compatible');
    const url = screen.getByLabelText(/Base URL/) as HTMLInputElement;
    const model = screen.getByLabelText(/模型 ID/) as HTMLInputElement;
    expect(url.required).toBe(true); expect(url.value).toBe('');
    expect(model.required).toBe(true); expect(model.value).toBe('');
    expect((screen.getByLabelText(/API Key/) as HTMLInputElement).value).toBe('');
    expect((screen.getByLabelText('凭据保存方式') as HTMLSelectElement).value).toBe('session');
    expect((screen.getByLabelText('语义向量') as HTMLInputElement).disabled).toBe(false);
    await user.click(screen.getByRole('button', { name: '保存连接' }));
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/codex/status']);
  });

  it('binds and tests CLI capabilities through the existing workflow', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url, options) => url === '/api/settings' ? JSON.parse(options!.body as string) : { ok: true, message: 'CLI 查询理解测试通过' });
    renderApp(<Settings />, appContext({ bootstrap: { ...bootstrap, provider_profiles: [codex] } }));
    expect(screen.getByText('本地 Codex CLI · ChatGPT 订阅')).toBeTruthy();
    expect(screen.getByText('订阅额度 · 费用未知')).toBeTruthy();
    expect(within(screen.getByLabelText('分配语义向量模型')).queryByRole('option', { name: /我的 Codex/ })).toBeNull();
    await user.selectOptions(screen.getByLabelText('分配查询理解模型'), codex.id);
    await user.click(screen.getByRole('button', { name: '保存分配' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string).bindings).toEqual({ query: codex.id });
    await user.selectOptions(screen.getByLabelText('测试 我的 Codex 的能力'), 'query');
    await user.click(screen.getByRole('button', { name: '连接测试' }));
    await screen.findByText('CLI 查询理解测试通过');
    expect(fetch.mock.calls[1][0]).toBe('/api/profiles/codex-test/test');
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string)).toEqual({ capability: 'query' });
  });

  it('removes a stale embedding assignment after its profile changes to CLI', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((_url, options) => JSON.parse(options!.body as string));
    const context = appContext({ bootstrap: { ...bootstrap, provider_profiles: [{ ...profile, capabilities: ['vision', 'embedding'] }], settings: { ...bootstrap.settings, bindings: { vision: profile.id, embedding: profile.id } } } });
    const view = renderApp(<Settings />, context);
    expect((screen.getByLabelText('分配语义向量模型') as HTMLSelectElement).value).toBe(profile.id);
    view.rerender(<AppContext.Provider value={{ ...context, bootstrap: { ...context.bootstrap, provider_profiles: [{ ...codex, id: profile.id }] } }}><Settings /></AppContext.Provider>);
    await waitFor(() => expect((screen.getByLabelText('分配语义向量模型') as HTMLSelectElement).value).toBe(''));
    await user.click(screen.getByRole('button', { name: '保存分配' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string).bindings).toEqual({ vision: profile.id, embedding: null });
  });

  it('requires unbinding embedding before converting its API connection to CLI', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => ready);
    const context = appContext({ bootstrap: { ...bootstrap, provider_profiles: [{ ...profile, capabilities: ['vision', 'embedding'] }], settings: { ...bootstrap.settings, bindings: { embedding: profile.id } } } });
    renderApp(<Settings />, context);
    await user.click(screen.getByRole('button', { name: `编辑 ${profile.name}` }));
    await user.selectOptions(screen.getByLabelText('接入方式'), 'codex_cli');
    await screen.findByText('已就绪');
    expect(screen.getByText(/请先在能力分配中解除「语义向量」的绑定并保存/)).toBeTruthy();
    expect((screen.getByRole('button', { name: '保存连接' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: '关闭' }));
    expect((screen.getByLabelText('分配语义向量模型') as HTMLSelectElement).value).toBe(profile.id);
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/codex/status']);
  });

  it('uses request budgets for CLI jobs and explicitly disables an inherited dollar limit', async () => {
    const user = userEvent.setup();
    const estimate = { duration_ms: 12000, estimated_frames: 8, estimated_requests: 2, cost_estimate: null, warnings: [] };
    const fetch = mockFetch((url) => url.endsWith('/estimate') ? estimate : { id: 'codex-job' });
    const onClose = vi.fn();
    renderApp(<JobModal asset={asset} onClose={onClose} />, appContext({ bootstrap: { ...bootstrap, provider_profiles: [codex], settings: { bindings: { vision: codex.id }, defaults: { ...DEFAULTS, max_cost: 10 } } } }));
    expect(screen.getByText(/Codex 默认模型 · 发送采样画面/)).toBeTruthy();
    expect(screen.queryByText(/尚未分配模型/)).toBeNull();
    expect((screen.getByLabelText(/费用上限（USD/) as HTMLInputElement).disabled).toBe(true);
    expect(screen.getByText(/金额预算（包括默认费用上限）已停用/)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: '估算用量' }));
    await screen.findByText('重新估算');
    await user.click(screen.getByRole('button', { name: /开始分析/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(fetch.mock.calls.map((call) => JSON.parse(call[1]!.body as string).max_cost)).toEqual([null, null]);
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string).max_requests).toBe(DEFAULTS.max_requests);
  });
});

describe('Analysis budgeting and independent stages', () => {
  const estimate = { duration_ms: 12000, estimated_frames: 8, estimated_requests: 2, cost_estimate: null, warnings: [] };
  it('does not start a job with a missing model binding, even after a returned estimate', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => estimate);
    renderApp(<JobModal asset={asset} onClose={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /估算用量/ }));
    await screen.findByText(/重新估算/);
    expect((screen.getByRole('button', { name: /开始分析/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/jobs/estimate']);
  });

  it('requires re-estimation after a sampling change and submits only the selected stage', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url) => url.endsWith('/estimate') ? estimate : { id: 'synthetic-job', status: 'queued' });
    const onClose = vi.fn();
    const context = appContext({ bootstrap: { ...bootstrap, provider_profiles: [profile], settings: { ...bootstrap.settings, bindings: { vision: profile.id } } } });
    renderApp(<JobModal asset={asset} onClose={onClose} />, context);
    const startButton = screen.getByRole('button', { name: /开始分析/ }) as HTMLButtonElement;
    expect(startButton.disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: /估算用量/ }));
    await waitFor(() => expect(startButton.disabled).toBe(false));
    const requests = screen.getByLabelText(/最多模型请求数/);
    await user.clear(requests); await user.type(requests, '50');
    expect(startButton.disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: /估算用量/ }));
    await waitFor(() => expect(startButton.disabled).toBe(false));
    await user.click(startButton);
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    const jobCall = fetch.mock.calls.find((call) => call[0] === '/api/jobs');
    expect(jobCall).toBeTruthy();
    expect(JSON.parse(jobCall![1]!.body as string)).toMatchObject({ asset_id: asset.id, stages: ['vision'], max_requests: 50, max_cost: null });
  });

  it('allows retrying partial jobs with revised cumulative budgets', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => ({}));
    const job: Job = { id: 'partial-job', type: 'analysis', status: 'partial', progress: 0.5, completed: 1, total: 2, asset_id: asset.id, request_count: 2, created_at: asset.created_at, budget: { max_requests: 2, max_cost: 1 } };
    renderApp(<Jobs />, appContext({ jobs: [job] }));
    expect(screen.getByText('部分完成')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: /重试/ }));
    const dialog = screen.getByRole('dialog');
    const requests = within(dialog).getByLabelText(/累计请求次数上限/);
    await user.clear(requests); await user.type(requests, '20');
    expect((within(dialog).getByLabelText(/累计费用上限/) as HTMLInputElement).value).toBe('1');
    await user.clear(within(dialog).getByLabelText(/累计费用上限/));
    await user.click(within(dialog).getByRole('button', { name: /继续任务/ }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetch.mock.calls[0][0]).toBe('/api/jobs/partial-job/retry');
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toEqual({ max_requests: 20, max_cost: null });
  });
});

describe('Evidence retrieval and session authentication', () => {
  it('shows degraded search with playable evidence and saves its source record', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url) => url === '/api/search' ? { results: [{ id: 'cue-result', record_id: 'cue-source', asset_id: asset.id, title: asset.title, kind: 'subtitle', start_ms: 3500, end_ms: 5700, text: '下一站，我们再见。', review_status: 'reviewed', match_type: 'full', match_reason: '原字幕逐字匹配' }], degraded: ['语义索引尚未构建，已使用字面检索'] } : {});
    renderApp(<SearchPage />);
    await user.type(screen.getByRole('textbox', { name: /用自然语言搜索/ }), '下一站');
    await user.click(screen.getByRole('button', { name: /^检索/ }));
    await screen.findByText('下一站，我们再见。');
    expect(screen.getByText(/语义索引尚未构建/)).toBeTruthy();
    expect(screen.getByRole('link', { name: /查看原片/ }).getAttribute('href')).toBe('#/asset/synthetic-asset?t=3500');
    await user.click(screen.getByRole('button', { name: /收藏片段/ }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    expect(fetch.mock.calls[1][0]).toBe('/api/assets/synthetic-asset/annotations');
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string)).toEqual({ record_id: 'cue-source', favorite: true });
    expect(fetch.mock.calls.map((call) => call[0]).some((url) => /jobs|analy|recogn/.test(url))).toBe(false);
  });

  it('exchanges a startup token before reading the library and removes it from the URL', async () => {
    window.history.replaceState({}, '', '/?token=synthetic-local-token#/library');
    const fetch = mockFetch((url) => url === '/api/session' ? { ok: true } : url === '/api/bootstrap' ? bootstrap : []);
    render(<App />);
    await screen.findByRole('heading', { name: '我的资料库' });
    expect(window.location.search).toBe('');
    expect(fetch.mock.calls[0][0]).toBe('/api/session');
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toEqual({ token: 'synthetic-local-token' });
    expect(window.localStorage.length).toBe(0);
    expect(screen.queryByText('synthetic-local-token')).toBeNull();
  });
});
