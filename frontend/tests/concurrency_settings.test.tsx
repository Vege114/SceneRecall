import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, DEFAULT_EXECUTION, type Asset, type ExecutionOptions, type Profile } from '../src/types';
import Settings from '../src/components/Settings';
import JobModal from '../src/components/JobModal';

const asset: Asset = { id: 'one', work_id: 'work', title: '测试影片', kind: 'movie', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const shared: Profile = { id: 'api', name: '共享连接', provider_type: 'openai_compatible', base_url: 'https://example.invalid/v1', model: 'test-model', capabilities: ['preparation', 'vision', 'subtitle'], secret_mode: 'env', env_var: 'SYNTHETIC_KEY', timeout_s: 120, max_concurrency: 4 };
const other: Profile = { id: 'other', name: '检索连接', provider_type: 'openai_compatible', base_url: 'https://example.invalid/v1', model: 'embedding-model', capabilities: ['embedding'], secret_mode: 'none', timeout_s: 30, max_concurrency: 2 };
const mixed: ExecutionOptions = { transcript: 'serial', preparation: 'parallel', subtitle: 'serial', max_concurrency: 5 };

function context(execution: ExecutionOptions = DEFAULT_EXECUTION, profiles = [shared, other]): AppContextValue {
  return { bootstrap: { library_path: '/tmp/concurrency-ui-tests', settings: { defaults: { ...DEFAULTS, max_requests: 37, max_cost: 0.25, execution }, bindings: { preparation: shared.id, vision: shared.id, subtitle: shared.id, embedding: other.id } }, provider_profiles: profiles, stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' }, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function mount(app: AppContextValue) { return render(<AppContext.Provider value={app}><Settings /></AppContext.Provider>); }
function mockFetch(app: AppContextValue, failure?: string) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const payload = options?.body ? JSON.parse(options.body as string) : {};
    if (url === '/api/concurrency') {
      if (failure) return new Response(JSON.stringify({ detail: failure }), { status: 503 });
      return new Response(JSON.stringify({ settings: { ...app.bootstrap.settings, defaults: { ...app.bootstrap.settings.defaults, execution: payload.execution } }, provider_profiles: app.bootstrap.provider_profiles.map((profile) => ({ ...profile, max_concurrency: payload.profiles[profile.id] })) }), { status: 200 });
    }
    if (url === '/api/settings') return new Response(JSON.stringify({ ...app.bootstrap.settings, ...payload, defaults: { ...app.bootstrap.settings.defaults, ...payload.defaults } }), { status: 200 });
    return new Response(JSON.stringify(shared), { status: 200 });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
function body(fetch: ReturnType<typeof mockFetch>, path = '/api/concurrency') { return JSON.parse(fetch.mock.calls.filter(([url]) => url === path).at(-1)![1]!.body as string); }
async function openConcurrency(user: ReturnType<typeof userEvent.setup>) { await user.click(screen.getByRole('button', { name: '执行与并发' })); }
async function setNumber(user: ReturnType<typeof userEvent.setup>, label: string, value: number | string) { const input = screen.getByLabelText(label); await user.clear(input); await user.type(input, String(value)); }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Central execution and concurrency settings', () => {
  it('shows one ceiling per shared connection and the effective minimum for each bound stage', async () => {
    const user = userEvent.setup(); const app = context(); mount(app); await openConcurrency(user);
    expect(screen.getAllByLabelText(`${shared.name} 连接并发上限`)).toHaveLength(1);
    expect(screen.getAllByLabelText(`${other.name} 连接并发上限`)).toHaveLength(1);
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).value).toBe('16');
    expect((screen.getByLabelText(`${shared.name} 连接并发上限`) as HTMLInputElement).value).toBe('4');
    expect(screen.getAllByText('最多同时 4 个请求（任务 16 · 连接 4）')).toHaveLength(3);
    await setNumber(user, '并发请求上限', 3);
    expect(screen.getAllByText('最多同时 3 个请求（任务 3 · 连接 4）')).toHaveLength(3);
    await user.selectOptions(screen.getByLabelText('台词整理执行方式'), 'serial');
    expect(screen.getByText('串行 · 最多同时 1 个请求')).toBeTruthy();
    expect(screen.getAllByText('最多同时 3 个请求（任务 3 · 连接 4）')).toHaveLength(2);
    await user.selectOptions(screen.getByLabelText('字幕识别执行方式'), 'serial');
    await user.selectOptions(screen.getByLabelText('视觉分块执行方式'), 'serial');
    expect(screen.getAllByText('串行 · 最多同时 1 个请求')).toHaveLength(3);
  });

  it('sets all ceilings to 16 in one save while preserving execution modes and unrelated settings', async () => {
    const app = context(mixed); const original = structuredClone(app.bootstrap); const fetch = mockFetch(app); const user = userEvent.setup(); mount(app); await openConcurrency(user);
    await user.click(screen.getByRole('button', { name: '全部设为 16' }));
    expect(fetch).not.toHaveBeenCalled();
    for (const label of ['并发请求上限', `${shared.name} 连接并发上限`, `${other.name} 连接并发上限`]) expect((screen.getByLabelText(label) as HTMLInputElement).value).toBe('16');
    expect((screen.getByLabelText('台词整理执行方式') as HTMLSelectElement).value).toBe('serial');
    expect((screen.getByLabelText('字幕识别执行方式') as HTMLSelectElement).value).toBe('serial');
    await user.click(screen.getByRole('button', { name: '保存执行与并发' }));
    await waitFor(() => expect(app.refresh).toHaveBeenCalledOnce());
    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch.mock.calls[0][0]).toBe('/api/concurrency'); expect(fetch.mock.calls[0][1]!.method).toBe('PUT');
    expect(body(fetch)).toEqual({ execution: { ...mixed, max_concurrency: 16 }, profiles: { api: 16, other: 16 } });
    expect(app.bootstrap).toEqual(original);
    expect(app.notify).toHaveBeenCalledWith('执行与并发已保存');
  });

  it('uses the preparation connection for both transcript and visual partitioning limits', async () => {
    const app = context({ ...DEFAULT_EXECUTION, max_concurrency: 6 }, [shared, { ...other, capabilities: ['vision'], max_concurrency: 8 }]);
    app.bootstrap.settings.bindings = { preparation: shared.id, vision: other.id, subtitle: shared.id };
    const user = userEvent.setup(); mount(app); await openConcurrency(user);
    expect(within(screen.getByLabelText('台词整理默认并发')).getByText('最多同时 4 个请求（任务 6 · 连接 4）')).toBeTruthy();
    expect(within(screen.getByLabelText('视觉分块默认并发')).getByText('最多同时 4 个请求（任务 6 · 连接 4）')).toBeTruthy();
    expect(within(screen.getByLabelText('字幕 OCR默认并发')).getByText('最多同时 4 个请求（任务 6 · 连接 4）')).toBeTruthy();
  });

  it('saves an individual connection ceiling above 16 without imposing a numeric maximum', async () => {
    const app = context(); const fetch = mockFetch(app); const user = userEvent.setup(); mount(app); await openConcurrency(user);
    const input = screen.getByLabelText(`${shared.name} 连接并发上限`) as HTMLInputElement;
    expect(input.hasAttribute('max')).toBe(false);
    await setNumber(user, `${shared.name} 连接并发上限`, 64);
    expect(screen.getAllByText('最多同时 16 个请求（任务 16 · 连接 64）')).toHaveLength(3);
    await user.click(screen.getByRole('button', { name: '保存执行与并发' }));
    await waitFor(() => expect(app.refresh).toHaveBeenCalledOnce());
    expect(body(fetch)).toEqual({ execution: DEFAULT_EXECUTION, profiles: { api: 64, other: 2 } });
  });

  it('persists execution defaults through the central panel for the next task modal', async () => {
    const app = context(); const fetch = mockFetch(app); const user = userEvent.setup(); const view = mount(app); await openConcurrency(user);
    await user.selectOptions(screen.getByLabelText('台词整理执行方式'), mixed.transcript);
    await user.selectOptions(screen.getByLabelText('字幕识别执行方式'), mixed.subtitle);
    await setNumber(user, '并发请求上限', mixed.max_concurrency);
    await user.click(screen.getByRole('button', { name: '保存执行与并发' }));
    await waitFor(() => expect(app.refresh).toHaveBeenCalledOnce());
    expect(body(fetch).execution).toEqual(mixed);
    view.unmount();
    const updated = { ...app, bootstrap: { ...app.bootstrap, settings: { ...app.bootstrap.settings, defaults: { ...app.bootstrap.settings.defaults, execution: body(fetch).execution } } } };
    render(<AppContext.Provider value={updated}><JobModal asset={asset} onClose={vi.fn()} /></AppContext.Provider>);
    expect((screen.getByLabelText('台词整理执行方式') as HTMLSelectElement).value).toBe('serial');
    expect((screen.getByLabelText('视觉分块执行方式') as HTMLSelectElement).value).toBe('parallel');
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).value).toBe('5');
  });

  it.each(['并发请求上限', `${shared.name} 连接并发上限`])('rejects invalid positive integers in %s before writing', async (label) => {
    const app = context(); const fetch = mockFetch(app); const user = userEvent.setup(); mount(app); await openConcurrency(user);
    for (const value of ['0', '0.5', '9007199254740992']) {
      await setNumber(user, label, value);
      await user.click(screen.getByRole('button', { name: '保存执行与并发' }));
      expect(fetch).not.toHaveBeenCalled();
      expect((screen.getByLabelText(label) as HTMLInputElement).validity.customError).toBe(true);
    }
  });

  it('keeps changes editable when the unified save fails', async () => {
    const app = context(); const fetch = mockFetch(app, '并发设置写入失败'); const user = userEvent.setup(); mount(app); await openConcurrency(user);
    await user.click(screen.getByRole('button', { name: '全部设为 16' }));
    await user.click(screen.getByRole('button', { name: '保存执行与并发' }));
    await screen.findByText('并发设置写入失败');
    expect(fetch).toHaveBeenCalledOnce(); expect(app.refresh).not.toHaveBeenCalled();
    expect(app.notify).not.toHaveBeenCalledWith('执行与并发已保存');
    expect((screen.getByLabelText(`${shared.name} 连接并发上限`) as HTMLInputElement).value).toBe('16');
    expect((screen.getByRole('button', { name: '保存执行与并发' }) as HTMLButtonElement).disabled).toBe(false);
  });

  it('retains successfully saved ceilings if refresh fails and other settings are then saved', async () => {
    const app = context(); vi.mocked(app.refresh).mockRejectedValue(new Error('刷新失败')); const fetch = mockFetch(app); const user = userEvent.setup(); mount(app); await openConcurrency(user);
    await user.click(screen.getByRole('button', { name: '全部设为 16' }));
    await user.click(screen.getByRole('button', { name: '保存执行与并发' }));
    await screen.findByText('执行与并发已保存，但刷新失败：刷新失败');
    expect(app.notify).toHaveBeenCalledWith('执行与并发已保存，但刷新失败：刷新失败', 'error');
    await user.click(screen.getByRole('button', { name: '分析默认值' }));
    expect(screen.queryByRole('spinbutton', { name: '并发请求上限' })).toBeNull();
    await setNumber(user, '最大模型请求数', 50);
    await user.click(screen.getByRole('button', { name: '保存默认值' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/settings')).toBe(true));
    expect(body(fetch, '/api/settings').defaults.max_requests).toBe(50);
    expect(body(fetch, '/api/settings').defaults).not.toHaveProperty('execution');
    await openConcurrency(user);
    expect((screen.getByLabelText('并发请求上限') as HTMLInputElement).value).toBe('16');
    expect((screen.getByLabelText(`${shared.name} 连接并发上限`) as HTMLInputElement).value).toBe('16');
    expect(fetch.mock.calls.filter(([url]) => url === '/api/concurrency')).toHaveLength(1);
  });

  it('keeps unsaved concurrency out of budget and model binding saves', async () => {
    const app = context(mixed); const fetch = mockFetch(app); const user = userEvent.setup(); mount(app); await openConcurrency(user);
    await user.click(screen.getByRole('button', { name: '全部设为 16' }));
    await user.selectOptions(screen.getByLabelText('台词整理执行方式'), 'parallel');
    await user.click(screen.getByRole('button', { name: '分析默认值' }));
    expect(screen.queryByRole('combobox', { name: '台词整理执行方式' })).toBeNull();
    await setNumber(user, '最大模型请求数', 50);
    await user.click(screen.getByRole('button', { name: '保存默认值' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    const budgetPatch = body(fetch, '/api/settings');
    expect(budgetPatch).not.toHaveProperty('bindings'); expect(budgetPatch.defaults).not.toHaveProperty('execution');
    await user.click(screen.getByRole('button', { name: '模型连接' }));
    await user.selectOptions(screen.getByLabelText('分配画面字幕 OCR模型'), '');
    await user.click(screen.getByRole('button', { name: '保存分配' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    expect(body(fetch, '/api/settings')).toEqual({ bindings: { ...app.bootstrap.settings.bindings, subtitle: null } });
    expect(fetch.mock.calls.some(([url]) => url === '/api/concurrency')).toBe(false);
  });

  it('uses 16 for an unspecified connection and keeps old saved ceilings in the central panel', async () => {
    const app = context(DEFAULT_EXECUTION, [{ ...shared, max_concurrency: undefined }, other]); const user = userEvent.setup(); mount(app); await openConcurrency(user);
    expect((screen.getByLabelText(`${shared.name} 连接并发上限`) as HTMLInputElement).value).toBe('16');
    expect((screen.getByLabelText(`${other.name} 连接并发上限`) as HTMLInputElement).value).toBe('2');
  });

  it('keeps concurrency out of model editing payloads while allowing existing credentials to remain', async () => {
    const app = context(); const fetch = mockFetch(app); const user = userEvent.setup(); mount(app);
    await user.click(screen.getByRole('button', { name: `编辑 ${shared.name}` }));
    const dialog = within(screen.getByRole('dialog', { name: '编辑模型连接' }));
    expect(dialog.queryByLabelText('最大并发')).toBeNull();
    expect(dialog.getByText(/执行与并发/)).toBeTruthy();
    await user.click(dialog.getByRole('button', { name: '保存连接' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    const payload = body(fetch, '/api/profiles');
    expect(payload).not.toHaveProperty('max_concurrency'); expect(payload).not.toHaveProperty('api_key');
    expect(payload).toMatchObject({ id: shared.id, timeout_s: shared.timeout_s, env_var: shared.env_var, model: shared.model });
  });
});
