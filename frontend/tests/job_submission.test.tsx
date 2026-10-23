import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type Job, type Profile } from '../src/types';
import JobModal from '../src/components/JobModal';
import { DeepenSceneButton } from '../src/components/SceneAnalysis';

const asset: Asset = { id: 'one', work_id: 'work', title: '测试影片', kind: 'movie', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const model: Profile = { id: 'api', name: '测试连接', provider_type: 'openai_compatible', base_url: 'https://example.invalid/v1', model: 'test-model', capabilities: ['preparation', 'vision', 'subtitle'], secret_mode: 'none' };
function context(assets = [asset]): AppContextValue {
  return { bootstrap: { library_path: '/tmp/submission-ui-tests', settings: { defaults: DEFAULTS, bindings: { preparation: model.id, vision: model.id, subtitle: model.id } }, provider_profiles: [model], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' }, assets, jobs: [], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function mount(ui: ReactNode, app = context()) { return render(<AppContext.Provider value={app}>{ui}</AppContext.Provider>); }
function response(value: unknown, status = 200) { return new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } }); }
function mockFetch(handler: (url: string) => unknown | Promise<unknown>) {
  const fetch = vi.fn(async (url: string, _options?: RequestInit) => {
    const result = await handler(url);
    return result instanceof Response ? result : response(result);
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Submission completion is independent of library refresh', () => {
  it.each(['single', 'batch'])('acknowledges a %s job as soon as creation succeeds while refresh is still pending', async (kind) => {
    const selected = kind === 'batch' ? [asset, { ...asset, id: 'two' }] : [asset];
    const path = kind === 'batch' ? '/api/jobs/batch' : '/api/jobs';
    const creation = deferred<unknown>(); const refresh = deferred<void>();
    const fetch = mockFetch((url) => creation.promise);
    const app = context(selected); vi.mocked(app.refresh).mockReturnValue(refresh.promise);
    const close = vi.fn(); const user = userEvent.setup();
    mount(<JobModal assets={selected} onClose={close} />, app);
    const start = screen.getByRole('button', { name: '开始任务' }) as HTMLButtonElement;
    await user.dblClick(start);
    expect(fetch.mock.calls.filter(([url]) => url === path)).toHaveLength(1);
    expect(start.disabled).toBe(true);
    expect(screen.getByText('正在提交')).toBeTruthy();
    expect(close).not.toHaveBeenCalled();
    expect(app.notify).not.toHaveBeenCalled();
    expect(app.refresh).not.toHaveBeenCalled();

    await act(async () => { creation.resolve({ id: 'job' }); });
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    expect(window.location.hash).toBe('#/jobs');
    expect(app.notify).toHaveBeenCalledWith(`${selected.length} 个分析任务已加入队列`);
    expect(app.refresh).toHaveBeenCalledOnce();
    expect(screen.queryByText('正在提交')).toBeNull();
    await user.click(start);
    expect(fetch.mock.calls.filter(([url]) => url === path)).toHaveLength(1);

    await act(async () => { refresh.reject(new Error('资料库刷新失败')); });
    await waitFor(() => expect(app.notify).toHaveBeenCalledWith(expect.stringContaining('任务已创建，但列表刷新失败'), 'error'));
    expect(close).toHaveBeenCalledOnce();
    expect(window.location.hash).toBe('#/jobs');
    expect(screen.queryByText('资料库刷新失败')).toBeNull();
    await user.click(start);
    expect(fetch.mock.calls.filter(([url]) => url === path)).toHaveLength(1);
  });

  it.each(['single', 'batch'])('keeps a failed %s creation open and allows a successful retry without waiting for refresh', async (kind) => {
    const selected = kind === 'batch' ? [asset, { ...asset, id: 'two' }] : [asset];
    const path = kind === 'batch' ? '/api/jobs/batch' : '/api/jobs';
    let attempts = 0;
    const refresh = deferred<void>();
    const fetch = mockFetch((url) => ++attempts === 1 ? response({ detail: '任务队列不可用' }, 503) : { id: 'job' });
    const app = context(selected); vi.mocked(app.refresh).mockReturnValue(refresh.promise);
    const close = vi.fn(); const user = userEvent.setup();
    mount(<JobModal assets={selected} onClose={close} />, app);
    await user.click(screen.getByRole('button', { name: '开始任务' }));
    await screen.findByText('任务队列不可用');
    expect(close).not.toHaveBeenCalled();
    expect(window.location.hash).toBe('');
    expect(app.notify).not.toHaveBeenCalled();
    expect(app.refresh).not.toHaveBeenCalled();
    expect((screen.getByRole('button', { name: '开始任务' }) as HTMLButtonElement).disabled).toBe(false);

    await user.click(screen.getByRole('button', { name: '开始任务' }));
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    expect(fetch.mock.calls.filter(([url]) => url === path)).toHaveLength(2);
    expect(window.location.hash).toBe('#/jobs');
    expect(screen.queryByText('任务队列不可用')).toBeNull();
    await act(async () => { refresh.resolve(undefined); });
  });

  it('shows a creation error immediately after saving defaults even if their refresh remains pending', async () => {
    const refresh = deferred<void>();
    const fetch = mockFetch((url) => url === '/api/jobs' ? response({ detail: '任务队列不可用' }, 503) : {});
    const app = context(); vi.mocked(app.refresh).mockReturnValue(refresh.promise);
    const close = vi.fn(); const user = userEvent.setup();
    mount(<JobModal asset={asset} onClose={close} />, app);
    await user.click(screen.getByRole('checkbox', { name: '设为默认值' }));
    await user.click(screen.getByRole('button', { name: '开始任务' }));
    await screen.findByText(/默认任务配置已保存，但任务创建失败：任务队列不可用/);
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/settings', '/api/jobs']);
    expect(app.refresh).toHaveBeenCalledOnce();
    expect(close).not.toHaveBeenCalled();
    expect(app.notify).not.toHaveBeenCalled();
    expect(screen.queryByText('正在提交')).toBeNull();
    expect((screen.getByRole('button', { name: '开始任务' }) as HTMLButtonElement).disabled).toBe(false);

    await act(async () => { refresh.reject(new Error('默认设置刷新失败')); });
    expect(screen.getByText(/默认任务配置已保存，但任务创建失败：任务队列不可用/)).toBeTruthy();
    expect(screen.queryByText('默认设置刷新失败')).toBeNull();
  });

  it('closes scene deepening and calls onStarted before refresh completes, keeping refresh errors separate', async () => {
    const creation = deferred<unknown>(); const refresh = deferred<void>();
    const fetch = mockFetch(() => creation.promise);
    const app = context(); vi.mocked(app.refresh).mockReturnValue(refresh.promise);
    const started = vi.fn(); const user = userEvent.setup();
    const ui = <DeepenSceneButton assetId={asset.id} chunkId="chunk" title="片段" onStarted={started} />;
    const view = mount(ui, app);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    const start = screen.getByRole('button', { name: '开始深入分析' }) as HTMLButtonElement;
    await user.dblClick(start);
    expect(fetch).toHaveBeenCalledOnce();
    expect(start.disabled).toBe(true);
    expect(screen.getByText('正在提交')).toBeTruthy();
    expect(started).not.toHaveBeenCalled();
    expect(app.refresh).not.toHaveBeenCalled();

    await act(async () => { creation.resolve({ id: 'deep-job' }); });
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(started).toHaveBeenCalledOnce();
    expect(app.notify).toHaveBeenCalledWith('片段深入分析任务已加入队列，收藏与引用会保留');
    expect(app.refresh).toHaveBeenCalledOnce();
    const entry = screen.getByRole('button', { name: '片段分析中' }) as HTMLButtonElement;
    expect(entry.disabled).toBe(true);
    await user.click(entry);
    expect(screen.queryByRole('dialog')).toBeNull();

    await act(async () => { refresh.reject(new Error('资料库刷新失败')); });
    await waitFor(() => expect(app.notify).toHaveBeenCalledWith(expect.stringContaining('任务已创建，但列表刷新失败'), 'error'));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(started).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledOnce();
    expect(entry.disabled).toBe(true);
    await user.click(entry);
    expect(screen.queryByRole('dialog')).toBeNull();

    const job: Job = { id: 'deep-job', type: 'scene_deepen', asset_id: asset.id, chunk_id: 'chunk', status: 'running', progress: 0.5, completed: 1, total: 2, created_at: asset.created_at };
    view.rerender(<AppContext.Provider value={{ ...app, jobs: [job] }}>{ui}</AppContext.Provider>);
    expect((screen.getByRole('button', { name: '片段分析中' }) as HTMLButtonElement).disabled).toBe(true);
    view.rerender(<AppContext.Provider value={{ ...app, jobs: [{ ...job, status: 'completed', progress: 1, completed: 2 }] }}>{ui}</AppContext.Provider>);
    expect((screen.getByRole('button', { name: '深入分析此片段' }) as HTMLButtonElement).disabled).toBe(false);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(fetch).toHaveBeenCalledOnce();
  });

  it('treats a rejected scene refresh as a successful submission, not a creation error', async () => {
    const fetch = mockFetch(() => ({ id: 'deep-job' }));
    const app = context(); vi.mocked(app.refresh).mockRejectedValue(new Error('资料库刷新失败'));
    const started = vi.fn(); const user = userEvent.setup();
    mount(<DeepenSceneButton assetId={asset.id} chunkId="chunk" title="片段" onStarted={started} />, app);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    await user.click(screen.getByRole('button', { name: '开始深入分析' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(started).toHaveBeenCalledOnce();
    expect(app.notify).toHaveBeenCalledWith('片段深入分析任务已加入队列，收藏与引用会保留');
    expect(app.notify).toHaveBeenCalledWith(expect.stringContaining('任务已创建，但列表刷新失败'), 'error');
    expect(fetch).toHaveBeenCalledOnce();
  });

  it('keeps scene deepening open on an actual POST failure and permits retry', async () => {
    let attempts = 0;
    const refresh = deferred<void>();
    const fetch = mockFetch(() => ++attempts === 1 ? response({ detail: '片段任务创建失败' }, 503) : { id: 'deep-job' });
    const app = context(); vi.mocked(app.refresh).mockReturnValue(refresh.promise);
    const started = vi.fn(); const user = userEvent.setup();
    mount(<DeepenSceneButton assetId={asset.id} chunkId="chunk" title="片段" onStarted={started} />, app);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    await user.click(screen.getByRole('button', { name: '开始深入分析' }));
    await screen.findByText('片段任务创建失败');
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(started).not.toHaveBeenCalled();
    expect(app.notify).not.toHaveBeenCalled();
    expect(app.refresh).not.toHaveBeenCalled();
    expect((screen.getByRole('button', { name: '开始深入分析' }) as HTMLButtonElement).disabled).toBe(false);

    await user.click(screen.getByRole('button', { name: '开始深入分析' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(started).toHaveBeenCalledOnce();
    await act(async () => { refresh.resolve(undefined); });
  });
});
