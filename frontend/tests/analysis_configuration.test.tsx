import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type Profile } from '../src/types';
import JobModal from '../src/components/JobModal';
import Settings from '../src/components/Settings';

const asset: Asset = { id: 'video', title: '示例', kind: 'movie', duration_ms: 90000, width: 640, height: 360, subtitle_mode: 'embedded', source_available: true, source_changed: false, status: 'ready', created_at: '2026-10-23' };
const api: Profile = { id: 'api', name: '本地视觉模型', provider_type: 'openai_compatible', base_url: 'http://localhost:8999/v1', model: 'local-vlm', capabilities: ['preparation', 'vision', 'subtitle'], secret_mode: 'none' };
const codex: Profile = { ...api, id: 'codex', name: 'Codex 订阅', provider_type: 'codex_cli', base_url: '', model: 'configured-model' };
function context(): AppContextValue {
  return { bootstrap: { library_path: '/tmp/analysis-ui', provider_profiles: [api, codex], settings: { bindings: { preparation: api.id, vision: api.id, subtitle: api.id }, defaults: { ...DEFAULTS } }, stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test', visual_encoder: { available: true, model: 'SigLIP 2' } }, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function mockFetch(app: AppContextValue) {
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    let result: unknown = { id: 'job-created' };
    if (url === '/api/codex/models') result = { models: [{ id: 'fast-model', name: '快速模型', reasoning_efforts: ['low'], default_reasoning_effort: 'low' }], default_model: 'configured-model' };
    if (url === '/api/settings') { app.bootstrap.settings = { ...app.bootstrap.settings, defaults: { ...app.bootstrap.settings.defaults, ...body.defaults } }; result = app.bootstrap.settings; }
    return new Response(JSON.stringify(result), { status: 200 });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
const payload = (fetch: ReturnType<typeof mockFetch>, url: string) => JSON.parse(String(fetch.mock.calls.find(([path]) => path === url)?.[1]?.body));
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Shared analysis defaults and one-step task entry', () => {
  it('saves the same settings used by the task dialog, including stage connections and Codex options', async () => {
    const app = context(); const fetch = mockFetch(app); const user = userEvent.setup();
    window.location.hash = '/settings?tab=defaults';
    const view = render(<AppContext.Provider value={app}><Settings /></AppContext.Provider>);
    await user.selectOptions(screen.getByLabelText('画面理解连接'), 'codex');
    await screen.findByRole('option', { name: '快速模型 · fast-model' });
    await user.selectOptions(screen.getByLabelText('画面理解模型'), 'model:fast-model');
    await user.selectOptions(screen.getByLabelText('画面理解思考等级'), 'low');
    await user.selectOptions(screen.getByLabelText('台词整理执行方式'), 'serial');
    const limit = screen.getByLabelText('每个视频最多模型请求数'); await user.clear(limit); await user.type(limit, '123');
    const settingsLabels = within(screen.getByRole('group', { name: '任务处理配置' })).getAllByRole('combobox').map((select) => (select as HTMLSelectElement).value);
    await user.click(screen.getByRole('button', { name: '保存默认值' }));
    await waitFor(() => expect(app.notify).toHaveBeenCalledWith('设置已保存'));
    expect(payload(fetch, '/api/settings').defaults.model_bindings).toEqual({ vision: 'codex' });
    view.unmount();
    render(<AppContext.Provider value={app}><JobModal asset={asset} onClose={vi.fn()} /></AppContext.Provider>);
    await screen.findByRole('option', { name: '快速模型 · fast-model' });
    expect(within(screen.getByRole('group', { name: '任务处理配置' })).getAllByRole('combobox').map((select) => (select as HTMLSelectElement).value)).toEqual(settingsLabels);
    expect(screen.queryByRole('button', { name: '扫描关键帧' })).toBeNull();
    expect(screen.queryByRole('button', { name: '开始角色确认' })).toBeNull();
    await user.click(screen.getByRole('button', { name: '开始任务' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/jobs')).toBe(true));
    expect(payload(fetch, '/api/jobs')).toMatchObject({ task_pipeline: 'unified-v1', max_requests: 123, model_bindings: { vision: 'codex' }, model_overrides: { vision: { profile_id: 'codex', model: 'fast-model', reasoning_effort: 'low' } }, execution: { transcript: 'serial' }, start_ms: 0, end_ms: null, max_cost: null });
    expect(fetch.mock.calls.some(([url]) => url.includes('keyframe-plans') || url.includes('/estimate') || url.includes('/confirmation'))).toBe(false);
  });

  it('submits local retrieval with no inherited model calls and shows fixed local methods as locked options', async () => {
    const app = context(); app.bootstrap.settings.defaults = { ...DEFAULTS, analysis_strategy: 'retrieval-first-v1', retrieval_mode: 'core', model_bindings: { vision: 'codex' }, model_overrides: { vision: { provider_type: 'codex_cli', profile_id: 'codex', model: 'fast-model' } } };
    const fetch = mockFetch(app); const user = userEvent.setup();
    render(<AppContext.Provider value={app}><JobModal asset={asset} onClose={vi.fn()} /></AppContext.Provider>);
    for (const label of ['关键帧划分方式', '角色确认方式', '人物检测与匹配模型', '画面索引模型', '文字字幕处理方式']) expect((screen.getByLabelText(label) as HTMLSelectElement).disabled).toBe(true);
    expect(screen.getByLabelText('画面索引模型').textContent).toContain('SigLIP 2');
    await user.click(screen.getByRole('button', { name: '开始任务' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(payload(fetch, '/api/jobs')).toMatchObject({ analysis_strategy: 'retrieval-first-v1', retrieval_mode: 'core', task_pipeline: 'unified-v1', model_bindings: {}, model_overrides: {}, stages: ['vision'], end_ms: null });
  });

  it('keeps explicit unavailable connections visible and blocks submission until corrected', async () => {
    const app = context(); app.bootstrap.settings.defaults = { ...DEFAULTS, stages: ['vision'], model_bindings: { vision: 'removed' } };
    const fetch = mockFetch(app); const user = userEvent.setup();
    render(<AppContext.Provider value={app}><JobModal asset={asset} onClose={vi.fn()} /></AppContext.Provider>);
    expect(screen.getByRole('option', { name: '原连接已不可用，请重新选择' })).toBeTruthy();
    expect((screen.getByRole('button', { name: '开始任务' }) as HTMLButtonElement).disabled).toBe(true);
    await user.selectOptions(screen.getByLabelText('画面理解连接'), 'api');
    await user.click(screen.getByRole('button', { name: '开始任务' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(payload(fetch, '/api/jobs').model_bindings.vision).toBe('api');
  });
});
