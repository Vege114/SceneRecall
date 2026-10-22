import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type DeletionPreview, type TrashEntry, type Job } from '../src/types';
import Jobs from '../src/components/Jobs';
import Library from '../src/components/Library';
import AssetDetail from '../src/components/AssetDetail';
import DeleteModal from '../src/components/DeleteModal';
import Trash from '../src/components/Trash';
import App from '../src/App';

const asset: Asset = { id: 'ep1', work_id: 'work1', title: '第一集', kind: 'animation', project_id: 'project1', project_name: '星海', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'container', source_available: true, source_changed: false, created_at: '2026-10-13T00:00:00Z', status: 'ready' };
const second: Asset = { ...asset, id: 'ep2', title: '第二集' };
const baseJob: Job = { id: 'job1', type: 'analysis', status: 'completed', asset_id: 'ep1', progress: 1, completed: 2, total: 2, created_at: asset.created_at };
function context(overrides: Partial<AppContextValue> = {}): AppContextValue {
  return { bootstrap: { library_path: '/tmp/ui', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: '0.4.0' }, assets: [asset, second], projects: [{ id: 'project1', name: '星海', kind: 'animation', folder_id: 'child', tags: ['科幻'], asset_count: 2 }], folders: [{ id: 'parent', name: '动画', parent_id: null }, { id: 'child', name: '长篇', parent_id: 'parent' }], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openJob: vi.fn(), openJobs: vi.fn(), openImport: vi.fn(), ...overrides };
}
function show(ui: ReactNode, value = context()) { return render(<AppContext.Provider value={value}>{ui}</AppContext.Provider>); }
function mockFetch(handler: (url: string, body: any, options?: RequestInit) => unknown | Promise<unknown>) {
  const mock = vi.fn(async (url: string, options?: RequestInit) => { const value = await handler(url, options?.body ? JSON.parse(options.body as string) : {}, options); return value instanceof Response ? value : new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } }); }); vi.stubGlobal('fetch', mock); return mock;
}
const preview: DeletionPreview = { token: 'scope-token', title: '第一集', asset_count: 1, project_count: 0, folder_count: 0, assets: [{ id: asset.id, title: asset.title }], projects: [], folders: [], blocked_jobs: [], source_files_preserved: true, delete_source: false, source_file_count: 1, source_files: [{ asset_ids: [asset.id], path: '/movies/第一集.mkv', status: 'ready', message: '' }] };
const trash: TrashEntry = { ...preview, id: 'trash1', kind: 'assets', deleted_at: asset.created_at, delete_contents: false, source_results: [] };
const failure = (detail: string, status = 409) => new Response(JSON.stringify({ detail }), { status, headers: { 'Content-Type': 'application/json' } });
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Historical jobs after deletion', () => {
  const imported: Job = { ...baseJob, type: 'import', asset_id: undefined, items: [{ item_id: 'one', name: asset.title, status: 'completed', asset_id: asset.id }, { item_id: 'two', name: second.title, status: 'completed', asset_id: second.id }] };
  it('only offers assets that remain in the library for batch analysis', async () => {
    const user = userEvent.setup(); const value = context({ assets: [second], jobs: [imported] }); show(<Jobs />, value);
    await user.click(screen.getByRole('button', { name: '分析这批' })); expect(value.openJobs).toHaveBeenCalledWith([second]);
  });
  it('hides batch analysis after every imported asset is deleted', () => {
    show(<Jobs />, context({ assets: [], jobs: [imported] })); expect(screen.queryByRole('button', { name: '分析这批' })).toBeNull();
  });
  it('retains single-asset history and links to the trash instead of an unavailable asset', () => {
    show(<Jobs />, context({ assets: [], jobs: [baseJob] })); expect(screen.getByRole('link', { name: /前往回收站/ }).getAttribute('href')).toBe('#/trash'); expect(document.querySelector('a[href="#/asset/ep1"]')).toBeNull();
  });
});

describe('Deletion entry points and safe scope confirmation', () => {
  it('offers an always-present card action and defaults to application data only', async () => {
    const user = userEvent.setup(); const fetch = mockFetch((url) => url === '/api/deletions/preview' ? preview : { trash }); const value = context();
    show(<Library />, value); await user.click(screen.getByRole('button', { name: '删除 第一集' }));
    const dialog = await screen.findByRole('dialog', { name: '删除 1 个视频' });
    await within(dialog).findByText('1 个视频');
    expect((screen.getByRole('radio', { name: /仅删除应用内影片资料/ }) as HTMLInputElement).checked).toBe(true);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toEqual({ kind: 'assets', asset_ids: ['ep1'], delete_source: false });
    await user.click(screen.getByRole('button', { name: '确认移入回收站' }));
    await waitFor(() => expect(value.refresh).toHaveBeenCalled());
    expect(fetch.mock.calls[1][0]).toBe('/api/assets/ep1?delete_source=false&expected_token=scope-token'); expect(fetch.mock.calls[1][1]?.method).toBe('DELETE');
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });
  it('deletes exactly the selected videos with one batch request', async () => {
    const user = userEvent.setup(); const scope = { ...preview, asset_count: 2, assets: [asset, second] };
    const fetch = mockFetch((url) => url.endsWith('/preview') ? scope : { trash: { ...trash, ...scope } });
    show(<Library />); await user.click(screen.getByRole('checkbox', { name: '选择 第一集' })); await user.click(screen.getByRole('checkbox', { name: '选择 第二集' })); await user.click(screen.getByRole('button', { name: '删除所选视频' }));
    await screen.findByText('2 个视频'); await user.click(screen.getByRole('button', { name: '确认移入回收站' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    expect(fetch.mock.calls[1][0]).toBe('/api/assets/delete'); expect(JSON.parse(fetch.mock.calls[1][1]!.body as string)).toEqual({ asset_ids: ['ep1', 'ep2'], delete_source: false, expected_token: 'scope-token' });
  });
  it('supports deletion from the detail page and returns to the library', async () => {
    const user = userEvent.setup(); mockFetch((url, _body, options) => options?.method === 'DELETE' ? { trash } : url.endsWith('/preview') ? preview : { asset, observations: [], subtitles: [], annotations: [], runs: [] });
    show(<AssetDetail id={asset.id} />); await user.click(await screen.findByRole('button', { name: '删除视频' })); await screen.findByText('1 个视频'); await user.click(screen.getByRole('button', { name: '确认移入回收站' }));
    await waitFor(() => expect(window.location.hash).toBe('#/library'));
  });
  it('previews folder-only deletion first, then explicitly includes descendant content', async () => {
    const user = userEvent.setup(); const fetch = mockFetch((url, body) => url.endsWith('/preview') ? { ...preview, asset_count: body.delete_contents ? 2 : 0, assets: body.delete_contents ? [asset, second] : [], project_count: body.delete_contents ? 1 : 0, projects: body.delete_contents ? [{ id: 'project1', name: '星海' }] : [], folder_count: body.delete_contents ? 2 : 1, folders: [{ id: 'parent', name: '动画' }, ...(body.delete_contents ? [{ id: 'child', name: '长篇' }] : [])] } : { trash });
    show(<DeleteModal target={{ kind: 'folder', id: 'parent', name: '动画' }} onClose={vi.fn()} onDeleted={vi.fn()} />);
    await screen.findByText('0 个视频'); expect(screen.queryByRole('radio')).toBeNull(); expect(screen.getByText(/内部项目及下级文件夹移至上一级/)).toBeTruthy();
    await user.click(screen.getByRole('checkbox', { name: /同时删除内部视频/ })); await screen.findByText('2 个视频');
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string)).toEqual({ kind: 'folder', id: 'parent', delete_contents: true, delete_source: false });
    await user.click(screen.getByRole('button', { name: '确认移入回收站' })); await waitFor(() => expect(fetch).toHaveBeenCalledTimes(3));
    expect(fetch.mock.calls[2][0]).toBe('/api/folders/parent?delete_contents=true&delete_source=false&expected_token=scope-token');
  });
  it('uses the shared confirmation for project management', async () => {
    const user = userEvent.setup(); const fetch = mockFetch((url) => url.endsWith('/preview') ? { ...preview, asset_count: 0, assets: [], project_count: 1, projects: [{ id: 'project1', name: '星海' }] } : { trash });
    show(<Library />); await user.selectOptions(screen.getByLabelText('项目'), 'project1'); await user.click(screen.getByRole('button', { name: '删除项目' }));
    await screen.findByText('0 个视频'); expect(screen.getByText(/变为未归类/)).toBeTruthy(); await user.click(screen.getByRole('button', { name: '确认移入回收站' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2)); expect(fetch.mock.calls[1][0]).toContain('/api/projects/project1?delete_contents=false&delete_source=false&expected_token=');
  });
  it('does not mutate anything when the confirmation is cancelled', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => preview); const close = vi.fn(); show(<DeleteModal target={{ kind: 'assets', asset_ids: ['ep1'] }} onClose={close} onDeleted={vi.fn()} />);
    await screen.findByText('1 个视频'); await user.click(screen.getByRole('button', { name: '取消' })); expect(close).toHaveBeenCalledOnce(); expect(fetch).toHaveBeenCalledTimes(1);
  });
  it('blocks deletion while a related task has not stopped', async () => {
    mockFetch(() => ({ ...preview, blocked_jobs: [{ id: 'scan1', status: 'running', message: '第一集正在扫描' }] })); show(<DeleteModal target={{ kind: 'assets', asset_ids: ['ep1'] }} onClose={vi.fn()} onDeleted={vi.fn()} />);
    await screen.findByText('第一集正在扫描'); expect((screen.getByRole('button', { name: '确认移入回收站' }) as HTMLButtonElement).disabled).toBe(true); expect(screen.getByRole('link', { name: '任务队列' }).getAttribute('href')).toBe('#/jobs');
  });
  it('refreshes an expired preview and requires another deliberate submission', async () => {
    const user = userEvent.setup(); let previews = 0; const fetch = mockFetch((url) => url.endsWith('/preview') ? { ...preview, token: ++previews === 1 ? 'old' : 'new', asset_count: previews === 1 ? 1 : 2 } : failure('范围已变化，请重新预览'));
    show(<DeleteModal target={{ kind: 'assets', asset_ids: ['ep1'] }} onClose={vi.fn()} onDeleted={vi.fn()} />); await screen.findByText('1 个视频'); await user.click(screen.getByRole('button', { name: '确认移入回收站' }));
    await screen.findByText('2 个视频'); expect(screen.getByText(/范围已变化/)).toBeTruthy(); expect(fetch.mock.calls.filter((call) => call[1]?.method === 'DELETE')).toHaveLength(1);
    await user.click(screen.getByRole('button', { name: '确认移入回收站' })); await waitFor(() => expect(fetch.mock.calls.filter((call) => call[1]?.method === 'DELETE')).toHaveLength(2)); expect(fetch.mock.calls[3][0]).toContain('expected_token=new');
  });
});

describe('Explicit permanent source-video deletion', () => {
  it('requires the exact source list and irreversible confirmation before deleting source files', async () => {
    const user = userEvent.setup(); const done = vi.fn(); const fetch = mockFetch((url, body) => url.endsWith('/preview') ? { ...preview, delete_source: !!body.delete_source, source_files_preserved: !body.delete_source } : { trash: { ...trash, delete_source: true, source_results: [{ asset_ids: ['ep1'], status: 'deleted', message: '' }] } });
    show(<DeleteModal target={{ kind: 'assets', asset_ids: ['ep1'] }} onClose={vi.fn()} onDeleted={done} />); await screen.findByText('1 个视频'); await user.click(screen.getByRole('radio', { name: /同时永久删除源视频文件/ }));
    await screen.findByText('/movies/第一集.mkv'); expect((screen.getByRole('button', { name: '确认删除资料与源文件' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole('checkbox', { name: /我已核对以上 1 个文件/ })); await user.click(screen.getByRole('button', { name: '确认删除资料与源文件' })); await waitFor(() => expect(done).toHaveBeenCalled());
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string).delete_source).toBe(true); expect(fetch.mock.calls[2][0]).toContain('delete_source=true');
  });
  it.each(['missing', 'changed', 'shared', 'unsafe'] as const)('blocks an unsafe source with status %s but permits application-only deletion', async (status) => {
    const user = userEvent.setup(); mockFetch((_url, body) => ({ ...preview, delete_source: !!body.delete_source, source_files: [{ ...preview.source_files[0], status, message: '文件状态冲突' }] }));
    show(<DeleteModal target={{ kind: 'assets', asset_ids: ['ep1'] }} onClose={vi.fn()} onDeleted={vi.fn()} />); await screen.findByText('1 个视频'); await user.click(screen.getByRole('radio', { name: /同时永久删除源视频文件/ }));
    await screen.findByText(/存在无法安全删除的源文件/); expect((screen.getByRole('button', { name: '确认删除资料与源文件' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole('radio', { name: /仅删除应用内影片资料/ })); await screen.findByText('1 个视频'); expect((screen.getByRole('button', { name: '确认移入回收站' }) as HTMLButtonElement).disabled).toBe(false);
  });
  it('ignores a stale preview after the source handling choice changes', async () => {
    const user = userEvent.setup(); let resolve!: (value: unknown) => void; const deferred = new Promise((done) => { resolve = done; }); mockFetch((_url, body) => body.delete_source ? { ...preview, delete_source: true } : deferred);
    show(<DeleteModal target={{ kind: 'assets', asset_ids: ['ep1'] }} onClose={vi.fn()} onDeleted={vi.fn()} />); await user.click(screen.getByRole('radio', { name: /同时永久删除源视频文件/ })); await screen.findByText('/movies/第一集.mkv');
    await act(async () => { resolve({ ...preview, source_file_count: 999 }); await deferred; }); expect(screen.getByText('将永久删除 1 个源视频文件')).toBeTruthy(); expect(screen.queryByText(/999/)).toBeNull();
  });
  it('clears irreversible confirmation after preview conflict', async () => {
    const user = userEvent.setup(); mockFetch((url, body) => url.endsWith('/preview') ? { ...preview, delete_source: !!body.delete_source } : failure('源文件已改变'));
    show(<DeleteModal target={{ kind: 'assets', asset_ids: ['ep1'] }} onClose={vi.fn()} onDeleted={vi.fn()} />); await screen.findByText('1 个视频'); await user.click(screen.getByRole('radio', { name: /同时永久删除源视频文件/ }));
    await user.click(await screen.findByRole('checkbox', { name: /我已核对以上/ })); await user.click(screen.getByRole('button', { name: '确认删除资料与源文件' }));
    await screen.findByText(/源文件已改变/); await screen.findByText('/movies/第一集.mkv'); expect((screen.getByRole('checkbox', { name: /我已核对以上/ }) as HTMLInputElement).checked).toBe(false);
  });
  it('keeps partial source failures visible after application records are archived', async () => {
    const user = userEvent.setup(); const done = vi.fn(); const value = context(); mockFetch((url, body) => url.endsWith('/preview') ? { ...preview, delete_source: !!body.delete_source } : { trash: { ...trash, delete_source: true, source_results: [{ asset_ids: ['ep1'], status: 'error', message: '文件被占用，未能删除' }] } });
    show(<DeleteModal target={{ kind: 'assets', asset_ids: ['ep1'] }} onClose={vi.fn()} onDeleted={done} />, value); await screen.findByText('1 个视频'); await user.click(screen.getByRole('radio', { name: /同时永久删除源视频文件/ })); await user.click(await screen.findByRole('checkbox', { name: /我已核对以上/ })); await user.click(screen.getByRole('button', { name: '确认删除资料与源文件' }));
    await screen.findByRole('dialog', { name: '应用资料已移入回收站' }); expect(screen.getByText(/文件被占用，未能删除/)).toBeTruthy(); expect(screen.getByText(/1 个未完成删除/)).toBeTruthy(); expect(done).not.toHaveBeenCalled(); await waitFor(() => expect(value.notify).toHaveBeenCalledWith(expect.stringContaining('部分源文件未删除'), 'error')); await user.click(screen.getByRole('button', { name: '完成' })); expect(done).toHaveBeenCalled();
  });
});

describe('Persistent trash and restore', () => {
  it('shows an empty state and a stable library route', async () => {
    mockFetch(() => []); show(<Trash />); await screen.findByText('回收站是空的'); expect(screen.getByRole('link', { name: '返回资料库' }).getAttribute('href')).toBe('#/library');
  });
  it('loads saved batches and restores the complete batch after confirmation', async () => {
    const user = userEvent.setup(); let restored = false; const fetch = mockFetch((url) => url.endsWith('/restore') ? (restored = true, { restored: trash }) : restored ? [] : [trash]); const value = context(); show(<Trash />, value);
    await user.click(await screen.findByRole('button', { name: '恢复' })); expect(fetch).toHaveBeenCalledTimes(1); await user.click(screen.getByRole('button', { name: '确认恢复' })); await screen.findByText('回收站是空的'); expect(fetch.mock.calls[1][0]).toBe('/api/trash/trash1/restore'); expect(value.refresh).toHaveBeenCalled();
  });
  it('retains a batch and shows restore conflicts without pretending it was restored', async () => {
    const user = userEvent.setup(); mockFetch((url) => url.endsWith('/restore') ? failure('项目已移动，无法覆盖后续修改') : [trash]); const value = context(); show(<Trash />, value);
    await user.click(await screen.findByRole('button', { name: '恢复' })); await user.click(screen.getByRole('button', { name: '确认恢复' })); await screen.findByText('项目已移动，无法覆盖后续修改'); expect(screen.getByRole('dialog', { name: '恢复这批资料' })).toBeTruthy(); expect(value.notify).not.toHaveBeenCalled();
  });
  it('warns that restoring application records cannot restore permanently deleted source files', async () => {
    const user = userEvent.setup(); mockFetch(() => [{ ...trash, delete_source: true, source_results: [{ asset_ids: ['ep1'], status: 'deleted', message: '' }] }]); show(<Trash />); await screen.findByText(/已删除 1 个文件/); await user.click(screen.getByRole('button', { name: '恢复' })); expect(within(screen.getByRole('dialog')).getByText(/本次只恢复应用内资料/)).toBeTruthy();
  });
  it('makes the trash accessible from the main navigation', async () => {
    vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined); window.location.hash = '#/trash'; const value = context(); mockFetch((url) => url.endsWith('/bootstrap') ? value.bootstrap : url.endsWith('/assets') ? value.assets : url.endsWith('/projects') ? value.projects : url.endsWith('/folders') ? value.folders : []);
    render(<App />); await screen.findByText('回收站是空的'); expect(screen.getByRole('link', { name: '回收站' }).getAttribute('href')).toBe('#/trash');
  });
});
