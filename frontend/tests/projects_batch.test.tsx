import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type ImportInput, type Job, type KeyframePlan } from '../src/types';
import ImportModal from '../src/components/ImportModal';
import JobModal from '../src/components/JobModal';
import Library from '../src/components/Library';
import SearchPage from '../src/components/Search';
import Jobs from '../src/components/Jobs';
import Settings from '../src/components/Settings';

const asset: Asset = { id: 'ep1', work_id: 'work1', title: '第一集', kind: 'animation', project_id: 'project1', project_name: '星海', folder_id: 'child', tags: ['科幻'], duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'container', source_available: true, source_changed: false, created_at: '2026-10-13T00:00:00Z', status: 'ready' };
const second: Asset = { ...asset, id: 'ep2', title: '第二集', duration_ms: 8000, subtitle_mode: 'embedded' };
const baseJob: Job = { id: 'job1', type: 'analysis', status: 'queued', asset_id: 'ep1', progress: 0, completed: 0, total: 2, created_at: asset.created_at };
const plan = (id: string): KeyframePlan => ({ id: `plan-${id}`, job_id: `scan-${id}`, status: 'completed', frame_count: 1, batch_count: 1, frames: [{ id: `frame-${id}`, at_ms: 500, thumbnail_url: `/api/frames/${id}`, reason: '稳定姿态' }] });
const noSubtitles = { duration_ms: 12000, tracks: [], recommended_stream_index: null, external_candidates: [], recommended_mode: 'auto', recommended_subtitle_path: null, reason: '未找到字幕', scan_complete: true };
const estimate = { duration_ms: 12000, estimated_frames: 1, estimated_requests: 1, cost_estimate: null, warnings: [] };
const input = (id: string): ImportInput => ({ item_id: id, video_path: `/tmp/${id}.mkv`, title: id, kind: 'animation', series: '星海', season: 1, episode: null, version: '', subtitle_mode: 'auto', subtitle_path: null, subtitle_offset_ms: 0 });
function context(overrides: Partial<AppContextValue> = {}): AppContextValue {
  return { bootstrap: { library_path: '/tmp/ui', settings: { bindings: { vision: 'model', preparation: 'model', subtitle: 'model' }, defaults: DEFAULTS }, provider_profiles: [{ id: 'model', name: 'test', model: 'test', base_url: 'http://localhost', secret_mode: 'none', capabilities: ['vision', 'subtitle'] }], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: '0.4.0' }, assets: [asset, second], projects: [{ id: 'project1', name: '星海', kind: 'animation', folder_id: 'child', tags: ['科幻'], asset_count: 2, anime_profile: { mode: 'scratch' } }, { id: 'project2', name: '短片集', kind: 'movie', folder_id: null, tags: ['收藏'], asset_count: 0 }], folders: [{ id: 'parent', name: '动画', parent_id: null }, { id: 'child', name: '长篇', parent_id: 'parent' }], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openJob: vi.fn(), openJobs: vi.fn(), openImport: vi.fn(), ...overrides };
}
function show(ui: ReactNode, value = context()) { return render(<AppContext.Provider value={value}>{ui}</AppContext.Provider>); }
function mockFetch(handler: (url: string, body: any, options?: RequestInit) => unknown | Promise<unknown>) {
  const mock = vi.fn(async (url: string, options?: RequestInit) => { const value = url === '/api/media/subtitle-detection' ? noSubtitles : await handler(url, options?.body ? JSON.parse(options.body as string) : {}, options); return value instanceof Response ? value : new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } }); }); vi.stubGlobal('fetch', mock); return mock;
}
function requests(fetch: ReturnType<typeof mockFetch>) { return fetch.mock.calls.filter((call) => call[0] !== '/api/media/subtitle-detection'); }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); window.history.replaceState({}, '', '/'); });

describe('Native multi-file selection and import', () => {
  it('chooses multiple videos through Finder and submits their local paths in episode order', async () => {
    const user = userEvent.setup();
    const selected = ['/movies/Anime [03].mkv', '/movies/Anime [01].mkv', '/movies/Anime [02].mkv'];
    const fetch = mockFetch((url, body) => url === '/api/local-files/pick' ? { paths: selected, cancelled: false } : { ...baseJob, type: 'import', status: 'completed', items: body.items.map((entry: ImportInput) => ({ item_id: entry.item_id, name: entry.title, status: 'completed', asset_id: entry.item_id })) });
    show(<ImportModal projectId="project1" onClose={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    expect(within(screen.getByLabelText('待导入视频')).getAllByRole('button').map((row) => row.textContent)).toEqual(['第 1 集Anime [01].mkv', '第 2 集Anime [02].mkv', '第 3 集Anime [03].mkv']);
    expect(JSON.parse(requests(fetch)[0][1]!.body as string)).toEqual({ kind: 'video', multiple: true });
    expect(document.querySelector('input[type="file"]')).toBeNull();
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await screen.findByRole('dialog', { name: '批量导入进度' });
    const payload = JSON.parse(requests(fetch).find((call) => call[0] === '/api/imports')![1]!.body as string);
    expect(payload.items.map((entry: ImportInput) => entry.video_path)).toEqual(['/movies/Anime [01].mkv', '/movies/Anime [02].mkv', '/movies/Anime [03].mkv']);
    expect(requests(fetch).map((call) => call[0])).toEqual(['/api/local-files/pick', '/api/imports']);
  });

  it('keeps already chosen videos when the next Finder selection is cancelled', async () => {
    const user = userEvent.setup(); let picks = 0;
    const fetch = mockFetch(() => ++picks === 1 ? { paths: ['/movies/Anime [01].mkv', '/movies/Anime [02].mkv'], cancelled: false } : { paths: [], cancelled: true });
    show(<ImportModal projectId="project1" onClose={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    const rows = within(screen.getByLabelText('待导入视频')).getAllByRole('button').map((row) => row.textContent);
    await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    expect(within(screen.getByLabelText('待导入视频')).getAllByRole('button').map((row) => row.textContent)).toEqual(rows);
    expect(requests(fetch).map((call) => call[0])).toEqual(['/api/local-files/pick', '/api/local-files/pick']);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('deduplicates pasted paths, preserves individual subtitles, and never starts analysis on import', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url, body) => url === '/api/local-files/pick' ? { paths: ['/tmp/ep2.srt'], cancelled: false } : { ...baseJob, id: 'import1', type: 'import', status: 'partial', items: body.items.map((entry: ImportInput, index: number) => ({ item_id: entry.item_id, name: entry.title, status: index ? 'failed' : 'completed', ...(index ? { error: '字幕文件不可读' } : { asset_id: 'ep1' }) })) });
    show(<ImportModal onClose={vi.fn()} projectId="project1" />);
    await user.click(screen.getByText('粘贴多个绝对路径'));
    await user.type(screen.getByLabelText('每行一个视频路径'), '/tmp/ep1.mkv\n/tmp/ep2.mkv\n/tmp/ep1.mkv');
    await user.click(screen.getByRole('button', { name: '添加路径列表' }));
    await user.click(within(screen.getByLabelText('待导入视频')).getByRole('button', { name: /ep2/ }));
    await user.click(screen.getByRole('button', { name: /外挂字幕文件/ }));
    await user.click(screen.getByRole('button', { name: '选择本机字幕文件（可多选）' }));
    expect((screen.getByLabelText(/这集的外挂字幕/) as HTMLSelectElement).value).toBe('/tmp/ep2.srt');
    await user.clear(screen.getByLabelText(/时间偏移/)); await user.type(screen.getByLabelText(/时间偏移/), '-0.125');
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await screen.findByText('字幕文件不可读');
    const body = JSON.parse(requests(fetch).find((call) => call[0] === '/api/imports')![1]!.body as string);
    expect(body.project_id).toBe('project1'); expect(body.items).toHaveLength(2); expect(body.request_id).toBeTruthy();
    expect(body.items[0]).toMatchObject({ video_path: '/tmp/ep1.mkv', subtitle_mode: 'auto' });
    expect(body.items[1]).toMatchObject({ subtitle_mode: 'external', subtitle_path: '/tmp/ep2.srt', subtitle_offset_ms: -125 });
    expect(requests(fetch).map((call) => call[0])).toEqual(['/api/local-files/pick', '/api/imports']);
    expect(screen.getByRole('button', { name: '分析这批' })).toBeTruthy();
  });

  it('reloads a failed import and retries only the corrected failed item', async () => {
    const user = userEvent.setup(); const entries = [input('ok'), input('bad')].map((row) => ({ ...row, anime_profile: { mode: 'scratch' as const } }));
    const job = { ...baseJob, id: 'import1', type: 'import', status: 'partial', items: [{ item_id: 'ok', name: 'ok', status: 'completed', asset_id: 'ep1' }, { item_id: 'bad', name: 'bad', status: 'failed', error: '无法读取原视频' }] };
    const fetch = mockFetch((url, body) => url === '/api/imports/import1' ? { job, items: entries, project_id: 'project1' } : { ...job, status: 'queued', items: job.items.map((entry) => entry.item_id === body.items[0].item_id ? { ...entry, status: 'pending' } : entry) });
    show(<ImportModal retryJobId="import1" onClose={vi.fn()} />);
    await screen.findByDisplayValue('/tmp/bad.mkv');
    await user.clear(screen.getByLabelText(/视频绝对路径/)); await user.type(screen.getByLabelText(/视频绝对路径/), '/fixed/bad.mkv');
    await user.click(screen.getByRole('button', { name: /重试失败文件/ }));
    await waitFor(() => expect(requests(fetch)).toHaveLength(2));
    expect(requests(fetch)[1][0]).toBe('/api/imports/import1/retry');
    const body = JSON.parse(requests(fetch)[1][1]!.body as string); expect(body.items).toHaveLength(1); expect(body.items[0]).toMatchObject({ item_id: 'bad', video_path: '/fixed/bad.mkv' });
  });

  it('preserves separate archive selections and legacy kinds when retrying an older project import', async () => {
    const user = userEvent.setup();
    const entries: ImportInput[] = [{ ...input('one'), episode: 1, anime_profile: { mode: 'bangumi', subject_id: 42 } }, { ...input('two'), episode: 2, anime_profile: { mode: 'scratch' } }, { ...input('film'), kind: 'movie', series: '', episode: null }];
    const job = { ...baseJob, id: 'legacy-import', type: 'import', status: 'failed', items: entries.map((entry) => ({ item_id: entry.item_id, name: entry.title, status: 'failed', error: '原视频不可读' })) };
    const fetch = mockFetch((url) => url === '/api/imports/legacy-import' ? { job, items: entries, project_id: 'project1' } : { ...job, status: 'completed' });
    show(<ImportModal retryJobId="legacy-import" onClose={vi.fn()} />);
    await screen.findByText('已选择 Bangumi 条目 #42，导入时读取现成档案。');
    expect((screen.getByLabelText(/^作品名称/) as HTMLInputElement).disabled).toBe(true);
    await user.clear(screen.getByLabelText(/视频绝对路径/)); await user.type(screen.getByLabelText(/视频绝对路径/), '/fixed/one.E99.mkv');
    expect((screen.getByLabelText('集（可选）') as HTMLInputElement).value).toBe('1');
    await user.click(within(screen.getByLabelText('待导入视频')).getByRole('button', { name: /^two/ }));
    expect(screen.getByText('从零建立档案 · 仅为能够确认姓名的人物建档，后续识别补充出场证据。')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: /重试失败文件/ })); await waitFor(() => expect(requests(fetch)).toHaveLength(2));
    const body = JSON.parse(requests(fetch)[1][1]!.body as string);
    expect(body.items.map((entry: ImportInput) => [entry.title, entry.kind, entry.episode, entry.anime_profile])).toEqual([['one', 'animation', 1, { mode: 'bangumi', subject_id: 42 }], ['two', 'animation', 2, { mode: 'scratch' }], ['film', 'movie', null, undefined]]);
  });

  it('naturally sorts 2 before 10, applies continuous episode numbering, and preserves a per-file override', async () => {
    const user = userEvent.setup(); const fetch = mockFetch((_url, body) => ({ ...baseJob, type: 'import', status: 'completed', items: body.items.map((entry: ImportInput) => ({ item_id: entry.item_id, name: entry.title, status: 'completed', asset_id: entry.item_id })) }));
    show(<ImportModal projectId="project1" onClose={vi.fn()} />);
    await user.click(screen.getByText('粘贴多个绝对路径')); await user.type(screen.getByLabelText('每行一个视频路径'), '/tmp/10.mkv\n/tmp/2.mkv'); await user.click(screen.getByRole('button', { name: '添加路径列表' }));
    await user.click(screen.getByText('批量设置季号与连续集号')); await user.clear(screen.getByLabelText('批量季号')); await user.type(screen.getByLabelText('批量季号'), '2'); await user.clear(screen.getByLabelText('起始集号')); await user.type(screen.getByLabelText('起始集号'), '3'); await user.click(screen.getByRole('button', { name: '应用连续集号' }));
    await user.click(within(screen.getByLabelText('待导入视频')).getByRole('button', { name: /10/ })); await user.clear(screen.getByLabelText('集（可选）')); await user.type(screen.getByLabelText('集（可选）'), '10'); await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(requests(fetch)).toHaveLength(1)); const body = JSON.parse(requests(fetch)[0][1]!.body as string); expect(body.items.map((entry: ImportInput) => entry.video_path)).toEqual(['/tmp/2.mkv', '/tmp/10.mkv']); expect(body.items.map((entry: ImportInput) => [entry.season, entry.episode])).toEqual([[2, 3], [2, 10]]);
  });

  it('keeps a shared title and manual episode settings when replacing a file or adding more files', async () => {
    const user = userEvent.setup(); const fetch = mockFetch((_url, body) => ({ ...baseJob, type: 'import', status: 'completed', items: body.items.map((entry: ImportInput) => ({ item_id: entry.item_id, name: entry.title, status: 'completed', asset_id: entry.item_id })) }));
    show(<ImportModal projectId="project1" onClose={vi.fn()} />);
    await user.click(screen.getByText('粘贴多个绝对路径')); await user.click(screen.getByLabelText('每行一个视频路径')); await user.paste('/tmp/[A] Anime [10][1080p].mkv\n/tmp/[Z] Anime [02][x265].mkv'); await user.click(screen.getByRole('button', { name: '添加路径列表' }));
    const rows = within(screen.getByLabelText('待导入视频'));
    expect(rows.getAllByRole('button').map((row) => row.textContent)).toEqual(['第 2 集[Z] Anime [02][x265].mkv', '第 10 集[A] Anime [10][1080p].mkv']);
    expect(screen.getAllByLabelText(/^作品名称/)).toHaveLength(1);
    await user.clear(screen.getByLabelText(/^作品名称/)); await user.type(screen.getByLabelText(/^作品名称/), '共同动画名');
    await user.click(screen.getByRole('button', { name: '从零建档' }));
    await user.click(rows.getByRole('button', { name: /第 10 集/ }));
    await user.type(screen.getByLabelText('季（可选）'), '3'); await user.clear(screen.getByLabelText('集（可选）')); await user.type(screen.getByLabelText('集（可选）'), '20');
    await user.clear(screen.getByLabelText(/视频绝对路径/)); await user.type(screen.getByLabelText(/视频绝对路径/), '/fixed/Other.S02E04.mkv');
    expect((screen.getByLabelText(/^作品名称/) as HTMLInputElement).value).toBe('共同动画名');
    expect((screen.getByLabelText('季（可选）') as HTMLInputElement).value).toBe('3'); expect((screen.getByLabelText('集（可选）') as HTMLInputElement).value).toBe('20');
    await user.click(screen.getByLabelText('每行一个视频路径')); await user.paste('/tmp/Anime [01].mkv'); await user.click(screen.getByRole('button', { name: '添加路径列表' })); await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(requests(fetch)).toHaveLength(1)); const body = JSON.parse(requests(fetch)[0][1]!.body as string);
    expect(body.items.map((entry: ImportInput) => entry.title)).toEqual(['共同动画名', '共同动画名', '共同动画名']);
    expect(body.items.map((entry: ImportInput) => [entry.season, entry.episode])).toEqual([[null, 1], [null, 2], [3, 20]]);
    expect(body.items[2].video_path).toBe('/fixed/Other.S02E04.mkv');
  });

  it('explicitly clears project membership and legacy series metadata when choosing unassigned', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => ({ ...baseJob, type: 'import', status: 'completed', items: [] }));
    show(<ImportModal projectId="project1" onClose={vi.fn()} />);
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/ep1.mkv'); await user.selectOptions(screen.getByLabelText('归属项目'), ''); await user.click(screen.getByRole('button', { name: '从零建档' })); await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(requests(fetch)).toHaveLength(1)); const body = JSON.parse(requests(fetch)[0][1]!.body as string); expect(body.project_id).toBeNull(); expect(body.items[0].series).toBe('');
  });

  it('reuses a submission request ID after a network failure', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => { throw new Error('network'); });
    show(<ImportModal onClose={vi.fn()} />);
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/ep1.mkv');
    await user.click(screen.getByRole('button', { name: '从零建档' }));
    await user.click(screen.getByRole('button', { name: /加入资料库/ })); await screen.findByRole('alert');
    await user.click(screen.getByRole('button', { name: /加入资料库/ })); await waitFor(() => expect(requests(fetch)).toHaveLength(2));
    expect(JSON.parse(requests(fetch)[0][1]!.body as string).request_id).toBe(JSON.parse(requests(fetch)[1][1]!.body as string).request_id);
  });
});

describe('Projects, hierarchy, and search', () => {
  it('filters descendant projects and assigns selected videos without copying sources', async () => {
    const user = userEvent.setup(); const value = context(); const fetch = mockFetch(() => ({})); show(<Library />, value);
    await user.selectOptions(screen.getByLabelText('文件夹'), 'parent');
    expect(screen.getByText('第一集')).toBeTruthy(); expect(screen.getByText('第二集')).toBeTruthy();
    await user.click(screen.getByRole('checkbox', { name: '选择当前 2 个视频' }));
    await user.click(screen.getByRole('button', { name: '分析所选视频' })); expect(value.openJobs).toHaveBeenCalledWith([asset, second]);
    await user.selectOptions(screen.getByLabelText('移动到项目'), 'project2'); await user.click(screen.getByRole('button', { name: '调整归属' }));
    await waitFor(() => expect(value.refresh).toHaveBeenCalledOnce());
    expect(requests(fetch)[0][0]).toBe('/api/projects/assign'); expect(JSON.parse(requests(fetch)[0][1]!.body as string)).toEqual({ asset_ids: ['ep1', 'ep2'], project_id: 'project2' });
  });

  it('creates a tagged project in the selected nested folder', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => ({ id: 'new-project' })); show(<Library />);
    await user.selectOptions(screen.getByLabelText('文件夹'), 'child'); await user.click(screen.getByRole('button', { name: '新建项目' }));
    await user.type(screen.getByLabelText('项目名称'), '新动画'); await user.type(within(screen.getByRole('dialog')).getByLabelText(/项目标签/), '科幻，收藏,科幻');
    await user.click(screen.getByRole('button', { name: '保存分类' })); await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(JSON.parse(requests(fetch)[0][1]!.body as string)).toEqual({ name: '新动画', kind: 'animation', folder_id: 'child', tags: ['科幻', '收藏'] });
  });

  it('submits project, recursive folder, and tag filters with the search', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => ({ results: [], degraded: [] })); show(<SearchPage />);
    await user.click(screen.getByRole('button', { name: /筛选条件/ })); await user.selectOptions(screen.getByLabelText('项目'), 'project1'); await user.selectOptions(screen.getByLabelText('文件夹'), 'parent'); await user.selectOptions(screen.getByLabelText('项目标签'), '科幻'); await user.type(screen.getByRole('textbox', { name: /用自然语言搜索/ }), '雨中的车站'); await user.click(screen.getByRole('button', { name: /^检索/ }));
    await waitFor(() => expect(requests(fetch)).toHaveLength(1)); expect(JSON.parse(requests(fetch)[0][1]!.body as string).filters).toEqual({ project_id: 'project1', folder_id: 'parent', tag: '科幻' });
  });
});

describe('Content keyframe review and batch budgets', () => {
  it('requires a scanned preview and builds independent per-episode stages and budgets', async () => {
    const user = userEvent.setup(); const fetch = mockFetch((url, body) => url === '/api/keyframe-plans' ? plan(body.asset_id) : url.endsWith('/estimate') ? { ...estimate, items: body.items.map((entry: any) => ({ ...estimate, asset_id: entry.asset_id })) } : { jobs: [] }); const close = vi.fn(); show(<JobModal assets={[asset, second]} onClose={close} />);
    expect((screen.getByRole('button', { name: '估算用量' }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByLabelText(/画面窗口|每窗口画面/)).toBeNull();
    await user.click(screen.getByRole('button', { name: '先试运行前 60 秒' })); await user.click(screen.getByRole('button', { name: '扫描关键帧' })); await screen.findByAltText('第二集 · 视频 ep2 00:00:00 关键帧');
    expect(requests(fetch).map((call) => call[0])).toEqual(['/api/keyframe-plans', '/api/keyframe-plans']);
    expect(requests(fetch).map((call) => JSON.parse(call[1]!.body as string).end_ms)).toEqual([12000, 8000]);
    await user.clear(screen.getByLabelText(/每集最多模型请求数/)); await user.type(screen.getByLabelText(/每集最多模型请求数/), '50');
    await user.click(screen.getByRole('button', { name: '估算用量' })); await screen.findByRole('button', { name: '重新估算' }); await user.click(screen.getByRole('button', { name: '开始分析' })); await waitFor(() => expect(close).toHaveBeenCalledOnce());
    const body = JSON.parse(requests(fetch).find((call) => call[0] === '/api/jobs/batch')![1]!.body as string);
    expect(body.items.map((entry: any) => entry.stages)).toEqual([['vision'], ['vision', 'subtitle']]); expect(body.items.map((entry: any) => entry.max_requests)).toEqual([50, 50]); expect(body.items.map((entry: any) => entry.keyframe_plan_id)).toEqual(['plan-ep1', 'plan-ep2']); expect(body.items[0]).not.toHaveProperty('window_ms'); expect(body.items[0]).not.toHaveProperty('frames_per_window');
  });

  it('invalidates the selected keyframe plan when the range changes', async () => {
    const user = userEvent.setup(); mockFetch((_url, body) => plan(body.asset_id)); show(<JobModal asset={asset} onClose={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: '扫描关键帧' })); await screen.findByAltText('第一集 · 视频 ep1 00:00:00 关键帧'); await user.type(screen.getByLabelText(/结束时间/), '5');
    expect(screen.getByText('分析范围已变化，请重新扫描关键帧。')).toBeTruthy(); expect((screen.getByRole('button', { name: '估算用量' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('loads the preview of an already cached scan before allowing estimates', async () => {
    const user = userEvent.setup(); const fetch = mockFetch((url) => url === '/api/keyframe-plans' ? { id: 'plan-ep1', status: 'completed', job_id: 'cached' } : plan('ep1')); show(<JobModal asset={asset} onClose={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: '扫描关键帧' })); await screen.findByAltText('第一集 · 视频 ep1 00:00:00 关键帧'); expect(requests(fetch).map((call) => call[0])).toEqual(['/api/keyframe-plans', '/api/keyframe-plans/plan-ep1']);
  });

  it('polls a pending local scan and does not call model endpoints during scanning', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url) => url === '/api/keyframe-plans' ? { ...plan('ep1'), status: 'queued', frames: undefined } : plan('ep1'));
    show(<JobModal asset={asset} onClose={vi.fn()} />); await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
    await waitFor(() => expect(screen.getByAltText('第一集 · 视频 ep1 00:00:00 关键帧')).toBeTruthy(), { timeout: 3000 });
    expect(screen.getByAltText('第一集 · 视频 ep1 00:00:00 关键帧')).toBeTruthy(); expect(requests(fetch).every((call) => call[0].startsWith('/api/keyframe-plans'))).toBe(true);
  });

  it('shows scan progress and cancels a local scan without creating analysis work', async () => {
    const user = userEvent.setup(); const fetch = mockFetch((url) => url.endsWith('/cancel') ? { status: 'cancelled', message: '已取消' } : { ...plan('ep1'), status: 'running', progress: 0.25, message: '本地关键帧扫描 3 / 12 秒', frames: [] });
    show(<JobModal asset={asset} onClose={vi.fn()} />); await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
    expect(await screen.findByText('本地关键帧扫描 3 / 12 秒')).toBeTruthy(); expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('25');
    await user.click(screen.getByRole('button', { name: '取消「第一集 · 视频 ep1」扫描' })); await screen.findByText('已取消');
    expect(requests(fetch).map((call) => call[0])).toEqual(['/api/keyframe-plans', '/api/jobs/scan-ep1/cancel']); expect((screen.getByRole('button', { name: '估算用量' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('controls the batch without replaying successful child jobs', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => ({ jobs: [] })); show(<Jobs />, context({ jobs: [{ ...baseJob, batch_id: 'batch1' }, { ...baseJob, id: 'job2', asset_id: 'ep2', batch_id: 'batch1', status: 'completed' }] }));
    await user.click(screen.getByRole('button', { name: '暂停整批' })); await waitFor(() => expect(requests(fetch)).toHaveLength(1)); expect(requests(fetch)[0][0]).toBe('/api/jobs/batches/batch1/pause'); expect(screen.getByText(/完成 1 \/ 2/)).toBeTruthy();
  });

  it('shows per-episode failures after a partially successful batch control', async () => {
    const user = userEvent.setup(); const value = context({ jobs: [{ ...baseJob, batch_id: 'batch1' }, { ...baseJob, id: 'job2', asset_id: 'ep2', batch_id: 'batch1' }] });
    mockFetch(() => ({ jobs: [{ ...baseJob, status: 'paused' }], failures: [{ job_id: 'job2', message: '项目归属已变化，请重新创建分析任务' }] })); show(<Jobs />, value);
    await user.click(screen.getByRole('button', { name: '暂停整批' })); await screen.findByText(/第二集 · 视频 ep2：项目归属已变化/);
    expect(screen.getByRole('alert').textContent).toContain('1 集未处理成功'); expect(value.notify).toHaveBeenCalledWith('批次部分完成，1 集未能处理，请查看原因', 'error');
  });

  it('removes visual interval settings while keeping independent OCR cadence', async () => {
    const user = userEvent.setup(); show(<Settings />); await user.click(screen.getByRole('button', { name: '分析默认值' })); expect(screen.queryByLabelText(/最长分析窗口|每窗口画面数/)).toBeNull(); expect(screen.getByLabelText(/画面字幕 OCR 采样/)).toBeTruthy();
  });
});
