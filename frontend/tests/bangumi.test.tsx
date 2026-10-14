import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type BangumiDossier, type ImportInput, type Project } from '../src/types';
import ImportModal from '../src/components/ImportModal';
import CharacterProfiles from '../src/components/CharacterProfiles';

const dossier = (id = 42, name = '星海'): BangumiDossier => ({ source: 'bangumi', subject: { id, name, name_cn: name, date: '2026-01-01', summary: '来自 Bangumi 的作品简介。', image_url: 'https://lain.bgm.tv/pic/cover.jpg', url: `https://bgm.tv/subject/${id}`, infobox: [] }, characters: [{ id: 9, name: '小夏', name_cn: '小夏', aliases: ['夏同学'], description: '角色现成资料。', image_url: 'https://lain.bgm.tv/pic/character.jpg', url: 'https://bgm.tv/character/9' }], fetched_at: '2026-10-14T00:00:00Z' });
const project: Project = { id: 'anime', name: '星海', kind: 'animation', folder_id: null, tags: [], asset_count: 0 };
function context(projects: Project[] = []): AppContextValue {
  return { bootstrap: { library_path: '/tmp/bangumi-ui', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: '0.3' }, assets: [], projects, jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function show(projects: Project[] = [], projectId?: string) {
  return render(<AppContext.Provider value={context(projects)}><ImportModal onClose={vi.fn()} projectId={projectId} /></AppContext.Provider>);
}
function mockFetch(handler: (url: string, body: any) => unknown | Promise<unknown>) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const body = options?.body ? JSON.parse(String(options.body)) : {};
    const result = url === '/api/media/subtitle-detection' ? { duration_ms: 12000, tracks: [], recommended_stream_index: null, external_candidates: [], recommended_mode: 'auto', recommended_subtitle_path: null, reason: '未找到字幕', scan_complete: true } : await handler(url, body);
    return result instanceof Response ? result : new Response(JSON.stringify(result), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
const importJob = (body: any) => ({ id: 'import-1', type: 'import', status: 'completed', progress: 1, completed: body.items.length, total: body.items.length, created_at: '2026-10-14T00:00:00Z', items: body.items.map((row: ImportInput) => ({ item_id: row.item_id, name: row.title, status: 'completed', asset_id: row.item_id })) });
const importButton = () => screen.getByRole('button', { name: /加入资料库/ }) as HTMLButtonElement;
async function animation(user: ReturnType<typeof userEvent.setup>, title = '星海') {
  await user.type(screen.getByLabelText(/视频绝对路径/), `/tmp/${title}.mkv`);
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Bangumi animation import', () => {
  it('applies the initial project kind, automatically reads an exact match, and shares the selection across episodes', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url, body) => url === '/api/bangumi/search' ? { status: 'matches', candidates: [dossier().subject] } : url === '/api/bangumi/subject' ? dossier() : importJob(body));
    show([project], project.id);
    expect(screen.queryByLabelText('作品类型')).toBeNull();
    expect(importButton().disabled).toBe(true);
    await screen.findByRole('link', { name: '查看 Bangumi 条目' });
    expect(JSON.parse(String(fetch.mock.calls[0][1]?.body))).toEqual({ query: '星海' });
    expect(JSON.parse(String(fetch.mock.calls[1][1]?.body))).toEqual({ reference: '42' });
    await user.click(screen.getByText('粘贴多个绝对路径'));
    await user.type(screen.getByLabelText('每行一个视频路径'), '/tmp/1.mkv\n/tmp/2.mkv');
    await user.click(screen.getByRole('button', { name: '添加路径列表' }));
    await user.click(importButton());
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => url === '/api/imports')).toHaveLength(1));
    const body = JSON.parse(String(fetch.mock.calls.find(([url]) => url === '/api/imports')![1]?.body));
    expect(body.items.map((row: ImportInput) => row.anime_profile)).toEqual([{ mode: 'bangumi', subject_id: 42 }, { mode: 'bangumi', subject_id: 42 }]);
    expect(JSON.stringify(body)).not.toContain('dossier');
    expect(fetch.mock.calls.filter(([url]) => url === '/api/bangumi/search')).toHaveLength(1);
  });

  it('requires explicit selection for a fuzzy match, even when there is only one candidate', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url, body) => url === '/api/bangumi/search' ? { status: 'matches', candidates: [dossier(43, '星海 第二季').subject] } : url === '/api/bangumi/subject' ? dossier(43, '星海 第二季') : importJob(body));
    show(); await animation(user);
    const candidate = await screen.findByRole('button', { name: /使用「星海 第二季」的条目/ });
    expect(importButton().disabled).toBe(true);
    expect(fetch.mock.calls.some(([url]) => url === '/api/bangumi/subject')).toBe(false);
    await user.click(candidate); await screen.findByRole('link', { name: '查看 Bangumi 条目' });
    expect(importButton().disabled).toBe(false);
    await user.click(importButton());
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/imports')).toBe(true));
    const body = JSON.parse(String(fetch.mock.calls.find(([url]) => url === '/api/imports')![1]?.body));
    expect(body.items[0].anime_profile).toEqual({ mode: 'bangumi', subject_id: 43 });
  });

  it('offers a URL fallback after no match and reads the supplied reference before import', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url, body) => url === '/api/bangumi/search' ? { status: 'not_found', candidates: [] } : url === '/api/bangumi/subject' ? dossier(51) : importJob(body));
    show(); await animation(user); expect(fetch).not.toHaveBeenCalled();
    await screen.findByText('没有找到对应的 Bangumi 条目。你可以提供链接，或从零建立档案。');
    expect(importButton().disabled).toBe(true);
    await user.type(screen.getByLabelText(/Bangumi 条目链接/), 'https://bgm.tv/subject/51');
    await user.click(screen.getByRole('button', { name: '读取链接' }));
    await screen.findByRole('link', { name: '查看 Bangumi 条目' });
    expect(JSON.parse(String(fetch.mock.calls.find(([url]) => url === '/api/bangumi/subject')![1]?.body))).toEqual({ reference: 'https://bgm.tv/subject/51' });
    await user.click(importButton());
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/imports')).toBe(true));
  });

  it('allows scratch after a lookup error and keeps a late successful lookup from replacing that choice', async () => {
    const user = userEvent.setup(); let finish!: (value: unknown) => void;
    const pending = new Promise((resolve) => { finish = resolve; });
    const fetch = mockFetch((url, body) => url === '/api/bangumi/search' ? { status: 'error', candidates: [], error: 'Bangumi 暂时不可用' } : url === '/api/bangumi/subject' ? pending : importJob(body));
    show(); await animation(user);
    await screen.findByText('Bangumi 暂时不可用');
    await user.type(screen.getByLabelText(/Bangumi 条目链接/), 'https://bgm.tv/subject/42');
    await user.click(screen.getByRole('button', { name: '读取链接' }));
    await user.click(screen.getByRole('button', { name: '从零建档' }));
    await act(async () => { finish(dossier()); });
    expect(screen.queryByRole('link', { name: '查看 Bangumi 条目' })).toBeNull();
    expect(importButton().disabled).toBe(false);
    await user.click(importButton());
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/imports')).toBe(true));
    expect(JSON.parse(String(fetch.mock.calls.find(([url]) => url === '/api/imports')![1]?.body)).items[0].anime_profile).toEqual({ mode: 'scratch' });
  });

  it('ignores a stale automatic dossier when the animation title changes', async () => {
    const user = userEvent.setup(); let finish!: (value: unknown) => void;
    const pending = new Promise((resolve) => { finish = resolve; });
    const fetch = mockFetch((url, body) => url === '/api/bangumi/search' ? body.query === '星海' ? { status: 'matches', candidates: [dossier().subject] } : { status: 'not_found', candidates: [] } : url === '/api/bangumi/subject' ? pending : importJob(body));
    show(); await animation(user);
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/bangumi/subject')).toBe(true));
    await user.clear(screen.getByLabelText(/作品名称/)); await user.type(screen.getByLabelText(/作品名称/), '新动画');
    await act(async () => { finish(dossier()); });
    await screen.findByText('没有找到对应的 Bangumi 条目。你可以提供链接，或从零建立档案。');
    expect(screen.queryByRole('link', { name: '查看 Bangumi 条目' })).toBeNull();
    expect(importButton().disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: '从零建档' }));
    expect(importButton().disabled).toBe(false);
  });

  it('reuses a saved project archive without lookup and resets when switching projects', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url, body) => url === '/api/bangumi/search' ? { status: 'not_found', candidates: [] } : importJob(body));
    show([{ ...project, anime_profile: { mode: 'bangumi', dossier: dossier() } }, { ...project, id: 'other', name: '另一部动画' }], project.id);
    expect(screen.getByRole('link', { name: '查看 Bangumi 条目' })).toBeTruthy();
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/1.mkv');
    expect(fetch).not.toHaveBeenCalled(); expect(importButton().disabled).toBe(false);
    await user.selectOptions(screen.getByLabelText('归属项目'), 'other');
    expect(importButton().disabled).toBe(true); expect(screen.queryByRole('link', { name: '查看 Bangumi 条目' })).toBeNull();
    await screen.findByText('没有找到对应的 Bangumi 条目。你可以提供链接，或从零建立档案。');
    expect(JSON.parse(String(fetch.mock.calls.find(([url]) => url === '/api/bangumi/search')![1]?.body))).toEqual({ query: '星海' });
    expect((screen.getByLabelText(/^作品名称/) as HTMLInputElement).value).toBe('星海');
  });

  it('shares one unassigned batch archive, keeps it when adding or removing files, and resets all rows on a title change', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url, body) => url === '/api/bangumi/search' ? { status: 'not_found', candidates: [] } : importJob(body));
    show(); await user.click(screen.getByText('粘贴多个绝对路径'));
    await user.click(screen.getByLabelText('每行一个视频路径')); await user.paste('/tmp/星海 [02].mkv\n/tmp/星海 [01].mkv');
    await user.click(screen.getByRole('button', { name: '添加路径列表' }));
    await screen.findByText('没有找到对应的 Bangumi 条目。你可以提供链接，或从零建立档案。');
    expect(fetch.mock.calls.filter(([url]) => url === '/api/bangumi/search')).toHaveLength(1);
    await user.click(screen.getByRole('button', { name: '从零建档' }));
    expect(importButton().disabled).toBe(false);
    const rows = within(screen.getByLabelText('待导入视频'));
    await user.click(rows.getByRole('button', { name: /第 2 集/ }));
    expect(importButton().disabled).toBe(false);
    await user.click(screen.getByLabelText('每行一个视频路径')); await user.paste('/tmp/星海 [03].mkv');
    await user.click(screen.getByRole('button', { name: '添加路径列表' }));
    expect(importButton().disabled).toBe(false);
    await user.click(rows.getByRole('button', { name: /第 1 集/ }));
    await user.click(screen.getByRole('button', { name: '移除当前视频' }));
    expect(importButton().disabled).toBe(false);
    await user.clear(screen.getByLabelText(/作品名称/)); await user.type(screen.getByLabelText(/作品名称/), '另一部动画');
    expect(importButton().disabled).toBe(true);
    expect(screen.queryByText('从零建立档案 · 仅为能够确认姓名的人物建档，后续识别补充出场证据。')).toBeNull();
    await user.click(screen.getByRole('button', { name: '从零建档' }));
    await user.click(importButton());
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/imports')).toBe(true));
    const body = JSON.parse(String(fetch.mock.calls.find(([url]) => url === '/api/imports')![1]?.body));
    expect(body.items).toHaveLength(2);
    expect(body.items.map((row: ImportInput) => row.title)).toEqual(['另一部动画', '另一部动画']);
    expect(body.items.map((row: ImportInput) => row.kind)).toEqual(['animation', 'animation']);
    expect(body.items.map((row: ImportInput) => row.anime_profile)).toEqual([{ mode: 'scratch' }, { mode: 'scratch' }]);
    expect(body.items.map((row: ImportInput) => row.episode)).toEqual([2, 3]);
  });

  it('preserves the original archive selection when retrying a failed import', async () => {
    const user = userEvent.setup();
    const input: ImportInput = { item_id: 'failed', title: '星海', video_path: '/tmp/missing.mkv', kind: 'animation', series: '', season: null, episode: null, version: '', subtitle_mode: 'auto', subtitle_path: null, subtitle_offset_ms: 0, anime_profile: { mode: 'bangumi', subject_id: 42 } };
    const job = { ...importJob({ items: [input] }), id: 'failed-job', status: 'failed', items: [{ item_id: input.item_id, name: input.title, status: 'failed', error: '原视频无法读取' }] };
    const fetch = mockFetch((url, body) => url === '/api/imports/failed-job' ? { job, items: [input] } : importJob(body));
    render(<AppContext.Provider value={context()}><ImportModal retryJobId="failed-job" onClose={vi.fn()} /></AppContext.Provider>);
    await screen.findByText('已选择 Bangumi 条目 #42，导入时读取现成档案。');
    expect(screen.queryByRole('button', { name: '从零建档' })).toBeNull();
    await user.clear(screen.getByLabelText(/视频绝对路径/)); await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/fixed.mkv');
    await user.click(screen.getByRole('button', { name: '重试失败文件' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    expect(fetch.mock.calls[1][0]).toBe('/api/imports/failed-job/retry');
    expect(JSON.parse(String(fetch.mock.calls[1][1]?.body)).items[0].anime_profile).toEqual({ mode: 'bangumi', subject_id: 42 });
  });
});

it('labels seeded public character references separately from video proof', async () => {
  const asset = { id: 'ep1', work_id: 'work1', title: '星海', kind: 'animation' as const, duration_ms: 1000, width: 640, height: 360, subtitle_mode: 'container' as const, source_available: true, source_changed: false, created_at: '2026-10-14T00:00:00Z', status: 'ready' };
  mockFetch(() => ({ scope_id: 'work1', revision: 1, assets: [asset], profiles: [{ id: 'c9', name: '小夏', aliases: [], description: '角色现成资料。', notes: '', appearances: [], appearance_count: 0, external_source: { source: 'bangumi', subject_id: 42, character_id: 9, url: 'https://bgm.tv/character/9', image_url: 'https://lain.bgm.tv/pic/character.jpg' } }] }));
  render(<AppContext.Provider value={context()}><CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={vi.fn()} /></AppContext.Provider>);
  await screen.findByText('Bangumi 初始资料');
  expect(screen.getByRole('link', { name: 'Bangumi 角色资料' }).getAttribute('href')).toBe('https://bgm.tv/character/9');
  expect(screen.getByRole('img', { name: '小夏的 Bangumi 公开参考图' })).toBeTruthy();
  expect((screen.getByRole('button', { name: '查看小夏的当前形象原图' }) as HTMLButtonElement).disabled).toBe(true);
  expect(screen.queryByRole('img', { name: '小夏的当前形象' })).toBeNull();
});
