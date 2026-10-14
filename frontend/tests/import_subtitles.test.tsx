import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type ImportInput, type SubtitleDetection } from '../src/types';
import { pairSubtitles } from '../src/importSubtitles';
import { inferAnimeFilename } from '../src/importFilenames';
import ImportModal from '../src/components/ImportModal';

const detection: SubtitleDetection = { duration_ms: 3000, tracks: [], recommended_stream_index: null, external_candidates: [], recommended_mode: 'auto', recommended_subtitle_path: null, reason: '未发现匹配字幕', scan_complete: true };
const container: SubtitleDetection = { ...detection, recommended_mode: 'container', recommended_stream_index: 2, reason: '已自动选择字幕轨', tracks: [{ index: 2, codec: 'ass', language: 'zho', title: '简体', default: true, forced: false, supported: true }] };
const row = (path: string, id = path): ImportInput => ({ item_id: id, video_path: path, ...inferAnimeFilename(path), kind: 'animation', series: '', version: '', subtitle_mode: 'auto', subtitle_path: null, subtitle_offset_ms: 0 });
const context = (): AppContextValue => ({ bootstrap: { library_path: '/tmp/test', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' }, assets: [], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() });
function show() { const value = context(); render(<AppContext.Provider value={value}><ImportModal onClose={vi.fn()} /></AppContext.Provider>); return value; }
function mockFetch(handler: (url: string, body: any) => unknown | Promise<unknown>) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => new Response(JSON.stringify(await handler(url, options?.body ? JSON.parse(String(options.body)) : {})), { status: 200, headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetch); return fetch;
}
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((done) => { resolve = done; }); return { promise, resolve }; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Multiple subtitle pairing', () => {
  it('pairs reversed Finder selections by file and episode identity', () => {
    const rows = [row('/video/[Video] Show [01][1080p].mkv', 'one'), row('/video/[Video] Show [02][1080p].mkv', 'two')];
    expect(pairSubtitles(rows, ['/subs/[Subs] Show [02].chs.ass', '/subs/[Subs] Show [01].chs.ass'])).toEqual({ one: '/subs/[Subs] Show [01].chs.ass', two: '/subs/[Subs] Show [02].chs.ass' });
  });
  it('leaves competing editions, other seasons and unrelated shows for manual selection', () => {
    const rows = [row('/video/Show.S01E02.mkv', 'one')];
    expect(pairSubtitles(rows, ['/subs/Show.S02E02.ass', '/subs/Other.S01E02.ass'])).toEqual({});
    expect(pairSubtitles(rows, ['/a/Show.S01E02.ass', '/b/Show.S01E02.ass'])).toEqual({});
    expect(pairSubtitles([row('/video/Show[2020].E01.mkv', 'one')], ['/subs/Show[2021].E01.ass'])).toEqual({});
  });
  it('does not reuse one subtitle for two indistinguishable videos', () => {
    expect(pairSubtitles([row('/a/Show.E01.mkv', 'one'), row('/b/Show.E01.mkv', 'two')], ['/subs/Show.E01.ass'])).toEqual({});
  });
});

describe('Automatic detection with native file selection', () => {
  it('automatically detects every selected video and still allows manual source changes', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url) => url.endsWith('/pick') ? { paths: ['/video/Show.E02.mkv', '/video/Show.E01.mkv'], cancelled: false } : url.endsWith('/subtitle-detection') ? container : { status: 'not_found', candidates: [] });
    show(); await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => url.endsWith('/subtitle-detection'))).toHaveLength(2));
    expect((await screen.findByLabelText(/视频内的字幕轨/)) as HTMLSelectElement).toHaveProperty('value', '2');
    await user.click(within(screen.getByLabelText('待导入视频')).getByRole('button', { name: /第 2 集/ }));
    expect((screen.getByLabelText(/视频内的字幕轨/) as HTMLSelectElement).value).toBe('2');
    await user.click(screen.getByRole('button', { name: /外挂字幕文件/ }));
    expect(screen.getByRole('button', { name: /外挂字幕文件/ }).getAttribute('aria-pressed')).toBe('true');
    expect(screen.queryByLabelText(/字幕文件绝对路径/)).toBeNull();
    expect(fetch.mock.calls[0][0]).toBe('/api/local-files/pick');
    expect(JSON.parse(String(fetch.mock.calls[0][1]?.body))).toEqual({ kind: 'video', multiple: true });
  });
  it('automatically selects a recursive external match and offers alternate candidates', async () => {
    const user = userEvent.setup(); const path = '/video/Subs/Show.E01.chs.ass';
    mockFetch((url) => url.endsWith('/pick') ? { paths: ['/video/Show.E01.mkv'], cancelled: false } : url.endsWith('/subtitle-detection') ? { ...detection, recommended_mode: 'external', recommended_subtitle_path: path, reason: '已匹配子文件夹外挂字幕', external_candidates: [{ path, name: 'Show.E01.chs.ass', relative_path: 'Subs/Show.E01.chs.ass', match: 'exact', language: 'zh-Hans', score: 100 }] } : { status: 'not_found', candidates: [] });
    show(); await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    expect((await screen.findByLabelText(/这集的外挂字幕/)) as HTMLSelectElement).toHaveProperty('value', path);
    expect(screen.getByRole('option', { name: 'Subs/Show.E01.chs.ass' })).toBeTruthy();
    await user.click(screen.getByRole('button', { name: /自动提取字幕轨/ }));
    expect(screen.getByRole('button', { name: /自动提取字幕轨/ }).getAttribute('aria-pressed')).toBe('true');
  });
  it('keeps a manual external choice when an older automatic detection finishes', async () => {
    const user = userEvent.setup(); const pending = deferred<SubtitleDetection>();
    const fetch = mockFetch((url, body) => url.endsWith('/pick') ? { paths: body.kind === 'video' ? ['/video/Show.E01.mkv'] : ['/chosen/manual.ass'], cancelled: false } : url.endsWith('/subtitle-detection') ? pending.promise : { status: 'not_found', candidates: [] });
    show(); await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.endsWith('/subtitle-detection'))).toBe(true));
    await user.click(screen.getByRole('button', { name: /外挂字幕文件/ }));
    await user.click(screen.getByRole('button', { name: '选择本机字幕文件（可多选）' }));
    await act(async () => { pending.resolve(container); await pending.promise; });
    expect(screen.getByRole('button', { name: /外挂字幕文件/ }).getAttribute('aria-pressed')).toBe('true');
    expect((screen.getByLabelText(/这集的外挂字幕/) as HTMLSelectElement).value).toBe('/chosen/manual.ass');
  });
  it('assigns multiple Finder subtitles across episodes and preserves their individual offsets', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url, body) => url.endsWith('/pick') ? { paths: body.kind === 'video' ? ['/video/Show.E01.mkv', '/video/Show.E02.mkv'] : ['/subs/Show.E02.ass', '/subs/Show.E01.ass'], cancelled: false } : url.endsWith('/subtitle-detection') ? detection : url === '/api/imports' ? { id: 'import', type: 'import', status: 'completed', items: body.items.map((value: ImportInput) => ({ item_id: value.item_id, status: 'completed', asset_id: value.item_id })) } : { status: 'not_found', candidates: [] });
    show(); await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    await user.click(screen.getByRole('button', { name: /外挂字幕文件/ }));
    await user.click(screen.getByRole('button', { name: '选择本机字幕文件（可多选）' }));
    expect((screen.getByLabelText(/这集的外挂字幕/) as HTMLSelectElement).value).toBe('/subs/Show.E01.ass');
    await user.clear(screen.getByLabelText(/时间偏移/)); await user.type(screen.getByLabelText(/时间偏移/), '0.25');
    await user.click(within(screen.getByLabelText('待导入视频')).getByRole('button', { name: /第 2 集/ }));
    expect((screen.getByLabelText(/这集的外挂字幕/) as HTMLSelectElement).value).toBe('/subs/Show.E02.ass');
    expect((screen.getByLabelText(/时间偏移/) as HTMLInputElement).value).toBe('0');
    await user.click(screen.getByRole('button', { name: '从零建档' }));
    await waitFor(() => expect((screen.getByRole('button', { name: /加入资料库/ }) as HTMLButtonElement).disabled).toBe(false));
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/imports')).toBe(true));
    const body = JSON.parse(String(fetch.mock.calls.find(([url]) => url === '/api/imports')![1]?.body));
    expect(body.items.map((value: ImportInput) => [value.subtitle_path, value.subtitle_offset_ms])).toEqual([['/subs/Show.E01.ass', 250], ['/subs/Show.E02.ass', 0]]);
  });
  it('treats a cancelled native picker as unchanged input', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => ({ paths: [], cancelled: true })); const value = show();
    await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    expect((screen.getByLabelText(/视频绝对路径/) as HTMLInputElement).value).toBe('');
    expect(screen.queryByRole('alert')).toBeNull(); expect(value.notify).not.toHaveBeenCalled();
    expect(fetch).toHaveBeenCalledOnce();
  });
  it('does not keep submission blocked after removing a video with an unfinished probe', async () => {
    const user = userEvent.setup(); const pending = deferred<SubtitleDetection>();
    const fetch = mockFetch((url, body) => url.endsWith('/pick') ? { paths: ['/video/Show.E01.mkv', '/video/Show.E02.mkv'], cancelled: false } : url.endsWith('/subtitle-detection') ? body.video_path.includes('E01') ? pending.promise : detection : { status: 'not_found', candidates: [] });
    show(); await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => url.endsWith('/subtitle-detection'))).toHaveLength(2));
    await user.click(screen.getByRole('button', { name: '移除当前视频' }));
    await user.click(screen.getByRole('button', { name: '从零建档' }));
    await waitFor(() => expect((screen.getByRole('button', { name: /加入资料库/ }) as HTMLButtonElement).disabled).toBe(false));
    await act(async () => { pending.resolve(container); await pending.promise; });
    expect(screen.queryByLabelText(/视频内的字幕轨/)).toBeNull();
    expect(screen.getByText('Show.E02.mkv', { selector: '.selected-local-file strong' })).toBeTruthy();
  });
  it('limits automatic probes to three active videos and continues through the queue', async () => {
    const user = userEvent.setup(); const pending = Array.from({ length: 4 }, () => deferred<SubtitleDetection>()); let probes = 0;
    mockFetch((url) => url.endsWith('/pick') ? { paths: [1, 2, 3, 4].map((value) => `/video/Show.E0${value}.mkv`), cancelled: false } : url.endsWith('/subtitle-detection') ? pending[probes++].promise : { status: 'not_found', candidates: [] });
    show(); await user.click(screen.getByRole('button', { name: '选择多个视频' }));
    await waitFor(() => expect(probes).toBe(3));
    await act(async () => { pending[0].resolve(detection); await pending[0].promise; });
    await waitFor(() => expect(probes).toBe(4));
    await act(async () => { pending.slice(1).forEach((value) => value.resolve(detection)); await Promise.all(pending.map((value) => value.promise)); });
  });
});
