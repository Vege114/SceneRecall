import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type AssetDetail as Detail, type SubtitleTracks } from '../src/types';
import ImportModal from '../src/components/ImportModal';
import JobModal from '../src/components/JobModal';
import AssetDetail from '../src/components/AssetDetail';
import ExtractSubtitlesModal from '../src/components/ExtractSubtitlesModal';

const asset: Asset = { id: 'synthetic-asset', work_id: 'synthetic-work', title: '合成双语片段', kind: 'animation', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'embedded', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const tracks: SubtitleTracks = { duration_ms: 12000, recommended_stream_index: 2, tracks: [
  { index: 2, codec: 'ass', language: 'zho', title: '简体', default: true, forced: false, supported: true },
  { index: 3, codec: 'ass', language: 'eng', title: 'English', default: false, forced: true, supported: true },
  { index: 4, codec: 'hdmv_pgs_subtitle', language: 'jpn', title: '图像字幕', default: false, forced: false, supported: false, reason: '图像轨暂不支持' },
] };
function context(): AppContextValue {
  return { bootstrap: { library_path: '/tmp/subtitle-ui-tests', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: '0.2.0' }, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function renderApp(ui: ReactNode, value = context()) { return render(<AppContext.Provider value={value}>{ui}</AppContext.Provider>); }
function mockFetch(handler: (url: string, options?: RequestInit) => unknown | Promise<unknown>) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const data = await handler(url, options);
    return data instanceof Response ? data : new Response(JSON.stringify(data), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((done) => { resolve = done; }); return { promise, resolve }; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Local container subtitle import', () => {
  it('defaults to automatic import without a model, an external file, or a preliminary probe', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => ({ ...asset, subtitle_mode: 'container' }));
    const onClose = vi.fn(); const value = context();
    renderApp(<ImportModal onClose={onClose} />, value);
    expect(screen.getByRole('button', { name: /自动提取字幕轨/ }).getAttribute('aria-pressed')).toBe('true');
    await user.type(screen.getByLabelText(/视频绝对路径/), '"/tmp/synthetic.mkv"');
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/assets']);
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({ video_path: '/tmp/synthetic.mkv', subtitle_mode: 'auto', subtitle_path: null, subtitle_offset_ms: 0 });
    expect(value.refresh).toHaveBeenCalledOnce();
  });

  it('shows track metadata, disables bitmap tracks and imports an explicitly selected stream and offset without a model', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url) => url.endsWith('/subtitle-tracks') ? tracks : { ...asset, subtitle_mode: 'container' });
    const onClose = vi.fn(); renderApp(<ImportModal onClose={onClose} />);
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/synthetic.mkv');
    await user.click(screen.getByRole('button', { name: '探测字幕轨' }));
    const select = await screen.findByLabelText(/视频内的字幕轨/);
    expect(screen.getByRole('option', { name: '轨道 2 · zho · 简体 · ass · 默认' })).toBeTruthy();
    expect((screen.getByRole('option', { name: /hdmv_pgs_subtitle/ }) as HTMLOptionElement).disabled).toBe(true);
    await user.selectOptions(select, '3');
    await user.clear(screen.getByLabelText(/时间偏移/)); await user.type(screen.getByLabelText(/时间偏移/), '-0.125');
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string)).toMatchObject({ subtitle_mode: 'container', subtitle_stream_index: 3, subtitle_offset_ms: -125 });
  });

  it('clears selected tracks on path change and ignores an older probe that finishes after the new probe', async () => {
    const user = userEvent.setup(); const stale = deferred<SubtitleTracks>();
    const fetch = mockFetch((url, options) => {
      if (url === '/api/assets') return asset;
      return JSON.parse(options!.body as string).video_path === '/tmp/a.mkv' ? stale.promise : { duration_ms: 12000, recommended_stream_index: 7, tracks: [{ ...tracks.tracks[0], index: 7, title: '新文件轨道' }] };
    });
    const onClose = vi.fn(); renderApp(<ImportModal onClose={onClose} />);
    const path = screen.getByLabelText(/视频绝对路径/);
    await user.type(path, '/tmp/a.mkv'); await user.click(screen.getByRole('button', { name: '探测字幕轨' }));
    await user.clear(path); await user.type(path, '/tmp/b.mkv');
    await user.click(screen.getByRole('button', { name: '探测字幕轨' }));
    const select = await screen.findByLabelText(/视频内的字幕轨/);
    await user.selectOptions(select, '7');
    await act(async () => { stale.resolve(tracks); await stale.promise; });
    expect((select as HTMLSelectElement).value).toBe('7');
    expect(screen.queryByRole('option', { name: /English/ })).toBeNull();
    await user.clear(path); await user.type(path, '/tmp/c.mkv');
    expect(screen.queryByLabelText(/视频内的字幕轨/)).toBeNull();
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    const body = JSON.parse(fetch.mock.calls.at(-1)![1]!.body as string);
    expect(body).toMatchObject({ video_path: '/tmp/c.mkv', subtitle_mode: 'auto' });
    expect(body).not.toHaveProperty('subtitle_stream_index');
  });

  it('allows unmodelled import with no tracks but blocks automatic import for bitmap-only tracks', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((_url, options) => JSON.parse(options!.body as string).video_path === '/tmp/empty.mkv' ? { ...tracks, tracks: [], recommended_stream_index: null } : { ...tracks, tracks: [tracks.tracks[2]], recommended_stream_index: null });
    renderApp(<ImportModal onClose={vi.fn()} />);
    const path = screen.getByLabelText(/视频绝对路径/);
    await user.type(path, '/tmp/empty.mkv'); await user.click(screen.getByRole('button', { name: '探测字幕轨' }));
    await screen.findByText(/未发现容器字幕轨/);
    expect((screen.getByRole('button', { name: /加入资料库/ }) as HTMLButtonElement).disabled).toBe(false);
    await user.clear(path); await user.type(path, '/tmp/bitmap.mkv'); await user.click(screen.getByRole('button', { name: '探测字幕轨' }));
    await screen.findByText(/图像字幕轨不能直接作为画面 OCR/);
    expect((screen.getByRole('button', { name: /加入资料库/ }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: /外挂字幕文件/ }));
    expect((screen.getByRole('button', { name: /加入资料库/ }) as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByLabelText(/字幕文件绝对路径/) as HTMLInputElement).required).toBe(true);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('surfaces probe errors and allows retry without reusing a stale path error', async () => {
    const user = userEvent.setup(); let failed = true;
    mockFetch(() => failed ? new Response(JSON.stringify({ detail: '视频路径不可读' }), { status: 400 }) : tracks);
    renderApp(<ImportModal onClose={vi.fn()} />);
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/missing.mkv');
    await user.click(screen.getByRole('button', { name: '探测字幕轨' }));
    await screen.findByRole('alert');
    await user.clear(screen.getByLabelText(/视频绝对路径/));
    expect(screen.queryByText('视频路径不可读')).toBeNull();
    failed = false;
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/synthetic.mkv');
    await user.click(screen.getByRole('button', { name: '探测字幕轨' }));
    await screen.findByLabelText(/视频内的字幕轨/);
  });

  it('does not carry a hidden text-track offset into explicit OCR import', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => asset); const onClose = vi.fn(); const value = context();
    value.bootstrap.settings.bindings.subtitle = 'subtitle-test';
    renderApp(<ImportModal onClose={onClose} />, value);
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/burned.mp4');
    await user.clear(screen.getByLabelText(/时间偏移/)); await user.type(screen.getByLabelText(/时间偏移/), '2.5');
    await user.click(screen.getByRole('button', { name: /画面字幕 OCR/ }));
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({ subtitle_mode: 'embedded', subtitle_offset_ms: 0 });
  });
});

describe('Existing asset extraction and OCR exclusion', () => {
  it('converts an existing OCR asset, reloads current subtitles, and shows the local source', async () => {
    const user = userEvent.setup(); const value = context();
    let detail: Detail = { asset, observations: [], subtitles: [], annotations: [], runs: [] };
    const fetch = mockFetch((url, options) => {
      if (url.endsWith('/subtitle-tracks')) return tracks;
      if (url.endsWith('/subtitles/extract')) {
        const body = JSON.parse(options!.body as string);
        detail = { ...detail, asset: { ...asset, subtitle_mode: 'container', subtitle_track: tracks.tracks[1], subtitle_stream_index: body.subtitle_stream_index }, subtitles: [{ id: 'local-text', start_ms: 1000, end_ms: 2000, text: 'Synthetic searchable dialogue', language: 'eng', source: 'container', review_status: 'confirmed' }] };
        return detail.asset;
      }
      return detail;
    });
    renderApp(<AssetDetail id={asset.id} />, value);
    await user.click(await screen.findByRole('button', { name: '提取 / 切换字幕轨' }));
    const modal = within(screen.getByRole('dialog'));
    const select = await modal.findByLabelText(/视频内的字幕轨/);
    expect((select as HTMLSelectElement).value).toBe('2');
    await user.selectOptions(select, '3');
    await user.clear(modal.getByLabelText(/时间偏移/)); await user.type(modal.getByLabelText(/时间偏移/), '1.25');
    await user.click(modal.getByRole('button', { name: '提取并更新台词' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(screen.getByText('容器字幕轨')).toBeTruthy();
    expect(screen.getByText('eng · English · ass')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: /台词 1/ }));
    expect(screen.getByText('Synthetic searchable dialogue')).toBeTruthy();
    expect(value.refresh).toHaveBeenCalledOnce();
    expect(fetch.mock.calls.map((call) => call[0])).toEqual([`/api/assets/${asset.id}`, `/api/assets/${asset.id}/subtitle-tracks`, `/api/assets/${asset.id}/subtitles/extract`, `/api/assets/${asset.id}`]);
    expect(JSON.parse(fetch.mock.calls[2][1]!.body as string)).toEqual({ subtitle_stream_index: 3, subtitle_offset_ms: 1250 });
  });

  it('keeps the current supported stream and offset when reopening the extraction dialog', async () => {
    mockFetch(() => tracks);
    renderApp(<ExtractSubtitlesModal asset={{ ...asset, subtitle_mode: 'container', subtitle_stream_index: 3, subtitle_offset_ms: -125 }} onClose={vi.fn()} onSaved={vi.fn()} />);
    const select = await screen.findByLabelText(/视频内的字幕轨/);
    expect((select as HTMLSelectElement).value).toBe('3');
    expect((screen.getByLabelText(/时间偏移/) as HTMLInputElement).value).toBe('-0.125');
  });

  it.each(['missing', 'running'])('disables extraction for %s source or active analysis', async (state) => {
    const value = context();
    if (state === 'running') value.jobs = [{ id: 'analysis', asset_id: asset.id, type: 'analysis', status: 'running', progress: 0, total: 1, completed: 0, created_at: '2026-01-01T00:00:00Z' }];
    mockFetch(() => ({ asset: { ...asset, source_available: state !== 'missing' }, observations: [], subtitles: [], annotations: [], runs: [] }));
    renderApp(<AssetDetail id={asset.id} />, value);
    expect((await screen.findByRole('button', { name: '提取 / 切换字幕轨' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('preserves review warnings when a container cue was clipped to the video timeline', async () => {
    const user = userEvent.setup();
    mockFetch(() => ({ asset: { ...asset, subtitle_mode: 'container' }, observations: [], subtitles: [{ id: 'clipped', start_ms: 11000, end_ms: 12000, text: 'Synthetic clipped cue', language: 'eng', source: 'container', review_status: 'needs_review' }], annotations: [], runs: [] }));
    renderApp(<AssetDetail id={asset.id} />);
    await user.click(await screen.findByRole('button', { name: /台词 1/ }));
    expect(screen.getByText('待复核')).toBeTruthy();
  });

  it('retries asset probing after failure and leaves extraction disabled when no text tracks are available', async () => {
    const user = userEvent.setup(); let failed = true;
    const fetch = mockFetch(() => failed ? new Response(JSON.stringify({ detail: '媒体暂时不可读' }), { status: 409 }) : { ...tracks, tracks: [], recommended_stream_index: null });
    renderApp(<ExtractSubtitlesModal asset={asset} onClose={vi.fn()} onSaved={vi.fn()} />);
    await screen.findByText('媒体暂时不可读');
    expect((screen.getByRole('button', { name: '提取并更新台词' }) as HTMLButtonElement).disabled).toBe(true);
    failed = false;
    await user.click(screen.getByRole('button', { name: '重新探测字幕轨' }));
    await screen.findByText(/原视频未发现容器字幕轨/);
    expect((screen.getByRole('button', { name: '提取并更新台词' }) as HTMLButtonElement).disabled).toBe(true);
    expect(fetch.mock.calls.every((call) => call[0].endsWith('/subtitle-tracks'))).toBe(true);
  });

  it('disables OCR for container subtitles and submits only the independent visual stage', async () => {
    const user = userEvent.setup(); const value = context(); value.bootstrap.settings.bindings.vision = 'vision-test';
    const fetch = mockFetch((url) => url.endsWith('/estimate') ? { duration_ms: 12000, estimated_frames: 8, estimated_requests: 2, cost_estimate: null, warnings: [] } : { id: 'visual-job' });
    const onClose = vi.fn();
    renderApp(<JobModal asset={{ ...asset, subtitle_mode: 'container' }} onClose={onClose} />, value);
    const ocr = screen.getByRole('checkbox', { name: /画面字幕 OCR/ }) as HTMLInputElement;
    expect(ocr.disabled).toBe(true); expect(ocr.checked).toBe(false);
    expect(screen.getByText('容器文本字幕已在本地提取，无需发送字幕图片')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: '估算用量' }));
    await screen.findByRole('button', { name: '重新估算' });
    await user.click(screen.getByRole('button', { name: /开始分析/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(fetch.mock.calls.map((call) => JSON.parse(call[1]!.body as string).stages)).toEqual([['vision'], ['vision']]);
  });
});
