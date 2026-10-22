import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset } from '../src/types';
import AssetDetail from '../src/components/AssetDetail';
import Settings from '../src/components/Settings';
import { localFileDirectory, pickLocalFiles } from '../src/lib/localFiles';

const asset: Asset = { id: 'local-video', work_id: 'local-work', title: '本机片段', kind: 'movie', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: false, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const detail = { asset, observations: [], subtitles: [], annotations: [], runs: [] };
function context(): AppContextValue {
  return { bootstrap: { library_path: '/tmp/native-files-tests', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: '0.4.0' }, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function mockFetch(handler: (url: string, options?: RequestInit) => unknown | Promise<unknown>) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const data = await handler(url, options);
    return data instanceof Response ? data : new Response(JSON.stringify(data), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((done) => { resolve = done; }); return { promise, resolve }; }
async function showRelocate(value = context()) {
  const user = userEvent.setup();
  render(<AppContext.Provider value={value}><AssetDetail id={asset.id} /></AppContext.Provider>);
  await user.click(await screen.findByRole('button', { name: '定位原文件' }));
  return { user, modal: within(screen.getByRole('dialog', { name: '重新定位原视频' })) };
}
async function showBackup(value = context()) {
  const user = userEvent.setup();
  render(<AppContext.Provider value={value}><Settings /></AppContext.Provider>);
  await user.click(screen.getByRole('button', { name: '本地数据' }));
  return user;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Native local file selection', () => {
  it.each([
    ['C:\\episode.mkv', 'C:\\'],
    ['D:\\动画\\episode.mkv', 'D:\\动画'],
    ['C:/episode.mkv', 'C:/'],
    ['\\\\server\\share\\episode.mkv', '\\\\server\\share'],
    ['/episode.mkv', '/'],
    ['/tmp/动画/episode.mkv', '/tmp/动画'],
    ['', undefined],
    ['episode.mkv', undefined],
  ])('keeps an absolute starting directory for %s', (path, expected) => {
    expect(localFileDirectory(path!)).toBe(expected);
  });

  it('requests multiple subtitle files in the requested starting folder', async () => {
    const paths = ['/tmp/字幕/E01.zh.ass', '/tmp/字幕/E02.zh.ass'];
    const fetch = mockFetch(() => ({ paths, cancelled: false }));
    expect(await pickLocalFiles('subtitle', true, '/tmp/字幕')).toEqual(paths);
    expect(fetch.mock.calls[0][0]).toBe('/api/local-files/pick');
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toEqual({ kind: 'subtitle', multiple: true, directory: '/tmp/字幕' });
  });

  it('treats cancelled selection as an empty result', async () => {
    mockFetch(() => ({ paths: [], cancelled: true }));
    expect(await pickLocalFiles('video')).toEqual([]);
  });

  it('selects an original video through the native picker and validates before saving its location', async () => {
    const path = '/tmp/移动后/本机片段.mkv';
    const fetch = mockFetch((url) => url === '/api/local-files/pick' ? { paths: [path], cancelled: false } : detail);
    const value = context(); const { user, modal } = await showRelocate(value);
    expect((modal.getByRole('button', { name: '校验并更新位置' }) as HTMLButtonElement).disabled).toBe(true);
    expect(modal.queryByLabelText('新的视频绝对路径')).toBeNull();
    await user.click(modal.getByRole('button', { name: '选择原视频' }));
    expect((modal.getByLabelText('已选择的视频') as HTMLInputElement).value).toBe(path);
    expect((modal.getByLabelText('已选择的视频') as HTMLInputElement).readOnly).toBe(true);
    expect(fetch.mock.calls.some((call) => call[0].endsWith('/relocate'))).toBe(false);
    await user.click(modal.getByRole('button', { name: '校验并更新位置' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    const pick = fetch.mock.calls.find((call) => call[0] === '/api/local-files/pick')!;
    expect(JSON.parse(pick[1]!.body as string)).toEqual({ kind: 'video', multiple: false });
    const relocation = fetch.mock.calls.find((call) => call[0].endsWith('/relocate'))!;
    expect(JSON.parse(relocation[1]!.body as string)).toEqual({ video_path: path });
    expect(value.refresh).toHaveBeenCalledOnce();
    expect(value.notify).toHaveBeenCalledWith('原视频位置已更新');
  });

  it('keeps a chosen video after cancelling a replacement and blocks duplicate selection while Finder is open', async () => {
    const pending = deferred<{ paths: string[]; cancelled: boolean }>(); let picks = 0;
    const fetch = mockFetch((url) => url === '/api/local-files/pick' ? ++picks === 1 ? { paths: ['/tmp/original.mkv'], cancelled: false } : pending.promise : detail);
    const { user, modal } = await showRelocate();
    await user.click(modal.getByRole('button', { name: '选择原视频' }));
    await user.click(modal.getByRole('button', { name: '重新选择原视频' }));
    expect((modal.getByRole('button', { name: '正在选择文件' }) as HTMLButtonElement).disabled).toBe(true);
    expect((modal.getByRole('button', { name: '校验并更新位置' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(modal.getByRole('button', { name: '关闭' }));
    expect(screen.getByRole('dialog')).toBeTruthy();
    await act(async () => pending.resolve({ paths: [], cancelled: true }));
    expect((modal.getByLabelText('已选择的视频') as HTMLInputElement).value).toBe('/tmp/original.mkv');
    expect(fetch.mock.calls.filter((call) => call[0] === '/api/local-files/pick')).toHaveLength(2);
  });

  it('shows native picker errors and preserves backend fingerprint rejection until another selection', async () => {
    let picks = 0;
    const fetch = mockFetch((url) => url === '/api/local-files/pick' ? ++picks === 1 ? new Response(JSON.stringify({ detail: '无法打开本机文件选择器' }), { status: 503 }) : { paths: ['/tmp/different.mkv'], cancelled: false } : url.endsWith('/relocate') ? new Response(JSON.stringify({ detail: '所选视频与原文件不一致' }), { status: 409 }) : detail);
    const value = context(); const { user, modal } = await showRelocate(value);
    await user.click(modal.getByRole('button', { name: '选择原视频' }));
    expect(await modal.findByText('无法打开本机文件选择器')).toBeTruthy();
    await user.click(modal.getByRole('button', { name: '选择原视频' }));
    expect(modal.queryByText('无法打开本机文件选择器')).toBeNull();
    await user.click(modal.getByRole('button', { name: '校验并更新位置' }));
    expect(await modal.findByText('所选视频与原文件不一致')).toBeTruthy();
    expect(screen.getByRole('dialog')).toBeTruthy(); expect(value.refresh).not.toHaveBeenCalled();
    expect(fetch.mock.calls.filter((call) => call[0].endsWith('/relocate'))).toHaveLength(1);
  });

  it('restores a ZIP selected through Finder by its local path and shows the report', async () => {
    const path = '/tmp/备份/scenerecall.zip'; const report = { restored: 4, conflicts: [] };
    const fetch = mockFetch((url) => url === '/api/local-files/pick' ? { paths: [path], cancelled: false } : report);
    const value = context(); const user = await showBackup(value);
    expect(document.querySelector('input[type="file"]')).toBeNull();
    expect((screen.getByRole('button', { name: '恢复所选备份' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: '选择资料库备份文件' }));
    expect((screen.getByLabelText('已选择的备份') as HTMLInputElement).value).toBe(path);
    expect(fetch.mock.calls).toHaveLength(1);
    await user.click(screen.getByRole('button', { name: '恢复所选备份' }));
    await waitFor(() => expect(screen.queryByLabelText('已选择的备份')).toBeNull());
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toEqual({ kind: 'backup', multiple: false });
    expect(fetch.mock.calls[1][0]).toBe('/api/backups/restore-local');
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string)).toEqual({ path });
    expect(screen.getByText(/"restored": 4/)).toBeTruthy();
    expect(value.refresh).toHaveBeenCalledOnce();
    expect(value.notify).toHaveBeenCalledWith('备份恢复已完成，请查看恢复报告');
  });

  it('preserves the chosen backup when a replacement dialog is cancelled', async () => {
    let picks = 0;
    mockFetch(() => ++picks === 1 ? { paths: ['/tmp/backup.zip'], cancelled: false } : { paths: [], cancelled: true });
    const user = await showBackup();
    await user.click(screen.getByRole('button', { name: '选择资料库备份文件' }));
    await user.click(screen.getByRole('button', { name: '重新选择备份文件' }));
    expect((screen.getByLabelText('已选择的备份') as HTMLInputElement).value).toBe('/tmp/backup.zip');
    expect((screen.getByRole('button', { name: '恢复所选备份' }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('holds backup selection busy and reports invalid archives without clearing the selection', async () => {
    const pending = deferred<{ paths: string[]; cancelled: boolean }>();
    const fetch = mockFetch((url) => url === '/api/local-files/pick' ? pending.promise : new Response(JSON.stringify({ detail: '不是有效的 SceneRecall 备份' }), { status: 400 }));
    const value = context(); const user = await showBackup(value);
    await user.click(screen.getByRole('button', { name: '选择资料库备份文件' }));
    expect((screen.getByRole('button', { name: '正在选择文件' }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: '恢复所选备份' }) as HTMLButtonElement).disabled).toBe(true);
    await act(async () => pending.resolve({ paths: ['/tmp/invalid.zip'], cancelled: false }));
    await user.click(screen.getByRole('button', { name: '恢复所选备份' }));
    expect(await screen.findByText('不是有效的 SceneRecall 备份')).toBeTruthy();
    expect((screen.getByLabelText('已选择的备份') as HTMLInputElement).value).toBe('/tmp/invalid.zip');
    expect(value.refresh).not.toHaveBeenCalled(); expect(value.notify).not.toHaveBeenCalled();
    expect(fetch.mock.calls).toHaveLength(2);
  });
});
