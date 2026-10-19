import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type ImportInput, type Job } from '../src/types';
import AssetDetail from '../src/components/AssetDetail';
import ImportModal from '../src/components/ImportModal';
import Jobs from '../src/components/Jobs';

const warning = '已忽略 2 段 ASS 特效字幕，正常台词已保留。';
const asset: Asset = { id: 'ep1', work_id: 'anime', title: '合成动画', kind: 'animation', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', subtitle_import_warnings: [warning], source_available: true, source_changed: false, created_at: '2026-10-14T00:00:00Z', status: 'ready' };
const inputs: ImportInput[] = [1, 2].map((episode) => ({ item_id: `item-${episode}`, title: '合成动画', video_path: `/tmp/episode-${episode}.mkv`, kind: 'animation', series: '', season: null, episode, version: '', subtitle_mode: 'external', subtitle_path: `/tmp/episode-${episode}.ass`, subtitle_offset_ms: 0, anime_profile: { mode: 'scratch' } }));
const job: Job = { id: 'import1', type: 'import', status: 'completed', progress: 1, completed: 2, total: 2, created_at: '2026-10-14T00:00:00Z', items: [{ item_id: 'item-1', name: 'episode-1.mkv', status: 'completed', asset_id: asset.id, warnings: [warning] }, { item_id: 'item-2', name: 'episode-2.mkv', status: 'completed', asset_id: 'ep2' }] };

function context(): AppContextValue {
  return { bootstrap: { library_path: '/tmp/import-warning-tests', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' }, assets: [asset], jobs: [job], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function show(ui: ReactNode) { return render(<AppContext.Provider value={context()}>{ui}</AppContext.Provider>); }
function mockFetch(data: unknown) {
  const fetch = vi.fn(async () => new Response(JSON.stringify(data), { status: 200, headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetch); return fetch;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('ASS import warnings', () => {
  it('shows the selected imported file warning without leaking it to another episode', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch({ job, items: inputs });
    show(<ImportModal retryJobId={job.id} onClose={vi.fn()} />);
    await screen.findByText(warning);
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.queryByRole('button', { name: /重试失败文件/ })).toBeNull();
    await user.click(within(screen.getByLabelText('待导入视频')).getByRole('button', { name: /episode-2/ }));
    expect(screen.queryByText(warning)).toBeNull();
    await user.click(within(screen.getByLabelText('待导入视频')).getByRole('button', { name: /episode-1/ }));
    expect(screen.getByText(warning)).toBeTruthy();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('keeps a completed import successful in the queue while showing its file warning', () => {
    show(<Jobs />);
    const warningElement = screen.getByText(warning);
    const row = warningElement.parentElement!;
    expect(within(row).getByText('episode-1.mkv')).toBeTruthy();
    expect(within(row).getByText('已完成')).toBeTruthy();
    expect(within(row).queryByText('失败')).toBeNull();
    expect(screen.queryByRole('button', { name: '重试' })).toBeNull();
  });

  it('keeps the warning visible on the saved asset while allowing subtitle extraction', async () => {
    mockFetch({ asset, observations: [], scenes: [], subtitles: [], annotations: [], runs: [] });
    show(<AssetDetail id={asset.id} />);
    await screen.findByText(warning);
    expect(screen.queryByRole('alert')).toBeNull();
    expect((screen.getByRole('button', { name: '提取 / 切换字幕轨' }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.getByRole('button', { name: '开始画面分析' })).toBeTruthy();
  });
});
