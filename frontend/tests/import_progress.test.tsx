import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, within } from '@testing-library/react';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type ImportInput, type Job } from '../src/types';
import ImportModal from '../src/components/ImportModal';
import Jobs from '../src/components/Jobs';

const inputs: ImportInput[] = [1, 2].map((episode) => ({ item_id: `item-${episode}`, title: '测试动画', video_path: `D:/videos/episode-${episode}.mkv`, kind: 'animation', series: '', season: 1, episode, version: '', subtitle_mode: 'external', subtitle_path: `D:/videos/episode-${episode}.ass`, subtitle_offset_ms: 0, anime_profile: { mode: 'scratch' } }));
const baseJob: Job = { id: 'import1', type: 'import', status: 'running', progress: 0.5, completed: 1, total: 2, created_at: '2026-10-18T00:00:00Z', items: [{ item_id: 'item-1', name: 'episode-1.mkv', status: 'completed', asset_id: 'asset-1' }, { item_id: 'item-2', name: 'episode-2.mkv', status: 'running', asset_id: 'asset-2' }] };

function context(job: Job): AppContextValue {
  return { bootstrap: { library_path: 'D:/import-tests', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' }, assets: [], jobs: [job], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function show(ui: ReactNode, job = baseJob) { return render(<AppContext.Provider value={context(job)}>{ui}</AppContext.Provider>); }
function mockImport(job: Job) {
  const fetch = vi.fn(async () => new Response(JSON.stringify({ job, items: inputs }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Import phase progress', () => {
  it.each([
    ['import_profile', '正在同步作品档案'],
    ['import_index', '正在建立搜索索引'],
  ])('shows %s in the dialog and counts only finished files', async (stage, label) => {
    const job = { ...baseJob, stage };
    mockImport(job);
    show(<ImportModal retryJobId={job.id} onClose={vi.fn()} />, job);
    expect(await screen.findByText('1 / 2 个视频已导入')).toBeTruthy();
    expect(screen.getAllByText(label)).toHaveLength(2);
    expect(screen.queryByText('正在本机校验与提取字幕')).toBeNull();
    const rows = within(screen.getByLabelText('待导入视频'));
    expect(within(rows.getByRole('button', { name: /episode-2/ })).getByText(label)).toBeTruthy();
    expect(within(rows.getByRole('button', { name: /episode-1/ })).queryByText(label)).toBeNull();
  });

  it('shows the current file message while using its explicit phase in the list', async () => {
    const job: Job = { ...baseJob, stage: 'import', message: '正在建立搜索索引：episode-2.mkv', items: baseJob.items?.map((item) => item.status === 'running' ? { ...item, phase: 'import_index' } : item) };
    mockImport(job);
    show(<ImportModal retryJobId={job.id} onClose={vi.fn()} />, job);
    expect(await screen.findByText(job.message!)).toBeTruthy();
    expect(within(screen.getByLabelText('待导入视频')).getByText('正在建立搜索索引')).toBeTruthy();
    expect(screen.queryByText('正在本机校验与提取字幕')).toBeNull();
  });

  it('keeps running imports without phase fields readable', async () => {
    mockImport(baseJob);
    show(<ImportModal retryJobId={baseJob.id} onClose={vi.fn()} />);
    await screen.findByText('1 / 2 个视频已导入');
    expect(screen.getAllByText('正在本机校验与提取字幕')).toHaveLength(2);
  });

  it.each([
    ['import_profile', '同步作品档案', '正在同步作品档案'],
    ['import_index', '建立搜索索引', '正在建立搜索索引'],
  ])('shows %s in the task queue without raw stage names', (stage, title, label) => {
    const job = { ...baseJob, stage };
    show(<Jobs />, job);
    expect(screen.getByText(new RegExp(`批量导入 · ${title}`))).toBeTruthy();
    expect(screen.getAllByText(label)).toHaveLength(2);
    expect(screen.queryByText(stage)).toBeNull();
    expect(screen.getByText('1 / 2')).toBeTruthy();
  });

  it('keeps completed legacy imports from displaying an active phase', () => {
    const job: Job = { ...baseJob, status: 'completed', progress: 1, completed: 2, items: baseJob.items?.map((item) => ({ ...item, status: 'completed' })) };
    show(<Jobs />, job);
    expect(screen.getByText('导入完成')).toBeTruthy();
    expect(screen.queryByText('正在本机校验与提取字幕')).toBeNull();
  });
});
