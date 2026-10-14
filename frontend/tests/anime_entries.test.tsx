import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset } from '../src/types';
import Library from '../src/components/Library';

const anime: Asset = { id: 'anime', work_id: 'work', title: '动画第一集', kind: 'animation', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'container', source_available: true, source_changed: false, created_at: '2026-10-13T00:00:00Z', status: 'ready' };
const legacy: Asset = { ...anime, id: 'legacy', work_id: 'legacy-work', title: '旧资料', kind: 'movie', project_id: 'legacy-project' };
function context(): AppContextValue {
  return { bootstrap: { library_path: '/tmp/ui', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: '0.3' }, assets: [anime, legacy], projects: [{ id: 'legacy-project', name: '已有项目', kind: 'movie', folder_id: null, tags: [], asset_count: 1 }], folders: [], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openJob: vi.fn(), openJobs: vi.fn(), openImport: vi.fn() };
}
function show() { return render(<AppContext.Provider value={context()}><Library /></AppContext.Provider>); }
function mockFetch() {
  const fetch = vi.fn(async (_url: string, _options?: RequestInit) => new Response(JSON.stringify({ id: 'saved-project' }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetch); return fetch;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Animation-focused entry points', () => {
  it('removes movie and series navigation while keeping existing assets accessible', async () => {
    const user = userEvent.setup(); show();
    expect(screen.queryByRole('button', { name: '电影' })).toBeNull();
    expect(screen.queryByRole('button', { name: '剧集' })).toBeNull();
    expect(screen.getByText('旧资料')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: '动画' }));
    expect(screen.getByText('动画第一集')).toBeTruthy();
    expect(screen.queryByText('旧资料')).toBeNull();
    await user.click(screen.getByRole('button', { name: /全部作品/ }));
    expect(screen.getByText('旧资料')).toBeTruthy();
  });

  it('creates animation projects without a movie or series type selector', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(); show();
    await user.click(screen.getByRole('button', { name: '新建项目' }));
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).queryByLabelText('项目类型')).toBeNull();
    expect(within(dialog).getByText(/TV 动画、不同季度和剧场版/)).toBeTruthy();
    await user.type(within(dialog).getByLabelText('项目名称'), '动画剧场版');
    await user.click(within(dialog).getByRole('button', { name: '保存分类' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(fetch.mock.calls[0][0]).toBe('/api/projects');
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({ name: '动画剧场版', kind: 'animation' });
  });

  it('preserves legacy project kind when its name or tags are edited', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(); show();
    await user.selectOptions(screen.getByLabelText('项目'), 'legacy-project');
    await user.click(screen.getByRole('button', { name: '编辑项目与标签' }));
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).queryByLabelText('项目类型')).toBeNull();
    await user.type(within(dialog).getByLabelText(/项目标签/), '收藏');
    await user.click(within(dialog).getByRole('button', { name: '保存分类' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(fetch.mock.calls[0][0]).toBe('/api/projects/legacy-project');
    expect(fetch.mock.calls[0][1]!.method).toBe('PATCH');
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({ kind: 'movie', tags: ['收藏'] });
  });
});
