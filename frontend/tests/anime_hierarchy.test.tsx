import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type BangumiDossier } from '../src/types';
import Library from '../src/components/Library';
import SearchPage from '../src/components/Search';
import ImportModal from '../src/components/ImportModal';

const base: Asset = { id: 'ep1', work_id: 'work1', anime_id: 'anime_tv', anime_title: '星海 TV', title: '星海 TV', kind: 'animation', project_id: 'series', project_name: '星海系列', season: 1, episode: 1, duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-10-14T00:00:00Z', status: 'ready', anime_profile: { mode: 'scratch' } };
const episodes = Array.from({ length: 12 }, (_, i) => ({ ...base, id: `ep${i + 1}`, episode: i + 1 }));
const film = { ...base, id: 'film', anime_id: 'anime_film', anime_title: '星海 剧场版', title: '星海 剧场版', season: null, episode: null };
const elsewhere = { ...base, id: 'elsewhere', project_id: 'other', anime_id: 'anime_other', title: '其他动画' };
function context(): AppContextValue { return { bootstrap: { library_path: '/tmp/hierarchy', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: '0.4.0' }, assets: [...episodes, film, elsewhere], projects: [{ id: 'series', name: '星海系列', kind: 'animation', folder_id: null, tags: [], asset_count: 13 }, { id: 'other', name: '其他系列', kind: 'animation', folder_id: null, tags: [], asset_count: 1 }], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn(), openJobs: vi.fn() }; }
function show(ui: ReactNode, value = context()) { render(<AppContext.Provider value={value}>{ui}</AppContext.Provider>); return value; }
function mockFetch() { const fetch = vi.fn(async (_url: string, _options?: RequestInit) => new Response(JSON.stringify({ results: [], degraded: [], status: 'not_found', candidates: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })); vi.stubGlobal('fetch', fetch); return fetch; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Project, anime, and episode navigation', () => {
  it('filters all restored episodes by the displayed animation name', async () => {
    const value = context(); value.assets = episodes.map((asset) => ({ ...asset, anime_title: '少女歌剧', title: `English Release [${asset.episode}]` }));
    const user = userEvent.setup(); show(<Library />, value);
    await user.type(screen.getByLabelText('按名称筛选作品'), '少女歌剧');
    expect(screen.getByText('12 集 · 12 个视频')).toBeTruthy();
    expect(screen.getByText('少女歌剧 · 第 1 季')).toBeTruthy();
    expect(screen.getByRole('checkbox', { name: '选择当前 12 个视频' })).toBeTruthy();
  });
  it('presents twelve episodes as one closed anime group and keeps the film separate', async () => {
    const user = userEvent.setup(); show(<Library />);
    const series = screen.getByRole('region', { name: '星海系列' });
    expect(series.querySelectorAll('details.anime-group')).toHaveLength(2);
    const tv = within(series).getByText('星海 TV · 第 1 季').closest('details') as HTMLDetailsElement;
    expect(tv.open).toBe(false);
    expect(within(series).getByText('12 集 · 12 个视频')).toBeTruthy();
    await user.click(within(tv).getByText('星海 TV · 第 1 季'));
    expect(tv.open).toBe(true);
    const links = within(tv).getAllByRole('link', { name: '检索本集' });
    expect(links).toHaveLength(12);
    expect(links[0].getAttribute('href')).toContain('asset_id=ep1');
    expect(links[11].getAttribute('href')).toContain('asset_id=ep12');
    expect(within(tv).getByRole('link', { name: '检索这部动画' }).getAttribute('href')).toBe('#/search?project_id=series&anime_id=anime_tv');
  });

  it('cascades scope choices and submits global, project, anime, and exact episode filters', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(); show(<SearchPage />);
    await user.type(screen.getByRole('textbox', { name: /用自然语言搜索/ }), '约定');
    const submit = () => user.click(screen.getByRole('button', { name: /^检索/ }));
    await submit();
    await user.selectOptions(screen.getByLabelText('项目'), 'series'); await submit();
    expect(within(screen.getByLabelText('动画 / 季度')).queryByRole('option', { name: /其他动画/ })).toBeNull();
    await user.selectOptions(screen.getByLabelText('动画 / 季度'), 'anime_tv'); await submit();
    expect(within(screen.getByLabelText('单集 / 视频')).queryByRole('option', { name: /剧场版/ })).toBeNull();
    await user.selectOptions(screen.getByLabelText('单集 / 视频'), 'ep12'); await submit();
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(4));
    expect(fetch.mock.calls.map((call) => JSON.parse(call[1]!.body as string).filters)).toEqual([{}, { project_id: 'series' }, { project_id: 'series', anime_id: 'anime_tv' }, { project_id: 'series', anime_id: 'anime_tv', asset_id: 'ep12' }]);
    await user.selectOptions(screen.getByLabelText('项目'), 'other');
    expect((screen.getByLabelText('动画 / 季度') as HTMLSelectElement).value).toBe('');
    expect((screen.getByLabelText('单集 / 视频') as HTMLSelectElement).value).toBe('');
  });

  it('opens the exact episode scope from a library link and can restore global search', async () => {
    window.location.hash = '#/search?project_id=series&anime_id=anime_tv&asset_id=ep2';
    const user = userEvent.setup(); show(<SearchPage />);
    expect((screen.getByLabelText('单集 / 视频') as HTMLSelectElement).value).toBe('ep2');
    await user.click(screen.getByRole('button', { name: '恢复全局范围' }));
    expect((screen.getByLabelText('项目') as HTMLSelectElement).value).toBe('');
    expect((screen.getByLabelText('动画 / 季度') as HTMLSelectElement).value).toBe('');
    expect((screen.getByLabelText('单集 / 视频') as HTMLSelectElement).value).toBe('');
  });

  it('reuses the selected anime archive and starts a separate archive for a new film in the same project', async () => {
    const dossier: BangumiDossier = { source: 'bangumi', subject: { id: 42, name: '星海 TV', name_cn: '星海 TV', url: 'https://bgm.tv/subject/42' }, characters: [], fetched_at: '2026-10-14T00:00:00Z' };
    const value = context(); value.assets = [{ ...base, anime_profile: { mode: 'bangumi', dossier } }];
    const user = userEvent.setup(); const fetch = mockFetch(); show(<ImportModal projectId="series" onClose={vi.fn()} />, value);
    await user.selectOptions(screen.getByLabelText(/^已有动画 \/ 季度/), 'anime_tv');
    expect((screen.getByLabelText(/^作品名称/) as HTMLInputElement).value).toBe('星海 TV');
    expect(screen.getByRole('link', { name: '查看 Bangumi 条目' })).toBeTruthy();
    await user.clear(screen.getByLabelText(/^作品名称/)); await user.type(screen.getByLabelText(/^作品名称/), '星海 剧场版');
    expect(screen.queryByRole('link', { name: '查看 Bangumi 条目' })).toBeNull();
    await screen.findByText('没有找到对应的 Bangumi 条目。你可以提供链接，或从零建立档案。');
    expect(JSON.parse(fetch.mock.calls.at(-1)![1]!.body as string)).toEqual({ query: '星海 剧场版' });
  });
});
