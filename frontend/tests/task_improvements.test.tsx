import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type Job, type KeyframePlan, type Profile } from '../src/types';
import { assetTaskLabel } from '../src/taskHelpers';
import JobModal from '../src/components/JobModal';
import Jobs from '../src/components/Jobs';
import CharacterProfiles from '../src/components/CharacterProfiles';
import { DeepenSceneButton, StageEstimates } from '../src/components/SceneAnalysis';

const asset: Asset = { id: 'episode-1', work_id: 'work', title: '少女歌剧', anime_title: '少女歌剧', season: 1, episode: 1, source_filename: 'revuestarlight.S01E01.mkv', kind: 'animation', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const model: Profile = { id: 'api', name: '测试连接', base_url: 'https://example.invalid/v1', model: 'test-model', capabilities: ['preparation', 'vision', 'subtitle'], secret_mode: 'none' };
const second = { ...asset, id: 'episode-2', episode: 2, source_filename: 'revuestarlight.S01E02.mkv' };
const plan = (assetId: string): KeyframePlan => ({ id: `plan-${assetId}`, job_id: `scan-${assetId}`, status: 'completed', frames: [], frame_count: 1 });
function context(assets = [asset]): AppContextValue {
  return { bootstrap: { library_path: '/tmp/task-ui-tests', settings: { defaults: DEFAULTS, bindings: { preparation: model.id, vision: model.id, subtitle: model.id } }, provider_profiles: [model], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' }, assets, jobs: [], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function mount(ui: ReactNode, app = context()) { return render(<AppContext.Provider value={app}>{ui}</AppContext.Provider>); }
function response(value: unknown, status = 200) { return new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } }); }
function mockFetch(handler: (url: string, body: any) => unknown | Promise<unknown>) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const result = await handler(url, options?.body ? JSON.parse(options.body as string) : {});
    return result instanceof Response ? result : response(result);
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((done) => { resolve = done; }); return { promise, resolve }; }
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Parallel local scans', () => {
  it('submits several episodes before the first response, limits in-flight requests, and isolates failures', async () => {
    const selected = Array.from({ length: 6 }, (_, index) => ({ ...asset, id: `episode-${index + 1}`, episode: index + 1 }));
    const pending = selected.map(() => deferred<unknown>());
    const fetch = mockFetch((_url, body) => pending[selected.findIndex((entry) => entry.id === body.asset_id)].promise);
    const user = userEvent.setup(); mount(<JobModal assets={selected} onClose={vi.fn()} />, context(selected));
    await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(4));
    pending[0].resolve(response({ detail: '第一集读取失败' }, 500));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(5));
    pending[1].resolve(plan(selected[1].id));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(6));
    pending.slice(2).forEach((request, index) => request.resolve(plan(selected[index + 2].id)));
    await screen.findByText('第一集读取失败');
    await waitFor(() => expect(screen.getAllByText(/选中 1 张关键帧/)).toHaveLength(5));
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    expect(fetch.mock.calls.every(([url]) => url === '/api/keyframe-plans')).toBe(true);
    expect(screen.getByText('少女歌剧 · 第 1 季 · 第 6 集')).toBeTruthy();
  });

  it('polls other episodes while an earlier poll is still pending', async () => {
    const firstPoll = deferred<unknown>();
    const fetch = mockFetch((url, body) => url === '/api/keyframe-plans' ? { ...plan(body.asset_id), status: 'queued' } : url.endsWith('episode-1') ? firstPoll.promise : plan(second.id));
    const user = userEvent.setup(); mount(<JobModal assets={[asset, second]} onClose={vi.fn()} />, context([asset, second]));
    await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => url.startsWith('/api/keyframe-plans/'))).toHaveLength(2), { timeout: 3000 });
    expect(await screen.findByText(/选中 1 张关键帧/)).toBeTruthy();
    expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(true);
    firstPoll.resolve(plan(asset.id));
    await waitFor(() => expect((screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement).disabled).toBe(false));
  });
});

describe('Optional estimates preserve submission validation', () => {
  it('starts with the latest budget after scanning without ever requesting an estimate', async () => {
    const fetch = mockFetch((url, body) => url === '/api/keyframe-plans' ? plan(body.asset_id) : { id: 'job' });
    const user = userEvent.setup(); const close = vi.fn(); mount(<JobModal asset={asset} onClose={close} />);
    const start = screen.getByRole('button', { name: '开始分析' }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
    await waitFor(() => expect(start.disabled).toBe(false));
    await user.clear(screen.getByLabelText(/最多模型请求数/)); await user.type(screen.getByLabelText(/最多模型请求数/), '25');
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), '0.5');
    await user.click(start);
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/keyframe-plans']);
    await user.clear(screen.getByLabelText('并发请求上限')); await user.type(screen.getByLabelText('并发请求上限'), '8');
    await user.click(start); await waitFor(() => expect(close).toHaveBeenCalledOnce());
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/keyframe-plans', '/api/jobs']);
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string)).toMatchObject({ max_requests: 25, execution: { max_concurrency: 8 }, keyframe_plan_id: `plan-${asset.id}` });
  });

  it('starts subtitle-only work without a keyframe scan or estimate', async () => {
    const fetch = mockFetch(() => ({ id: 'job' })); const user = userEvent.setup();
    const embedded = { ...asset, subtitle_mode: 'embedded' as const }; mount(<JobModal asset={embedded} onClose={vi.fn()} />, context([embedded]));
    await user.click(screen.getByRole('checkbox', { name: /画面理解/ }));
    await user.click(screen.getByRole('button', { name: '开始分析' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(fetch.mock.calls[0][0]).toBe('/api/jobs');
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string).stages).toEqual(['subtitle']);
  });

  it('starts scene deepening directly while rejecting an invalid budget', async () => {
    const fetch = mockFetch(() => ({ id: 'deep' })); const user = userEvent.setup();
    mount(<DeepenSceneButton assetId={asset.id} chunkId="chunk" title="片段" />);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    await user.clear(screen.getByLabelText(/最多模型请求数/)); await user.type(screen.getByLabelText(/最多模型请求数/), '0');
    await user.click(screen.getByRole('button', { name: '开始深入分析' })); expect(fetch).not.toHaveBeenCalled();
    await user.clear(screen.getByLabelText(/最多模型请求数/)); await user.type(screen.getByLabelText(/最多模型请求数/), '10');
    await user.click(screen.getByRole('button', { name: '开始深入分析' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/assets/episode-1/chunks/chunk/deepen']);
  });

  it('prepares multiple episodes concurrently on direct character reanalysis without estimating', async () => {
    const first = deferred<unknown>();
    const library = { scope_id: 'work', revision: 7, profiles: [], assets: [asset, second] };
    const fetch = mockFetch((url, body) => url === '/api/assets/episode-1/characters' ? library : url === '/api/keyframe-plans' ? body.asset_id === asset.id ? first.promise : plan(second.id) : { jobs: [{ id: 'one' }, { id: 'two' }], revision: 7 });
    const user = userEvent.setup(); mount(<CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={vi.fn().mockResolvedValue(undefined)} />);
    await user.click(await screen.findByRole('button', { name: '按新版档案重新识别' }));
    await user.selectOptions(screen.getByLabelText('重新识别范围'), 'series');
    await user.click(screen.getByRole('button', { name: '开始重新识别' }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => url === '/api/keyframe-plans')).toHaveLength(2));
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/reanalyze'))).toBe(false);
    first.resolve(plan(asset.id));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/estimate'))).toBe(false);
    const request = fetch.mock.calls.find(([url]) => url.endsWith('/reanalyze'))!;
    expect(JSON.parse(request[1]!.body as string)).toMatchObject({ expected_revision: 7, keyframe_plan_ids: { 'episode-1': 'plan-episode-1', 'episode-2': 'plan-episode-2' } });
  });
});

describe('Identifiable episodes and wall time', () => {
  it('distinguishes same-title jobs by season, episode, version and filename fallback', () => {
    const noEpisode = { ...asset, id: 'special', season: null, episode: null, source_filename: 'special.mkv' };
    const selected = [asset, { ...second, version: '导演剪辑' }, noEpisode]; const app = context(selected);
    app.jobs = selected.map((entry, index): Job => ({ id: `job-${index}`, type: index === 0 ? 'keyframe_scan' : 'analysis', asset_id: entry.id, status: 'running', progress: 0, completed: 0, total: 1, created_at: asset.created_at }));
    mount(<Jobs />, app);
    expect(screen.getByRole('link', { name: '少女歌剧 · 第 1 季 · 第 1 集' })).toBeTruthy();
    expect(screen.getByRole('link', { name: '少女歌剧 · 第 1 季 · 第 2 集 · 导演剪辑' })).toBeTruthy();
    expect(screen.getByRole('link', { name: '少女歌剧 · special.mkv' })).toBeTruthy();
    expect(assetTaskLabel({ ...noEpisode, source_filename: undefined })).toBe('少女歌剧 · 视频 special');
    expect(assetTaskLabel({ ...noEpisode, season: 1 })).toBe('少女歌剧 · 第 1 季 · special.mkv');
    expect(assetTaskLabel({ ...noEpisode, id: 'asset_0123456789abcdef', source_filename: undefined })).toBe('少女歌剧 · 视频 89abcdef');
  });

  it('shows predicted wall duration with stage concurrency and scheduling waves', () => {
    mount(<StageEstimates estimate={{ duration_ms: 120000, estimated_frames: 112, estimated_requests: 10, cost_estimate: null, warnings: [], estimated_wall_ms_min: 30000, estimated_wall_ms_max: 90000, timing_basis: '耗时已计入并行与连接上限。', stage_estimates: { transcript: { requests_min: 4, requests_max: 4, concurrency: 4, waves_min: 1, waves_max: 1, estimated_wall_ms_min: 30000, estimated_wall_ms_max: 30000 } } }} />);
    const estimates = within(screen.getByLabelText('分阶段请求估算'));
    expect(estimates.getByText('预计处理耗时 30 秒–1 分 30 秒')).toBeTruthy();
    expect(estimates.getByText('最多同时 4 个请求 · 1 轮')).toBeTruthy();
    expect(estimates.getByText('耗时已计入并行与连接上限。')).toBeTruthy();
  });
});
