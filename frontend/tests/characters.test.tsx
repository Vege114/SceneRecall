import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type CharacterLibrary, type CharacterProfile } from '../src/types';
import CharacterProfiles from '../src/components/CharacterProfiles';
import AssetDetail from '../src/components/AssetDetail';

const asset: Asset = { id: 'episode-one', work_id: 'work-one', title: '星海 · 第一集', kind: 'animation', series: '星海', episode: 1, duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const episodeTwo: Asset = { ...asset, id: 'episode-two', title: '星海 · 第二集', episode: 2 };
const first: CharacterProfile = { id: 'person-one', name: '小夏', aliases: ['夏同学'], description: '短发，绿色围巾。', notes: '与姐姐外貌相似。', user_edited: true, representative: { asset_id: asset.id, frame_id: 'frame-one' }, appearance: ['绿色围巾'], appearance_count: 2, appearances: [{ asset_id: asset.id, asset_title: asset.title, record_id: 'record-one', start_ms: 1000, end_ms: 3000, entity_id: 'e1', frame_ids: ['frame-one'] }, { asset_id: episodeTwo.id, asset_title: episodeTwo.title, record_id: 'record-two', start_ms: 4000, end_ms: 6000, entity_id: 'e1', frame_ids: ['frame-two'] }] };
const second: CharacterProfile = { id: 'person-two', name: '绿围巾少女', aliases: [], description: '绿围巾，短发。', notes: '', appearance_count: 1, appearances: [] };
const library: CharacterLibrary = { scope_id: 'shared-series', revision: 7, profiles: [first, second], assets: [asset, episodeTwo] };
const estimate = { duration_ms: 12000, estimated_frames: 8, estimated_requests: 2, cost_estimate: 0.02, warnings: [] };

function context(overrides: Partial<AppContextValue> = {}): AppContextValue {
  return {
    bootstrap: { library_path: '/tmp/character-ui-tests', settings: { defaults: DEFAULTS, bindings: { vision: 'vision' } }, provider_profiles: [{ id: 'vision', name: '画面模型', model: 'vision-test', base_url: 'https://example.invalid/v1', capabilities: ['vision'], secret_mode: 'session' }], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: '0.2.0' },
    assets: [asset, episodeTwo], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn(), ...overrides,
  };
}
function setup(customContext = context()) {
  const onSeek = vi.fn(); const onSaved = vi.fn().mockResolvedValue(undefined);
  render(<AppContext.Provider value={customContext}><CharacterProfiles asset={asset} onSeek={onSeek} onSaved={onSaved} /></AppContext.Provider>);
  return { user: userEvent.setup(), onSeek, onSaved, context: customContext };
}
function mockFetch(handler: (url: string, options?: RequestInit) => unknown | Promise<unknown> = () => library) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const result = await handler(url, options);
    return result instanceof Response ? result : new Response(JSON.stringify(result), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Cross-shot character dossiers', () => {
  it('shows evidence, searches aliases and opens appearances in the current or another episode', async () => {
    mockFetch(); const { user, onSeek } = setup();
    await screen.findByText('2 份角色档案');
    expect(screen.getByText('同一作品的 2 个视频共享档案 · 修订 7')).toBeTruthy();
    const portraitUrl = new URL(screen.getByRole('img', { name: '小夏的当前形象' }).getAttribute('src')!, 'http://localhost');
    expect(portraitUrl.pathname).toBe('/api/assets/episode-one/characters/person-one/portrait');
    expect(portraitUrl.searchParams.get('revision')).toBe('7');
    expect(JSON.parse(portraitUrl.searchParams.get('representative')!)).toEqual(first.representative);
    await user.click(screen.getByRole('button', { name: '查看小夏的当前形象原图' }));
    expect(within(screen.getByRole('dialog')).getByRole('img').getAttribute('src')).toBe('/api/assets/episode-one/frames/frame-one');
    await user.click(screen.getByRole('button', { name: '关闭' }));
    await user.type(screen.getByRole('textbox', { name: '搜索角色档案' }), '夏同学');
    expect(screen.queryByRole('article', { name: '绿围巾少女的角色档案' })).toBeNull();
    await user.click(screen.getByRole('button', { name: '查看小夏的出场记录' }));
    await user.click(screen.getByRole('button', { name: '播放小夏在00:00:01的出场' }));
    expect(onSeek).toHaveBeenCalledWith(1000);
    expect(screen.getByRole('link', { name: '前往星海 · 第二集查看小夏' }).getAttribute('href')).toBe('#/asset/episode-two?t=4000');
    await user.click(screen.getByRole('button', { name: '放大小夏在星海 · 第一集的证据画面' }));
    expect(screen.getByRole('dialog', { name: '角色出场证据' })).toBeTruthy();
  });

  it('saves a revised name, deduplicated aliases, appearance and notes with the expected revision', async () => {
    let finish!: (value: unknown) => void;
    const pending = new Promise((resolve) => { finish = resolve; });
    const fetch = mockFetch((_url, options) => options?.method === 'PATCH' ? pending : library);
    const { user, onSaved } = setup();
    await user.click(await screen.findByRole('button', { name: '编辑小夏的档案' }));
    const dialog = within(screen.getByRole('dialog', { name: '编辑角色档案' }));
    await user.clear(dialog.getByLabelText('角色姓名')); await user.type(dialog.getByLabelText('角色姓名'), '夏未央');
    await user.clear(dialog.getByLabelText(/^别名/)); await user.type(dialog.getByLabelText(/^别名/), '小夏、小夏，夏同学');
    await user.clear(dialog.getByLabelText(/^外貌与识别特征/)); await user.type(dialog.getByLabelText(/^外貌与识别特征/), '左眼下有痣。');
    await user.clear(dialog.getByLabelText(/^已积累的外观特征/)); await user.type(dialog.getByLabelText(/^已积累的外观特征/), '左眼下有痣\n深色短发');
    await user.clear(dialog.getByLabelText(/^角色补充说明/)); await user.type(dialog.getByLabelText(/^角色补充说明/), '姐姐没有痣。');
    await user.click(dialog.getByRole('button', { name: '保存档案' }));
    expect((dialog.getByRole('button', { name: '保存中' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(dialog.getByRole('button', { name: '关闭' }));
    expect(screen.getByRole('dialog')).toBeTruthy();
    const call = fetch.mock.calls.find((entry) => entry[1]?.method === 'PATCH')!;
    expect(call[0]).toBe('/api/assets/episode-one/characters/person-one');
    expect(JSON.parse(call[1]!.body as string)).toEqual({ name: '夏未央', aliases: ['小夏', '夏同学'], description: '左眼下有痣。', appearance: ['左眼下有痣', '深色短发'], notes: '姐姐没有痣。', expected_revision: 7 });
    finish({});
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(onSaved).toHaveBeenCalledOnce();
  });

  it('keeps a conflicting edit visible and requires fresh dossiers before retrying', async () => {
    const fetch = mockFetch((_url, options) => options?.method === 'PATCH' ? new Response(JSON.stringify({ detail: '档案已更新，请刷新后重试。' }), { status: 409 }) : library);
    const { user } = setup();
    await user.click(await screen.findByRole('button', { name: '编辑小夏的档案' }));
    await user.type(screen.getByLabelText('角色姓名'), '（修订）');
    await user.click(screen.getByRole('button', { name: '保存档案' }));
    await screen.findByRole('alert');
    expect(screen.getByText('档案已更新，请刷新后重试。')).toBeTruthy();
    expect((screen.getByRole('button', { name: '保存档案' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: '刷新档案后重新操作' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetch.mock.calls.filter((call) => !call[1]?.method)).toHaveLength(2);
  });

  it('changes only fields the user edited so AI can continue enriching the others', async () => {
    const fetch = mockFetch(); const { user } = setup();
    await user.click(await screen.findByRole('button', { name: '编辑小夏的档案' }));
    expect((screen.getByRole('button', { name: '保存档案' }) as HTMLButtonElement).disabled).toBe(true);
    await user.clear(screen.getByLabelText('角色姓名'));
    await user.type(screen.getByLabelText('角色姓名'), '夏未央');
    await user.click(screen.getByRole('button', { name: '保存档案' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    const call = fetch.mock.calls.find((entry) => entry[1]?.method === 'PATCH')!;
    expect(JSON.parse(call[1]!.body as string)).toEqual({ name: '夏未央', expected_revision: 7 });
  });

  it('refreshes portraits when AI adds new evidence without a human revision change', async () => {
    let current = library;
    mockFetch(() => current);
    const app = context();
    const view = render(<AppContext.Provider value={app}><CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={vi.fn()} /></AppContext.Provider>);
    const initialImage = await screen.findByRole('img', { name: '小夏的当前形象' });
    const firstUrl = initialImage.getAttribute('src');
    current = { ...library, profiles: [{ ...first, updated_at: '2026-01-02T00:00:00Z', representative: { asset_id: episodeTwo.id, frame_id: 'frame-latest', box: [0.1, 0.1, 0.3, 0.4] } }, second] };
    const working = { ...app, jobs: [{ id: 'active', type: 'analysis', status: 'running', progress: 0.5, completed: 1, total: 2, asset_id: asset.id, created_at: asset.created_at }] };
    view.rerender(<AppContext.Provider value={working}><CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={vi.fn()} /></AppContext.Provider>);
    await waitFor(() => expect(screen.getByRole('img', { name: '小夏的当前形象' }).getAttribute('src')).not.toBe(firstUrl));
    const updated = new URL(screen.getByRole('img', { name: '小夏的当前形象' }).getAttribute('src')!, 'http://localhost');
    expect(updated.searchParams.get('revision')).toBe('7');
    expect(updated.searchParams.get('updated_at')).toBe('2026-01-02T00:00:00Z');
    expect(JSON.parse(updated.searchParams.get('representative')!)).toEqual(current.profiles[0].representative);
  });

  it('previews merge direction and excludes the source character from the destination choices', async () => {
    const fetch = mockFetch(); const { user } = setup();
    await user.click(await screen.findByRole('button', { name: '合并绿围巾少女的档案' }));
    const dialog = within(screen.getByRole('dialog', { name: '合并角色档案' }));
    expect(dialog.queryByRole('option', { name: /绿围巾少女/ })).toBeNull();
    expect((dialog.getByRole('button', { name: '确认合并' }) as HTMLButtonElement).disabled).toBe(true);
    await user.selectOptions(dialog.getByLabelText('保留哪一份角色档案'), first.id);
    expect(dialog.getByText('保留此档案')).toBeTruthy();
    await user.click(dialog.getByRole('button', { name: '确认合并' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    const call = fetch.mock.calls.find((entry) => entry[0].endsWith('/merge'))!;
    expect(call[0]).toBe('/api/assets/episode-one/characters/person-two/merge');
    expect(JSON.parse(call[1]!.body as string)).toEqual({ target_id: 'person-one', expected_revision: 7 });
  });

  it('only deletes after the graphical confirmation and preserves the revision guard', async () => {
    const fetch = mockFetch(); const { user } = setup();
    await user.click(await screen.findByRole('button', { name: '删除小夏的档案' }));
    expect(fetch).toHaveBeenCalledOnce();
    expect(screen.getByText(/原始画面记录和证据图片仍会保留/)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: '取消' }));
    expect(fetch).toHaveBeenCalledOnce();
    await user.click(screen.getByRole('button', { name: '删除小夏的档案' }));
    await user.click(screen.getByRole('button', { name: '确认删除' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetch.mock.calls.find((entry) => entry[1]?.method === 'DELETE')?.[0]).toBe('/api/assets/episode-one/characters/person-one?expected_revision=7');
  });

  it('recovers from a list error and offers a useful empty state', async () => {
    let attempts = 0;
    mockFetch(() => ++attempts === 1 ? new Response(JSON.stringify({ detail: '暂时无法读取档案' }), { status: 503 }) : { ...library, profiles: [] });
    const { user, context: app } = setup();
    await screen.findByText('暂时无法读取档案');
    await user.click(screen.getByRole('button', { name: '重新加载角色档案' }));
    await screen.findByText('角色故事，从第一次登场开始');
    await user.click(screen.getByRole('button', { name: '开始画面分析' }));
    expect(app.openJob).toHaveBeenCalledWith(asset);
  });

  it('opens linked dossiers from the asset detail entity tags', async () => {
    const user = userEvent.setup();
    mockFetch((url) => url.endsWith('/characters') ? library : { asset, observations: [{ id: 'obs', start_ms: 0, end_ms: 2000, summary: '小夏走来', entities: [{ id: 'e1', character_id: first.id, character_name: first.name }], evidence_frame_ids: [], uncertainties: [], events: [] }], subtitles: [], annotations: [], runs: [] });
    render(<AppContext.Provider value={context()}><AssetDetail id={asset.id} /></AppContext.Provider>);
    await user.click(await screen.findByRole('button', { name: '查看小夏的角色档案' }));
    await screen.findByText('2 份角色档案');
    expect(screen.getByRole('button', { name: '查看小夏的出场记录' }).getAttribute('aria-expanded')).toBe('true');
    expect(screen.getByRole('button', { name: '角色档案' }).className).toBe('active');
  });
});

describe('Reinterpret video with revised dossiers', () => {
  it('estimates every selected episode, invalidates stale budgets and submits the dossier revision', async () => {
    const fetch = mockFetch((url) => url.endsWith('/estimate') ? estimate : url.endsWith('/reanalyze') ? { jobs: [{ id: 'job-one' }, { id: 'job-two' }], revision: 7 } : library);
    const { user, context: app } = setup();
    await user.click(await screen.findByRole('button', { name: '按新版档案重新识别' }));
    const dialog = within(screen.getByRole('dialog', { name: '按新版档案重新识别' }));
    const start = dialog.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    await user.selectOptions(dialog.getByLabelText('重新识别范围'), 'series');
    await user.click(dialog.getByRole('button', { name: '估算重新识别用量' }));
    await waitFor(() => expect(start.disabled).toBe(false));
    const estimateCalls = fetch.mock.calls.filter((entry) => entry[0] === '/api/jobs/estimate');
    expect(estimateCalls.map((entry) => JSON.parse(entry[1]!.body as string).asset_id)).toEqual([asset.id, episodeTwo.id]);
    expect(JSON.parse(estimateCalls[0][1]!.body as string)).toMatchObject({ stages: ['vision'], start_ms: 0, end_ms: null, force: true });
    await user.clear(dialog.getByLabelText(/^每个视频最多模型请求数/)); await user.type(dialog.getByLabelText(/^每个视频最多模型请求数/), '50');
    expect(start.disabled).toBe(true);
    await user.click(dialog.getByRole('button', { name: '估算重新识别用量' }));
    await waitFor(() => expect(start.disabled).toBe(false));
    await user.click(start);
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    const call = fetch.mock.calls.find((entry) => entry[0].endsWith('/reanalyze'))!;
    expect(JSON.parse(call[1]!.body as string)).toEqual({ scope: 'series', max_requests: 50, max_cost: null, expected_revision: 7 });
    expect(app.refresh).toHaveBeenCalledOnce();
  });

  it('does not allow inaccessible episodes or missing models to begin reanalysis', async () => {
    mockFetch(() => ({ ...library, assets: [asset, { ...episodeTwo, source_available: false }] }));
    const app = context(); app.bootstrap.settings.bindings = {};
    const { user } = setup(app);
    await user.click(await screen.findByRole('button', { name: '按新版档案重新识别' }));
    await user.selectOptions(screen.getByLabelText('重新识别范围'), 'series');
    expect(screen.getByText(/这些视频的原文件暂不可用：星海 · 第二集/)).toBeTruthy();
    expect(screen.getByText(/中分配画面理解模型/)).toBeTruthy();
    expect((screen.getByRole('button', { name: '估算重新识别用量' }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('disables money budgets for Codex and hides the series option for a single video', async () => {
    const fetch = mockFetch((url) => url.endsWith('/estimate') ? { ...estimate, cost_estimate: null } : url.endsWith('/reanalyze') ? { jobs: [{ id: 'job' }], revision: 7 } : { ...library, assets: [asset] });
    const app = context(); app.bootstrap.provider_profiles[0].provider_type = 'codex_cli'; app.bootstrap.settings.defaults = { ...DEFAULTS, max_cost: 10 };
    const { user } = setup(app);
    await user.click(await screen.findByRole('button', { name: '按新版档案重新识别' }));
    expect(screen.queryByRole('option', { name: /同一作品全部/ })).toBeNull();
    expect((screen.getByLabelText(/^每个视频费用上限/) as HTMLInputElement).disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: '估算重新识别用量' }));
    await waitFor(() => expect((screen.getByRole('button', { name: '开始重新识别' }) as HTMLButtonElement).disabled).toBe(false));
    await user.click(screen.getByRole('button', { name: '开始重新识别' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetch.mock.calls.filter((entry) => entry[1]?.method === 'POST').map((entry) => JSON.parse(entry[1]!.body as string).max_cost)).toEqual([null, null]);
  });
});
