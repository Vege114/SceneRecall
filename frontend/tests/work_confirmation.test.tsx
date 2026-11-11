import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CharacterConfirmation from '../src/components/CharacterConfirmation';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS } from '../src/types';
import type { Asset, CharacterProfile } from '../src/types';
import type { BangumiReferenceCacheResult, CharacterConfirmationBatchResult, CharacterConfirmationCandidate, CharacterConfirmationResult } from '../src/identityTypes';

const asset: Asset = { id: 'synthetic-episode', title: '合成第一集', kind: 'animation', series: '合成作品', duration_ms: 120000,
  width: 1280, height: 720, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '', status: 'ready' };
const profile: CharacterProfile = { id: 'summer', name: '小夏', aliases: [], description: '', notes: '', appearance_count: 0,
  appearances: [], external_source: { source: 'bangumi', subject_id: 1, character_id: 2, url: 'https://bgm.tv/character/2', image_url: '/synthetic/summer.jpg' } };
const endpoint = `/api/assets/${asset.id}/identity/confirmation`;
const queue = (): CharacterConfirmationResult => ({ asset_id: asset.id, scope_id: 'synthetic-work', roster_revision: 3,
  roles: [{ character_id: profile.id, recognition_enabled: true, profile, status: 'pending', trusted_reference_count: 0,
    matching_reference_count: 0, candidates: [], available_count: 0 }],
  summary: { total: 1, confirmed: 0, pending: 1, skipped: 0, candidates: 0 }, sampling: null, sampling_available: true });
const cached = (next_cursor: string | null = null): BangumiReferenceCacheResult => ({ imported: 1, existing: 0, skipped: 0, errors: [], next_cursor });
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
function mockApi(handler: (url: string, init?: RequestInit) => unknown | Promise<unknown>) {
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    const result = await handler(url, init);
    return result instanceof Response ? result : new Response(JSON.stringify(result), { headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
function setup() {
  const onSaved = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
  render(<CharacterConfirmation asset={asset} onSaved={onSaved} onClose={vi.fn()} />);
  return { user: userEvent.setup(), onSaved };
}
const writes = (fetch: ReturnType<typeof mockApi>) => fetch.mock.calls.filter(([, options]) => options?.method === 'PUT' || options?.method === 'POST');

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Episode names in generation progress', () => {
  it('uses the application asset titles before any region exists, including zero-frame episodes', async () => {
    const state = queue();
    state.sampling = { status: 'completed', completed: 0, total: 0, asset_id: asset.id, assets: [
      { asset_id: asset.id, status: 'completed', completed: 0, total: 0 },
      { asset_id: 'synthetic-second-id', status: 'unavailable', completed: 0, total: 0 },
    ] };
    const context: AppContextValue = { assets: [{ ...asset, title: '应用内第一集名称' }, { ...asset, id: 'synthetic-second-id', title: '应用内第二集名称' }], jobs: [],
      bootstrap: { library_path: '/synthetic', settings: { defaults: DEFAULTS, bindings: {} }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' },
      refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
    const fetch = mockApi(() => state);
    render(<AppContext.Provider value={context}><CharacterConfirmation asset={asset} onSaved={vi.fn().mockResolvedValue(undefined)} onClose={vi.fn()} /></AppContext.Provider>);
    const progress = within(await screen.findByRole('region', { name: '候选生成进度' }));
    expect(progress.getByText('应用内第一集名称：0 / 0 帧 · completed')).toBeTruthy();
    expect(progress.getByText('应用内第二集名称：0 / 0 帧 · unavailable')).toBeTruthy();
    expect(progress.queryByText(/synthetic-second-id/)).toBeNull();
    expect(writes(fetch)).toHaveLength(0);
  });

  it('uses current, pool and role titles without an application context and keeps an unknown id as the last fallback', async () => {
    const state = queue();
    const crop = { region_id: 'name-region', asset_id: 'synthetic-role-id', asset_title: '角色候选中的剧集名', frame_id: 'name-frame', at_ms: 5000,
      box: [0, 0, 0.4, 0.8], crop_url: '/synthetic/name.jpg', shot_id: 'name-shot', quality: { accepted: true } };
    state.roles[0].candidates = [crop];
    state.review_pool = [{ ...crop, region_id: 'pool-name-region', asset_id: 'synthetic-pool-id', asset_title: '待判定池中的剧集名', assignment_state: 'pending' }];
    state.sampling = { status: 'completed', completed: 0, total: 0, asset_id: asset.id, assets: [
      ...[asset.id, 'synthetic-pool-id', 'synthetic-role-id', 'synthetic-unknown-id'].map((id) => ({ asset_id: id, status: 'completed', completed: 0, total: 0 })),
    ] };
    const fetch = mockApi(() => state);
    setup();
    const progress = within(await screen.findByRole('region', { name: '候选生成进度' }));
    for (const title of [asset.title, '待判定池中的剧集名', '角色候选中的剧集名', 'synthetic-unknown-id']) {
      expect(progress.getByText(`${title}：0 / 0 帧 · completed`)).toBeTruthy();
    }
    expect(progress.queryByText(/synthetic-pool-id|synthetic-role-id/)).toBeNull();
    expect(writes(fetch)).toHaveLength(0);
  });
});

describe('Reference detection diagnostics', () => {
  it('explains that a cached image with no detected person did not participate in matching and gives the next action', async () => {
    const state = queue();
    Object.assign(state.roles[0], { matching_reference_count: 1, usable_matching_reference_count: 0,
      reference_detection_status: 'failed', reference_model: 'synthetic-detector', reference_issues: [
        { reference_id: 'synthetic-reference', status: 'no_person', person_count: 0, reason: '该参考图未检出人物，未参与本次自动匹配。',
          action: '在参考图管理中替换清晰单人图后重新生成。', model_version: 'synthetic-model-v1' },
      ], coverage: { status: 'reference_invalid', reason: '参考图未检出可用的单人人物。', candidate_count: 0, review_pool_count: 7, sampled_frames: 12, sampled_assets: 2 } });
    const fetch = mockApi(() => state);
    setup();
    const review = within(await screen.findByRole('region', { name: '小夏的角色确认' }));
    const diagnostics = within(review.getByRole('region', { name: '小夏的参考图检测' }));
    expect(diagnostics.getByText(/本地匹配参考 1 张 · 检测可用 0 张/)).toBeTruthy();
    expect(diagnostics.getByText('未检测到人物')).toBeTruthy();
    expect(diagnostics.getByText(/未参与本次自动匹配/)).toBeTruthy();
    expect(diagnostics.getByText(/下一步：在参考图管理中替换清晰单人图后重新生成/)).toBeTruthy();
    expect(diagnostics.getByText(/增加正片采样不会修复参考图/)).toBeTruthy();
    expect(review.getByText(/可供人工核对的待判定 7 张（不代表该角色已被找到）/)).toBeTruthy();
    expect(review.queryByText(/^覆盖：/)).toBeNull();
    expect(writes(fetch)).toHaveLength(0);
  });

  it.each([
    { status: 'not_checked', issue: 'not_checked', label: '尚未检测', reason: '尚无当前模型的检测记录。', action: '生成候选以检测已有参考。' },
    { status: 'model_unavailable', issue: 'model_unavailable', label: '检测模型不可用', reason: '本地视觉模型未就绪，参考尚未完成检测。', action: '检查本地模型环境后重试生成，无需撤销参考。' },
  ] as const)('distinguishes $status from a reference image detection failure', async ({ status, issue, label, reason, action }) => {
    const state = queue();
    Object.assign(state.roles[0], { matching_reference_count: 1, usable_matching_reference_count: 0, reference_detection_status: status,
      reference_issues: [{ reference_id: 'synthetic-pending-reference', status: issue, person_count: null, reason, action, model_version: null }] });
    const fetch = mockApi(() => state);
    setup();
    const review = within(await screen.findByRole('region', { name: '小夏的角色确认' }));
    const diagnostics = within(review.getByRole('region', { name: '小夏的参考图检测' }));
    expect(diagnostics.getByText(label)).toBeTruthy();
    expect(diagnostics.getByText(`下一步：${action}`)).toBeTruthy();
    expect(diagnostics.queryByText('未检测到人物')).toBeNull();
    expect(diagnostics.queryByText(/当前参考图未得到可用的单人人物检测/)).toBeNull();
    expect(writes(fetch)).toHaveLength(0);
  });

  it('keeps the usable reference count while explaining a separate image containing multiple people', async () => {
    const state = queue();
    Object.assign(state.roles[0], { matching_reference_count: 2, usable_matching_reference_count: 1, reference_detection_status: 'usable',
      reference_issues: [{ reference_id: 'synthetic-group-reference', status: 'multiple_people', person_count: 2,
        reason: '该参考图检出 2 个人物，未参与单人匹配。', action: '裁剪或替换为单人参考后重新生成。', model_version: 'synthetic-model-v1' }] });
    const fetch = mockApi(() => state);
    setup();
    const review = within(await screen.findByRole('region', { name: '小夏的角色确认' }));
    const diagnostics = within(review.getByRole('region', { name: '小夏的参考图检测' }));
    expect(diagnostics.getByText(/本地匹配参考 2 张 · 检测可用 1 张/)).toBeTruthy();
    expect(diagnostics.getByText(/已有检测可用的单人参考图/)).toBeTruthy();
    expect(diagnostics.getByText('检测到多个人物')).toBeTruthy();
    expect(diagnostics.getByText(/下一步：裁剪或替换为单人参考/)).toBeTruthy();
    expect(writes(fetch)).toHaveLength(0);
  });
});

describe('Explicit work target generation', () => {
  it('waits for the saved work targets and every reference cache page before starting bounded cross-episode sampling', async () => {
    const saved = deferred<CharacterConfirmationResult>();
    const firstCache = deferred<BangumiReferenceCacheResult>();
    const secondCache = deferred<BangumiReferenceCacheResult>();
    const sampled = deferred<CharacterConfirmationResult>();
    let cacheCalls = 0;
    const fetch = mockApi((url, options) => {
      if (url.endsWith('/selection')) return saved.promise;
      if (url.endsWith('/cache-bangumi')) return ++cacheCalls === 1 ? firstCache.promise : secondCache.promise;
      if (url.endsWith('/sample')) return sampled.promise;
      return queue();
    });
    const { user } = setup();
    await user.click(await screen.findByRole('button', { name: '保存名单并生成候选' }));
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/selection`]);
    expect(JSON.parse(writes(fetch)[0][1]!.body as string)).toEqual({ character_ids: ['summer'], expected_revision: 3, scope_id: 'synthetic-work' });
    expect((screen.getByRole('button', { name: '保存名单并生成候选' }) as HTMLButtonElement).disabled).toBe(true);
    await act(async () => { saved.resolve({ ...queue(), roster_revision: 4 }); });
    await waitFor(() => expect(cacheCalls).toBe(1));
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/selection`, '/api/assets/synthetic-episode/references/cache-bangumi']);
    expect(JSON.parse(writes(fetch)[1][1]!.body as string)).toEqual({ max_images: 20 });
    await act(async () => { firstCache.resolve(cached('cursor-page-two')); });
    await waitFor(() => expect(cacheCalls).toBe(2));
    expect(JSON.parse(writes(fetch)[2][1]!.body as string)).toEqual({ max_images: 20, cursor: 'cursor-page-two' });
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/sample'))).toBe(false);
    await act(async () => { secondCache.resolve(cached()); });
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.endsWith('/sample'))).toBe(true));
    expect(JSON.parse(writes(fetch)[3][1]!.body as string)).toEqual({ scope: 'work', max_frames: 120, max_assets: 24, frames_per_asset: 12 });
    await act(async () => { sampled.resolve({ ...queue(), roster_revision: 4, sampling: { status: 'completed', completed: 12, total: 12, asset_id: asset.id } }); });
    await waitFor(() => expect((screen.getByRole('button', { name: '保存名单并生成候选' }) as HTMLButtonElement).disabled).toBe(false));
  });

  it('keeps a rejected target revision visible and does not cache or sample it', async () => {
    const fetch = mockApi((url) => url.endsWith('/selection')
      ? new Response(JSON.stringify({ detail: '识别名单已被其他窗口修改' }), { status: 409 }) : queue());
    const { user, onSaved } = setup();
    await user.click(await screen.findByRole('button', { name: '保存名单并生成候选' }));
    await screen.findByText(/识别名单已被其他窗口修改/);
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/selection`]);
    expect(onSaved).not.toHaveBeenCalled();
    expect((screen.getByRole('checkbox', { name: '识别小夏' }) as HTMLInputElement).checked).toBe(true);
  });

  it('stops generation on reference download failure after targets have been saved', async () => {
    const fetch = mockApi((url) => url.endsWith('/cache-bangumi')
      ? new Response(JSON.stringify({ detail: '合成参考下载失败' }), { status: 503 }) : queue());
    const { user, onSaved } = setup();
    await user.click(await screen.findByRole('button', { name: '保存名单并生成候选' }));
    await screen.findByText(/合成参考下载失败/);
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/selection`, '/api/assets/synthetic-episode/references/cache-bangumi']);
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/sample'))).toBe(false);
    expect(onSaved).toHaveBeenCalledOnce();
    expect((screen.getByRole('button', { name: '保存名单并生成候选' }) as HTMLButtonElement).disabled).toBe(false);
  });

  it('reconciles a durably saved target set after its response is lost and releases the old local draft lock', async () => {
    const state = queue();
    const fetch = mockApi((url) => {
      if (url.endsWith('/selection')) {
        state.roles[0].recognition_enabled = false;
        state.roster_revision += 1;
        state.summary = { ...state.summary, total: 0, pending: 0 };
        return new Response(JSON.stringify({ detail: '合成已落盘响应丢失' }), { status: 503 });
      }
      return state;
    });
    const { user } = setup();
    await user.click(await screen.findByRole('checkbox', { name: '识别小夏' }));
    await user.click(screen.getByRole('button', { name: '保存名单并生成候选' }));
    await screen.findByText(/合成已落盘响应丢失/);
    await user.click(screen.getByRole('button', { name: '重新读取确认进度' }));
    await screen.findByText('本作品未选择识别角色');
    await waitFor(() => expect(screen.queryByText('识别名单修改尚未保存。')).toBeNull());
    expect((screen.getByRole('checkbox', { name: '识别小夏' }) as HTMLInputElement).checked).toBe(false);
    expect((screen.getByRole('button', { name: '保存进度并返回' }) as HTMLButtonElement).disabled).toBe(false);
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/selection`]);
  });

  it('rebuilds the target draft from a successful read when a lost write did not change the server revision', async () => {
    const state = queue();
    const fetch = mockApi((url) => url.endsWith('/selection')
      ? new Response(JSON.stringify({ detail: '合成名单请求未到达' }), { status: 503 }) : state);
    const { user } = setup();
    await user.click(await screen.findByRole('checkbox', { name: '识别小夏' }));
    await user.click(screen.getByRole('button', { name: '保存名单并生成候选' }));
    await screen.findByText(/合成名单请求未到达/);
    await user.click(screen.getByRole('button', { name: '重新读取确认进度' }));
    await waitFor(() => expect((screen.getByRole('checkbox', { name: '识别小夏' }) as HTMLInputElement).checked).toBe(true));
    expect(screen.queryByText('识别名单修改尚未保存。')).toBeNull();
    expect((screen.getByRole('button', { name: '保存进度并返回' }) as HTMLButtonElement).disabled).toBe(false);
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/selection`]);
  });

  it('saves an explicitly empty work target set without starting a generation request', async () => {
    const state = queue();
    const fetch = mockApi((url) => {
      if (url.endsWith('/selection')) {
        state.roles[0].recognition_enabled = false;
        state.summary = { total: 0, confirmed: 0, pending: 0, skipped: 0, candidates: 0 };
        state.roster_revision++;
      }
      return state;
    });
    const { user } = setup();
    await user.click(await screen.findByRole('checkbox', { name: '识别小夏' }));
    await user.click(screen.getByRole('button', { name: '保存名单并生成候选' }));
    await screen.findByText('本作品未选择识别角色');
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/selection`]);
    expect(JSON.parse(writes(fetch)[0][1]!.body as string).character_ids).toEqual([]);
  });

  it('keeps a lost generation response unresolved and requires a successful reread before another generation', async () => {
    let reads = 0;
    const reconcile = deferred<CharacterConfirmationResult>();
    const fetch = mockApi((url) => {
      if (url.endsWith('/sample')) throw new Error('synthetic generation response lost');
      if (url.endsWith('/cache-bangumi')) return cached();
      if (url === endpoint) return ++reads === 1 ? queue() : reconcile.promise;
      return queue();
    });
    const { user } = setup();
    await user.click(await screen.findByRole('button', { name: '保存名单并生成候选' }));
    await screen.findByText(/名单已保存，候选生成尚未完成/);
    expect((screen.getByRole('button', { name: '保存名单并生成候选' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByText('候选不足？补充参考图与少量采样'));
    expect((screen.getByRole('button', { name: '补充少量本地采样' }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: '缓存 Bangumi 参考图' }) as HTMLButtonElement).disabled).toBe(true);
    const writesBeforeReread = writes(fetch).length;
    await user.click(screen.getByRole('button', { name: '重新读取确认进度' }));
    expect((screen.getByRole('button', { name: '保存名单并生成候选' }) as HTMLButtonElement).disabled).toBe(true);
    expect(writes(fetch)).toHaveLength(writesBeforeReread);
    await act(async () => { reconcile.resolve(queue()); });
    await waitFor(() => expect((screen.getByRole('button', { name: '保存名单并生成候选' }) as HTMLButtonElement).disabled).toBe(false));
    expect(writes(fetch)).toHaveLength(writesBeforeReread);
  });

  it('warns before unloading a changed target draft and releases that warning after cancel', async () => {
    const fetch = mockApi(() => queue());
    const { user } = setup();
    await user.click(await screen.findByRole('checkbox', { name: '识别小夏' }));
    const unsaved = new Event('beforeunload', { cancelable: true });
    window.dispatchEvent(unsaved);
    expect(unsaved.defaultPrevented).toBe(true);
    await user.click(screen.getByRole('button', { name: '取消修改' }));
    const cancelled = new Event('beforeunload', { cancelable: true });
    window.dispatchEvent(cancelled);
    expect(cancelled.defaultPrevented).toBe(false);
    expect(writes(fetch)).toHaveLength(0);
  });
});

describe('Pending identity review pool', () => {
  const pendingCandidate = (): CharacterConfirmationCandidate => ({ region_id: 'pool-region', asset_id: 'synthetic-second-episode',
    asset_title: '合成第二集', frame_id: 'pool-frame', at_ms: 6000, box: [0.1, 0.1, 0.4, 0.8], crop_url: '/synthetic/pool.jpg',
    shot_id: 'pool-shot', distance: 0.8, quality: { accepted: true }, source_available: true,
    assignment_state: 'pending', confidence_state: 'uncalibrated', suggested_character_id: 'summer',
    eligible_character_ids: ['summer', 'spring'], raw_candidates: [{ character_id: 'summer', distance: 0.8 }, { character_id: 'spring', distance: 0.9 }] });
  const poolQueue = (): CharacterConfirmationResult => {
    const state = queue();
    state.roles[0].matching_reference_count = 1;
    state.roles.push({ ...state.roles[0], character_id: 'spring', profile: { ...profile, id: 'spring', name: '小春',
      external_source: { ...profile.external_source!, character_id: 3, image_url: '/synthetic/spring.jpg' } } });
    state.roles.push({ ...state.roles[0], character_id: 'disabled', recognition_enabled: false, profile: { ...profile, id: 'disabled', name: '小秋' } });
    state.summary = { total: 2, confirmed: 0, pending: 2, skipped: 0, candidates: 0 };
    state.review_pool = [pendingCandidate()];
    state.review_pool_count = 1;
    return state;
  };

  it('requires manual role choice, crop verification and an explicit submit before saving an uncalibrated region', async () => {
    const state = poolQueue();
    Object.assign(state.roles[1], { usable_matching_reference_count: 0, reference_detection_status: 'failed', reference_issues: [
      { reference_id: 'synthetic-pool-role-reference', status: 'no_person', person_count: 0, reason: '所选角色的参考图未检出人物。',
        action: '替换清晰单人参考图，或对当前裁剪人工核对。', model_version: 'synthetic-model-v1' },
    ] });
    const fetch = mockApi((url, options) => {
      if (url.endsWith('/confirm-batch')) {
        const result = poolQueue(); result.review_pool = []; result.review_pool_count = 0;
        result.roles[1].status = 'confirmed'; result.roles[1].trusted_reference_count = 1;
        result.summary.confirmed = 1; result.summary.pending = 1;
        return { queue: result, results: [{ source_asset_id: 'synthetic-second-episode', region_id: 'pool-region', status: 'confirmed' }],
          summary: { requested: 1, confirmed: 1, failed: 0 } } satisfies CharacterConfirmationBatchResult;
      }
      return state;
    });
    const { user, onSaved } = setup();
    const pool = within(await screen.findByRole('region', { name: '待判定候选池' }));
    const choice = pool.getByRole('combobox', { name: '为合成第二集 00:00:06选择核对角色' }) as HTMLSelectElement;
    expect(choice.value).toBe('');
    expect(pool.queryByRole('checkbox')).toBeNull();
    expect(within(choice).queryByRole('option', { name: '小秋' })).toBeNull();
    expect(writes(fetch)).toHaveLength(0);
    await user.selectOptions(choice, 'spring');
    expect(pool.getByAltText('小春的角色头像')).toBeTruthy();
    const diagnostics = within(pool.getByRole('region', { name: '小春的参考图检测' }));
    expect(diagnostics.getByText('未检测到人物')).toBeTruthy();
    expect(diagnostics.getByText(/下一步：替换清晰单人参考图，或对当前裁剪人工核对/)).toBeTruthy();
    expect(pool.getAllByRole('img').filter((element) => element.getAttribute('src') === '/synthetic/pool.jpg')).toHaveLength(1);
    const checked = pool.getByRole('checkbox', { name: /已核对：这是小春/ }) as HTMLInputElement;
    expect(checked.checked).toBe(false);
    expect(writes(fetch)).toHaveLength(0);
    await user.click(checked);
    expect(writes(fetch)).toHaveLength(0);
    expect(onSaved).not.toHaveBeenCalled();
    await user.click(pool.getByRole('button', { name: /确认所选 1 张/ }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/characters/spring/confirm-batch`]);
    expect(JSON.parse(writes(fetch)[0][1]!.body as string)).toEqual({ candidates: [
      { source_asset_id: 'synthetic-second-episode', region_id: 'pool-region' },
    ], single_character: true, use_as_reference: true });
    expect(screen.queryByRole('combobox', { name: '为合成第二集 00:00:06选择核对角色' })).toBeNull();
  });

  it('removes excluded roles from manual choices and never promotes a suggestion into a preselected identity', async () => {
    const state = poolQueue();
    state.review_pool![0].excluded_character_ids = ['summer'];
    mockApi(() => state);
    setup();
    const pool = within(await screen.findByRole('region', { name: '待判定候选池' }));
    const choice = pool.getByRole('combobox') as HTMLSelectElement;
    expect(choice.value).toBe('');
    expect(within(choice).queryByRole('option', { name: '小夏' })).toBeNull();
    expect(within(choice).getByRole('option', { name: '小春' })).toBeTruthy();
    expect(pool.queryByRole('checkbox')).toBeNull();
  });

  it('shows the manually selected role own distance and rank instead of the strongest suggested role metrics', async () => {
    const state = poolQueue();
    state.review_pool![0].candidate_rank = 1;
    state.review_pool![0].candidate_total = 2;
    const fetch = mockApi(() => state);
    const { user } = setup();
    const pool = within(await screen.findByRole('region', { name: '待判定候选池' }));
    await user.selectOptions(pool.getByRole('combobox'), 'spring');
    expect(pool.getByText(/匹配距离 0\.9000/)).toBeTruthy();
    expect(pool.getByText(/第 2 \/ 2 名/)).toBeTruthy();
    expect(pool.queryByText(/匹配距离 0\.8000/)).toBeNull();
    expect(pool.queryByText(/第 1 \/ 2 名/)).toBeNull();
    expect(writes(fetch)).toHaveLength(0);
  });

  it('omits saved match metrics for an eligible role absent from raw TopK and still allows explicit confirmation of the original region', async () => {
    const state = poolQueue();
    state.roles[2] = { ...state.roles[2], character_id: 'autumn', recognition_enabled: true,
      profile: { ...state.roles[2].profile, id: 'autumn' } };
    state.summary.total = 3; state.summary.pending = 3;
    state.review_pool![0].eligible_character_ids!.push('autumn');
    state.review_pool![0].candidate_rank = 1;
    state.review_pool![0].candidate_total = 2;
    const fetch = mockApi((url) => {
      if (url.endsWith('/confirm-batch')) {
        const final = { ...state, review_pool: [], review_pool_count: 0 };
        return { queue: final, results: [{ source_asset_id: 'synthetic-second-episode', region_id: 'pool-region', status: 'confirmed' }],
          summary: { requested: 1, confirmed: 1, failed: 0 } } satisfies CharacterConfirmationBatchResult;
      }
      return state;
    });
    const { user, onSaved } = setup();
    const pool = within(await screen.findByRole('region', { name: '待判定候选池' }));
    await user.selectOptions(pool.getByRole('combobox'), 'autumn');
    expect(pool.queryByText(/匹配距离\s*\d+\.\d+/)).toBeNull();
    expect(pool.queryByText(/第\s*\d+\s*\/\s*\d+\s*名/)).toBeNull();
    expect(writes(fetch)).toHaveLength(0);
    const checked = pool.getByRole('checkbox', { name: /已核对：这是小秋/ }) as HTMLInputElement;
    expect(checked.disabled).toBe(false);
    await user.click(checked);
    expect(writes(fetch)).toHaveLength(0);
    await user.click(pool.getByRole('button', { name: /确认所选 1 张/ }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/characters/autumn/confirm-batch`]);
    expect(JSON.parse(writes(fetch)[0][1]!.body as string)).toEqual({ candidates: [
      { source_asset_id: 'synthetic-second-episode', region_id: 'pool-region' },
    ], single_character: true, use_as_reference: true });
  });

  it('keeps checked crops across cursor pages and confirms them together without showing an old crop on the new page', async () => {
    const firstPage = poolQueue();
    firstPage.review_pool_count = 2;
    firstPage.review_pool_offset = 0;
    firstPage.review_pool_limit = 20;
    firstPage.review_pool_next_cursor = 'synthetic-next:20?';
    firstPage.review_pool_previous_cursor = null;
    const secondPage = poolQueue();
    // A concurrent matching refresh can move an older pool region to a role.
    // Its manual draft still reserves that region until explicitly resolved.
    secondPage.roles[0].candidates = [{ ...pendingCandidate(), assignment_state: 'assigned' }];
    secondPage.review_pool = [{ ...pendingCandidate(), region_id: 'pool-region-two', frame_id: 'pool-frame-two',
      crop_url: '/synthetic/pool-two.jpg', at_ms: 12000 }];
    secondPage.review_pool_count = 2;
    secondPage.review_pool_offset = 1;
    secondPage.review_pool_limit = 20;
    secondPage.review_pool_next_cursor = null;
    secondPage.review_pool_previous_cursor = 'synthetic-first:0?';
    const nextPage = deferred<CharacterConfirmationResult>();
    const fetch = mockApi((url, options) => {
      if (url.includes('?review_pool_cursor=')) return nextPage.promise;
      if (url.endsWith('/confirm-batch')) {
        const confirmed = JSON.parse(options!.body as string).candidates as { source_asset_id: string; region_id: string }[];
        const final = poolQueue(); final.review_pool = []; final.review_pool_count = 0;
        final.roles[1].status = 'confirmed'; final.roles[1].trusted_reference_count = 2;
        final.summary.confirmed = 1; final.summary.pending = 1;
        return { queue: final, results: confirmed.map((value) => ({ ...value, status: 'confirmed' as const })),
          summary: { requested: confirmed.length, confirmed: confirmed.length, failed: 0 } } satisfies CharacterConfirmationBatchResult;
      }
      return firstPage;
    });
    const { user, onSaved } = setup();
    const firstPool = within(await screen.findByRole('region', { name: '待判定候选池' }));
    await user.selectOptions(firstPool.getByRole('combobox'), 'spring');
    await user.click(firstPool.getByRole('checkbox', { name: /已核对：这是小春/ }));
    await user.click(firstPool.getByRole('button', { name: '下一页待判定' }));
    expect((firstPool.getByRole('button', { name: '下一页待判定' }) as HTMLButtonElement).disabled).toBe(true);
    const pagedRead = fetch.mock.calls.find(([url]) => url.includes('?review_pool_cursor='))!;
    expect(pagedRead[0]).toBe(`${endpoint}?review_pool_cursor=synthetic-next%3A20%3F&review_pool_limit=20`);
    expect(pagedRead[1]?.method).toBeUndefined();
    await act(async () => { nextPage.resolve(secondPage); });
    const currentPool = within(screen.getByRole('region', { name: '待判定候选池' }));
    await waitFor(() => expect(currentPool.getByRole('combobox').getAttribute('aria-label')).toBe('为合成第二集 00:00:12选择核对角色'));
    expect(screen.queryByAltText('小春的待确认正片裁剪 00:00:06')).toBeNull();
    expect(screen.queryByAltText('小夏的待确认正片裁剪 00:00:06')).toBeNull();
    expect(screen.queryByRole('article', { name: '待判定候选 合成第二集 00:00:06' })).toBeNull();
    expect(screen.getByRole('button', { name: '提交全部已核对 1 张' })).toBeTruthy();
    expect(writes(fetch)).toHaveLength(0);
    await user.selectOptions(currentPool.getByRole('combobox'), 'spring');
    await user.click(currentPool.getByRole('checkbox', { name: /已核对：这是小春/ }));
    await user.click(screen.getByRole('button', { name: '提交全部已核对 2 张' }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(writes(fetch).map(([url]) => url)).toEqual([`${endpoint}/characters/spring/confirm-batch`]);
    expect(JSON.parse(writes(fetch)[0][1]!.body as string).candidates).toEqual([
      { source_asset_id: 'synthetic-second-episode', region_id: 'pool-region' },
      { source_asset_id: 'synthetic-second-episode', region_id: 'pool-region-two' },
    ]);
  });

  it('preserves the current page, chosen role and crop draft when a cursor request fails', async () => {
    const firstPage = poolQueue(); firstPage.review_pool_count = 2; firstPage.review_pool_next_cursor = 'synthetic-next';
    const fetch = mockApi((url) => url.includes('?review_pool_cursor=')
      ? new Response(JSON.stringify({ detail: '合成分页读取失败' }), { status: 503 }) : firstPage);
    const { user } = setup();
    const pool = within(await screen.findByRole('region', { name: '待判定候选池' }));
    await user.selectOptions(pool.getByRole('combobox'), 'spring');
    await user.click(pool.getByRole('checkbox', { name: /已核对：这是小春/ }));
    await user.click(pool.getByRole('button', { name: '下一页待判定' }));
    await screen.findByText(/合成分页读取失败/);
    expect((pool.getByRole('combobox') as HTMLSelectElement).value).toBe('spring');
    expect((pool.getByRole('checkbox', { name: /已核对：这是小春/ }) as HTMLInputElement).checked).toBe(true);
    expect(screen.getByRole('button', { name: '提交全部已核对 1 张' })).toBeTruthy();
    expect(writes(fetch)).toHaveLength(0);
    await user.click(screen.getByRole('button', { name: '重新读取确认进度' }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => url === endpoint)).toHaveLength(2));
    expect((pool.getByRole('checkbox', { name: /已核对：这是小春/ }) as HTMLInputElement).checked).toBe(true);
    expect(pool.getAllByRole('img').filter((element) => element.getAttribute('src') === '/synthetic/pool.jpg')).toHaveLength(1);
  });
});
