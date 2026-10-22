import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CharacterConfirmation from '../src/components/CharacterConfirmation';
import type { Asset, CharacterProfile } from '../src/types';
import type { CharacterConfirmationCandidate, CharacterConfirmationResult } from '../src/identityTypes';

const asset: Asset = { id: 'ep-one', title: '第一集', kind: 'animation', series: '星海', duration_ms: 120000, width: 1280, height: 720, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-10-22', status: 'ready' };
const profile: CharacterProfile = { id: 'c1', name: '小夏', name_ja: 'ナツ', aliases: ['夏同学'], description: '短发，绿色围巾。', notes: '', appearance_count: 0, appearances: [], external_source: { source: 'bangumi', subject_id: 1, character_id: 2, url: 'https://bgm.tv/character/2', image_url: 'https://lain.bgm.tv/character/2.jpg' } };
const candidate = (index: number): CharacterConfirmationCandidate => ({ region_id: `r${index}`, asset_id: index === 2 ? 'ep-two' : asset.id, asset_title: index === 2 ? '第二集' : asset.title, frame_id: `f${index}`, at_ms: index * 5000, box: [0, 0, 0.4, 0.8], crop_url: `/api/crops/r${index}`, shot_id: `shot${index}`, distance: 0.1, quality: { accepted: true }, source_available: true });
const initial = (): CharacterConfirmationResult => ({ scope_id: 'work-one', asset_id: asset.id, summary: { total: 1, pending: 1, skipped: 0, confirmed: 0, candidates: 3 }, roles: [{ character_id: profile.id, profile, status: 'pending', trusted_reference_count: 0, candidates: [candidate(1), candidate(2), candidate(3)], available_count: 3 }], sampling: null, sampling_available: true });
const endpoint = `/api/assets/${asset.id}/identity/confirmation`;
function mockApi(handler: (url: string, init?: RequestInit) => unknown | Promise<unknown>) {
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    const result = await handler(url, init);
    return result instanceof Response ? result : new Response(JSON.stringify(result), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
function setup(currentAsset = asset, onSaved = vi.fn<() => Promise<void>>().mockResolvedValue(undefined)) {
  const onClose = vi.fn();
  const view = render(<CharacterConfirmation asset={currentAsset} onSaved={onSaved} onClose={onClose} />);
  return { user: userEvent.setup(), onSaved, onClose, view };
}
function deferred<T>() {
  let resolve!: (value: T) => void; let reject!: (error: Error) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Pre-analysis character confirmation', () => {
  it('allows the next confirmation while outer refresh is pending and ignores an older refresh failure', async () => {
    const state = initial(); const firstRefresh = deferred<void>();
    const refresh = vi.fn<() => Promise<void>>().mockImplementationOnce(() => firstRefresh.promise).mockResolvedValue(undefined);
    const fetch = mockApi((_url, options) => {
      if (options?.method === 'POST') {
        const decision = JSON.parse(options.body as string);
        state.roles[0].candidates = state.roles[0].candidates.filter((item) => item.region_id !== decision.region_id);
        state.roles[0].status = 'confirmed'; state.roles[0].trusted_reference_count += 1;
        state.summary.confirmed = 1; state.summary.pending = 0;
      }
      return state;
    });
    const { user, onClose } = setup(asset, refresh);
    await screen.findByText('正片里，哪个是小夏？');
    for (let i = 0; i < 2; i += 1) {
      const card = within(screen.getAllByRole('article')[0]);
      await user.click(card.getByRole('checkbox'));
      await user.click(card.getByRole('button', { name: '确认并用作后续参考' }));
      await screen.findByText(`可信参考 ${i + 1} 张`);
      expect(screen.queryByText('正在保存确认进度')).toBeNull();
      expect((screen.getByRole('button', { name: '保存进度并返回' }) as HTMLButtonElement).disabled).toBe(false);
    }
    expect(refresh).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls.filter((call) => call[1]?.method === 'POST')).toHaveLength(2);
    await act(async () => { firstRefresh.reject(new Error('旧快照刷新失败')); });
    expect(screen.queryByText(/旧快照刷新失败/)).toBeNull();
    expect(screen.getByText(/已确认「小夏」的这张正片裁剪/)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: '保存进度并返回' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('reports refresh failure separately and retries only the refresh without repeating a saved confirmation', async () => {
    const refresh = vi.fn<() => Promise<void>>().mockRejectedValueOnce(new Error('作品列表暂不可用')).mockResolvedValue(undefined);
    const fetch = mockApi(() => initial());
    const { user } = setup(asset, refresh);
    await screen.findByText('正片里，哪个是小夏？');
    const card = within(screen.getAllByRole('article')[0]);
    await user.click(card.getByRole('checkbox'));
    await user.click(card.getByRole('button', { name: '确认并用作后续参考' }));
    await screen.findByText('确认已保存，但外层列表刷新失败：作品列表暂不可用');
    expect(screen.getByText(/已确认「小夏」的这张正片裁剪/)).toBeTruthy();
    expect(screen.queryByText('正在保存确认进度')).toBeNull();
    await user.click(screen.getByRole('button', { name: '重试刷新外层列表' }));
    await waitFor(() => expect(screen.queryByText(/外层列表刷新失败/)).toBeNull());
    expect(refresh).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls.filter((call) => call[1]?.method === 'POST')).toHaveLength(1);
  });

  it('starts invalidation before closing and handles a late refresh rejection after unmount', async () => {
    const pending = deferred<void>();
    const refresh = vi.fn<() => Promise<void>>(() => pending.promise);
    const fetch = mockApi(() => initial());
    const { user, onClose, view } = setup(asset, refresh);
    await screen.findByText('正片里，哪个是小夏？');
    const card = within(screen.getAllByRole('article')[0]);
    await user.click(card.getByRole('checkbox'));
    await user.click(card.getByRole('button', { name: '确认并用作后续参考' }));
    await screen.findByText(/已确认「小夏」的这张正片裁剪/);
    await user.click(screen.getByRole('button', { name: '保存进度并返回' }));
    expect(onClose).toHaveBeenCalledOnce(); expect(refresh).toHaveBeenCalledOnce();
    view.unmount();
    await act(async () => { pending.reject(new Error('卸载后刷新失败')); });
    expect(screen.queryByText(/卸载后刷新失败/)).toBeNull();
    expect(refresh).toHaveBeenCalledOnce();
    expect(fetch.mock.calls.filter((call) => call[1]?.method === 'POST')).toHaveLength(1);
  });

  it('reads existing cross-episode candidates without starting inference and requires single-character consent', async () => {
    const state = initial();
    const fetch = mockApi((_url, options) => {
      if (options?.method === 'POST') { state.roles[0].status = 'confirmed'; state.roles[0].trusted_reference_count = 1; state.roles[0].candidates = [candidate(1), candidate(3)]; state.summary.confirmed = 1; state.summary.pending = 0; }
      return state;
    });
    const { user, onSaved } = setup();
    await screen.findByText('正片里，哪个是小夏？');
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe(endpoint);
    expect(screen.getByAltText('小夏的 Bangumi 档案参考图').getAttribute('src')).toBe(profile.external_source!.image_url);
    const cards = screen.getAllByRole('article'); expect(cards).toHaveLength(3);
    const card = within(cards[1]);
    expect((card.getByRole('button', { name: '确认并用作后续参考' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(card.getByRole('checkbox', { name: /已核对：这是小夏/ }));
    await user.click(card.getByRole('button', { name: '确认并用作后续参考' }));
    await screen.findByText(/已确认「小夏」的这张正片裁剪/);
    const mutation = fetch.mock.calls.find((call) => call[1]?.method === 'POST')!;
    expect(mutation[0]).toBe(`${endpoint}/characters/c1/decision`);
    expect(JSON.parse(mutation[1]!.body as string)).toEqual({ action: 'confirm', source_asset_id: 'ep-two', region_id: 'r2', single_character: true, use_as_reference: true });
    expect(onSaved).toHaveBeenCalledOnce();
    expect(screen.getByText('已确认 1 / 1 个角色')).toBeTruthy();
    expect(screen.getAllByRole('checkbox').every((element) => !(element as HTMLInputElement).checked)).toBe(true);
  });

  it('persists a rejection and each unclear candidate when changing the group', async () => {
    const state = initial();
    const fetch = mockApi((_url, options) => {
      if (options?.method === 'POST') {
        const decision = JSON.parse(options.body as string);
        state.roles[0].candidates = state.roles[0].candidates.filter((item) => item.region_id !== decision.region_id);
      }
      return state;
    });
    const { user } = setup();
    await screen.findByText('正片里，哪个是小夏？');
    await user.click(screen.getAllByRole('button', { name: '不是这个角色' })[0]);
    await screen.findByText('已记住这个否定结果，不会再为该角色推荐这张候选。');
    expect(screen.getAllByRole('article')).toHaveLength(2);
    await user.click(screen.getByRole('button', { name: '看不清／换一组' }));
    await screen.findByText('暂时没有合适候选');
    expect(fetch.mock.calls.filter((call) => call[1]?.method === 'POST').map((call) => JSON.parse(call[1]!.body as string))).toEqual([
      { action: 'reject', source_asset_id: 'ep-one', region_id: 'r1' },
      { action: 'unclear', source_asset_id: 'ep-two', region_id: 'r2' },
      { action: 'unclear', source_asset_id: 'ep-one', region_id: 'r3' },
    ]);
  });

  it('saves skipped roles and resumes them without requiring every role to appear', async () => {
    const state = initial(); state.roles[0].candidates = [];
    const fetch = mockApi((_url, options) => {
      if (options?.method === 'POST') {
        const action = JSON.parse(options.body as string).action;
        state.roles[0].status = action === 'skip' ? 'skipped' : 'pending';
        state.summary.skipped = action === 'skip' ? 1 : 0;
        state.summary.pending = action === 'skip' ? 0 : 1;
      }
      return state;
    });
    const { user, onClose } = setup();
    await screen.findByText('暂时没有合适候选');
    await user.click(screen.getByRole('button', { name: '暂时跳过这个角色' }));
    await screen.findByText('这个角色已暂时跳过');
    await user.click(screen.getByRole('button', { name: '继续确认这个角色' }));
    await screen.findByText('暂时没有合适候选');
    expect(fetch.mock.calls.filter((call) => call[1]?.method === 'POST').map((call) => JSON.parse(call[1]!.body as string))).toEqual([{ action: 'skip' }, { action: 'resume' }]);
    await user.click(screen.getByRole('button', { name: '保存进度并返回' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('caches Bangumi references explicitly and reports partial errors without granting trust', async () => {
    const fetch = mockApi((url) => url.endsWith('/references/cache-bangumi') ? { imported: 2, existing: 1, skipped: 1, errors: [{ character_id: 'c1', reason: '远端图片下载失败' }] } : initial());
    const { user } = setup();
    await user.click(await screen.findByText('候选不足？补充参考图与少量采样'));
    await user.click(await screen.findByRole('button', { name: '缓存 Bangumi 参考图' }));
    await screen.findByText(/Bangumi 参考图：新增 2 张/);
    expect(screen.getByText('小夏：远端图片下载失败')).toBeTruthy();
    expect(screen.getByText('可信参考 0 张')).toBeTruthy();
    const call = fetch.mock.calls.find((entry) => entry[1]?.method === 'POST')!;
    expect(call[0]).toBe('/api/assets/ep-one/references/cache-bangumi');
    expect(JSON.parse(call[1]!.body as string)).toEqual({ max_images: 20 });
  });

  it('starts a bounded sample only on demand and recovers from unavailable models', async () => {
    let attempt = 0;
    const fetch = mockApi((url) => url.endsWith('/sample') ? { ...initial(), sampling: ++attempt === 1 ? { status: 'unavailable', completed: 0, total: 12, asset_id: asset.id, error: '未找到本地 CCIP 模型' } : { status: 'completed', completed: 12, total: 12, asset_id: asset.id } } : initial());
    const { user } = setup();
    await user.click(await screen.findByText('候选不足？补充参考图与少量采样'));
    await user.click(await screen.findByRole('button', { name: '补充少量本地采样' }));
    await screen.findByText('未找到本地 CCIP 模型');
    await user.click(screen.getByRole('button', { name: '补充少量本地采样' }));
    await screen.findByRole('button', { name: '继续少量本地采样' });
    const calls = fetch.mock.calls.filter((entry) => entry[1]?.method === 'POST');
    expect(calls).toHaveLength(2);
    expect(JSON.parse(calls[0][1]!.body as string)).toEqual({ max_frames: 12 });
    expect(screen.getByText(/没有匹配候选不代表角色一定未出场/)).toBeTruthy();
  });

  it('polls only lightweight progress and reloads the roster after sampling completes', async () => {
    vi.useFakeTimers();
    let polls = 0;
    const fetch = mockApi((url) => url.endsWith('/sampling') ? { sampling: { status: ++polls === 1 ? 'running' : 'completed', completed: polls === 1 ? 4 : 12, total: 12, asset_id: asset.id } } : { ...initial(), sampling: { status: polls >= 2 ? 'completed' : 'running', completed: polls >= 2 ? 12 : 0, total: 12, asset_id: asset.id } });
    setup();
    await act(async () => { await Promise.resolve(); });
    fireEvent.click(screen.getByText('候选不足？补充参考图与少量采样'));
    expect(screen.getByText('本地采样中 · 0 / 12 帧')).toBeTruthy();
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(screen.getByText('本地采样中 · 4 / 12 帧')).toBeTruthy();
    expect(fetch.mock.calls.filter((entry) => entry[0] === endpoint)).toHaveLength(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(fetch.mock.calls.filter((entry) => entry[0] === endpoint)).toHaveLength(2);
    expect(screen.getByRole('button', { name: '继续少量本地采样' })).toBeTruthy();
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(polls).toBe(2);
  });

  it('seeks the source clip, pauses at the preview end and offers a playback fallback', async () => {
    mockApi(() => initial());
    const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    const { user } = setup();
    await screen.findByText('正片里，哪个是小夏？');
    await user.click(screen.getAllByRole('button', { name: '播放前后片段' })[0]);
    const video = screen.getByLabelText('第一集在00:00:05的角色确认片段') as HTMLVideoElement;
    expect(video.getAttribute('src')).toBe('/api/assets/ep-one/media');
    Object.defineProperty(video, 'duration', { value: 120 });
    fireEvent.loadedMetadata(video); expect(video.currentTime).toBe(3);
    video.currentTime = 9.2; Object.defineProperty(video, 'paused', { value: false });
    fireEvent.timeUpdate(video); expect(pause).toHaveBeenCalledOnce(); expect(video.currentTime).toBe(9);
    fireEvent.error(video);
    expect(screen.getByText(/浏览器暂时无法播放这份原片/)).toBeTruthy();
    expect(screen.getByRole('link', { name: '查看完整原始画面' }).getAttribute('href')).toBe('/api/assets/ep-one/frames/f1');
    expect(screen.getByRole('link', { name: '前往剧集核对' }).getAttribute('href')).toBe('#/asset/ep-one?t=5000');
  });

  it('keeps saved candidate images usable when original files are unavailable', async () => {
    const state = initial(); state.roles[0].candidates = [{ ...candidate(1), source_available: false }];
    mockApi(() => state); const { user } = setup({ ...asset, source_available: false });
    await screen.findByText('正片里，哪个是小夏？');
    await user.click(screen.getByText('候选不足？补充参考图与少量采样'));
    expect((screen.getByRole('button', { name: '播放前后片段' }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: '补充少量本地采样' }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole('link', { name: '原始画面' }).getAttribute('href')).toBe('/api/assets/ep-one/frames/f1');
    expect((screen.getByRole('checkbox') as HTMLInputElement).disabled).toBe(false);
  });

  it('recovers read errors and keeps failed decisions visible without claiming success', async () => {
    let reads = 0;
    mockApi((_url, options) => options?.method === 'POST' ? new Response(JSON.stringify({ detail: '候选已更新，请重新核对' }), { status: 409 }) : ++reads === 1 ? new Response(JSON.stringify({ detail: '确认进度暂不可用' }), { status: 503 }) : initial());
    const { user, onSaved } = setup();
    await screen.findByText('确认进度暂不可用');
    await user.click(screen.getByRole('button', { name: '重新读取确认进度' }));
    await screen.findByText('正片里，哪个是小夏？');
    const card = within(screen.getAllByRole('article')[0]);
    await user.click(card.getByRole('checkbox'));
    await user.click(card.getByRole('button', { name: '确认并用作后续参考' }));
    await screen.findByText('候选已更新，请重新核对');
    expect(onSaved).not.toHaveBeenCalled();
    expect(screen.getAllByRole('article')).toHaveLength(3);
    await waitFor(() => expect((card.getByRole('button', { name: '确认并用作后续参考' }) as HTMLButtonElement).disabled).toBe(false));
  });
});
