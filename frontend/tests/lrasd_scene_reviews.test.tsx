import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import LRASDPreview from '../src/components/LRASDPreview';
import { type LRASDSceneReviewRun, type LRASDSceneReviewsResult } from '../src/identityTypes';
import { type CharacterProfile } from '../src/types';

const profile: CharacterProfile = { id: 'kaguya', name: '辉夜', aliases: [], description: '', notes: '', appearances: [], appearance_count: 0 };
const props = { endpoint: '/assets/episode-1/audio', durationMs: 120000, initialStartMs: 0,
  sourceAvailable: true, hasAudio: true, configuration: null, settingsDirty: false, disabled: false,
  profiles: [profile], onSeek: vi.fn(), onRunningChange: vi.fn() };
const completedRun: LRASDSceneReviewRun = {
  run_id: 'scene-run-1', job_id: 'full-job-1', source: 'full_scene_job', status: 'completed', created_at: '2026-10-18T06:00:00Z',
  coverage: { mode: 'bounded_scene_sample', max_windows: 3, selected_windows: 1,
    selected_duration_ms: 4000, analysis_duration_ms: 120000, full_episode_scored: false }, degraded: [],
  reviews: [{ scene_id: 'scene-2', scene_start_ms: 10000, scene_end_ms: 20000, start_ms: 12000, end_ms: 16000,
    status: 'review_required', review_status: 'candidate', model_version: 'LR-ASD human-v2', checkpoint_sha256: 'a'.repeat(64),
    subtitle_ids: ['subtitle-1'], frame_ids: ['frame-1'], audio_record_ids: [], tracks: [{ id: 'person-track-1', start_ms: 12200, end_ms: 15800,
      association_status: 'candidate', active_speaker_probability: .8234, frame_probabilities: [{ at_ms: 12200, probability: .8 }],
      candidates: [{ character_id: 'kaguya', distance: .23456, reference_ids: ['ref-1'] }] }] }],
};

function mockRequests(reviews: () => Promise<LRASDSceneReviewsResult> | LRASDSceneReviewsResult) {
  const fetch = vi.fn(async (url: string, _options?: RequestInit) => new Response(JSON.stringify(url.endsWith('/lrasd-scene-reviews')
    ? await reviews() : { status: 'not_started' }), { status: 200 }));
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('full analysis LR-ASD saved scene reviews', () => {
  it('shows bounded coverage, actual scene/model status, speaker scores and candidate identities with read-only navigation', async () => {
    const fetch = mockRequests(() => ({ status: 'completed', runs: [completedRun] }));
    const onSeek = vi.fn();
    render(<LRASDPreview {...props} onSeek={onSeek} />);
    const section = within(await screen.findByRole('region', { name: '正式任务 LR-ASD 候选复核' }));
    await section.findByText('抽样处理完成 · 待人工核对');
    expect(section.getByText(/这是抽样复核，不代表整集逐帧/)).toBeTruthy();
    expect(section.getByText('已选 1 / 最多 3 个窗口 · 抽样 4.00 秒 / 本次分析 120.00 秒 · 覆盖范围：抽样窗口')).toBeTruthy();
    expect(section.getByText('scene-2')).toBeTruthy();
    expect(section.getByText('窗口评分完成 · 候选待核对')).toBeTruthy();
    expect(section.getByText(/模型：LR-ASD human-v2/)).toBeTruthy();
    expect(section.getByText('a'.repeat(64))).toBeTruthy();
    const person = within(section.getByRole('article', { name: 'LR-ASD 抽样人物轨迹 person-track-1' }));
    expect(person.getByText('平均发言概率：0.823')).toBeTruthy();
    expect(person.getByText('角色候选：辉夜 · CCIP 距离 0.2346 · 候选待确认')).toBeTruthy();
    expect(person.getByText('候选 · 待人工核对')).toBeTruthy();
    fireEvent.click(section.getByRole('button', { name: '原片 00:00:12.000 — 00:00:16.000' }));
    expect(onSeek).toHaveBeenCalledWith(12000);
    expect(screen.getByRole('region', { name: 'LR-ASD 短片段试跑' })).toBeTruthy();
    expect(section.queryByRole('button', { name: /确认说话人|确认人物|启动|试跑/ })).toBeNull();
    expect(fetch.mock.calls.every(([, options]) => !options?.method || options.method === 'GET')).toBe(true);
  });

  it('refreshes only saved results and shows partial/unavailable windows without fabricated identities', async () => {
    let calls = 0;
    const run = { ...completedRun, status: 'partial' as const, degraded: ['第二个窗口模型不可用'],
      reviews: [{ ...completedRun.reviews[0], status: 'unavailable' as const, reason: '本机权重不可用', tracks: [] }] };
    const fetch = mockRequests(() => { calls += 1; return { status: 'partial', runs: [run] }; });
    render(<LRASDPreview {...props} />);
    await screen.findByText('部分窗口完成 · 待人工核对');
    expect(screen.getByText('第二个窗口模型不可用')).toBeTruthy();
    expect(screen.getByText('本机权重不可用')).toBeTruthy();
    expect(screen.queryByRole('article')).toBeNull();
    expect(screen.queryByText('窗口评分完成 · 候选待核对')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: '刷新抽样结果' }));
    await waitFor(() => expect(calls).toBe(2));
    expect(fetch.mock.calls.every(([, options]) => !options?.method)).toBe(true);
  });

  it('polls an in-progress full run until it completes', async () => {
    let calls = 0;
    mockRequests(() => { calls += 1; return calls === 1 ? { status: 'running', runs: [{ ...completedRun, status: 'running', reviews: [] }] }
      : { status: 'completed', runs: [completedRun] }; });
    render(<LRASDPreview {...props} />);
    await screen.findByText('正在生成抽样复核结果');
    await screen.findByText('抽样处理完成 · 待人工核对', {}, { timeout: 3500 });
    expect(calls).toBe(2);
  });

  it('does not replace the separate short-preview error/result state when saved scene reviews fail', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => new Response(JSON.stringify(url.endsWith('/lrasd-scene-reviews')
      ? { detail: '已保存结果暂不可读' } : { status: 'no_tracks', reason: '独立短片中没有连续轨迹' }),
    { status: url.endsWith('/lrasd-scene-reviews') ? 503 : 200 })));
    render(<LRASDPreview {...props} />);
    expect(await screen.findByText('抽样结果读取失败：已保存结果暂不可读')).toBeTruthy();
    expect(await screen.findByText('独立短片中没有连续轨迹')).toBeTruthy();
    expect(screen.getByRole('button', { name: '刷新抽样结果' })).toBeTruthy();
  });

  it('ignores saved results from the previous asset after navigation', async () => {
    let resolveOld: (value: LRASDSceneReviewsResult) => void = () => {};
    const old = new Promise<LRASDSceneReviewsResult>((resolve) => { resolveOld = resolve; });
    vi.stubGlobal('fetch', vi.fn(async (url: string) => new Response(JSON.stringify(url.endsWith('/lrasd-scene-reviews')
      ? url.includes('/episode-1/') ? await old : { status: 'not_started', runs: [] }
      : { status: 'not_started' }), { status: 200 })));
    const { rerender } = render(<LRASDPreview {...props} />);
    rerender(<LRASDPreview {...props} endpoint="/assets/episode-2/audio" />);
    await screen.findByText('尚无正式任务的 LR-ASD 抽样结果');
    resolveOld({ status: 'completed', runs: [completedRun] });
    await waitFor(() => expect(screen.queryByText('scene-2')).toBeNull());
  });
});
