import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import LRASDPreview from '../src/components/LRASDPreview';
import AudioIdentity from '../src/components/AudioIdentity';
import { type Asset, type CharacterProfile } from '../src/types';
import { type AudioConfiguration } from '../src/identityTypes';

const asset: Asset = { id: 'kaguya', work_id: 'work', title: '辉夜姬', kind: 'animation', duration_ms: 120000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '', status: 'ready' };
const profile: CharacterProfile = { id: 'kaguya-profile', name: '辉夜', aliases: [], description: '', notes: '', appearances: [], appearance_count: 0, external_source: { source: 'bangumi', subject_id: 1, character_id: 2, url: 'https://bgm.tv/character/2', image_url: 'https://lain.bgm.tv/pic/crt/l/example.jpg' } };
const configuration: AudioConfiguration = {
  settings: { enabled: true, stream_index: null, language: 'ja', dub_version: 'original', diarization_model_path: '', embedding_model_path: '', asr_model_path: '', asr_enabled: false, device: 'cuda', sync_enabled: true, sync_repository_path: '/local/lrasd', sync_checkpoint_path: '/local/bundle', sync_animation_verified: false, calibration_path: '', sync_model_type: 'lrasd', sync_python_path: '/local/python', sync_detector_path: '/local/person.onnx' },
  availability: { diarization: { available: false, reason: '未配置说话人分段' }, embedding: { available: false, reason: '' }, asr: { available: false, reason: '' }, sync: { available: true, reason: '候选需复核' }, reasons: [] },
};
const defaultProps = { endpoint: '/assets/kaguya/audio', durationMs: 120000, initialStartMs: 10000, sourceAvailable: true, hasAudio: true, configuration, settingsDirty: false, disabled: false, profiles: [profile], onSeek: vi.fn(), onRunningChange: vi.fn() };

function mockFetch(handler: (url: string, options?: RequestInit) => unknown) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => new Response(JSON.stringify(await handler(url, options)), { status: 200 }));
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('LR-ASD short-preview workflow', () => {
  it('runs without diarization or confirmed tracks, polls and separates probability from identity candidates', async () => {
    let posted = false;
    const fetch = mockFetch((_url, options) => {
      if (options?.method === 'POST') { posted = true; return { status: 'running', job_id: 'preview-1', start_ms: 10000, end_ms: 14000 }; }
      return posted ? { status: 'review_required', job_id: 'preview-1', start_ms: 10000, end_ms: 14000, cache_hit: false, timings: { total_ms: 1250 }, tracks: [{ id: 'P1', start_ms: 10200, end_ms: 13800, thumbnail_url: '/api/assets/kaguya/audio/lrasd-previews/preview-1/tracks/P1/thumbnail', active_speaker_probability: .8234, frame_probabilities: [{ at_ms: 10200, probability: .8 }], candidates: [{ character_id: profile.id, distance: .23456, reference_ids: ['ref1'] }], association_status: 'candidate' }] } : { status: 'not_started' };
    });
    const onRunningChange = vi.fn(); const onSeek = vi.fn();
    render(<LRASDPreview {...defaultProps} onRunningChange={onRunningChange} onSeek={onSeek} />);
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    expect(fetch.mock.calls.some(([, options]) => options?.method === 'POST')).toBe(false);
    const button = screen.getByRole('button', { name: '试跑 LR-ASD 短片段' }) as HTMLButtonElement;
    expect(button.disabled).toBe(false);
    fireEvent.click(button);
    await screen.findByText('试跑完成 · 待人工核对', {}, { timeout: 3000 });
    const request = fetch.mock.calls.find(([, options]) => options?.method === 'POST')!;
    expect(request[0]).toBe('/api/assets/kaguya/audio/lrasd-preview');
    expect(JSON.parse(request[1]!.body as string)).toEqual({ start_ms: 10000, end_ms: 14000 });
    const track = within(screen.getByRole('article', { name: 'LR-ASD 人物轨迹 P1' }));
    expect(track.getByText('人物1')).toBeTruthy();
    expect(track.getByText('平均发言概率：0.823')).toBeTruthy();
    expect(track.getByText('角色候选：辉夜 · CCIP 距离 0.2346')).toBeTruthy();
    expect(track.getByText(/不代表人物身份置信度/)).toBeTruthy();
    expect(track.getByRole('img').getAttribute('src')).toContain('/lrasd-previews/preview-1/');
    fireEvent.click(track.getByRole('button', { name: '00:00:10 — 00:00:13' }));
    expect(onSeek).toHaveBeenCalledWith(10200);
    expect(onRunningChange).toHaveBeenCalledWith(true);
    expect(onRunningChange).toHaveBeenLastCalledWith(false);
    expect(screen.queryByRole('button', { name: '确认说话人' })).toBeNull();
  });

  it('requires saved LR-ASD and local detector configuration, enforcing the 0.64–8 second range', async () => {
    const fetch = mockFetch(() => ({ status: 'not_started' }));
    const { rerender } = render(<LRASDPreview {...defaultProps} settingsDirty />);
    expect(screen.getByText('配置已修改，请先保存音频配置，再试跑。')).toBeTruthy();
    expect((screen.getByRole('button', { name: '试跑 LR-ASD 短片段' }) as HTMLButtonElement).disabled).toBe(true);
    rerender(<LRASDPreview {...defaultProps} configuration={{ ...configuration, settings: { ...configuration.settings, sync_detector_path: '' } }} />);
    expect((screen.getByRole('button', { name: '试跑 LR-ASD 短片段' }) as HTMLButtonElement).disabled).toBe(true);
    rerender(<LRASDPreview {...defaultProps} />);
    fireEvent.change(screen.getByLabelText('试跑终点（秒）'), { target: { value: '10.639' } });
    expect((screen.getByRole('button', { name: '试跑 LR-ASD 短片段' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText('试跑终点（秒）'), { target: { value: '18.001' } });
    expect((screen.getByRole('button', { name: '试跑 LR-ASD 短片段' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText('试跑终点（秒）'), { target: { value: '10.640' } });
    expect((screen.getByRole('button', { name: '试跑 LR-ASD 短片段' }) as HTMLButtonElement).disabled).toBe(false);
    expect(fetch.mock.calls.some(([, options]) => options?.method === 'POST')).toBe(false);
  });

  it('shows no-track and unavailable reasons without inventing successful identities', async () => {
    const fetch = mockFetch((_url, options) => options?.method === 'POST' ? { status: 'no_tracks', reason: '人物连续帧不足', tracks: [] } : { status: 'not_started' });
    render(<LRASDPreview {...defaultProps} />);
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole('button', { name: '试跑 LR-ASD 短片段' }));
    await screen.findByText('没有可评分的连续人物轨迹');
    expect(screen.getByText('人物连续帧不足')).toBeTruthy();
    expect(screen.queryByRole('article')).toBeNull();
    expect(screen.queryByText('试跑完成 · 待人工核对')).toBeNull();
  });

  it.each(['review_required', 'no_tracks'])('opens the saved local clip for a completed %s preview without autoplay', async (status) => {
    const clipUrl = '/api/assets/kaguya/audio/lrasd-previews/saved-preview/clip';
    const fetch = mockFetch(() => ({ status, start_ms: 10200, end_ms: 14200, tracks: [], clip_url: clipUrl }));
    render(<LRASDPreview {...defaultProps} />);
    const video = await screen.findByLabelText('本次 LR-ASD 试跑片段播放器') as HTMLVideoElement;
    expect(video.getAttribute('src')).toBe(clipUrl);
    expect(video.controls).toBe(true);
    expect(video.preload).toBe('metadata');
    expect(video.autoplay).toBe(false);
    expect(video.hasAttribute('autoplay')).toBe(false);
    expect(screen.getByText('本次试跑片段（从0开始，对应原片范围）')).toBeTruthy();
    expect(screen.getByText('原片 00:00:10.200 — 00:00:14.200')).toBeTruthy();
    expect(fetch.mock.calls.some(([, options]) => options?.method === 'POST')).toBe(false);
  });

  it('does not present a playable clip from an unsuccessful preview', async () => {
    mockFetch(() => ({ status: 'failed', tracks: [], clip_url: '/api/assets/kaguya/audio/lrasd-previews/failed/clip' }));
    render(<LRASDPreview {...defaultProps} />);
    await screen.findByText('短片段试跑失败');
    expect(screen.queryByLabelText('本次 LR-ASD 试跑片段播放器')).toBeNull();
  });

  it('shows only named millisecond timings and never formats feature counts as seconds', async () => {
    mockFetch(() => ({ status: 'review_required', tracks: [], timings: {
      clip_encode_ms: 1500, detector_load_ms: 30, feature_ms: 1900, total_ms: 3430,
      feature_cache_hits: 0, feature_extractions: 19, internal_stage_ms: 20,
    } }));
    render(<LRASDPreview {...defaultProps} />);
    const timings = within(await screen.findByLabelText('LR-ASD 试跑耗时'));
    expect(timings.getByText('短片编码')).toBeTruthy();
    expect(timings.getByText('检测模型加载')).toBeTruthy();
    expect(timings.getByText('参考图特征计算')).toBeTruthy();
    expect(timings.getByText('1.50 秒')).toBeTruthy();
    expect(timings.getByText('0.03 秒')).toBeTruthy();
    expect(timings.getByText('1.90 秒')).toBeTruthy();
    expect(timings.queryByText(/feature_cache_hits|feature_extractions|internal_stage_ms|clip_encode_ms|detector_load_ms|feature_ms/)).toBeNull();
    expect(timings.queryByText('0.00 秒')).toBeNull();
    expect(timings.queryByText('0.02 秒')).toBeNull();
  });

  it('shows compact person numbers while retaining opaque track IDs in accessible evidence labels', async () => {
    const ids = ['track_5fea10503390e75e5d4b', 'track_e899a133e7e32ec93140'];
    mockFetch(() => ({ status: 'review_required', tracks: ids.map((id) => ({
      id, start_ms: 10000, end_ms: 14000, active_speaker_probability: .4, candidates: [],
    })) }));
    render(<LRASDPreview {...defaultProps} />);
    for (const [index, id] of ids.entries()) {
      const track = within(await screen.findByRole('article', { name: 'LR-ASD 人物轨迹 ' + id }));
      expect(track.getByText('人物' + (index + 1))).toBeTruthy();
      expect(screen.queryByText(id)).toBeNull();
    }
  });

  it('caches only stored Bangumi images on explicit click and keeps downloaded references as candidates', async () => {
    const saved = vi.fn();
    const reference = { id: 'ref1', character_id: profile.id, status: 'candidate', source: { kind: 'external_cached', url: profile.external_source!.image_url }, quality: {}, version: 1, image_url: '/api/assets/kaguya/references/ref1/image' };
    const fetch = mockFetch((url, options) => {
      if (url.endsWith('/references/cache-bangumi')) return { imported: 1, existing: 0, skipped: 1, errors: [{ character_id: profile.id, reason: '另一张图片暂不可用' }], references: [reference] };
      if (url.endsWith('/audio/settings')) return configuration;
      if (url.endsWith('/audio/tracks')) return { tracks: [{ index: 1, language: 'ja', title: '', default: true, start_ms: 0 }] };
      if (url.endsWith('/audio/records')) return { status: 'not_started', records: [] };
      if (url.endsWith('/audio/lrasd-preview')) return { status: 'not_started' };
      if (url.endsWith('/characters')) return { profiles: [profile], revision: 1, assets: [asset] };
      if (url.endsWith('/identity/reviews')) return { status: 'not_started', regions: [] };
      if (url.endsWith('/identity/settings')) return { enabled: false, model: 'ccip-caformer-24-randaug-pruned', top_k: 5, calibration: null };
      throw new Error('Unexpected request: ' + url + ' ' + options?.method);
    });
    render(<AudioIdentity asset={asset} onSeek={vi.fn()} onSaved={saved} />);
    await screen.findByRole('button', { name: '缓存 Bangumi 角色图' });
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/references/cache-bangumi'))).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: '缓存 Bangumi 角色图' }));
    await screen.findByText(/新增 1 张，已有 0 张，跳过 1 项/);
    const request = fetch.mock.calls.find(([url]) => url.endsWith('/references/cache-bangumi'))!;
    expect(request[1]?.method).toBe('POST');
    expect(JSON.parse(request[1]!.body as string)).toEqual({ max_images: 20 });
    expect(screen.getByText('候选 · 待确认 · 已缓存外部图')).toBeTruthy();
    expect(screen.getByRole('img', { name: '辉夜 的本机角色参考图' }).getAttribute('src')).toBe(reference.image_url);
    expect(screen.getByText('辉夜：另一张图片暂不可用')).toBeTruthy();
    expect(saved).toHaveBeenCalled();
    expect(fetch.mock.calls.some(([, options]) => options?.method === 'PATCH')).toBe(false);
  });
});
