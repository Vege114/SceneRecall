import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AudioIdentity from '../src/components/AudioIdentity';
import CharacterReferences, { CreateCharacter } from '../src/components/CharacterReferences';
import { type Asset, type CharacterProfile } from '../src/types';
import { type AudioConfiguration, type AudioRecord, type CharacterReference } from '../src/identityTypes';

const asset: Asset = { id: 'episode', work_id: 'work', title: '动画', kind: 'animation', duration_ms: 120000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '', status: 'ready' };
const profiles: CharacterProfile[] = Array.from({ length: 20 }, (_, i) => ({ id: `c${i + 1}`, name: `角色 ${i + 1}`, aliases: [], description: '', notes: '', appearances: [], appearance_count: 0 }));
const config: AudioConfiguration = { settings: { enabled: false, stream_index: null, language: 'ja', dub_version: 'original', diarization_model_path: '', embedding_model_path: '', asr_model_path: '', asr_enabled: false, device: 'cpu', sync_enabled: false, sync_repository_path: '', sync_checkpoint_path: '', sync_animation_verified: false, calibration_path: '' }, availability: { diarization: { available: false, reason: '未安装 community-1' }, embedding: { available: false, reason: '未配置 ECAPA' }, asr: { available: false, reason: '' }, sync: { available: false, reason: '同步权重未验证' }, reasons: [] } };
const record: AudioRecord = { id: 'r1', asset_id: asset.id, start_ms: 1200, end_ms: 4500, speaker_id: 'episode:track-1:speaker-0', character_id: null, candidate_character_id: 'c20', association_status: 'candidate', association_source: 'voice_bank', candidates: [{ character_id: 'c20', distance: 0.23, sample_ids: ['v1'] }], subtitle_ids: ['cue1'], frame_ids: ['frame1'], text: '中文译文', language: 'ja', dub_version: 'original', audio_stream_index: 1, audio_ref: { path: 'clip.wav' }, model_versions: { embedding: 'ecapa-test' }, overlap: true, needs_identity_review: true };
const reference: CharacterReference = { id: 'ref1', character_id: 'c1', status: 'candidate', source: { kind: 'external_cached', url: 'https://example.org/character' }, quality: { width: 400, height: 600, sharpness: 100, accepted: true, single_character: false, reasons: [] }, version: 1, image_url: '/api/reference/ref1.png' };

function mockApi(overrides: (url: string, options?: RequestInit) => unknown = () => undefined) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    let result = await overrides(url, options);
    if (result === undefined) {
      if (url.endsWith('/audio/settings')) result = options?.method === 'PUT' ? { ...config, settings: JSON.parse(options.body as string) } : config;
      else if (url.endsWith('/audio/tracks')) result = { tracks: [{ index: 1, language: 'ja', title: '日语', default: true, start_ms: 350 }] };
      else if (url.endsWith('/audio/records')) result = { status: 'completed', records: [record], timings: { total_s: 1.7, model_load_s: 1.1 }, cache_hits: 2 };
      else if (url.endsWith('/identity/reviews')) result = { status: 'disabled', regions: [], degraded_reason: '视觉身份尚未启用' };
      else if (url.endsWith('/identity/settings')) result = { enabled: false, model: 'ccip-caformer-24-randaug-pruned', top_k: 5, calibration: null };
      else if (url.endsWith('/characters')) result = { profiles, revision: 1, assets: [asset] };
      else if (url.endsWith('/references')) result = { revision: 1, references: [reference] };
      else if (url.endsWith('/voice-samples')) result = { samples: [] };
      else result = {};
    }
    return result instanceof Response ? result : new Response(JSON.stringify(result), { status: 200 });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Audiovisual identity workflow', () => {
  it('shows disabled and unverified capability state, source track offsets and scoped evidence', async () => {
    mockApi(); render(<AudioIdentity asset={asset} onSeek={vi.fn()} onSaved={vi.fn()} />);
    await screen.findByText('视听身份与证据');
    expect((screen.getByRole('button', { name: '保存配置并识别这段音频' }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText('未安装 community-1')).toBeTruthy();
    expect(screen.getByText('同步权重未验证')).toBeTruthy();
    expect(screen.getByRole('option', { name: /起点 350 ms/ })).toBeTruthy();
    const article = within(screen.getByRole('article', { name: '声音记录 r1' }));
    expect(article.getByText('候选：角色 20')).toBeTruthy();
    expect(article.getByText('episode:track-1:speaker-0')).toBeTruthy();
    expect(article.getByText(/重叠发言/)).toBeTruthy();
    expect(article.getByLabelText('试听声音记录 r1').getAttribute('src')).toBe('/api/assets/episode/audio/records/r1/clip');
    expect(article.getByRole('link', { name: '字幕 1' }).getAttribute('href')).toContain('cue=cue1');
    expect(screen.getByText('模型加载')).toBeTruthy();
    expect(screen.getByText('1.70 秒')).toBeTruthy();
    expect(screen.getByText('1.10 秒')).toBeTruthy();
  });

  it('does not run when no audio track exists', async () => {
    const user = userEvent.setup(); const fetch = mockApi((url) => url.endsWith('/audio/tracks') ? { tracks: [] } : undefined);
    render(<AudioIdentity asset={asset} onSeek={vi.fn()} onSaved={vi.fn()} />);
    await screen.findByText('当前视频没有可用音轨，不能执行声音识别。');
    await user.click(screen.getByLabelText('启用本机音频角色识别'));
    expect((screen.getByRole('button', { name: '保存配置并识别这段音频' }) as HTMLButtonElement).disabled).toBe(true);
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/analyze'))).toBe(false);
  });

  it('saves explicit language and dub scope, then reports unavailable instead of claiming success', async () => {
    const user = userEvent.setup(); const fetch = mockApi((url) => url.endsWith('/analyze') ? { status: 'unavailable', records: [], degraded: ['ECAPA 模型目录不存在'], timings: { total_ms: 20 } } : undefined);
    render(<AudioIdentity asset={asset} onSeek={vi.fn()} onSaved={vi.fn()} />);
    await screen.findByText('视听身份与证据');
    await user.click(screen.getByLabelText('启用本机音频角色识别'));
    await user.clear(screen.getByLabelText('原音语言')); await user.type(screen.getByLabelText('原音语言'), 'zh');
    await user.clear(screen.getByLabelText(/^配音版本/)); await user.type(screen.getByLabelText(/^配音版本/), 'mandarin-2026');
    await user.selectOptions(screen.getByLabelText('识别音轨'), '1');
    await user.click(screen.getByRole('button', { name: '保存配置并识别这段音频' }));
    await screen.findByText('降级原因：ECAPA 模型目录不存在');
    const saved = fetch.mock.calls.find(([url, options]) => url.endsWith('/settings') && options?.method === 'PUT')!;
    expect(JSON.parse(saved[1]!.body as string)).toMatchObject({ enabled: true, language: 'zh', dub_version: 'mandarin-2026', stream_index: 1 });
    const run = fetch.mock.calls.find(([url]) => url.endsWith('/analyze'))!;
    expect(JSON.parse(run[1]!.body as string)).toEqual({ start_ms: 0, end_ms: 60000 });
    expect(screen.queryByText('声音记录已保存。请试听并核对候选说话人。')).toBeNull();
  });

  it('keeps the full cast selectable and requires manual listening to confirm or revoke an audio identity', async () => {
    const user = userEvent.setup(); const saved = vi.fn(); const fetch = mockApi();
    render(<AudioIdentity asset={asset} onSeek={vi.fn()} onSaved={saved} />);
    const article = within(await screen.findByRole('article', { name: '声音记录 r1' }));
    expect(article.getByRole('option', { name: '角色 20' })).toBeTruthy();
    expect((article.getByRole('button', { name: '确认说话人' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(article.getByLabelText('我已试听并核对这段声音的说话人'));
    await user.click(article.getByRole('button', { name: '确认说话人' }));
    await waitFor(() => expect(saved).toHaveBeenCalled());
    const confirmed = fetch.mock.calls.find(([url]) => url.endsWith('/r1/confirm'))!;
    expect(JSON.parse(confirmed[1]!.body as string).character_id).toBe('c20');
    await user.selectOptions(article.getByLabelText('确认声音记录 r1 的角色'), '');
    await user.click(article.getByLabelText('我已试听并核对这段声音的说话人'));
    await user.click(article.getByRole('button', { name: '保留未知 / 撤销关联' }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => url.endsWith('/r1/confirm'))).toHaveLength(2));
    const revoked = fetch.mock.calls.filter(([url]) => url.endsWith('/r1/confirm'))[1];
    expect(JSON.parse(revoked[1]!.body as string).character_id).toBeNull();
  });

  it('requests cooperative cancellation and preserves explicit audio duration budgets', async () => {
    const user = userEvent.setup(); let finish!: (result: unknown) => void;
    const pending = new Promise((resolve) => { finish = resolve; });
    const fetch = mockApi((url) => url.endsWith('/analyze') ? pending : url.endsWith('/cancel') ? { status: 'cancelling', reason: '在当前模型步骤结束后停止' } : undefined);
    render(<AudioIdentity asset={asset} onSeek={vi.fn()} onSaved={vi.fn()} />);
    await screen.findByText('视听身份与证据');
    await user.click(screen.getByLabelText('启用本机音频角色识别'));
    await user.clear(screen.getByLabelText(/^每次音频处理预算/)); await user.type(screen.getByLabelText(/^每次音频处理预算/), '90');
    await user.click(screen.getByRole('button', { name: '保存配置并识别这段音频' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.endsWith('/analyze'))).toBe(true));
    await user.click(screen.getByRole('button', { name: '取消音频识别' }));
    await screen.findByText('在当前模型步骤结束后停止');
    const configuration = fetch.mock.calls.find(([url, options]) => url.endsWith('/audio/settings') && options?.method === 'PUT')!;
    expect(JSON.parse(configuration[1]!.body as string).max_duration_ms).toBe(90000);
    finish({ status: 'cancelled', records: [], timings: {}, degraded: [] });
    await waitFor(() => expect(screen.queryByRole('button', { name: '等待当前模型步骤结束…' })).toBeNull());
    expect(screen.getAllByText('已取消，已完成的缓存保留').length).toBeGreaterThan(0);
  });

  it('keeps multiple visible regions separate and saves only the reviewed local identity', async () => {
    const user = userEvent.setup();
    const regions = [1, 2].map((index) => ({ id: `region${index}`, local_id: `P${index}`, frame_id: 'frame1', at_ms: 1000, box: [0.1, 0.1, 0.2, 0.6], crop_url: `/api/region/${index}.png`, quality: { accepted: true, reasons: [] }, match: { status: 'candidate', candidate_character_id: 'c20', distance: 0.18, margin: 0.01, source: 'ccip' } }));
    const fetch = mockApi((url) => url.endsWith('/identity/reviews') ? { status: 'completed', regions } : undefined);
    render(<AudioIdentity asset={asset} onSeek={vi.fn()} onSaved={vi.fn()} />);
    const first = within(await screen.findByRole('article', { name: '人物区域 P1 region1' }));
    expect(screen.getByRole('article', { name: '人物区域 P2 region2' })).toBeTruthy();
    expect(first.getByText(/CCIP 距离 0.1800 · 候选差距 0.0100/)).toBeTruthy();
    await user.click(first.getByLabelText('已核对当前人物区域'));
    await user.click(first.getByRole('button', { name: '确认人物身份' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.endsWith('/reviews/region1'))).toBe(true));
    const body = JSON.parse(fetch.mock.calls.find(([url]) => url.endsWith('/reviews/region1'))![1]!.body as string);
    expect(body).toMatchObject({ status: 'confirmed', character_id: 'c20' });
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/reviews/region2'))).toBe(false);
  });

  it('creates a named dossier before its first appearance without inventing evidence', async () => {
    const user = userEvent.setup(); const fetch = mockApi(); const saved = vi.fn();
    render(<CreateCharacter assetId={asset.id} onClose={vi.fn()} onSaved={saved} />);
    await user.type(screen.getByLabelText('角色姓名'), '新角色');
    await user.click(screen.getByRole('button', { name: '建立角色档案' }));
    await waitFor(() => expect(saved).toHaveBeenCalled());
    const call = fetch.mock.calls.find(([, options]) => options?.method === 'POST')!;
    expect(JSON.parse(call[1]!.body as string)).toEqual({ name: '新角色', aliases: [], description: '' });
  });

  it('uploads first reference independently of appearances, requiring explicit single-character confirmation', async () => {
    const user = userEvent.setup(); const fetch = mockApi((url, options) => url.endsWith('/references') && !options?.method ? { references: [] } : undefined);
    render(<CharacterReferences assetId={asset.id} profile={profiles[0]} onClose={vi.fn()} onSaved={vi.fn()} />);
    await screen.findByText(/尚无参考图片/);
    expect((screen.getByLabelText('人工确认身份，质量通过后加入可信库') as HTMLInputElement).disabled).toBe(true);
    await user.upload(screen.getByLabelText('选择参考图片'), new File(['image'], 'first.png', { type: 'image/png' }));
    await user.click(screen.getByLabelText('我已核对图片 / 裁剪中只有这个角色'));
    await user.click(screen.getByLabelText('人工确认身份，质量通过后加入可信库'));
    expect((screen.getByLabelText('选择参考图片') as HTMLInputElement).files?.length).toBe(1);
    // jsdom's native file-required check does not see userEvent's FileList.
    fireEvent.submit(screen.getByRole('button', { name: '保存参考图' }).closest('form')!);
    await waitFor(() => expect(fetch.mock.calls.some(([, options]) => options?.method === 'POST')).toBe(true));
    const body = fetch.mock.calls.find(([, options]) => options?.method === 'POST')![1]!.body as FormData;
    expect(body.get('confirmed')).toBe('true'); expect(body.get('single_character')).toBe('true');
    expect(body.get('source_kind')).toBe('upload'); expect(body.get('file')).toBeInstanceOf(File);
  });

  it('retains reference provenance and supports revocation without deleting evidence', async () => {
    const user = userEvent.setup(); const fetch = mockApi();
    render(<CharacterReferences assetId={asset.id} profile={profiles[0]} onClose={vi.fn()} onSaved={vi.fn()} />);
    await screen.findByText('已缓存外部图');
    expect(screen.getByRole('link', { name: '查看来源' }).getAttribute('href')).toBe(reference.source.url);
    expect((screen.getByRole('button', { name: '确认参考图' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: '撤销参考图' }));
    await waitFor(() => expect(fetch.mock.calls.some(([, options]) => options?.method === 'PATCH')).toBe(true));
    const body = JSON.parse(fetch.mock.calls.find(([, options]) => options?.method === 'PATCH')![1]!.body as string);
    expect(body.status).toBe('revoked'); expect(body.reason).toBeTruthy();
    expect(fetch.mock.calls.some(([, options]) => options?.method === 'DELETE')).toBe(false);
  });

  it('requires external voice rights, scopes by language and dub, and leaves unchecked samples proposed', async () => {
    const user = userEvent.setup(); const fetch = mockApi();
    render(<CharacterReferences assetId={asset.id} profile={profiles[0]} onClose={vi.fn()} onSaved={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: '声音样本' }));
    await screen.findByText(/暂无声音样本/);
    await user.selectOptions(screen.getByLabelText('声音样本来源'), 'external_licensed');
    expect((screen.getByLabelText('使用权与来源说明') as HTMLTextAreaElement).required).toBe(true);
    await user.upload(screen.getByLabelText(/^选择声音样本/), new File(['RIFF'], 'voice.wav', { type: 'audio/wav' }));
    await user.type(screen.getByLabelText('使用权与来源说明'), '自有授权样本');
    await user.clear(screen.getByLabelText('样本配音版本')); await user.type(screen.getByLabelText('样本配音版本'), 'tv-original');
    expect((screen.getByLabelText(/^选择声音样本/) as HTMLInputElement).files?.length).toBe(1);
    fireEvent.submit(screen.getByRole('button', { name: '保存声音样本' }).closest('form')!);
    await waitFor(() => expect(fetch.mock.calls.some(([, options]) => options?.method === 'POST')).toBe(true));
    const body = JSON.parse(fetch.mock.calls.find(([, options]) => options?.method === 'POST')![1]!.body as string);
    expect(body).toMatchObject({ source_kind: 'external_licensed', filename: 'voice.wav', language: 'ja', dub_version: 'tv-original', rights_note: '自有授权样本', confirmed: false });
    expect(body.audio_base64).toBeTruthy();
  });
});
