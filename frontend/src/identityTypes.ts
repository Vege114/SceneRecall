export type IdentityStatus = 'confirmed' | 'candidate' | 'unknown';
export interface ReferenceQuality {
  width?: number; height?: number; sharpness?: number; single_character?: boolean;
  accepted?: boolean; reasons?: string[];
}
export interface CharacterReference {
  id: string; character_id: string; status: 'candidate' | 'confirmed' | 'revoked';
  source: { kind: 'upload' | 'external_cached' | 'frame_crop'; url?: string; asset_id?: string; frame_id?: string; box?: number[] };
  quality: ReferenceQuality; version: number; image_url: string; created_at?: string; updated_at?: string;
}
export interface IdentityRegion {
  id: string; local_id: string; frame_id: string; at_ms: number; box: number[];
  crop_url: string; quality: ReferenceQuality;
  match: { status: IdentityStatus; character_id?: string | null; candidate_character_id?: string | null;
    distance?: number | null; margin?: number | null; source?: string;
    candidates?: { character_id: string; distance: number; reference_ids: string[] }[] };
}
export interface IdentityReview {
  regions: IdentityRegion[]; status: string; degraded_reason?: string;
  metrics?: Record<string, number>;
}
export interface AudioSettings {
  enabled: boolean; stream_index: number | null; language: string; dub_version: string;
  diarization_model_path: string; embedding_model_path: string; asr_model_path: string;
  asr_enabled: boolean; device: string; sync_enabled: boolean; sync_repository_path: string;
  sync_checkpoint_path: string; sync_animation_verified: boolean; calibration_path: string;
  sync_license_acknowledged?: boolean; max_duration_ms?: number; top_k?: number;
}
export interface AudioAvailability {
  diarization: { available: boolean; reason: string }; embedding: { available: boolean; reason: string };
  asr: { available: boolean; reason: string }; sync: { available: boolean; reason: string }; reasons: string[];
}
export interface AudioConfiguration { settings: AudioSettings; availability: AudioAvailability }
export interface AudioTrack { index: number; language: string; title: string; default: boolean; start_ms: number }
export interface VoiceSample {
  id: string; status: 'confirmed' | 'revoked' | 'proposed'; character_id: string;
  source_kind: 'original_clip' | 'external_licensed'; language: string; dub_version: string;
  version: number; quality?: Record<string, unknown>; provenance?: Record<string, unknown>; created_at?: string;
}
export interface AudioRecord {
  id: string; record_id?: string; asset_id: string; start_ms: number; end_ms: number;
  speaker_id: string; character_id?: string | null; candidate_character_id?: string | null;
  association_status: IdentityStatus; association_source: string;
  candidates: { character_id: string; distance: number; sample_ids: string[] }[];
  subtitle_ids: string[]; frame_ids: string[]; text: string; language: string; dub_version: string;
  audio_stream_index: number; audio_ref: unknown; model_versions: Record<string, string>;
  overlap: boolean; needs_identity_review: boolean;
}
export interface AudioResult {
  status: string; records: AudioRecord[]; timings?: Record<string, number>;
  degraded?: string[]; cache_hits?: number;
}
export const identityStatusLabel = (status?: string) => ({ confirmed: '已确认', candidate: '候选 · 待确认', unknown: '未知', revoked: '已撤销', proposed: '提议 · 未入可信库' }[status || ''] || '未知');
export const identitySourceLabel = (source?: string) => ({ manual: '人工确认', human: '人工确认', ccip: 'CCIP 图像匹配', voice: '声音匹配', voice_bank: '可信声音库匹配', unmatched: '尚未匹配', ecapa: 'ECAPA 声音匹配', upload: '人工上传', external_cached: '已缓存外部图', frame_crop: '原片裁剪', original_clip: '原片声音片段', external_licensed: '有权使用的外部声音' }[source || ''] || source || '未记录来源');
