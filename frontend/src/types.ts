export type Capability = 'vision' | 'subtitle' | 'embedding' | 'query' | 'decision' | 'answer';
export type AssetKind = 'movie' | 'animation' | 'series';
export type SubtitleMode = 'auto' | 'container' | 'external' | 'embedded';
export interface SubtitleTrack {
  index: number;
  codec: string;
  language: string;
  title: string;
  default: boolean;
  forced: boolean;
  supported: boolean;
  reason?: string;
}
export interface SubtitleTracks {
  tracks: SubtitleTrack[];
  recommended_stream_index: number | null;
  duration_ms: number;
}
export const SUBTITLE_SOURCE_LABELS = { container: '容器字幕轨', external: '外挂字幕', embedded: '画面烧录字幕 · OCR' };
export type ProviderType = 'openai_compatible' | 'codex_cli';
export interface Profile {
  id: string;
  name: string;
  provider_type?: ProviderType;
  base_url: string;
  model: string;
  capabilities: Capability[];
  secret_mode: 'session' | 'keyring' | 'env' | 'none';
  env_var?: string | null;
  timeout_s?: number;
  max_concurrency?: number;
  input_price_per_million?: number | null;
  output_price_per_million?: number | null;
  has_key?: boolean;
}
export interface CodexStatus {
  installed: boolean;
  authenticated: boolean;
  compatible?: boolean;
  auth_method: string | null;
  message: string;
  version?: string | null;
}
export function profileModelLabel(profile?: Profile): string {
  return profile?.model || (profile?.provider_type === 'codex_cli' ? 'Codex 默认模型' : '尚未分配模型');
}
export interface Defaults {
  window_ms: number;
  frames_per_window: number;
  subtitle_fps: number;
  subtitle_crop: number[];
  max_requests: number;
  max_cost: number | null;
}
export interface Settings {
  bindings: Partial<Record<Capability, string | null>>;
  defaults: Defaults;
}
export interface Bootstrap {
  library_path: string;
  settings: Settings;
  provider_profiles: Profile[];
  stats: Record<string, number>;
  tools: { ffmpeg: boolean; ffprobe: boolean };
  version: string;
}
export interface Asset {
  id: string;
  work_id: string;
  title: string;
  kind: AssetKind;
  series?: string;
  season?: number | null;
  episode?: number | null;
  version?: string;
  duration_ms: number;
  width: number;
  height: number;
  subtitle_mode: Exclude<SubtitleMode, 'auto'>;
  subtitle_import_mode?: SubtitleMode;
  subtitle_stream_index?: number | null;
  subtitle_track?: SubtitleTrack | null;
  subtitle_offset_ms?: number;
  subtitle_fallback_reason?: string;
  source_available: boolean;
  source_changed: boolean;
  created_at: string;
  status: string;
  thumbnail_url?: string;
}
export interface Entity {
  id?: string;
  entity_id?: string;
  local_id?: string;
  type?: string;
  name?: string;
  description?: string;
  appearance?: string;
  position?: string;
  character_id?: string;
  character_name?: string;
}
export interface CharacterAppearance {
  asset_id: string;
  asset_title: string;
  record_id: string;
  start_ms: number;
  end_ms: number;
  entity_id: string;
  frame_ids: string[];
}
export interface CharacterProfile {
  id: string;
  name: string;
  aliases: string[];
  description: string;
  notes: string;
  appearance?: string[];
  user_edited?: boolean;
  updated_at?: string;
  representative?: { asset_id: string; frame_id: string; box?: number[] | null } | null;
  appearances: CharacterAppearance[];
  appearance_count: number;
}
export interface CharacterLibrary {
  scope_id: string;
  revision: number;
  profiles: CharacterProfile[];
  assets: Asset[];
}
export interface Observation {
  id: string;
  window_id: string;
  shot_id: string;
  start_ms: number;
  end_ms: number;
  summary: string;
  entities: Entity[];
  events: { action: string; actor_ids?: string[]; target_ids?: string[]; start_ms?: number; end_ms?: number }[];
  spatial_observations?: { subject_id: string; relation: string; object_id?: string | null }[];
  uncertainties: string[];
  evidence_frame_ids: string[];
  thumbnail_url?: string;
  run_id: string;
}
export interface Subtitle {
  id: string;
  start_ms: number;
  end_ms: number;
  text: string;
  language: string;
  source: string;
  review_status: string;
}
export interface Annotation {
  record_id: string;
  summary?: string;
  note?: string;
  character_name?: string;
  entity_id?: string;
  aliases?: string[];
  favorite?: boolean;
  needs_reassociation?: boolean;
}
export interface Run {
  id: string;
  stage?: string;
  status?: string;
  created_at?: string;
  model?: string;
  profile_id?: string;
  [key: string]: unknown;
}
export interface AssetDetail {
  asset: Asset;
  observations: Observation[];
  subtitles: Subtitle[];
  annotations: Annotation[];
  runs: Run[];
}
export interface Job {
  id: string;
  type: string;
  status: string;
  progress: number;
  completed: number;
  total: number;
  asset_id?: string;
  stage?: string;
  message?: string;
  error?: string;
  usage?: Record<string, number>;
  request_count?: number;
  cost?: number | null;
  budget?: { max_requests?: number; max_cost?: number | null };
  uncertain_request_count?: number;
  cost_incomplete?: boolean;
  created_at: string;
}
export interface JobPayload extends Defaults {
  asset_id: string;
  stages: ('vision' | 'subtitle')[];
  start_ms: number;
  end_ms: number | null;
  force: boolean;
}
export interface Estimate {
  duration_ms: number;
  estimated_frames: number;
  estimated_requests: number;
  cost_estimate: number | null;
  warnings: string[];
}
export interface SearchResult {
  id: string;
  asset_id: string;
  work_id: string;
  title: string;
  series?: string;
  season?: number | null;
  episode?: number | null;
  version?: string;
  kind: 'visual' | 'subtitle';
  start_ms: number;
  end_ms: number;
  text: string;
  summary?: string;
  thumbnail_url?: string;
  evidence_frame_ids?: string[];
  record_id: string;
  source: string;
  review_status: string;
  score?: number;
  match_type?: 'full' | 'partial';
  match_reason?: string;
  subtitle_text?: string;
  media_url?: string;
  note?: string;
  favorite?: boolean;
  is_historical?: boolean;
}
export interface SearchResponse {
  results: SearchResult[];
  degraded: string[];
  explanation?: string;
}
export const CAPABILITIES: { key: Capability; name: string; description: string; data: string }[] = [
  { key: 'vision', name: '画面理解', description: '识别人物、动作、物体与空间关系', data: '发送带时间戳的画面图片' },
  { key: 'subtitle', name: '画面字幕 OCR', description: '转录烧录在画面中的文字；容器文本字幕无需模型', data: '发送裁剪后的字幕图片' },
  { key: 'embedding', name: '语义向量', description: '让相似表达也能找到同一片段', data: '发送文本记录与搜索词' },
  { key: 'query', name: '查询理解', description: '理解自然语言中的检索条件', data: '发送搜索词' },
  { key: 'decision', name: '匹配决策', description: '评估候选与问题的匹配程度，预留 Jev 接口', data: '发送搜索词与候选记录' },
  { key: 'answer', name: '结果解释', description: '基于已找到的证据组织回答（可选）', data: '发送搜索词与候选记录' },
];
export const DEFAULTS: Defaults = { window_ms: 6000, frames_per_window: 4, subtitle_fps: 2, subtitle_crop: [0, 0.65, 1, 0.35], max_requests: 1000, max_cost: null };
export const KIND_LABELS: Record<AssetKind, string> = { movie: '电影', animation: '动画', series: '剧集' };
