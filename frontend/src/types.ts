export type Capability = 'preparation' | 'vision' | 'subtitle' | 'embedding' | 'query' | 'decision' | 'answer';
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
export type TaskModelRole = 'preparation' | 'vision' | 'subtitle';
export interface TaskModelOverride {
  provider_type: 'codex_cli';
  profile_id: string;
  model?: string;
  reasoning_effort?: string | null;
}
export type TaskModelOverrides = Partial<Record<TaskModelRole, TaskModelOverride>>;
export interface CodexModelCatalog {
  models: { id: string; name: string; reasoning_efforts: string[]; default_reasoning_effort: string | null }[];
  default_model?: string | null;
  message?: string;
  source?: 'local_cache' | 'unavailable';
  fetched_at?: string | null;
}
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
export function taskModelLabel(profile?: Profile, selection?: TaskModelOverride): string {
  if (profile?.provider_type === 'codex_cli' && selection?.profile_id === profile.id && selection.model !== undefined) return selection.model || 'Codex 内置默认';
  return profileModelLabel(profile);
}
export interface Defaults {
  analysis_strategy?: 'scene-chunks-v1' | 'content-keyframes-v1';
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
export interface Folder { id: string; name: string; parent_id: string | null }
export interface Project { id: string; name: string; kind: AssetKind; folder_id: string | null; tags: string[]; asset_count: number }
export type DeletionTarget = { kind: 'assets'; asset_ids: string[] } | { kind: 'project' | 'folder'; id: string; name: string };
export interface DeletionScope {
  title: string; asset_count: number; project_count: number; folder_count: number;
  assets: { id: string; title: string }[]; projects: { id: string; name: string }[]; folders: { id: string; name: string }[];
}
export interface DeletionPreview extends DeletionScope {
  token: string; blocked_jobs: { id: string; status: string; message: string }[]; source_files_preserved: boolean;
  delete_source: boolean; source_file_count: number;
  source_files: { asset_ids: string[]; path: string; status: 'ready' | 'missing' | 'changed' | 'shared' | 'unsafe'; message: string }[];
}
export interface SourceDeletionResult { asset_ids: string[]; status: 'pending' | 'deleted' | 'error'; message: string }
export interface TrashEntry extends DeletionScope {
  id: string; kind: 'assets' | 'project' | 'folder'; deleted_at: string; delete_contents: boolean;
  delete_source: boolean; source_results: SourceDeletionResult[];
}
export interface ImportInput {
  item_id: string; video_path: string; title: string; kind: AssetKind; series: string;
  season: number | null; episode: number | null; version: string;
  subtitle_mode: SubtitleMode; subtitle_path: string | null; subtitle_stream_index?: number | null; subtitle_offset_ms: number;
}
export interface ImportItem { item_id: string; name: string; status: string; asset_id?: string; error?: string; input?: ImportInput }
export interface KeyframePlan {
  id: string; plan_id?: string; job_id: string; status: string; error?: string; progress?: number; message?: string;
  frames?: { id: string; at_ms: number; thumbnail_url: string; reason?: string }[];
  frame_count?: number; batch_count?: number; shots?: unknown[]; batches?: unknown[];
}
export interface Asset {
  project_id?: string | null;
  project_name?: string;
  folder_id?: string | null;
  tags?: string[];
  membership_revision?: number;
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
export interface SceneFact {
  text: string;
  frame_ids?: string[];
  cue_ids?: string[];
}
export interface SceneEvidence {
  chunk_id?: string;
  scene_key?: string;
  related_chunk_ids?: string[];
  understanding_level?: 'initial' | 'deep';
  context_stale?: boolean;
  stale_reasons?: string[];
  needs_deep?: boolean;
  visual_facts?: SceneFact[];
  dialogue_facts?: SceneFact[];
  inferences?: SceneFact[];
}
export interface SceneChunk extends Omit<Observation, 'window_id' | 'shot_id' | 'entities' | 'events' | 'uncertainties' | 'run_id'>, SceneEvidence {
  chunk_id: string;
  kind?: 'scene';
  shot_ids?: string[];
  frame_ids?: string[];
  entities?: Entity[];
  events?: Observation['events'];
  uncertainties?: string[];
  evidence_cue_ids?: string[];
  representative_frame_ids?: string[];
  version?: number;
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
  scenes?: SceneChunk[];
  subtitles: Subtitle[];
  annotations: Annotation[];
  runs: Run[];
}
export interface Job {
  task_models?: Partial<Record<TaskModelRole, { provider_type: ProviderType; profile_id: string; model: string; reasoning_effort?: string | null }>>;
  batch_id?: string;
  sampling_strategy?: string;
  analysis_strategy?: string;
  chunk_id?: string;
  items?: ImportItem[];
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
  stage_metrics?: Record<string, { request_count?: number; elapsed_ms?: number; cache_hits?: number; usage?: Record<string, number>; completed?: number; total?: number }>;
  stage_progress?: Record<string, { completed: number; total?: number | null; status: 'pending' | 'running' | 'completed' }>;
  created_at: string;
}
export interface JobPayload extends Omit<Defaults, 'window_ms' | 'frames_per_window'> {
  model_overrides?: TaskModelOverrides;
  keyframe_plan_id?: string;
  sampling_strategy?: string;
  analysis_strategy?: 'scene-chunks-v1';
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
  estimated_requests_min?: number;
  estimated_requests_max?: number;
  stage_estimates?: Record<string, { requests_min: number; requests_max: number }>;
  cost_estimate: number | null;
  cost_estimate_min?: number | null;
  cost_estimate_max?: number | null;
  warnings: string[];
}
export interface SearchResult extends SceneEvidence {
  id: string;
  asset_id: string;
  work_id: string;
  title: string;
  series?: string;
  season?: number | null;
  episode?: number | null;
  version?: string;
  kind: 'visual' | 'subtitle' | 'scene';
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
  { key: 'preparation', name: '分块与简要索引', description: '先整理台词，再结合全部关键帧建立场景与剧情索引', data: '发送台词与编号预览拼图；可与画面理解共用连接' },
  { key: 'vision', name: '画面理解', description: '识别人物、动作、物体与空间关系', data: '发送带时间戳的画面图片' },
  { key: 'subtitle', name: '画面字幕 OCR', description: '转录烧录在画面中的文字；容器文本字幕无需模型', data: '发送裁剪后的字幕图片' },
  { key: 'embedding', name: '语义向量', description: '让相似表达也能找到同一片段', data: '发送文本记录与搜索词' },
  { key: 'query', name: '查询理解', description: '理解自然语言中的检索条件', data: '发送搜索词' },
  { key: 'decision', name: '匹配决策', description: '评估候选与问题的匹配程度，预留 Jev 接口', data: '发送搜索词与候选记录' },
  { key: 'answer', name: '结果解释', description: '基于已找到的证据组织回答（可选）', data: '发送搜索词与候选记录' },
];
export const DEFAULTS: Defaults = { window_ms: 6000, frames_per_window: 4, subtitle_fps: 2, subtitle_crop: [0, 0.65, 1, 0.35], max_requests: 1000, max_cost: null };
export const KIND_LABELS: Record<AssetKind, string> = { movie: '电影', animation: '动画', series: '剧集' };

export function keyframeReason(reason?: string): string { return ({ stable_pose: '稳定姿态', representative_pose: '代表姿态', content_change: '画面内容变化', shot_representative: '镜头代表帧' } as Record<string, string>)[reason || ''] || '内容代表帧'; }

export function stageLabel(stage?: string): string {
  return ({ transcript: '台词整理', preparation: '视觉分块', context_refresh: '更新连续理解', deep_analysis: '深入理解', keyframes: '关键帧扫描', keyframe_scan: '关键帧扫描', vision: '深入理解', subtitle: '字幕识别', embedding: '语义向量', probe: '媒体探测', shots: '镜头检测', segmentation: '镜头检测', index: '更新索引', queued: '等待处理' } as Record<string, string>)[stage || ''] || stage || '准备中';
}

export function estimatedRequests(estimate: Estimate): string {
  const min = estimate.estimated_requests_min ?? estimate.estimated_requests;
  const max = estimate.estimated_requests_max ?? estimate.estimated_requests;
  return min === max ? min.toLocaleString() : `${min.toLocaleString()}–${max.toLocaleString()}`;
}

export function estimatedCost(estimate: Estimate): string {
  if (estimate.cost_estimate_min != null && estimate.cost_estimate_max != null && estimate.cost_estimate_min !== estimate.cost_estimate_max) return `$${estimate.cost_estimate_min.toFixed(3)}–$${estimate.cost_estimate_max.toFixed(3)}`;
  return estimate.cost_estimate == null ? '未知' : `$${estimate.cost_estimate.toFixed(3)}`;
}
