export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

function errorText(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((item) => typeof item === 'object' && item && 'msg' in item ? String(item.msg) : String(item)).join('；');
  if (detail && typeof detail === 'object') return JSON.stringify(detail);
  return '请求未能完成，请重试。';
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...options,
      credentials: 'same-origin',
      headers: { ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
    });
  } catch {
    throw new ApiError('无法连接本地服务，请确认 SceneRecall 服务正在运行。', 0);
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) window.dispatchEvent(new CustomEvent('session-expired'));
    throw new ApiError(errorText(data.detail || data.message || `服务返回错误 (${response.status})`), response.status);
  }
  return data as T;
}
export function post<T>(path: string, body: unknown = {}): Promise<T> {
  return api<T>(path, { method: 'POST', body: JSON.stringify(body) });
}
export function formatTime(ms: number): string {
  const total = Math.max(0, Math.floor((ms || 0) / 1000));
  return `${Math.floor(total / 3600).toString().padStart(2, '0')}:${Math.floor(total / 60 % 60).toString().padStart(2, '0')}:${(total % 60).toString().padStart(2, '0')}`;
}
export function duration(ms: number): string {
  const secs = Math.round((ms || 0) / 1000);
  return secs >= 3600 ? `${Math.floor(secs / 3600)} 小时 ${Math.floor(secs % 3600 / 60)} 分钟` : secs >= 60 ? `${Math.floor(secs / 60)} 分 ${secs % 60} 秒` : `${secs} 秒`;
}
export function friendlyDate(value?: string): string {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}
export function message(error: unknown): string { return error instanceof Error ? error.message : '操作失败，请重试。'; }
export function assetLink(id: string, atMs?: number): string { return `#/asset/${encodeURIComponent(id)}${atMs !== undefined ? `?t=${atMs}` : ''}`; }
export function subtitleLink(assetId: string, cueId: string, atMs = 0): string { return `${assetLink(assetId, atMs)}&cue=${encodeURIComponent(cueId)}`; }
export function sceneLink(assetId: string, chunkId: string, atMs = 0): string { return `${assetLink(assetId, atMs)}&chunk=${encodeURIComponent(chunkId)}`; }
