import { useId, useState } from 'react';
import { assetLink, formatTime } from '../api';
import type { RetrievalEvidence as Evidence, RetrievalProgress } from '../types';

const PREVIEW_FRAME_LIMIT = 8;

export function RetrievalFrames({ assetId, record, onSeek }: { assetId: string; record: Evidence; onSeek?: (at: number) => void }) {
  const stripId = useId();
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  if (record.evidence_type !== 'retrieval_window') return null;
  const frames = [...(record.ordered_frames || [])].sort((left, right) => left.at_ms - right.at_ms);
  const evidenceKey = JSON.stringify([assetId, record.hit_frame_id, frames.map((frame) => [frame.id, frame.at_ms])]);
  const expanded = expandedKey === evidenceKey;
  let visibleFrames = frames;
  if (!expanded && frames.length > PREVIEW_FRAME_LIMIT) {
    const selected = Array.from({ length: PREVIEW_FRAME_LIMIT }, (_, index) => Math.round(index * (frames.length - 1) / (PREVIEW_FRAME_LIMIT - 1)));
    const hitIndex = frames.findIndex((frame) => frame.id === record.hit_frame_id);
    if (hitIndex >= 0 && !selected.includes(hitIndex)) {
      // Keep both time boundaries and replace the closest interior representative.
      const replacement = selected.slice(1, -1).reduce((best, index) => Math.abs(index - hitIndex) < Math.abs(best - hitIndex) ? index : best);
      selected[selected.indexOf(replacement)] = hitIndex;
    }
    visibleFrames = selected.sort((left, right) => left - right).map((index) => frames[index]);
  }
  const frameUrl = (id: string) => `/api/assets/${encodeURIComponent(assetId)}/frames/${encodeURIComponent(id)}`;
  return <div className="retrieval-evidence" aria-label="检索画面证据">
    {record.seek_basis === 'local_event' && record.event_evidence && <p className="helper">按局部事件证据定位：{record.event_evidence.action} · {formatTime(record.event_evidence.start_ms)} — {formatTime(record.event_evidence.end_ms)}。请回原片核对。</p>}
    {record.hit_frame_id && <a href={frameUrl(record.hit_frame_id)} target="_blank" rel="noreferrer">查看命中原帧</a>}
    {!!frames.length && <><p className="helper">按原片时间排列的画面证据{frames.length > PREVIEW_FRAME_LIMIT && ` · 显示 ${visibleFrames.length} / ${frames.length} 帧`}</p><div id={stripId} className="evidence-strip">{visibleFrames.map((frame) => <a key={frame.id} href={assetLink(assetId, frame.at_ms)} onClick={onSeek ? (event) => { event.preventDefault(); onSeek(frame.at_ms); } : undefined} aria-label={`跳到证据画面 ${formatTime(frame.at_ms)}`} aria-current={frame.id === record.hit_frame_id ? 'true' : undefined}>
      <img src={frame.thumbnail_url || frameUrl(frame.id)} alt={`${formatTime(frame.at_ms)} 原片画面`} loading="lazy" /><span>{frame.id === record.hit_frame_id ? '命中 · ' : ''}{formatTime(frame.at_ms)}</span>
    </a>)}</div>{frames.length > PREVIEW_FRAME_LIMIT && <button className="text-button retrieval-evidence-toggle" type="button" aria-expanded={expanded} aria-controls={stripId} onClick={() => setExpandedKey(expanded ? null : evidenceKey)}>{expanded ? '收起' : `展开全部 ${frames.length} 帧`}</button>}</>}
    {(record.member_ids?.length || 0) > 1 && <p className="helper">合并了 {record.member_ids!.length} 个相邻检索窗口，保留各窗口的原片证据。</p>}
  </div>;
}

export function RetrievalState({ retrieval }: { retrieval?: RetrievalProgress }) {
  if (!retrieval) return null;
  const labels: Record<string, string> = { sampling: '画面采样', visual_index: '直接视觉索引', subtitle_index: '字幕索引', local_records: '局部事件', identity: '角色身份', final_publish: '索引发布' };
  const states: Record<string, string> = { pending: '待完成', running: '处理中', completed: '已完成', not_requested: '本次未启用', disabled: '已关闭', failed: '失败' };
  const incomplete = ['failed', 'partial', 'paused', 'cancelled'].includes(retrieval.status) ? '检索流程未完成' : '检索流程进行中';
  return <section className="job-execution-snapshot" aria-label="检索流程完成情况">
    <p className="helper">{retrieval.status === 'core_ready' ? '核心检索实验完成 · 默认完整流程未完成' : retrieval.default_completed_at ? '默认完整流程已完成' : `${incomplete} · 默认完整流程未完成`}</p>
    {retrieval.coverage && Number.isFinite(retrieval.coverage.total_units) && <p className="helper">已发布 {retrieval.coverage.published_units} / {retrieval.coverage.total_units} 个检索窗口 · 已处理 {retrieval.coverage.processed_units} 个</p>}
    {retrieval.capabilities && <p className="helper">{Object.entries(retrieval.capabilities).map(([key, state]) => `${labels[key] || key}：${states[state] || state}`).join(' · ')}</p>}
    {retrieval.identity?.reason && <p className="helper">本地人物匹配：{retrieval.identity.reason}</p>}
    {retrieval.first_searchable_at && <p className="helper">已有画面可搜索；完整能力以本任务的完成情况为准。</p>}
  </section>;
}
