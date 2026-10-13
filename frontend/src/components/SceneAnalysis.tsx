import { useState, type FormEvent } from 'react';
import { ArrowRight, Calculator, Maximize2, ScanLine } from 'lucide-react';
import { formatTime, message, post, sceneLink, subtitleLink } from '../api';
import { useApp } from '../context';
import { DEFAULTS, estimatedCost, estimatedRequests, taskModelLabel, stageLabel, type Estimate, type Job, type SceneChunk, type SceneEvidence, type SceneFact, type Subtitle } from '../types';
import { Alert, Field, Modal, Spinner } from './ui';
import TaskModels, { useTaskModels } from './TaskModels';

export function SceneBadges({ scene }: { scene: SceneEvidence }) {
  return <><span className={`tag scene-level ${scene.understanding_level === 'deep' ? 'deep' : ''}`}>{scene.understanding_level === 'deep' ? '深入理解' : '初步理解'}</span>{scene.context_stale && <span className="tag stale-tag" title={scene.stale_reasons?.join('；')}>上下文待更新</span>}</>;
}

export function RelatedScenes({ scene, scenes, assetId }: { scene: SceneChunk; scenes: SceneChunk[]; assetId: string }) {
  const [limit, setLimit] = useState(6);
  const related = [...new Set(scene.related_chunk_ids || [])].flatMap((id) => {
    const target = scenes.find((candidate) => candidate.chunk_id === id && candidate.id !== scene.id);
    return target ? [target] : [];
  }).sort((a, b) => a.start_ms - b.start_ms);
  if (!related.length) return null;
  return <nav className="scene-related" aria-label="可能相似的场景"><strong>可能相似的场景</strong>{scene.scene_key && <span className="helper">{scene.scene_key}</span>}<div>{related.slice(0, limit).map((target) => <a key={target.id} href={sceneLink(assetId, target.chunk_id, target.start_ms)} title={target.summary}>{formatTime(target.start_ms)} — {formatTime(target.end_ms)} <ArrowRight size={12} /></a>)}</div>{related.length > limit && <button type="button" className="text-button" onClick={() => setLimit((count) => count + 24)}>更多相似场景（剩余 {related.length - limit} 个）</button>}<p className="helper">这些链接用于画面对照，相似性尚未确认为同一地点、人物或事件。</p></nav>;
}

export function SceneFrames({ scene, assetId, onSelect }: { scene: SceneChunk; assetId: string; onSelect: (url: string) => void }) {
  const [extraCount, setExtraCount] = useState(0);
  const retained = [...new Set([...(scene.frame_ids || []), ...(scene.evidence_frame_ids || [])])];
  const representatives = [...new Set(scene.representative_frame_ids || [])].filter((id) => retained.includes(id)).slice(0, 8);
  const first = representatives.length ? representatives : retained.slice(0, 8);
  const remaining = retained.filter((id) => !first.includes(id));
  const shown = [...first, ...remaining.slice(0, extraCount)];
  if (!shown.length) return null;
  return <div className="scene-frame-gallery" aria-label="场景原始关键帧"><div className="evidence-strip">{shown.map((frameId) => {
    const url = `/api/assets/${encodeURIComponent(assetId)}/frames/${encodeURIComponent(frameId)}`;
    return <button key={frameId} onClick={() => onSelect(url)} aria-label="放大证据画面"><img src={url} alt={first.includes(frameId) ? '场景代表原帧' : '场景保留原帧'} loading="lazy" /><Maximize2 size={13} /></button>;
  })}</div><p className="helper">已显示 {shown.length} / {retained.length} 张原帧 · 代表帧优先</p>{extraCount < remaining.length && <button type="button" className="text-button" onClick={() => setExtraCount((count) => count + 24)}>显示更多原帧（剩余 {remaining.length - extraCount} 张）</button>}</div>;
}

export function SceneFacts({ scene, assetId, subtitles = [], startMs = 0 }: { scene: SceneEvidence; assetId: string; subtitles?: Subtitle[]; startMs?: number }) {
  const groups: [string, SceneFact[] | undefined][] = [['画面事实', scene.visual_facts], ['台词事实', scene.dialogue_facts], ['综合推断', scene.inferences]];
  if (!groups.some(([, facts]) => facts?.length)) return null;
  return <details className="scene-facts"><summary>查看事实与来源</summary>{groups.filter(([, facts]) => facts?.length).map(([label, facts]) => <section key={label}><strong>{label}</strong>{facts?.map((fact, index) => <div className="scene-fact" key={index}><p>{fact.text}</p>{fact.frame_ids?.length ? <div className="scene-fact-sources">画面来源：{fact.frame_ids.map((id, frameIndex) => <a href={`/api/assets/${encodeURIComponent(assetId)}/frames/${encodeURIComponent(id)}`} key={id} target="_blank" rel="noreferrer">原帧 {frameIndex + 1}</a>)}</div> : null}{fact.cue_ids?.length ? <div className="scene-fact-sources">台词来源：{fact.cue_ids.map((id, cueIndex) => { const cue = subtitles.find((entry) => entry.id === id); return cue ? <span key={id}>{formatTime(cue.start_ms)}「{cue.text}」</span> : <a key={id} href={subtitleLink(assetId, id, startMs)}>查看台词 {cueIndex + 1}</a>; })}</div> : null}</div>)}</section>)}<p className="helper">综合推断需结合原片核对；台词提及和此前记忆不代表当前画面已发生。</p></details>;
}

export function StageEstimates({ estimate }: { estimate: Estimate }) {
  if (!estimate.stage_estimates) return null;
  return <div className="stage-estimates" aria-label="分阶段请求估算">{Object.entries(estimate.stage_estimates).map(([stage, counts]) => <div key={stage}><span>{stageLabel(stage)}</span><strong>{counts.requests_min === counts.requests_max ? counts.requests_min : `${counts.requests_min}–${counts.requests_max}`} 次</strong></div>)}{(estimate.estimated_requests_max ?? 0) > (estimate.estimated_requests_min ?? estimate.estimated_requests) && <p className="helper">深入理解的数量在分块后确定，现阶段显示范围。</p>}</div>;
}

export function DeepenSceneButton({ assetId, chunkId, title, contextStale = false, onStarted }: { assetId: string; chunkId: string; title: string; contextStale?: boolean; onStarted?: () => void }) {
  const { jobs = [] } = useApp();
  const [open, setOpen] = useState(false);
  const active = jobs.some((job) => job.asset_id === assetId && job.chunk_id === chunkId && ['queued', 'pending', 'running', 'pausing'].includes(job.status));
  return <><button type="button" className="text-button scene-deepen" disabled={active} onClick={() => setOpen(true)}><ScanLine size={14} />{active ? '片段分析中' : '深入分析此片段'}</button>{open && <DeepenSceneModal assetId={assetId} chunkId={chunkId} title={title} contextStale={contextStale} onClose={() => setOpen(false)} onStarted={onStarted} />}</>;
}

function DeepenSceneModal({ assetId, chunkId, title, contextStale, onClose, onStarted }: { assetId: string; chunkId: string; title: string; contextStale: boolean; onClose: () => void; onStarted?: () => void }) {
  const { bootstrap, refresh, notify } = useApp();
  const defaults = { ...DEFAULTS, ...bootstrap.settings.defaults };
  const profile = bootstrap.provider_profiles.find((entry) => entry.id === bootstrap.settings.bindings.vision);
  const usesCodex = profile?.provider_type === 'codex_cli';
  const taskModels = useTaskModels(['vision']);
  const [requestLimit, setRequestLimit] = useState(defaults.max_requests);
  const [costLimit, setCostLimit] = useState(defaults.max_cost?.toString() || '');
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [estimateKey, setEstimateKey] = useState('');
  const [busy, setBusy] = useState<'estimate' | 'start' | null>(null);
  const [error, setError] = useState('');
  const budget = { ...(taskModels.overrides ? { model_overrides: taskModels.overrides } : {}), max_requests: requestLimit, max_cost: usesCodex || !costLimit ? null : Number(costLimit) };
  const key = JSON.stringify([budget, taskModels.sourceKey]);
  const endpoint = `/assets/${encodeURIComponent(assetId)}/chunks/${encodeURIComponent(chunkId)}/deepen`;
  const estimateValid = estimate && estimateKey === key;
  const estimateJob = async (event: FormEvent) => {
    event.preventDefault(); setBusy('estimate'); setError('');
    try { const result = await post<Estimate>(`${endpoint}/estimate`, budget); setEstimate(result); setEstimateKey(key); }
    catch (error) { setError(message(error)); } finally { setBusy(null); }
  };
  const start = async () => {
    if (!estimateValid || !profile || busy) return;
    setBusy('start'); setError('');
    try { await post<Job>(endpoint, budget); await refresh(); notify('片段深入分析任务已加入队列，收藏与引用会保留'); onStarted?.(); onClose(); }
    catch (error) { setError(message(error)); } finally { setBusy(null); }
  };
  return <Modal title="深入分析此片段" subtitle={title} onClose={onClose}><form onSubmit={estimateJob}><div className="modal-body"><p>使用保留的原始关键帧、对应台词与此前记忆升级当前片段。模型：{taskModelLabel(profile, taskModels.overrides?.vision)}。</p><TaskModels roles={['vision']} value={taskModels.overrides} onChange={taskModels.change} disabled={!!busy} />{contextStale && <Alert>此前内容已修订。更新这个片段后，受影响的后续内容会标为待更新。</Alert>}{!profile && <Alert>请在<a href="#/settings" onClick={onClose}>模型与设置</a>中分配画面理解模型。</Alert>}<Field label="最多模型请求数" hint="达到上限后保留已完成结果，可在任务队列提高预算继续"><input type="number" min="1" max="100000" required value={requestLimit} onChange={(event) => setRequestLimit(Number(event.target.value))} disabled={!!busy} /></Field><Field label="费用上限（USD，可选）" hint={usesCodex ? 'Codex CLI 使用请求数控制用量，美元费用未知' : '留空不设金额上限；金额控制需要配置模型单价'}><input type="number" min="0.01" step="0.01" value={usesCodex ? '' : costLimit} disabled={usesCodex || !!busy} onChange={(event) => setCostLimit(event.target.value)} /></Field>{estimateValid && <><div className="estimate"><div><span>预计请求</span><strong>{estimatedRequests(estimate)} <small>次</small></strong></div><div><span>估算费用</span><strong>{estimatedCost(estimate)}</strong></div>{estimate.warnings?.map((warning, index) => <p className="warning-text" key={index}>{warning}</p>)}</div><StageEstimates estimate={estimate} /></>}{error && <Alert tone="error">{error}</Alert>}</div><footer className="modal-footer"><button type="submit" className="button secondary" disabled={!!busy || !profile}>{busy === 'estimate' ? <Spinner label="正在估算" /> : <><Calculator size={15} />{estimateValid ? '重新估算' : '估算用量'}</>}</button><button type="button" className="button primary" onClick={() => void start()} disabled={!!busy || !estimateValid || !profile}>{busy === 'start' ? <Spinner label="正在提交" /> : <>开始深入分析 <ArrowRight size={15} /></>}</button></footer></form></Modal>;
}
