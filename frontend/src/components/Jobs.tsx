import { useState, type FormEvent } from 'react';
import { ArrowUpRight, CheckCircle2, Clock3, Pause, Play, RefreshCw, RotateCcw, Workflow, X } from 'lucide-react';
import { assetLink, friendlyDate, message, post } from '../api';
import { useApp } from '../context';
import { SERIAL_EXECUTION, stageLabel, type ExecutionStage, type Job } from '../types';
import { Alert, Empty, Field, Modal, PageHeader, Spinner, Status } from './ui';
import ImportModal from './ImportModal';
import { assetTaskLabel } from '../taskHelpers';
import { importItemProgress, importProgressMessage } from '../importProgress';
import { RetrievalState } from './RetrievalEvidence';

function StageMetrics({ job }: { job: Job }) {
  const stages = [...new Set([...Object.keys(job.stage_progress || {}), ...Object.keys(job.stage_metrics || {})])];
  if (!stages.length) return null;
  const statusLabels = { pending: '待开始', running: '进行中', completed: '已完成' };
  return <div className="job-stage-metrics" aria-label="分阶段用量">{stages.map((stage) => {
    const progress = job.stage_progress?.[stage];
    const metric = job.stage_metrics?.[stage];
    return <div key={stage} aria-label={`${stageLabel(stage)}阶段`}><strong>{stageLabel(stage)}</strong>{progress && <><span>{progress.total == null ? `已完成 ${progress.completed} · 待分块确定` : `已完成 ${progress.completed} / ${progress.total}`}</span><span>{statusLabels[progress.status]}</span></>}{metric && <><span>{!progress && metric.completed != null && metric.total != null ? `${metric.completed} / ${metric.total} · ` : ''}请求 {metric.request_count || 0} 次</span><span>{metric.wall_elapsed_ms != null ? '阶段耗时' : '耗时'} {((metric.wall_elapsed_ms ?? metric.elapsed_ms ?? 0) / 1000).toFixed(1)} 秒</span>{metric.wall_elapsed_ms != null && <span>请求累计 {((metric.elapsed_ms || 0) / 1000).toFixed(1)} 秒</span>}<span>复用 {metric.cache_hits || 0} 次</span>{metric.fallback_count != null && <span>回退 {metric.fallback_count} 次</span>}{metric.timeout_count != null && <span>超时 {metric.timeout_count} 次</span>}{metric.invalid_output_count != null && <span>无效输出 {metric.invalid_output_count} 次</span>}{metric.degraded_reason && <span className="warning-text">降级：{metric.degraded_reason}</span>}{metric.usage?.total_tokens != null && <span>{metric.usage.total_tokens.toLocaleString()} Token</span>}</>}</div>;
  })}</div>;
}

function JobExecution({ job }: { job: Job }) {
  if (!['analysis', 'analyze', 'scene_deepen', 'deepen'].includes(job.type)) return null;
  if (job.analysis_strategy === 'retrieval-first-v1') return <p className="helper job-execution-snapshot">画面检索{job.retrieval_mode === 'core' || job.retrieval?.status === 'core_ready' ? '核心实验' : ''} · 本机编码与增量发布</p>;
  if (job.chunk_id || ['scene_deepen', 'deepen'].includes(job.type)) return <p className="helper job-execution-snapshot">深入理解与连续理解更新：串行 · 依赖前序记忆</p>;
  const execution = job.execution || SERIAL_EXECUTION;
  const recordedStages = Object.keys(job.stage_progress || {});
  const hasSceneStages = job.analysis_strategy === 'scene-chunks-v1' || recordedStages.some((stage) => ['transcript', 'preparation', 'deep_analysis'].includes(stage));
  const stages: ExecutionStage[] = [...(hasSceneStages ? ['transcript' as const, 'preparation' as const] : []), ...(job.stages?.includes('subtitle') || recordedStages.includes('subtitle') || job.stage === 'subtitle' ? ['subtitle' as const] : [])];
  const parallel = stages.some((stage) => execution[stage] === 'parallel');
  return <div className="job-execution-snapshot" aria-label="任务创建时的执行方式"><p className="helper">{stages.length ? stages.map((stage) => `${stageLabel(stage)}：${execution[stage] === 'parallel' ? '并行' : '串行'}`).join(' · ') : '逐镜头画面分析：串行'}{parallel ? ` · 并发请求上限 ${execution.max_concurrency}` : ''}</p>{hasSceneStages && <p className="helper">深入理解与连续理解更新：串行</p>}{!job.execution && <p className="helper">历史任务未设置执行方式，沿用串行。</p>}</div>;
}

export default function Jobs() {
  const { jobs, assets, refreshJobs, notify, openJobs } = useApp();
  const [filter, setFilter] = useState('all');
  const [retryImport, setRetryImport] = useState<string | null>(null);
  const [busy, setBusy] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [resuming, setResuming] = useState<{ job: Job; action: 'resume' | 'retry' } | null>(null);
  const active = ['running', 'queued', 'pending', 'pausing'];
  const change = async (job: Job, action: string, budget: Record<string, number | null> = {}) => {
    setBusy(job.id);
    try { await post(`/jobs/${encodeURIComponent(job.id)}/${action}`, budget); await refreshJobs(true); notify(({ pause: '已请求暂停，当前请求完成后生效', resume: '任务已恢复', cancel: '已请求取消，已完成的资料会保留', retry: '失败任务已重新排队' })[action as 'pause']); return true; }
    catch (error) { notify(message(error), 'error'); return false; } finally { setBusy(''); }
  };
  const filtered = jobs.filter((job) => filter === 'all' || (filter === 'active' ? active.includes(job.status) || job.status === 'paused' : filter === 'failed' ? ['failed', 'partial'].includes(job.status) : ['completed', 'complete'].includes(job.status)));
  const typeLabel = (type: string) => ({ import: '批量导入', keyframe_scan: '本机关键帧扫描', scene_deepen: '片段深入分析', deepen: '片段深入分析', analysis: '影视分析', analyze: '影视分析', proxy: '播放代理', reindex: '索引重建', index: '索引重建' })[type] || type;
  return <><PageHeader eyebrow="WORK IN PROGRESS" title="任务队列" action={<button className="button secondary" disabled={refreshing} onClick={async () => { setRefreshing(true); try { await refreshJobs(true); } catch (error) { notify(message(error), 'error'); } finally { setRefreshing(false); } }}><RefreshCw size={16} className={refreshing ? 'spin' : ''} />刷新</button>}>分析在本机后台运行。暂停、恢复与重试，都保留已完成的资料。</PageHeader><div className="queue-stats"><div><Workflow size={21} /><strong>{jobs.filter((job) => active.includes(job.status)).length}</strong><span>进行中</span></div><div><CheckCircle2 size={21} /><strong>{jobs.filter((job) => ['completed', 'complete'].includes(job.status)).length}</strong><span>已完成</span></div><div><Clock3 size={21} /><strong>{jobs.filter((job) => job.status === 'paused').length}</strong><span>已暂停</span></div></div><div className="section-row"><div className="tabs">{[['all', '全部任务'], ['active', '进行中'], ['completed', '已完成'], ['failed', '失败']].map(([value, label]) => <button key={value} className={value === filter ? 'active' : ''} onClick={() => setFilter(value)}>{label}</button>)}</div><span className="helper">每 4 秒自动更新</span></div>{filtered.length === 0 ? <Empty icon={<Workflow size={29} />} title={jobs.length ? '这里暂时没有任务' : '还没有分析任务'} action={<a className="button secondary" href="#/library">返回资料库 <ArrowUpRight size={16} /></a>}>导入作品后，从作品详情页启动画面或字幕识别。</Empty> : <><div className="batch-summaries">{[...new Set(filtered.map((job) => job.batch_id).filter((id): id is string => !!id))].map((id) => <BatchSummary key={id} id={id} jobs={jobs.filter((job) => job.batch_id === id)} />)}</div><div className="job-list">{filtered.map((job) => {
    const asset = assets.find((asset) => asset.id === job.asset_id);
    const importedAssets = assets.filter((entry) => job.items?.some((item) => item.asset_id === entry.id));
    const progress = job.total > 0 ? Math.min(100, Math.round(job.completed / job.total * 100)) : job.progress > 1 ? Math.min(100, job.progress) : Math.round((job.progress || 0) * 100);
    return <article className="job-card" key={job.id}><div className="job-header"><div className="job-icon"><Workflow size={21} /></div><div><h3>{asset ? <a href={assetLink(asset.id)}>{assetTaskLabel(asset)}</a> : typeLabel(job.type)}</h3><p>{typeLabel(job.type)} · {stageLabel(job.stage)} · {friendlyDate(job.created_at)}</p></div><Status status={job.retrieval?.status === 'core_ready' ? 'core_ready' : job.status} /></div><RetrievalState retrieval={job.retrieval} /><div className="progress-label"><span>{job.type === 'import' ? importProgressMessage(job) : job.message || stageLabel(job.stage)}</span><span className="mono">{job.total ? `${job.completed} / ${job.total}` : `${progress}%`}</span></div><div className="progress-track"><div style={{ width: `${progress}%` }} /></div>{job.asset_id && !asset && <p className="helper">此视频已不在资料库。<a href="#/trash">前往回收站查看或恢复</a>，任务记录仍然保留。</p>}{job.sampling_strategy && /uniform|legacy/.test(job.sampling_strategy) && <p className="helper">历史定时采样任务 · 沿用原任务断点；新分析请使用关键帧预扫描。</p>}{job.items && <div className="import-job-items">{job.items.map((item) => <div key={item.item_id}><span>{item.name}</span><Status status={item.status} />{job.type === 'import' && importItemProgress(job, item) && <span>{importItemProgress(job, item)}</span>}{item.error && <small className="warning-text">{item.error}</small>}{item.warnings?.map((warning) => <small className="warning-text" role="status" key={warning}>{warning}</small>)}</div>)}</div>}{job.error && <div className="job-error">{job.error}</div>}{job.cost_incomplete && <p className="helper">上次请求中断，费用记录不完整{job.uncertain_request_count ? `；${job.uncertain_request_count} 次请求的完成状态未知` : ''}。请以服务商的实际用量记录为准。</p>}<JobExecution job={job} /><StageMetrics job={job} />{Object.keys(job.task_models || {}).length > 0 && <details className="advanced job-model-snapshot"><summary>任务创建时的模型与思考等级</summary>{Object.entries(job.task_models || {}).map(([role, model]) => <p className="helper" key={role}>{stageLabel(role)} · {model.provider_type === 'codex_cli' ? 'Codex CLI' : 'API'} · {model.model || 'Codex 内置默认'}{model.provider_type === 'codex_cli' ? ` · 思考等级：${model.reasoning_effort || '模型默认'}` : ''}</p>)}</details>}<div className="job-bottom"><div className="job-metrics">{['import', 'keyframe_scan'].includes(job.type) ? <span>本机处理 · 不调用模型</span> : <><span>模型请求 <strong>{job.request_count || 0}</strong></span><span>费用 <strong>{job.cost == null ? '未知' : `$${job.cost.toFixed(4)}`}</strong></span>{job.usage?.total_tokens != null && <span>Token <strong>{job.usage.total_tokens.toLocaleString()}</strong></span>}</>}</div><div className="job-actions">{busy === job.id ? <Spinner label="正在处理" /> : <>{job.type === 'import' && importedAssets.length > 0 && openJobs && <button className="button small primary" onClick={() => openJobs(importedAssets)}>分析这批</button>}{['running', 'queued', 'pending'].includes(job.status) && <button className="button small secondary" onClick={() => change(job, 'pause')}><Pause size={14} />暂停</button>}{['paused', 'interrupted'].includes(job.status) && <button className="button small primary" onClick={() => ['analysis', 'analyze', 'scene_deepen', 'deepen'].includes(job.type) ? setResuming({ job, action: 'resume' }) : void change(job, 'resume')}><Play size={14} />恢复</button>}{['failed', 'partial', 'cancelled', 'canceled'].includes(job.status) && <button className="button small secondary" onClick={() => job.type === 'import' ? setRetryImport(job.id) : ['analysis', 'analyze', 'scene_deepen', 'deepen'].includes(job.type) ? setResuming({ job, action: 'retry' }) : void change(job, 'retry')}><RotateCcw size={14} />重试</button>}{[...active, 'paused'].includes(job.status) && <button className="button small ghost" onClick={() => change(job, 'cancel')}><X size={14} />取消</button>}</>}</div></div></article>;
  })}</div></>}{retryImport && <ImportModal retryJobId={retryImport} onClose={() => setRetryImport(null)} />}{resuming && <ResumeModal job={resuming.job} action={resuming.action} onClose={() => setResuming(null)} onResume={async (budget) => { const ok = await change(resuming.job, resuming.action, budget); if (ok) setResuming(null); return ok; }} />}</>;
}


function ResumeModal({ job, action, onClose, onResume, failureMessages = [] }: { job: Job; action: 'resume' | 'retry'; onClose: () => void; onResume: (budget: Record<string, number | null>) => Promise<boolean>; failureMessages?: string[] }) {
  const [maxRequests, setMaxRequests] = useState(job.budget?.max_requests || Math.max(1000, (job.request_count || 0) + 100));
  const [maxCost, setMaxCost] = useState(job.budget?.max_cost?.toString() || '');
  const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true);
    await onResume({ max_requests: maxRequests, max_cost: maxCost === '' ? null : Number(maxCost) });
    setBusy(false);
  };
  return <Modal title={action === 'retry' ? '重试未完成的任务' : '恢复分析任务'} subtitle="复用已成功的结果，可在继续之前调整预算。" onClose={onClose}><form onSubmit={submit}><div className="modal-body">{failureMessages.length > 0 && <Alert tone="error">{failureMessages.length} 集未能继续：{failureMessages.join('；')}</Alert>}<Alert tone="info">预算是任务的累计上限。目前已请求 {job.request_count || 0} 次；若因预算暂停，请提高上限后继续。</Alert><p className="helper">继续使用各任务创建时的模型、思考等级与执行方式。</p><Field label="累计请求次数上限"><input type="number" min={(job.request_count || 0) + 1} max="100000" value={maxRequests} onChange={(event) => setMaxRequests(Number(event.target.value))} required /></Field><Field label="累计费用上限（USD）" hint="留空会移除原有费用限制；请求次数限制继续生效。"><input type="number" min="0.01" step="0.01" value={maxCost} onChange={(event) => setMaxCost(event.target.value)} placeholder="未设置" /></Field>{job.cost_incomplete && <Alert>中断导致费用无法完整核对。若继续保留费用上限，任务可能再次暂停；可以移除费用限制，并用请求次数限制控制用量。</Alert>}</div><footer className="modal-footer"><span>成功结果与人工修改会保留</span><button type="submit" className="button primary" disabled={busy}>{busy ? <Spinner label="提交中" /> : <><Play size={15} />继续任务</>}</button></footer></form></Modal>;
}

function BatchSummary({ id, jobs }: { id: string; jobs: Job[] }) {
  const { refreshJobs, notify, assets } = useApp(); const [busy, setBusy] = useState(false); const [failures, setFailures] = useState<{ job_id: string; message: string }[]>([]); const [resume, setResume] = useState<'resume' | 'retry' | null>(null);
  const control = async (action: string, budget: Record<string, number | null> = {}) => { setBusy(true); try { const result = await post<{ failures?: { job_id: string; message: string }[] }>(`/jobs/batches/${encodeURIComponent(id)}/${action}`, budget); setFailures(result.failures || []); await refreshJobs(true); if (result.failures?.length) { notify(`批次部分完成，${result.failures.length} 集未能处理，请查看原因`, 'error'); return false; } notify('批次操作已提交，各集保留已完成结果'); setResume(null); return true; } catch (error) { notify(message(error), 'error'); return false; } finally { setBusy(false); } };
  const failureMessages = failures.map((failure) => { const asset = assets.find((asset) => asset.id === jobs.find((job) => job.id === failure.job_id)?.asset_id); return `${asset ? assetTaskLabel(asset) : '视频任务'}：${failure.message}`; });
  const complete = jobs.filter((job) => job.status === 'completed').length;
  const readyCores = jobs.filter((job) => job.retrieval?.status === 'core_ready');
  const coreReady = readyCores.length;
  const modernCores = readyCores.filter((job) => (job.retrieval?.contract_version || 1) >= 2);
  const identityLabels: Record<string, string> = { completed: '已完成', running: '已启用（处理中）', pending: '已启用（待完成）', disabled: '已关闭', not_requested: '未启用', failed: '失败' };
  const identityCounts = modernCores.reduce<Record<string, number>>((counts, job) => {
    const state = job.retrieval?.capabilities?.identity || 'unrecorded';
    counts[state] = (counts[state] || 0) + 1;
    return counts;
  }, {});
  const coreExplanation = modernCores.length
    ? `${readyCores.length > modernCores.length ? `旧版 ${readyCores.length - modernCores.length} 集未启用局部事件与身份；` : ''}局部事件未启用；本地人物匹配：${Object.entries(identityCounts).map(([state, count]) => `${count} 集${identityLabels[state] || '状态未记录'}`).join('、')}，不等于默认完整流程`
    : '未启用局部事件与身份，不等于默认完整流程';
  const cost = jobs.every((job) => job.cost != null) ? jobs.reduce((sum, job) => sum + (job.cost || 0), 0) : null;
  const representative = { ...jobs[0], request_count: Math.max(...jobs.map((job) => job.request_count || 0)), budget: { max_requests: Math.max(...jobs.map((job) => job.budget?.max_requests || 1000)), max_cost: jobs[0].budget?.max_cost } };
  return <section className="batch-summary" aria-label="批量分析进度"><div><h3>批量分析 · {jobs.length} 集</h3><p className="helper">任务完成 {complete} / {jobs.length}{coreReady > 0 ? ` · 其中核心索引 ${coreReady}（${coreExplanation}）` : ''} · 累计请求 {jobs.reduce((sum, job) => sum + (job.request_count || 0), 0)} 次 · 总费用 {cost == null ? '未知' : `$${cost.toFixed(4)}`}</p></div><div className="button-row">{jobs.some((job) => ['queued', 'running'].includes(job.status)) && <button className="button secondary small" disabled={busy} onClick={() => void control('pause')}>暂停整批</button>}{jobs.some((job) => job.status === 'paused') && <button className="button secondary small" disabled={busy} onClick={() => setResume('resume')}>恢复整批</button>}{jobs.some((job) => ['failed', 'partial', 'cancelled'].includes(job.status)) && <button className="button secondary small" disabled={busy} onClick={() => setResume('retry')}>重试未完成集</button>}{jobs.some((job) => ['queued', 'running', 'paused'].includes(job.status)) && <button className="button ghost small" disabled={busy} onClick={() => void control('cancel')}>取消整批</button>}</div>{failureMessages.length > 0 && <Alert tone="error">{failureMessages.length} 集未处理成功：{failureMessages.join('；')}</Alert>}{resume && <ResumeModal failureMessages={failureMessages} job={representative} action={resume} onClose={() => setResume(null)} onResume={(budget) => control(resume, budget)} />}</section>;
}
