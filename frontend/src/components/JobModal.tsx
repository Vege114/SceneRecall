import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { ArrowRight, Calculator, ScanLine, Subtitles } from 'lucide-react';
import { api, duration, formatTime, message, post } from '../api';
import { useApp } from '../context';
import { DEFAULTS, CAPABILITIES, executionDefaults, estimatedCost, estimatedRequests, taskModelLabel, keyframeReason, type Asset, type Estimate, type ExecutionStage, type JobPayload, type KeyframePlan, type TaskModelRole } from '../types';
import { Alert, Field, Modal, Spinner, Status } from './ui';
import { StageEstimates } from './SceneAnalysis';
import TaskModels, { taskModelOptions, useTaskModels } from './TaskModels';
import TaskExecution, { saveExecutionDefaults } from './TaskExecution';
import { assetTaskLabel, settleConcurrent } from '../taskHelpers';

export default function JobModal({ asset, assets: selection, onClose }: { asset?: Asset; assets?: Asset[]; onClose: () => void }) {
  const { bootstrap, refresh, notify } = useApp();
  const assets = useMemo(() => selection || (asset ? [asset] : []), [selection, asset]);
  const defaults = { ...DEFAULTS, ...bootstrap.settings.defaults };
  const [strategy, setStrategy] = useState<'scene-chunks-v1' | 'retrieval-first-v1'>('scene-chunks-v1');
  const retrieval = strategy === 'retrieval-first-v1';
  const [useVlm, setUseVlm] = useState(false);
  const [vision, setVision] = useState(true);
  const [subtitle, setSubtitle] = useState(assets.some((item) => item.subtitle_mode === 'embedded'));
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState('');
  const [fps, setFps] = useState(defaults.subtitle_fps);
  const [crop, setCrop] = useState(defaults.subtitle_crop);
  const [requestLimit, setRequestLimit] = useState(defaults.max_requests);
  const [costLimit, setCostLimit] = useState(defaults.max_cost?.toString() || '');
  const [force, setForce] = useState(false);
  const [execution, setExecution] = useState(() => executionDefaults(defaults.execution));
  const [saveDefault, setSaveDefault] = useState(false);
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [estimateKey, setEstimateKey] = useState('');
  const [estimates, setEstimates] = useState<(Estimate & { asset_id: string })[]>([]);
  const [plans, setPlans] = useState<Record<string, KeyframePlan>>({});
  const [scanKey, setScanKey] = useState('');
  const [scanErrors, setScanErrors] = useState<Record<string, string>>({});
  const [cancelling, setCancelling] = useState('');
  const [previewCount, setPreviewCount] = useState(24);
  const [busy, setBusy] = useState<'estimate' | 'start' | 'scan' | null>(null);
  const [error, setError] = useState('');
  const request = useRef({ key: '', id: '' });
  const form = useRef<HTMLFormElement>(null);
  const plansRef = useRef(plans);
  plansRef.current = plans;
  const submitted = useRef(false);
  const live = useRef(true);
  useEffect(() => { live.current = true; return () => { live.current = false; }; }, []);
  const getProfile = (role: 'preparation' | 'vision' | 'subtitle') => bootstrap.provider_profiles.find((profile) => profile.id === bootstrap.settings.bindings[role]);
  const modelRoles = useMemo<TaskModelRole[]>(() => retrieval ? useVlm ? ['vision'] : [] : [...(vision ? ['preparation' as const, 'vision' as const] : []), ...(subtitle && assets.some((item) => item.subtitle_mode === 'embedded') ? ['subtitle' as const] : [])], [retrieval, useVlm, vision, subtitle, assets]);
  const taskModels = useTaskModels(modelRoles);
  const executionStages: ExecutionStage[] = retrieval ? [] : [...(vision ? ['transcript' as const, 'preparation' as const] : []), ...(subtitle && assets.some((item) => item.subtitle_mode === 'embedded') ? ['subtitle' as const] : [])];
  const connectionLimits = Object.fromEntries(executionStages.map((stage) => [stage, getProfile(stage === 'transcript' ? 'preparation' : stage)?.max_concurrency ?? 16]));
  const usesCodex = retrieval ? useVlm && getProfile('vision')?.provider_type === 'codex_cli' : ((vision && (getProfile('vision')?.provider_type === 'codex_cli' || getProfile('preparation')?.provider_type === 'codex_cli')) || (subtitle && assets.some((item) => item.subtitle_mode === 'embedded') && getProfile('subtitle')?.provider_type === 'codex_cli'));
  const ranges = useMemo(() => assets.map((item) => ({ asset_id: item.id, start_ms: Math.round(start * 1000), end_ms: end ? Math.min(Math.round(Number(end) * 1000), item.duration_ms) : null })), [assets, start, end]);
  const rangeKey = JSON.stringify(ranges);
  const scansReady = scanKey === rangeKey && assets.every((item) => plans[item.id]?.status === 'completed');
  const scanning = scanKey === rangeKey && Object.values(plans).some((plan) => ['queued', 'pending', 'running', 'cancelling', 'pausing'].includes(plan.status));
  useEffect(() => {
    if (!scanning) return;
    let active = true;
    let timer: number;
    const poll = async () => {
      const pending = Object.entries(plansRef.current).filter(([, plan]) => ['queued', 'pending', 'running', 'cancelling', 'pausing'].includes(plan.status));
      await settleConcurrent(pending, async ([assetId, plan]) => {
        if (!active) return;
        try { const value = await api<KeyframePlan>(`/keyframe-plans/${encodeURIComponent(plan.plan_id || plan.id)}`); if (active) setPlans((current) => ({ ...current, [assetId]: value })); }
        catch (error) { if (active) { setScanErrors((current) => ({ ...current, [assetId]: message(error) })); setPlans((current) => ({ ...current, [assetId]: { ...current[assetId], status: 'failed' } })); } }
      });
      if (active) timer = window.setTimeout(() => void poll(), 1500);
    };
    timer = window.setTimeout(() => void poll(), 1500);
    return () => { active = false; window.clearTimeout(timer); };
  }, [scanning, scanKey, rangeKey]);
  const payloads = useMemo<JobPayload[]>(() => assets.map((item, index) => ({ ...ranges[index], asset_id: item.id, ...(!retrieval ? { execution: { ...execution }, ...taskModelOptions(taskModels.overrides, item.subtitle_mode === 'embedded') } : useVlm ? taskModelOptions(taskModels.overrides, false) : {}), ...(retrieval ? { analysis_strategy: 'retrieval-first-v1' as const, retrieval_mode: useVlm ? 'complete' as const : 'core' as const } : vision ? { analysis_strategy: 'scene-chunks-v1' as const } : {}), stages: [...(retrieval || vision ? ['vision' as const] : []), ...(!retrieval && subtitle && item.subtitle_mode === 'embedded' ? ['subtitle' as const] : [])], ...(!retrieval && vision && scansReady ? { keyframe_plan_id: plans[item.id].plan_id || plans[item.id].id } : {}), subtitle_fps: fps, subtitle_crop: crop, max_requests: requestLimit, max_cost: (!retrieval || useVlm) && !usesCodex && costLimit ? Number(costLimit) : null, force })).filter((item) => item.stages.length), [assets, ranges, retrieval, useVlm, vision, subtitle, scansReady, plans, fps, crop, requestLimit, costLimit, usesCodex, force, taskModels.overrides, execution]);
  const currentKey = JSON.stringify([payloads, taskModels.sourceKey, connectionLimits]);
  const estimateValid = estimate && currentKey === estimateKey && (retrieval || !vision || scansReady);
  const missing = (retrieval ? useVlm ? ['vision' as const] : [] : [...new Set([...payloads.flatMap((item) => item.stages), ...(vision ? ['preparation' as const] : [])])]).filter((role) => !getProfile(role));
  const offline = assets.filter((item) => !item.source_available);
  const startReady = (retrieval || !vision || scansReady) && !missing.length && !!payloads.length && !offline.length && (!retrieval || bootstrap.visual_encoder?.available !== false);
  const scan = async () => {
    if (ranges.some((range, index) => !Number.isFinite(range.start_ms) || range.start_ms < 0 || (range.end_ms ?? assets[index].duration_ms) <= range.start_ms)) { setError('请设置有效的分析范围，开始时间应早于各集的结束时间。'); return; }
    setBusy('scan'); setError(''); setScanErrors({}); setPlans({}); setScanKey(rangeKey); setEstimate(null); setPreviewCount(24);
    await settleConcurrent(ranges, async (range) => {
      if (!live.current) return;
      try { let value = await post<KeyframePlan>('/keyframe-plans', range); if (value.status === 'completed' && !value.frames) value = await api<KeyframePlan>(`/keyframe-plans/${encodeURIComponent(value.plan_id || value.id)}`); if (!live.current) return; setPlans((current) => ({ ...current, [range.asset_id]: value })); }
      catch (error) { if (!live.current) return; setScanErrors((current) => ({ ...current, [range.asset_id]: message(error) })); }
    });
    if (live.current) setBusy(null);
  };
  const cancelScan = async (assetId: string) => { const plan = plans[assetId]; if (!plan?.job_id) return; setCancelling(assetId); try { const value = await post<{ status: string; message?: string }>(`/jobs/${encodeURIComponent(plan.job_id)}/cancel`); setPlans((current) => ({ ...current, [assetId]: { ...current[assetId], status: value.status, message: value.message } })); } catch (error) { setError(message(error)); } finally { setCancelling(''); } };
  const runEstimate = async (event: FormEvent) => {
    event.preventDefault(); setBusy('estimate'); setError('');
    try {
      if (!retrieval && vision && !scansReady) throw new Error('请先完成本机关键帧预扫描并查看候选画面。');
      if (payloads.length === 1) { const value = await post<Estimate>('/jobs/estimate', payloads[0]); setEstimate(value); setEstimates([{ ...value, asset_id: payloads[0].asset_id }]); }
      else { const value = await post<Estimate & { items: (Estimate & { asset_id: string })[] }>('/jobs/batch/estimate', { items: payloads }); setEstimate({ ...value, duration_ms: ranges.reduce((sum, range, index) => sum + Math.max(0, (range.end_ms ?? assets[index].duration_ms) - range.start_ms), 0) }); setEstimates(value.items); }
      setEstimateKey(currentKey);
    } catch (error) { setError(message(error)); } finally { setBusy(null); }
  };
  const startJob = async () => {
    if (busy || submitted.current || !startReady || !form.current?.reportValidity()) return;
    setBusy('start'); setError('');
    let defaultsSaved = false;
    try {
      if (!retrieval && saveDefault) { await saveExecutionDefaults(execution, executionStages); defaultsSaved = true; }
      if (request.current.key !== currentKey) request.current = { key: currentKey, id: crypto.randomUUID() };
      if (payloads.length === 1) await post('/jobs', payloads[0]);
      else await post('/jobs/batch', { request_id: request.current.id, items: payloads });
      submitted.current = true;
      let refreshError = '';
      try { await refresh(); } catch (error) { refreshError = message(error); }
      notify(`${payloads.length} 个分析任务已加入队列`); onClose(); window.location.hash = '/jobs';
      if (refreshError) notify(`任务已创建，但列表刷新失败：${refreshError}。请在任务队列刷新。`, 'error');
    } catch (error) {
      if (defaultsSaved) { try { await refresh(); } catch { /* Keep the creation error visible. */ } }
      setError(defaultsSaved ? `默认执行方式已保存，但任务创建失败：${message(error)}。请重试创建任务。` : message(error));
    } finally { setBusy(null); }
  };
  const subtitleCount = assets.filter((item) => item.subtitle_mode === 'embedded').length;
  return <Modal title={assets.length > 1 ? '分析这批视频' : '分析这部作品'} subtitle={assets.length > 1 ? `已选择 ${assets.length} 集 · 每集独立预算与进度` : assets[0] && assetTaskLabel(assets[0])} onClose={onClose} wide><form ref={form} onSubmit={runEstimate}>
    <div className="modal-body">
      <fieldset disabled={!!busy || scanning}>
        <Field label="处理策略"><select value={strategy} onChange={(event) => { const next = event.target.value as typeof strategy; setStrategy(next); if (next === 'retrieval-first-v1' && !end) setEnd(String(Math.min(start + 60, ...assets.map((item) => item.duration_ms / 1000)))); }}><option value="scene-chunks-v1">场景理解（当前默认）</option><option value="retrieval-first-v1">画面检索 · 可选 VLM</option></select></Field>
        {retrieval && <><label className="check-line"><input type="checkbox" checked={useVlm} onChange={(event) => setUseVlm(event.target.checked)} /><span>使用 VLM 生成局部描述与事件</span></label><Alert tone="info">{useVlm ? '本地画面索引之外，调用 VLM 生成局部人物、物体、场所及事件记录。本地人物匹配按独立识别设置执行；全部必需阶段完成后才报告完整流程完成。' : '不使用生成式 VLM：建立本地直接视觉索引、已有字幕索引。本地人物匹配按独立识别设置执行，关闭 VLM 不会将其跳过；未开启本地识别时需先启用。不会生成逐窗口描述或结构化事件。'}</Alert><p className="helper">{useVlm ? '局部记录会增加模型请求与处理时间；Codex 未显式选择模型时使用 gpt-6.1-sol / low。当前不承诺整集实时完成。' : '图文编码器与已启用的人物匹配在本机运行；角色档案和参考图在同一作品范围跨集共享。证据不足时显示候选或未知，不会直接把最相似角色当作实名。VLM 选项只控制描述与事件生成，不控制本地人物匹配。'}</p>{useVlm && <TaskModels roles={modelRoles} value={taskModels.overrides} onChange={taskModels.change} disabled={!!busy || scanning} />}{useVlm && missing.length > 0 && <Alert>请先在<a href="#/settings" onClick={onClose}>模型与设置</a>中为画面理解分配 vision 模型；取消勾选 VLM 可继续建立基础检索索引。</Alert>}{bootstrap.visual_encoder?.available === false && <Alert tone="error">本机视觉编码器尚不可用：{bootstrap.visual_encoder.reason || bootstrap.visual_encoder.message || '请先配置本机编码器与权重。'}</Alert>}</>}
        {!retrieval && <>
        <div className="choice-grid"><label className={`choice selectable ${vision ? 'selected' : ''}`}><input type="checkbox" checked={vision} onChange={(event) => setVision(event.target.checked)} /><ScanLine size={21} /><strong>画面理解</strong><span>{taskModelLabel(getProfile('vision'), taskModels.overrides?.vision)} · 仅发送内容关键帧</span></label><label className={`choice selectable ${subtitle ? 'selected' : ''} ${!subtitleCount ? 'disabled' : ''}`}><input type="checkbox" checked={subtitle} disabled={!subtitleCount} onChange={(event) => setSubtitle(event.target.checked)} /><Subtitles size={21} /><strong>画面字幕 OCR</strong><span>{subtitleCount ? `${taskModelLabel(getProfile('subtitle'), taskModels.overrides?.subtitle)} · 仅处理 ${subtitleCount} 集烧录字幕` : assets.length === 1 && assets[0].subtitle_mode === 'container' ? '容器文本字幕已在本地提取，无需发送字幕图片' : '外挂字幕已在导入时解析'}</span></label></div>
        {vision && <p className="helper scene-strategy">台词整理 → 视觉分块（{taskModelLabel(getProfile('preparation'), taskModels.overrides?.preparation)}）→ 复杂片段深入理解。全部关键帧保留，连续记忆在同项目内沿用。</p>}
        <TaskModels roles={modelRoles} value={taskModels.overrides} onChange={taskModels.change} disabled={!!busy || scanning} />
        <TaskExecution stages={executionStages} value={execution} onChange={setExecution} disabled={!!busy || scanning} connectionLimits={connectionLimits} saveDefault={saveDefault} onSaveDefaultChange={setSaveDefault} />
        {missing.length > 0 && <Alert>请先在<a href="#/settings" onClick={onClose}>模型与设置</a>中为{missing.map((role) => CAPABILITIES.find((item) => item.key === role)?.name).join('、')}分配模型。关键帧预扫描无需模型。</Alert>}
        {offline.length > 0 && <Alert tone="error">{offline.map(assetTaskLabel).join('、')}：原视频离线或已变化，请先定位原文件。</Alert>}
        </>}
        {retrieval && offline.length > 0 && <Alert tone="error">{offline.map(assetTaskLabel).join('、')}：原视频离线或已变化，请先定位原文件。</Alert>}
        <div className="section-row"><h3>分析范围</h3><div><button type="button" className="text-button" onClick={() => { setStart(0); setEnd(''); }}>处理全片</button><button type="button" className="text-button" onClick={() => { setStart(0); setEnd('60'); }}>先试运行前 60 秒</button></div></div>
        <div className="form-grid"><Field label="开始时间（秒）"><input type="number" min="0" step="0.001" value={start} onChange={(event) => setStart(Number(event.target.value))} required /></Field><Field label="结束时间（秒）" hint={assets.length > 1 ? '留空分析各集全片；指定时间自动裁剪到每集片尾' : `留空分析到结尾 · 全长 ${duration(assets[0]?.duration_ms || 0)}`}><input type="number" min={start + 0.001} step="0.001" placeholder="视频结尾" value={end} onChange={(event) => setEnd(event.target.value)} /></Field></div>
      </fieldset>
      {!retrieval && vision && <section className="keyframe-preview" aria-label="关键帧预扫描"><div className="section-row"><div><h3>先看关键帧，再分析</h3><p className="helper">本机扫描画面变化，减少重复和过渡画面。预扫描不调用模型。</p></div><button type="button" className="button secondary" disabled={!!busy || scanning || !!offline.length} onClick={() => void scan()}>{busy === 'scan' || scanning ? <Spinner label="正在扫描关键帧" /> : scansReady ? '重新扫描关键帧' : '扫描关键帧'}</button></div>
        {scanKey && scanKey !== rangeKey && <Alert>分析范围已变化，请重新扫描关键帧。</Alert>}
        {scanKey === rangeKey && assets.map((item) => { const plan = plans[item.id]; return <div className="keyframe-asset" key={item.id}><strong>{assetTaskLabel(item)}</strong>{plan && ['queued', 'pending', 'running', 'cancelling', 'pausing'].includes(plan.status) && <div className="scan-progress"><div className="progress-label"><span>{plan.message || '等待本机扫描'}</span><span>{Math.round((plan.progress || 0) * 100)}%</span></div><div className="progress-track" role="progressbar" aria-label={`${assetTaskLabel(item)}扫描进度`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round((plan.progress || 0) * 100)}><div style={{ width: `${Math.min(100, Math.max(0, (plan.progress || 0) * 100))}%` }} /></div><button type="button" className="text-button" disabled={cancelling === item.id || plan.status === 'cancelling'} onClick={() => void cancelScan(item.id)}>{plan.status === 'cancelling' ? '正在取消扫描' : `取消「${assetTaskLabel(item)}」扫描`}</button></div>}{plan && ['cancelled', 'paused'].includes(plan.status) && <Status status={plan.status} />}{scanErrors[item.id] && <Alert tone="error">{scanErrors[item.id]}</Alert>}{plan?.status === 'failed' && <Alert tone="error">{plan.error || '本机扫描失败，请检查视频后重新扫描。'}</Alert>}{plan?.status === 'completed' && <><p className="helper">选中 {plan.frame_count ?? plan.frames?.length ?? 0} 张关键帧 · 预计 {Math.ceil((plan.frame_count ?? plan.frames?.length ?? 0) / (getProfile('preparation')?.provider_type === 'codex_cli' ? 24 : 56))} 批视觉分块；深入理解数量在分块后确定</p><div className="keyframe-grid">{plan.frames?.slice(0, previewCount).map((frame) => <a key={frame.id} href={frame.thumbnail_url} target="_blank" rel="noreferrer" title={keyframeReason(frame.reason)}><img src={frame.thumbnail_url} alt={`${assetTaskLabel(item)} ${formatTime(frame.at_ms)} 关键帧`} loading="lazy" /><span>{formatTime(frame.at_ms)}</span><small>{keyframeReason(frame.reason)}</small></a>)}</div>{(plan.frames?.length || 0) > previewCount && <button type="button" className="text-button" onClick={() => setPreviewCount((count) => count + 48)}>显示更多关键帧</button>}</>}</div>; })}
      </section>}
      {(!retrieval || useVlm) && <>{!retrieval && <details className="advanced"><summary>字幕 OCR 与重跑选项</summary>{subtitle && <><Field label="字幕采样（帧 / 秒）" hint="仅控制烧录字幕的时间定位，不用于画面理解"><input type="number" min="0.2" max="10" step="0.1" value={fps} onChange={(event) => setFps(Number(event.target.value))} /></Field><div className="form-grid four">{['左侧 X', '顶部 Y', '宽度', '高度'].map((label, index) => <Field key={label} label={label}><input type="number" min={index > 1 ? 0.01 : 0} max="1" step="0.01" value={crop[index]} onChange={(event) => setCrop(crop.map((value, i) => i === index ? Number(event.target.value) : value))} /></Field>)}</div></>}<label className="check-line"><input type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} /><span>重新分析已成功的画面，保留旧版本与人工修改</span></label></details>}
      <div className="form-grid"><Field label={assets.length > 1 ? '每集最多模型请求数' : '最多模型请求数'} hint="每集累计上限，达到上限后可提高预算继续"><input type="number" min="1" max="100000" value={requestLimit} onChange={(event) => setRequestLimit(Number(event.target.value))} required /></Field><Field label={assets.length > 1 ? '每集费用上限（USD，可选）' : '费用上限（USD，可选）'} hint={usesCodex ? 'Codex CLI 无法使用美元费用上限' : '需要填写模型输入 / 输出单价'}><input type="number" min="0.01" step="0.01" disabled={usesCodex} value={usesCodex ? '' : costLimit} onChange={(event) => setCostLimit(event.target.value)} placeholder="未设置" /></Field></div>
      {assets.length > 1 && <p className="helper">整批请求预算最多 {requestLimit * payloads.length} 次；失败或暂停按集保留结果。</p>}
      {usesCodex && <Alert>本次分析包含 Codex CLI，金额预算（包括默认费用上限）已停用。请用最多模型请求数控制用量；Codex 订阅剩余额度和美元费用无法在此估算。</Alert>}
      </>}
      {estimateValid && <><div className="estimate"><div><span>分析时长</span><strong>{duration(estimate.duration_ms)}</strong></div><div><span>候选画面</span><strong>{estimate.estimated_frames.toLocaleString()} <small>帧</small></strong></div><div><span>预计请求</span><strong>≈ {estimatedRequests(estimate)} <small>次</small></strong></div><div><span>估算费用</span><strong>{estimatedCost(estimate)}</strong></div>{estimate.warnings?.map((warning, index) => <p className="warning-text" key={index}>{warning}</p>)}</div><StageEstimates estimate={estimate} />{assets.length > 1 && <div className="batch-estimates">{estimates.map((entry) => <div key={entry.asset_id}><strong>{assets.find((item) => item.id === entry.asset_id) ? assetTaskLabel(assets.find((item) => item.id === entry.asset_id)!) : entry.asset_id}</strong><span>{estimatedRequests(entry)} 次请求 · 费用 {estimatedCost(entry)}</span></div>)}</div>}</>}
      <p className="helper">{retrieval ? useVlm ? '本次包含 VLM 局部记录；处理范围、覆盖、请求量与完整完成状态会记录在任务中。' : '本次不使用生成式 VLM；处理范围、覆盖与首次可搜索时间会记录在任务中。' : '估算用量为可选步骤，完成关键帧扫描后即可开始分析。'}</p>
      {error && <Alert tone="error">{error}</Alert>}
    </div><footer className="modal-footer"><button type="submit" className="button secondary" disabled={!!busy || !payloads.length || (!retrieval && vision && !scansReady)}>{busy === 'estimate' ? <Spinner label="正在估算" /> : <><Calculator size={16} />{estimateValid ? '重新估算' : '估算用量'}</>}</button><button type="button" className="button primary" onClick={() => void startJob()} disabled={!!busy || !startReady}>{busy === 'start' ? <Spinner label="正在提交" /> : <>{retrieval ? useVlm ? '开始检索与 VLM 记录' : '开始检索索引（不使用 VLM）' : '开始分析'} <ArrowRight size={17} /></>}</button></footer>
  </form></Modal>;
}
