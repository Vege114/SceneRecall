import { useMemo, useRef, useState, type FormEvent } from 'react';
import { ArrowRight } from 'lucide-react';
import { api, duration, message, post } from '../api';
import { useApp } from '../context';
import { type Asset, type JobPayload } from '../types';
import { Alert, Field, Modal, Spinner } from './ui';
import { assetTaskLabel } from '../taskHelpers';
import AnalysisConfiguration, { analysisDefaults, analysisProfile, analysisRoles, resolvedAnalysisDefaults } from './AnalysisConfiguration';

export default function JobModal({ asset, assets: selection, onClose }: { asset?: Asset; assets?: Asset[]; onClose: () => void }) {
  const { bootstrap, refresh, notify } = useApp();
  const assets = useMemo(() => selection || (asset ? [asset] : []), [selection, asset]);
  const [draftOptions, setOptions] = useState(() => analysisDefaults(bootstrap.settings.defaults));
  const options = resolvedAnalysisDefaults(draftOptions, bootstrap.settings, bootstrap.provider_profiles);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState('');
  const [force, setForce] = useState(false);
  const [saveDefault, setSaveDefault] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const submitted = useRef(false);
  const submitting = useRef(false);
  const request = useRef({ key: '', id: '' });
  const retrieval = options.analysis_strategy === 'retrieval-first-v1';
  const hasEmbedded = assets.some((entry) => entry.subtitle_mode === 'embedded');
  const roles = analysisRoles(options, hasEmbedded);
  const missing = roles.some((role) => !analysisProfile(role, options, bootstrap.settings, bootstrap.provider_profiles));
  const offline = assets.filter((entry) => !entry.source_available);
  const payloads: JobPayload[] = assets.map((entry): JobPayload => {
    const activeRoles = analysisRoles(options, entry.subtitle_mode === 'embedded');
    const usesCodex = activeRoles.some((role) => analysisProfile(role, options, bootstrap.settings, bootstrap.provider_profiles)?.provider_type === 'codex_cli');
    return { asset_id: entry.id, task_pipeline: 'unified-v1', analysis_strategy: retrieval ? 'retrieval-first-v1' : 'scene-chunks-v1',
      ...(retrieval ? { retrieval_mode: options.retrieval_mode } : {}), execution: options.execution,
      stages: retrieval ? ['vision'] : (options.stages || []).filter((stage) => stage === 'vision' || entry.subtitle_mode === 'embedded'),
      model_bindings: Object.fromEntries(activeRoles.filter((role) => options.model_bindings?.[role]).map((role) => [role, options.model_bindings![role]!])),
      model_overrides: Object.fromEntries(activeRoles.filter((role) => options.model_overrides?.[role]).map((role) => [role, options.model_overrides![role]!])),
      start_ms: Math.round(start * 1000), end_ms: end ? Math.min(Math.round(Number(end) * 1000), entry.duration_ms) : null,
      subtitle_fps: options.subtitle_fps, subtitle_crop: options.subtitle_crop,
      max_requests: options.max_requests, max_cost: usesCodex || !activeRoles.length ? null : options.max_cost, force };
  }).filter((entry) => entry.stages.length);
  const startReady = !!payloads.length && !missing && !offline.length && (!retrieval || bootstrap.visual_encoder?.available !== false);
  const startJob = async (event: FormEvent) => {
    event.preventDefault();
    if (submitting.current || submitted.current || !startReady) return;
    if (payloads.some((entry) => !Number.isFinite(entry.start_ms) || entry.start_ms < 0 || (entry.end_ms ?? assets.find((asset) => asset.id === entry.asset_id)!.duration_ms) <= entry.start_ms)) {
      setError('请设置有效的分析范围，开始时间应早于各集的结束时间。'); return;
    }
    submitting.current = true; setBusy(true); setError('');
    let defaultsSaved = false;
    try {
      if (saveDefault) { await api('/settings', { method: 'PUT', body: JSON.stringify({ defaults: options }) }); defaultsSaved = true; }
      const key = JSON.stringify(payloads);
      if (request.current.key !== key) request.current = { key, id: crypto.randomUUID() };
      if (payloads.length === 1) await post('/jobs', payloads[0]);
      else await post('/jobs/batch', { request_id: request.current.id, items: payloads });
      submitted.current = true;
    } catch (caught) {
      if (defaultsSaved) void refresh().catch(() => {});
      setError(defaultsSaved ? `默认任务配置已保存，但任务创建失败：${message(caught)}。请重试创建任务。` : message(caught));
      return;
    } finally { submitting.current = false; setBusy(false); }
    notify(`${payloads.length} 个分析任务已加入队列`); onClose(); window.location.hash = '/jobs';
    void refresh().catch((caught) => notify(`任务已创建，但列表刷新失败：${message(caught)}。请在任务队列刷新。`, 'error'));
  };
  return <Modal title={assets.length > 1 ? '分析这批视频' : '分析这部作品'} subtitle={assets.length > 1 ? `已选择 ${assets.length} 集 · 每集独立预算与进度` : assets[0] && assetTaskLabel(assets[0])} onClose={onClose} wide>
    <form onSubmit={startJob}><div className="modal-body">
      <p className="helper">已载入默认任务配置。设置完成后直接开始；关键帧划分、角色确认与分析都在任务队列中继续。需要你判断角色时会显示“等待确认”。</p>
      <AnalysisConfiguration value={options} onChange={setOptions} disabled={busy} hasEmbedded={hasEmbedded} />
      {offline.length > 0 && <Alert tone="error">{offline.map(assetTaskLabel).join('、')}：原视频离线或已变化，请先定位原文件。</Alert>}
      {retrieval && bootstrap.visual_encoder?.available === false && <Alert tone="error">本机视觉编码器尚不可用：{bootstrap.visual_encoder.reason || bootstrap.visual_encoder.message || '请先配置本机编码器与权重。'}</Alert>}
      <fieldset disabled={busy}>
        <div className="section-row"><h3>分析范围</h3><div><button type="button" className="text-button" onClick={() => { setStart(0); setEnd(''); }}>处理全片</button><button type="button" className="text-button" onClick={() => { setStart(0); setEnd('60'); }}>先试运行前 60 秒</button></div></div>
        <div className="form-grid"><Field label="开始时间（秒）"><input type="number" min={0} step={0.001} required value={start} onChange={(event) => setStart(Number(event.target.value))} /></Field><Field label="结束时间（秒）" hint={assets.length > 1 ? '留空分析各集全片；指定时间自动裁剪到每集片尾' : `留空分析到结尾 · 全长 ${duration(assets[0]?.duration_ms || 0)}`}><input type="number" min={start + 0.001} step={0.001} placeholder="视频结尾" value={end} onChange={(event) => setEnd(event.target.value)} /></Field></div>
        <label className="check-line"><input type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} />忽略已有分析缓存，重新处理</label>
        <label className="check-line"><input type="checkbox" checked={saveDefault} onChange={(event) => setSaveDefault(event.target.checked)} />设为默认值</label>
        <p className="helper">保存处理策略、阶段、模型、执行方式、字幕参数与预算，供下次任务使用。分析范围和重新处理选项仅用于本次。</p>
      </fieldset>
      {!payloads.length && <Alert>请选择至少一个适用于这些视频的处理阶段。</Alert>}
      {error && <Alert tone="error">{error}</Alert>}
    </div><footer className="modal-footer"><span>开始后在任务队列统一查看进度</span><button className="button primary" type="submit" disabled={busy || !startReady}>{busy ? <Spinner label="正在提交" /> : <><ArrowRight size={16} />开始任务</>}</button></footer></form>
  </Modal>;
}
