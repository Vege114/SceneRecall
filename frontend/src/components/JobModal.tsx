import { useMemo, useState, type FormEvent } from 'react';
import { ArrowRight, Calculator, ScanLine, Subtitles } from 'lucide-react';
import { duration, message, post } from '../api';
import { useApp } from '../context';
import { DEFAULTS, profileModelLabel, type Asset, type Estimate, type JobPayload } from '../types';
import { Alert, Field, Modal, Spinner } from './ui';

export default function JobModal({ asset, onClose }: { asset: Asset; onClose: () => void }) {
  const { bootstrap, refresh, notify } = useApp();
  const defaults = { ...DEFAULTS, ...bootstrap.settings.defaults };
  const [vision, setVision] = useState(true);
  const [subtitle, setSubtitle] = useState(asset.subtitle_mode === 'embedded');
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState('');
  const [windowSeconds, setWindowSeconds] = useState(defaults.window_ms / 1000);
  const [frameCount, setFrameCount] = useState(defaults.frames_per_window);
  const [fps, setFps] = useState(defaults.subtitle_fps);
  const [crop, setCrop] = useState(defaults.subtitle_crop);
  const [requestLimit, setRequestLimit] = useState(defaults.max_requests);
  const [costLimit, setCostLimit] = useState(defaults.max_cost?.toString() || '');
  const [force, setForce] = useState(false);
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [estimateKey, setEstimateKey] = useState('');
  const [busy, setBusy] = useState<'estimate' | 'start' | null>(null);
  const [error, setError] = useState('');
  const getProfile = (role: 'vision' | 'subtitle') => bootstrap.provider_profiles.find((profile) => profile.id === bootstrap.settings.bindings[role]);
  const usesCodex = (vision && getProfile('vision')?.provider_type === 'codex_cli') || (subtitle && getProfile('subtitle')?.provider_type === 'codex_cli');
  const payload = useMemo<JobPayload>(() => ({ asset_id: asset.id, stages: [...(vision ? ['vision' as const] : []), ...(subtitle ? ['subtitle' as const] : [])], start_ms: Math.round(start * 1000), end_ms: end ? Math.round(Number(end) * 1000) : null, window_ms: Math.round(windowSeconds * 1000), frames_per_window: frameCount, subtitle_fps: fps, subtitle_crop: crop, max_requests: requestLimit, max_cost: !usesCodex && costLimit ? Number(costLimit) : null, force }), [asset.id, vision, subtitle, start, end, windowSeconds, frameCount, fps, crop, requestLimit, costLimit, usesCodex, force]);
  const currentKey = JSON.stringify(payload);
  const estimateValid = estimate && currentKey === estimateKey;
  const missing = payload.stages.filter((role) => !bootstrap.settings.bindings[role]);
  const runEstimate = async (event: FormEvent) => {
    event.preventDefault(); setBusy('estimate'); setError('');
    try { setEstimate(await post<Estimate>('/jobs/estimate', payload)); setEstimateKey(currentKey); } catch (error) { setError(message(error)); } finally { setBusy(null); }
  };
  const startJob = async () => {
    setBusy('start'); setError('');
    try { await post('/jobs', payload); await refresh(); notify('分析任务已加入队列'); onClose(); window.location.hash = '/jobs'; } catch (error) { setError(message(error)); } finally { setBusy(null); }
  };
  return <Modal title="分析这部作品" subtitle={asset.title} onClose={onClose} wide><form onSubmit={runEstimate}><div className="modal-body"><div className="choice-grid"><label className={`choice selectable ${vision ? 'selected' : ''}`}><input type="checkbox" checked={vision} onChange={(e) => setVision(e.target.checked)} /><ScanLine size={21} /><strong>画面理解</strong><span>{profileModelLabel(getProfile('vision'))} · 发送采样画面</span></label><label className={`choice selectable ${subtitle ? 'selected' : ''} ${asset.subtitle_mode !== 'embedded' ? 'disabled' : ''}`}><input type="checkbox" checked={subtitle} disabled={asset.subtitle_mode !== 'embedded'} onChange={(e) => setSubtitle(e.target.checked)} /><Subtitles size={21} /><strong>画面字幕 OCR</strong><span>{asset.subtitle_mode === 'embedded' ? `${profileModelLabel(getProfile('subtitle'))} · 发送字幕裁剪图` : asset.subtitle_mode === 'container' ? '容器文本字幕已在本地提取，无需发送字幕图片' : '外挂字幕已在导入时解析'}</span></label></div>{missing.length > 0 && <Alert>请先在<a href="#/settings" onClick={onClose}>模型与设置</a>中为{missing.map((role) => role === 'vision' ? '画面理解' : '字幕识别').join('、')}分配模型。</Alert>}<div className="section-row"><h3>分析范围</h3><button type="button" className="text-button" onClick={() => { setStart(0); setEnd(Math.min(60, asset.duration_ms / 1000).toString()); }}>先试运行前 60 秒</button></div><div className="form-grid"><Field label="开始时间（秒）"><input type="number" min="0" max={asset.duration_ms / 1000} step="0.1" value={start} onChange={(e) => setStart(Number(e.target.value))} required /></Field><Field label="结束时间（秒）" hint={`留空分析到结尾 · 全长 ${duration(asset.duration_ms)}`}><input type="number" min={start + 0.01} max={asset.duration_ms / 1000} step="0.1" placeholder="视频结尾" value={end} onChange={(e) => setEnd(e.target.value)} /></Field></div><details className="advanced"><summary>采样设置与重跑选项</summary><div className="form-grid three"><Field label="画面窗口（秒）"><input type="number" min="1" max="30" step="0.5" value={windowSeconds} onChange={(e) => setWindowSeconds(Number(e.target.value))} required /></Field><Field label="每窗口画面数"><input type="number" min="2" max="12" value={frameCount} onChange={(e) => setFrameCount(Number(e.target.value))} required /></Field><Field label="字幕采样（帧 / 秒）"><input type="number" min="0.2" max="10" step="0.1" value={fps} onChange={(e) => setFps(Number(e.target.value))} required /></Field></div>{subtitle && <><p className="helper">字幕区域使用 0–1 归一化坐标，默认读取底部 35%。</p><div className="form-grid four">{['左侧 X', '顶部 Y', '宽度', '高度'].map((label, index) => <Field key={label} label={label}><input type="number" min={index > 1 ? 0.01 : 0} max="1" step="0.01" value={crop[index]} onChange={(e) => setCrop(crop.map((value, i) => i === index ? Number(e.target.value) : value))} required /></Field>)}</div></>}<label className="check-line"><input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} /><span>重新分析已成功的窗口，保留旧版本与人工修改</span></label></details><div className="form-grid"><Field label="最多模型请求数" hint="到达上限后停止新增请求"><input type="number" min="1" max="100000" value={requestLimit} onChange={(e) => setRequestLimit(Number(e.target.value))} required /></Field><Field label="费用上限（USD，可选）" hint={usesCodex ? 'Codex CLI 无法使用美元费用上限' : '需要为所用模型填写输入 / 输出单价'}><input type="number" min="0.01" step="0.01" disabled={usesCodex} value={usesCodex ? '' : costLimit} onChange={(e) => setCostLimit(e.target.value)} placeholder="未设置" /></Field></div>{usesCodex && <Alert>本次分析包含 Codex CLI，金额预算（包括默认费用上限）已停用。请用最多模型请求数控制用量；Codex 订阅剩余额度和美元费用无法在此估算。</Alert>}{estimateValid && <div className="estimate"><div><span>分析时长</span><strong>{duration(estimate.duration_ms)}</strong></div><div><span>预计采样</span><strong>{estimate.estimated_frames.toLocaleString()} <small>帧</small></strong></div><div><span>预计请求</span><strong>≈ {estimate.estimated_requests.toLocaleString()} <small>次</small></strong></div><div><span>估算费用</span><strong>{estimate.cost_estimate === null ? '未知' : `$${estimate.cost_estimate.toFixed(3)}`}</strong></div><p>镜头数量、图像去重和实际 token 用量会影响最终请求及费用。</p>{estimate.warnings?.map((warning, index) => <p className="warning-text" key={index}>{warning}</p>)}</div>}{error && <Alert tone="error">{error}</Alert>}</div><footer className="modal-footer"><button type="submit" className="button secondary" disabled={!!busy || !payload.stages.length}>{busy === 'estimate' ? <Spinner label="正在估算" /> : <><Calculator size={16} />{estimateValid ? '重新估算' : '估算用量'}</>}</button><button type="button" className="button primary" onClick={startJob} disabled={!!busy || !estimateValid || missing.length > 0 || !payload.stages.length}>{busy === 'start' ? <Spinner label="正在提交" /> : <>开始分析 <ArrowRight size={17} /></>}</button></footer></form></Modal>;
}
