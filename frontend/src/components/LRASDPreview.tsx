import { useEffect, useRef, useState, type FormEvent } from 'react';
import { api, formatTime, message, post } from '../api';
import { type CharacterProfile } from '../types';
import { type AudioConfiguration, type LRASDPreviewResult } from '../identityTypes';
import { Alert, Field, Spinner } from './ui';

const previewStatus = {
  not_started: '尚未试跑', running: '正在本机试跑 LR-ASD', review_required: '试跑完成 · 待人工核对',
  no_tracks: '没有可评分的连续人物轨迹', unavailable: 'LR-ASD 暂不可用', failed: '短片段试跑失败',
};
const probability = (value?: number | null) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value.toFixed(3) : '不可用';
const frameTime = (ms: number) => formatTime(ms) + '.' + (Math.round(ms) % 1000).toString().padStart(3, '0');
const timingNames: Record<string, string> = { decode_ms: '片段解码', detection_tracking_ms: '人物检测与跟踪', asd_ms: 'LR-ASD 推理', matching_ms: '角色图像匹配', clip_encode_ms: '短片编码', detector_load_ms: '检测模型加载', feature_ms: '参考图特征计算', total_ms: '总耗时' };

export default function LRASDPreview({ endpoint, durationMs, initialStartMs, sourceAvailable, hasAudio, configuration, settingsDirty, disabled, profiles, onSeek, onRunningChange }: {
  endpoint: string; durationMs: number; initialStartMs: number; sourceAvailable: boolean; hasAudio: boolean;
  configuration: AudioConfiguration | null; settingsDirty: boolean; disabled: boolean; profiles: CharacterProfile[];
  onSeek: (ms: number) => void; onRunningChange: (running: boolean) => void;
}) {
  const [start, setStart] = useState(Math.min(initialStartMs, Math.max(0, durationMs - 640)) / 1000);
  const [end, setEnd] = useState(Math.min(durationMs, initialStartMs + 4000) / 1000);
  const [preview, setPreview] = useState<LRASDPreviewResult | null>(null);
  const [submitting, setSubmitting] = useState(false); const [error, setError] = useState('');
  const generation = useRef(0);
  const running = submitting || preview?.status === 'running';
  const settings = configuration?.settings;
  const configured = settings?.enabled && settings.sync_enabled && settings.sync_model_type === 'lrasd'
    && !!settings.sync_repository_path.trim() && !!settings.sync_checkpoint_path.trim()
    && !!settings.sync_python_path?.trim() && !!settings.sync_detector_path?.trim()
    && configuration?.availability?.sync?.available;
  const rangeMs = Math.round(end * 1000) - Math.round(start * 1000);
  const validRange = Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end * 1000 <= durationMs
    && rangeMs >= 640 && rangeMs <= 8000;
  const name = (id: string) => profiles.find((profile) => profile.id === id)?.name || id;

  useEffect(() => {
    const current = ++generation.current;
    setPreview(null); setSubmitting(false); setError('');
    setStart(Math.min(initialStartMs, Math.max(0, durationMs - 640)) / 1000);
    setEnd(Math.min(durationMs, initialStartMs + 4000) / 1000);
    void api<LRASDPreviewResult>(endpoint + '/lrasd-preview').then((result) => {
      if (generation.current === current && result.status) setPreview(result);
    }).catch((caught) => { if (generation.current === current) setError(message(caught)); });
    return () => { generation.current += 1; };
    // Keep these fields independent when the audio-analysis range changes.
  }, [endpoint]);
  useEffect(() => {
    onRunningChange(running);
    return () => onRunningChange(false);
  }, [running, onRunningChange]);
  useEffect(() => {
    if (preview?.status !== 'running') return;
    let cancelled = false; let timer: ReturnType<typeof setTimeout>;
    const current = generation.current;
    const poll = async () => {
      try {
        const result = await api<LRASDPreviewResult>(endpoint + '/lrasd-preview');
        if (cancelled || current !== generation.current) return;
        setPreview(result);
        if (result.status === 'running') timer = setTimeout(() => void poll(), 1000);
      } catch (caught) {
        if (!cancelled && current === generation.current) {
          setError(message(caught)); setPreview((previous) => previous ? { ...previous, status: 'failed' } : null);
        }
      }
    };
    timer = setTimeout(() => void poll(), 1000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [endpoint, preview?.status, preview?.job_id]);

  const run = async (event: FormEvent) => {
    event.preventDefault();
    if (running || disabled || !configured || settingsDirty || !validRange || !sourceAvailable || !hasAudio) return;
    generation.current += 1; const current = generation.current;
    setPreview(null); setSubmitting(true); setError('');
    try {
      const result = await post<LRASDPreviewResult>(endpoint + '/lrasd-preview', { start_ms: Math.round(start * 1000), end_ms: Math.round(end * 1000) });
      if (current === generation.current) setPreview(result);
    } catch (caught) { if (current === generation.current) setError(message(caught)); }
    finally { if (current === generation.current) setSubmitting(false); }
  };
  return <section className="audio-run lrasd-preview" aria-label="LR-ASD 短片段试跑">
    <h3>LR-ASD 短片段试跑</h3>
    <p className="helper">选择 0.64–8 秒原片，检测连续人物轨迹并计算发言概率。可直接试跑，无需先做说话人分段或确认人物区域。</p>
    {settingsDirty ? <Alert>配置已修改，请先保存音频配置，再试跑。</Alert> : !configured ? <Alert>请启用音频识别和音画复核，选择 LR-ASD，填写模型包、代码、Python 与人物检测模型路径，并保存音频配置。</Alert> : null}
    <form onSubmit={run}>
      <div className="form-grid">
        <Field label="试跑起点（秒）"><input type="number" min={0} max={durationMs / 1000} step={0.001} required value={start} disabled={running || disabled} onChange={(event) => setStart(Number(event.target.value))} /></Field>
        <Field label="试跑终点（秒）"><input type="number" min={0} max={durationMs / 1000} step={0.001} required value={end} disabled={running || disabled} onChange={(event) => setEnd(Number(event.target.value))} /></Field>
      </div>
      <button type="button" className="text-button" disabled={running || disabled} onClick={() => { const next = Math.min(initialStartMs, Math.max(0, durationMs - 640)); setStart(next / 1000); setEnd(Math.min(durationMs, next + 4000) / 1000); }}>使用当前音频起点，试跑 4 秒</button>
      {!validRange && <p className="warning-text">试跑时长须为 0.64–8 秒，并且位于原片范围内。</p>}
      <button type="submit" className="button primary" disabled={running || disabled || !configured || settingsDirty || !validRange || !sourceAvailable || !hasAudio}>{running ? <Spinner label="正在本机试跑 LR-ASD" /> : '试跑 LR-ASD 短片段'}</button>
    </form>
    {error && <Alert tone="error">{error}</Alert>}
    {preview && preview.status !== 'not_started' && <div aria-label="LR-ASD 试跑结果">
      <h4>{previewStatus[preview.status] || preview.status}</h4>
      {preview.reason && <Alert>{preview.reason}</Alert>}
      {preview.start_ms != null && preview.end_ms != null && <p className="helper">{formatTime(preview.start_ms)} — {formatTime(preview.end_ms)}{preview.cache_hit ? ' · 复用已保存的试跑结果' : ''}</p>}
      {(preview.status === 'review_required' || preview.status === 'no_tracks') && preview.clip_url && <div style={{ maxWidth: 480, width: '100%' }}>
        <p className="helper">本次试跑片段（从0开始，对应原片范围）</p>
        {preview.start_ms != null && preview.end_ms != null && <p className="helper">原片 {frameTime(preview.start_ms)} — {frameTime(preview.end_ms)}</p>}
        <div className="player-wrap"><video controls preload="metadata" src={preview.clip_url} aria-label="本次 LR-ASD 试跑片段播放器" /></div>
      </div>}
      {preview.timings && <div className="job-stage-metrics" aria-label="LR-ASD 试跑耗时">{Object.entries(preview.timings).filter(([stage, value]) => Object.prototype.hasOwnProperty.call(timingNames, stage) && Number.isFinite(value)).map(([stage, value]) => <div key={stage}><strong>{timingNames[stage]}</strong><span>{(value / 1000).toFixed(2)} 秒</span></div>)}</div>}
      {preview.tracks?.map((track, index) => <article className="identity-region" key={track.id} aria-label={'LR-ASD 人物轨迹 ' + track.id}>
        {track.thumbnail_url && <img src={track.thumbnail_url} alt={'轨迹 ' + track.id + ' 的原片人物裁剪'} loading="lazy" />}
        <div><div className="section-row"><strong>人物{index + 1}</strong><span className="tag">候选 · 待人工核对</span></div>
          <button className="record-time" type="button" onClick={() => onSeek(track.start_ms)}>{formatTime(track.start_ms)} — {formatTime(track.end_ms)}</button>
          <p>平均发言概率：{probability(track.active_speaker_probability)}</p>
          <p className="helper">发言概率描述这条人物轨迹的音画同步表现，不代表人物身份置信度。</p>
          {track.candidates?.length ? track.candidates.map((candidate) => <p className="helper" key={candidate.character_id}>角色候选：{name(candidate.character_id)} · CCIP 距离 {Number.isFinite(candidate.distance) ? candidate.distance.toFixed(4) : '不可用'}</p>) : <p className="helper">暂无可比较的角色参考图候选。</p>}
          {!!track.frame_probabilities?.length && <details><summary>逐帧发言概率</summary><p className="helper">{track.frame_probabilities.map((frame) => frameTime(frame.at_ms) + '：' + probability(frame.probability)).join(' · ')}</p></details>}
        </div>
      </article>)}
      <p className="helper">试跑结果保留为待核对候选。角色图像距离和发言概率各自提供线索，不会自动确认角色或说话人。</p>
    </div>}
  </section>;
}
