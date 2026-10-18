import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { api, formatTime, message, post, subtitleLink } from '../api';
import { type Asset, type CharacterLibrary, type CharacterProfile } from '../types';
import { type AudioConfiguration, type AudioRecord, type AudioResult, type AudioSettings, type AudioTrack, type BangumiReferenceCacheResult, type IdentityRegion, type IdentityReview, identitySourceLabel, identityStatusLabel } from '../identityTypes';
import { Alert, Field, Spinner } from './ui';
import LRASDPreview from './LRASDPreview';
import CharacterReferences from './CharacterReferences';

const audioStatusLabel = (status: string) => ({ disabled: '音频识别未启用', unavailable: '音频识别不可用', completed: '音频识别已完成', no_audio: '没有音轨', no_speech: '没有检测到语音', not_started: '尚未运行', pending: '尚未运行', empty: '尚无记录', partial: '部分完成', cancelled: '已取消，已完成的缓存保留', budget_exceeded: '处理范围超过音频时长预算' }[status] || status);
const timingLabel = (stage: string) => ({ extraction_s: '音频提取', diarization_s: '自动说话人分段（含加载）', transcription_s: '语音转写（含加载）', embedding_s: '声音特征（含加载）', matching_s: '声音匹配', total_s: '总耗时', model_load_s: '模型加载', sync_s: '音画复核', total_ms: '总耗时', total: '总耗时', extraction_ms: '音频提取', extract_ms: '音频提取', diarization_ms: '自动说话人分段', embedding_ms: '声音特征与匹配', matching_ms: '候选匹配', detection_ms: '人物检测', feature_ms: '视觉特征提取', model_load_ms: '模型加载', asr_ms: '语音转写', sync_ms: '音画复核' }[stage] || stage);
interface VisualSettings { enabled: boolean; model: string; top_k: number; calibration: Record<string, unknown> | null }

export default function AudioIdentity({ asset, initialAudio, onSeek, onSaved }: { asset: Asset; initialAudio?: string; onSeek: (ms: number) => void; onSaved: () => Promise<void> }) {
  const endpoint = `/assets/${encodeURIComponent(asset.id)}/audio`;
  const [configuration, setConfiguration] = useState<AudioConfiguration | null>(null);
  const [settings, setSettings] = useState<AudioSettings | null>(null);
  const [tracks, setTracks] = useState<AudioTrack[]>([]); const [trackLoaded, setTrackLoaded] = useState(false);
  const [result, setResult] = useState<AudioResult | null>(null);
  const [visual, setVisual] = useState<IdentityReview | null>(null); const [profiles, setProfiles] = useState<CharacterProfile[]>([]);
  const [visualSettings, setVisualSettings] = useState<VisualSettings | null>(null);
  const [calibration, setCalibration] = useState(''); const [frameIds, setFrameIds] = useState('');
  const [loading, setLoading] = useState(true); const [operation, setBusy] = useState('');
  const [previewRunning, setPreviewRunning] = useState(false);
  const busy = operation || (previewRunning ? 'preview' : '');
  const [bangumiCache, setBangumiCache] = useState<BangumiReferenceCacheResult | null>(null);
  const [referenceProfile, setReferenceProfile] = useState<CharacterProfile | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [errors, setErrors] = useState<string[]>([]); const [notice, setNotice] = useState('');
  const [startSeconds, setStartSeconds] = useState(0); const [endSeconds, setEndSeconds] = useState(Math.min(60, asset.duration_ms / 1000));
  const [recordLimit, setRecordLimit] = useState(30);
  const load = useCallback(async () => {
    const failures: string[] = [];
    const requests = await Promise.allSettled([
      api<AudioConfiguration>(`${endpoint}/settings`),
      api<{ tracks: AudioTrack[] }>(`${endpoint}/tracks`),
      api<AudioResult>(`${endpoint}/records`),
      api<CharacterLibrary>(`/assets/${encodeURIComponent(asset.id)}/characters`),
      api<IdentityReview>(`/assets/${encodeURIComponent(asset.id)}/identity/reviews`),
      api<VisualSettings>(`/assets/${encodeURIComponent(asset.id)}/identity/settings`),
    ]);
    const [configResult, trackResult, recordResult, profileResult, visualResult, visualConfigResult] = requests;
    if (configResult.status === 'fulfilled') { setConfiguration(configResult.value); setSettings(configResult.value.settings); } else failures.push(`音频配置：${message(configResult.reason)}`);
    if (trackResult.status === 'fulfilled') { setTracks(trackResult.value.tracks || []); setTrackLoaded(true); } else failures.push(`音轨读取：${message(trackResult.reason)}`);
    if (recordResult.status === 'fulfilled') setResult(recordResult.value); else failures.push(`声音记录：${message(recordResult.reason)}`);
    if (profileResult.status === 'fulfilled') setProfiles(profileResult.value.profiles || []); else failures.push(`角色档案：${message(profileResult.reason)}`);
    if (visualResult.status === 'fulfilled') setVisual(visualResult.value); else failures.push(`人物区域：${message(visualResult.reason)}`);
    if (visualConfigResult.status === 'fulfilled') { setVisualSettings(visualConfigResult.value); setCalibration(visualConfigResult.value.calibration ? JSON.stringify(visualConfigResult.value.calibration, null, 2) : ''); } else failures.push(`视觉身份配置：${message(visualConfigResult.reason)}`);
    setErrors(failures); setLoading(false);
  }, [endpoint, asset.id]);
  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (!initialAudio || !result?.records) return;
    const index = result.records.findIndex((record) => record.id === initialAudio);
    if (index >= 0) setRecordLimit((current) => Math.max(current, index + 1));
  }, [initialAudio, result]);
  useEffect(() => { if (initialAudio && !loading) document.getElementById(`audio-${initialAudio}`)?.scrollIntoView?.({ block: 'center' }); }, [initialAudio, loading, result, recordLimit]);
  const saveConfiguration = async () => {
    if (!settings) return;
    const saved = await api<AudioConfiguration>(`${endpoint}/settings`, { method: 'PUT', body: JSON.stringify(settings) });
    setConfiguration(saved); setSettings(saved.settings); return saved;
  };
  const save = async (event: FormEvent) => {
    event.preventDefault(); setBusy('save'); setErrors([]); setNotice('');
    try { await saveConfiguration(); setNotice('音频配置已保存，尚未启动识别。'); } catch (caught) { setErrors([message(caught)]); } finally { setBusy(''); }
  };
  const analyze = async (event: FormEvent) => {
    event.preventDefault(); if (!settings?.enabled || busy) return;
    if (endSeconds <= startSeconds) { setErrors(['结束时间必须晚于开始时间。']); return; }
    setBusy('analyze'); setErrors([]); setNotice('');
    try {
      await saveConfiguration();
      const analysis = await post<AudioResult>(`${endpoint}/analyze`, { start_ms: Math.round(startSeconds * 1000), end_ms: Math.round(endSeconds * 1000) });
      setResult(analysis); await onSaved();
      setNotice(analysis.status === 'completed' ? '声音记录已保存。请试听并核对候选说话人。' : audioStatusLabel(analysis.status));
    } catch (caught) { setErrors([message(caught)]); } finally { setBusy(''); setCancelling(false); }
  };
  const cancel = async () => {
    if (cancelling) return; setCancelling(true);
    try { const response = await post<{ status: string; reason?: string }>(`${endpoint}/cancel`); setNotice(response.reason || (response.status === 'cancelling' ? '正在请求取消，当前模型步骤结束后停止并保留缓存。' : '当前没有运行中的音频处理。')); }
    catch (caught) { setErrors([message(caught)]); setCancelling(false); }
  };
  const refreshRecords = async () => { setResult(await api<AudioResult>(`${endpoint}/records`)); await onSaved(); };
  const refreshVisual = async () => { setVisual(await api<IdentityReview>(`/assets/${encodeURIComponent(asset.id)}/identity/reviews`)); await onSaved(); };
  const saveVisual = async (run: boolean) => {
    if (!visualSettings || busy) return; setBusy('visual'); setErrors([]); setNotice('');
    try {
      const config = { ...visualSettings, calibration: calibration.trim() ? JSON.parse(calibration) : null };
      setVisualSettings(await api<VisualSettings>(`/assets/${encodeURIComponent(asset.id)}/identity/settings`, { method: 'PUT', body: JSON.stringify(config) }));
      if (run) { setVisual(await post<IdentityReview>(`/assets/${encodeURIComponent(asset.id)}/identity/run`, { frame_ids: frameIds.split(/[\s,，]+/).filter(Boolean) })); await onSaved(); }
      setNotice(run ? '已返回人物区域处理结果，请核对状态与候选。' : '视觉身份配置已保存。');
    } catch (caught) { setErrors([message(caught)]); } finally { setBusy(''); }
  };
  const setOption = <K extends keyof AudioSettings>(key: K, value: AudioSettings[K]) => setSettings((current) => current ? { ...current, [key]: value } : current);
  const refreshProfiles = async () => {
    const library = await api<CharacterLibrary>('/assets/' + encodeURIComponent(asset.id) + '/characters');
    setProfiles(library.profiles || []); await onSaved();
  };
  const cacheBangumi = async () => {
    if (busy) return;
    setBusy('bangumi'); setErrors([]); setNotice('');
    try {
      const cached = await post<BangumiReferenceCacheResult>('/assets/' + encodeURIComponent(asset.id) + '/references/cache-bangumi', { max_images: 20 });
      setBangumiCache(cached); await refreshProfiles();
      setNotice('Bangumi 角色图已处理：新增 ' + cached.imported + ' 张，已有 ' + cached.existing + ' 张，跳过 ' + cached.skipped + ' 项。新图保留为候选，请核对参考图库。');
    } catch (caught) { setErrors([message(caught)]); } finally { setBusy(''); }
  };
  const settingsDirty = !!settings && JSON.stringify(settings) !== JSON.stringify(configuration?.settings);
  if (loading) return <Spinner label="正在读取视听身份配置" />;
  return <div className="audio-identity"><div className="section-row"><h2>视听身份与证据</h2><button className="text-button" disabled={!!busy} onClick={() => void load()}>刷新身份记录</button></div>
    <p className="helper">先识别人物与说话人，再理解动作和对话。听到某个角色说话，不代表这个角色出现在当前画面。</p>
    {errors.map((error) => <Alert tone="error" key={error}>{error}</Alert>)}{notice && <Alert tone="info">{notice}</Alert>}
    {settings && <form onSubmit={save} className="audio-settings"><h3>当前视频的音频配置</h3><label className="check-line"><input type="checkbox" checked={settings.enabled} onChange={(event) => setOption('enabled', event.target.checked)} disabled={!!busy} />启用本机音频角色识别</label>
      {!settings.enabled && <p className="helper">未启用。原有字幕与画面流程仍可使用。</p>}
      {trackLoaded && !tracks.length && <Alert>当前视频没有可用音轨，不能执行声音识别。</Alert>}
      <div className="form-grid"><Field label="识别音轨"><select value={settings.stream_index ?? ''} onChange={(event) => setOption('stream_index', event.target.value === '' ? null : Number(event.target.value))} disabled={!!busy || !tracks.length}><option value="">自动选择默认音轨</option>{tracks.map((track) => <option key={track.index} value={track.index}>音轨 {track.index} · {[track.language, track.title].filter(Boolean).join(' · ') || '未标注语言'}{track.default ? ' · 默认' : ''} · 起点 {track.start_ms} ms</option>)}</select></Field><Field label="原音语言"><input value={settings.language} onChange={(event) => setOption('language', event.target.value)} required disabled={!!busy} placeholder="ja / zh / en" /></Field><Field label="配音版本" hint="声音库只匹配相同语言与配音版本。"><input value={settings.dub_version} onChange={(event) => setOption('dub_version', event.target.value)} required disabled={!!busy} placeholder="original" /></Field><Field label="每次音频处理预算（秒）" hint="超过预算会停止音频处理；扩大范围前请明确增加预算。"><input type="number" min={1} max={14400} step={1} value={(settings.max_duration_ms ?? 60000) / 1000} required disabled={!!busy} onChange={(event) => setOption('max_duration_ms', Math.round(Number(event.target.value) * 1000))} /></Field></div>
      <details className="identity-advanced"><summary>本机模型与音画复核配置</summary><p className="helper">填写已安装、具有使用权限的本机模型路径。此页面不会下载模型或上传原片。community-1 需要事先接受模型许可；ECAPA、ASR 与同步模型须分别配置。</p>
        <div className="form-grid"><Field label="说话人分段模型路径"><input value={settings.diarization_model_path} onChange={(event) => setOption('diarization_model_path', event.target.value)} disabled={!!busy} placeholder="pyannote community-1 本机目录" /></Field><Field label="ECAPA 声纹模型路径"><input value={settings.embedding_model_path} onChange={(event) => setOption('embedding_model_path', event.target.value)} disabled={!!busy} /></Field><Field label="运行设备"><input value={settings.device} onChange={(event) => setOption('device', event.target.value)} disabled={!!busy} placeholder="cpu / cuda" /></Field><Field label="声音匹配校准文件"><input value={settings.calibration_path} onChange={(event) => setOption('calibration_path', event.target.value)} disabled={!!busy} /></Field></div>
        <label className="check-line"><input type="checkbox" checked={settings.asr_enabled} onChange={(event) => setOption('asr_enabled', event.target.checked)} disabled={!!busy} />没有可靠字幕时使用本机 ASR</label><Field label="ASR 模型路径"><input value={settings.asr_model_path} onChange={(event) => setOption('asr_model_path', event.target.value)} disabled={!!busy} /></Field><p className="helper">译文字幕只做时间关联；不会把中文译文强制对齐到日语发音，一条字幕可能对应多位说话人。</p>
        <label className="check-line"><input type="checkbox" checked={settings.sync_enabled} onChange={(event) => setOption('sync_enabled', event.target.checked)} disabled={!!busy} />为歧义与冲突短片段启用音画复核</label>
        <div className="form-grid">
          <Field label="音画复核模型"><select value={settings.sync_model_type ?? 'animated_ad_lwtnet'} onChange={(event) => setOption('sync_model_type', event.target.value as AudioSettings['sync_model_type'])} disabled={!!busy}><option value="animated_ad_lwtnet">Animated_AD / LWTNet</option><option value="lrasd">LR-ASD</option></select></Field>
          <Field label="模型代码目录"><input value={settings.sync_repository_path} onChange={(event) => setOption('sync_repository_path', event.target.value)} disabled={!!busy} /></Field>
          <Field label={settings.sync_model_type === 'lrasd' ? 'LR-ASD 模型包目录' : '同步模型权重路径'}><input value={settings.sync_checkpoint_path} onChange={(event) => setOption('sync_checkpoint_path', event.target.value)} disabled={!!busy} /></Field>
          {settings.sync_model_type === 'lrasd' && <Field label="LR-ASD Python 路径" hint="填写已安装 LR-ASD 依赖的本机解释器。"><input value={settings.sync_python_path ?? ''} onChange={(event) => setOption('sync_python_path', event.target.value)} disabled={!!busy} /></Field>}
          {settings.sync_model_type === 'lrasd' && <Field label="LR-ASD 人物检测模型路径" hint="填写本机 person.onnx，用于短片段人物检测与跟踪。"><input value={settings.sync_detector_path ?? ''} onChange={(event) => setOption('sync_detector_path', event.target.value)} disabled={!!busy} /></Field>}
        </div>
        {settings.sync_model_type !== 'lrasd' && <label className="check-line"><input type="checkbox" checked={settings.sync_animation_verified} onChange={(event) => setOption('sync_animation_verified', event.target.checked)} disabled={!!busy} />已验证该权重适配动画及连续帧 / 音频预处理</label>}
        {settings.sync_model_type !== 'lrasd' && <label className="check-line"><input type="checkbox" checked={settings.sync_license_acknowledged ?? false} onChange={(event) => setOption('sync_license_acknowledged', event.target.checked)} disabled={!!busy} />已核实同步代码与权重许可证允许本次使用</label>}
        <p className="helper">LR-ASD 模型包须包含训练及独立验证记录；评分仍需人工核对。同步需要有真实时间对应关系的连续帧和音频。</p>
      </details>
      {configuration && <div className="identity-availability" aria-label="音频能力可用状态">{(['diarization', 'embedding', 'asr', 'sync'] as const).map((key) => { const item = configuration.availability?.[key]; return <p key={key}><strong>{{ diarization: '自动说话人分段', embedding: 'ECAPA 声音匹配', asr: 'ASR', sync: '音画同步复核' }[key]}</strong><span className={item?.available ? 'success-text' : 'warning-text'}>{item?.available ? '可用' : '未启用 / 未验证'}</span>{item?.reason && <small>{item.reason}</small>}</p>; })}</div>}
      {configuration?.availability?.reasons?.map((reason) => <Alert key={reason}>{reason}</Alert>)}
      <button type="submit" className="button secondary" disabled={!!busy}>{busy === 'save' ? <Spinner label="正在保存音频配置" /> : '保存音频配置'}</button>
    </form>}
    <form className="audio-run" onSubmit={analyze}><h3>运行一段原片音频</h3><div className="form-grid"><Field label="音频起点（秒）"><input type="number" min={0} max={asset.duration_ms / 1000} step={0.001} required value={startSeconds} onChange={(event) => setStartSeconds(Number(event.target.value))} disabled={!!busy} /></Field><Field label="音频终点（秒）"><input type="number" min={0} max={asset.duration_ms / 1000} step={0.001} required value={endSeconds} onChange={(event) => setEndSeconds(Number(event.target.value))} disabled={!!busy} /></Field></div><button className="button primary" type="submit" disabled={!!busy || !settings?.enabled || !tracks.length || !asset.source_available}>{busy === 'analyze' ? <Spinner label="正在本机分段与匹配，模型加载计入耗时" /> : '保存配置并识别这段音频'}</button>{busy === 'analyze' && <button type="button" className="button secondary" disabled={cancelling} onClick={() => void cancel()}>{cancelling ? '等待当前模型步骤结束…' : '取消音频识别'}</button>}<p className="helper">默认只处理前 60 秒；已保存特征可复用。耗时包括音频处理与模型加载。</p></form>
    {result && <section className="identity-audio-results" aria-label="声音识别结果"><h3>{audioStatusLabel(result.status)}</h3>{result.degraded?.map((reason) => <Alert key={reason}>降级原因：{reason}</Alert>)}{result.timings && <div className="job-stage-metrics" aria-label="音频阶段耗时">{Object.entries(result.timings).map(([stage, value]) => <div key={stage}><strong>{timingLabel(stage)}</strong><span>{(stage.endsWith('_s') ? value : value / 1000).toFixed(2)} 秒</span></div>)}</div>}{result.cache_hits != null && <p className="helper">缓存复用 {result.cache_hits} 次</p>}{!result.records?.length ? <p className="helper">尚无可核对的说话人记录。</p> : result.records.slice(0, recordLimit).map((record) => <SpeakerRecord key={record.id} record={record} profiles={profiles} endpoint={endpoint} onSeek={onSeek} onSaved={refreshRecords} />)}{(result.records?.length || 0) > recordLimit && <button className="button secondary" onClick={() => setRecordLimit(recordLimit + 30)}>显示更多声音记录</button>}</section>}
    <LRASDPreview endpoint={endpoint} durationMs={asset.duration_ms} initialStartMs={Math.round(startSeconds * 1000)} sourceAvailable={asset.source_available} hasAudio={tracks.length > 0} configuration={configuration} settingsDirty={settingsDirty} disabled={!!operation} profiles={profiles} onSeek={onSeek} onRunningChange={setPreviewRunning} />
    <section className="identity-visual-results" aria-label="视觉身份候选"><h3>人物区域与视觉身份</h3>
      <div className="audio-settings"><h4>Bangumi 角色参考图</h4><p className="helper">把当前角色档案中已有的 Bangumi 图片下载到本机候选图库。请核对图片中的角色，确认后再加入可信库。</p>
        <button type="button" className="button secondary" disabled={!!busy || !profiles.some((profile) => profile.external_source?.image_url)} onClick={() => void cacheBangumi()}>{busy === 'bangumi' ? <Spinner label="正在缓存 Bangumi 角色图" /> : '缓存 Bangumi 角色图'}</button>
        {!profiles.some((profile) => profile.external_source?.image_url) && <p className="helper">先在作品资料中关联 Bangumi 条目并读取角色档案。</p>}
        {bangumiCache?.errors?.map((item, index) => <Alert key={item.character_id + '-' + index}>{profiles.find((profile) => profile.id === item.character_id)?.name || item.character_id}：{item.reason}</Alert>)}
        {bangumiCache?.references?.map((reference) => { const profile = profiles.find((item) => item.id === reference.character_id); return <article className="identity-region" key={reference.id}>
          <img src={reference.image_url} alt={(profile?.name || reference.character_id) + ' 的本机角色参考图'} loading="lazy" />
          <div><strong>{profile?.name || reference.character_id}</strong><p className="helper">{identityStatusLabel(reference.status)} · {identitySourceLabel(reference.source.kind)}</p>
            <button type="button" className="button small secondary" disabled={!!busy || !profile} onClick={() => { if (profile) setReferenceProfile(profile); }}>核对{profile?.name || '角色'}的参考图</button>
          </div>
        </article>; })}
      </div>
      {visualSettings && <form className="audio-settings" onSubmit={(event) => { event.preventDefault(); void saveVisual(false); }}><label className="check-line"><input type="checkbox" checked={visualSettings.enabled} disabled={!!busy} onChange={(event) => setVisualSettings({ ...visualSettings, enabled: event.target.checked })} />启用 imgutils 人物检测与 CCIP 匹配</label><div className="form-grid"><Field label="CCIP 模型"><select value={visualSettings.model} disabled={!!busy} onChange={(event) => setVisualSettings({ ...visualSettings, model: event.target.value })}>{['ccip-caformer-24-randaug-pruned', 'ccip-caformer-6-randaug-pruned_fp32', 'ccip-caformer-5_fp32'].map((model) => <option key={model}>{model}</option>)}</select></Field><Field label="身份候选 Top-K" hint="先搜索作品允许的完整角色库，再选候选。"><input type="number" min={1} max={20} required value={visualSettings.top_k} disabled={!!busy} onChange={(event) => setVisualSettings({ ...visualSettings, top_k: Number(event.target.value) })} /></Field></div><details className="identity-advanced"><summary>标注集校准记录与局部重识别</summary><Field label="视觉匹配校准记录（JSON）" hint="留空只生成候选；记录须包括模型、数据集、评估 ID 与校准阈值。"><textarea rows={5} value={calibration} onChange={(event) => setCalibration(event.target.value)} disabled={!!busy} /></Field><Field label="仅重识别这些证据帧 ID" hint="用空格或逗号分隔。只运行人物检测与匹配，不重跑整集剧情。"><textarea rows={2} value={frameIds} onChange={(event) => setFrameIds(event.target.value)} disabled={!!busy} /></Field><button type="button" className="button secondary" disabled={!!busy || !visualSettings.enabled || !frameIds.trim() || !asset.source_available} onClick={() => void saveVisual(true)}>保存并重识别所选人物帧</button></details><p className="helper">只使用预先安装的模型；缺少权重时保留明确降级原因。CCIP 距离与候选差距不等于身份置信度。</p><button type="submit" className="button secondary" disabled={!!busy}>保存视觉身份配置</button></form>}
      {visual?.status && <p className="helper">人物匹配状态：{visual.status}</p>}{visual?.metrics && <div className="job-stage-metrics" aria-label="视觉身份阶段耗时">{Object.entries(visual.metrics).filter(([key]) => key.endsWith('_ms')).map(([key, value]) => <div key={key}><strong>{timingLabel(key)}</strong><span>{(value / 1000).toFixed(2)} 秒</span></div>)}{visual.metrics.feature_cache_hits != null && <div><strong>特征缓存复用</strong><span>{visual.metrics.feature_cache_hits} 次</span></div>}</div>}{visual?.degraded_reason && <Alert>视觉身份降级：{visual.degraded_reason}</Alert>}{!visual?.regions?.length ? <p className="helper">尚无独立人物区域。配置本机视觉匹配并执行画面分析后，可在这里核对 P1 / P2 与角色的映射。</p> : visual.regions.slice(0, recordLimit).map((region) => <RegionReview key={region.id} region={region} profiles={profiles} assetId={asset.id} onSeek={onSeek} onSaved={refreshVisual} />)}{(visual?.regions?.length || 0) > recordLimit && <button className="button secondary" onClick={() => setRecordLimit(recordLimit + 30)}>显示更多人物区域</button>}</section>
    {referenceProfile && <CharacterReferences assetId={asset.id} profile={referenceProfile} onClose={() => setReferenceProfile(null)} onSaved={refreshProfiles} />}
  </div>;
}

function SpeakerRecord({ record, profiles, endpoint, onSeek, onSaved }: { record: AudioRecord; profiles: CharacterProfile[]; endpoint: string; onSeek: (ms: number) => void; onSaved: () => Promise<void> }) {
  const [selected, setSelected] = useState(record.character_id || record.candidate_character_id || '');
  const [reviewed, setReviewed] = useState(false); const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const name = (id?: string | null) => profiles.find((profile) => profile.id === id)?.name || id || '未知说话人';
  const confirm = async () => { setBusy(true); setError(''); try { await post(`${endpoint}/records/${encodeURIComponent(record.id)}/confirm`, { character_id: selected || null, note: selected ? '用户试听并核对说话人' : '用户撤销关联，保留未知说话人' }); await onSaved(); setReviewed(false); } catch (caught) { setError(message(caught)); } finally { setBusy(false); } };
  return <article id={`audio-${record.id}`} className="identity-record" aria-label={`声音记录 ${record.id}`}><div className="section-row"><button className="record-time" onClick={() => onSeek(record.start_ms)}>{formatTime(record.start_ms)} — {formatTime(record.end_ms)}</button><span className="tag">{identityStatusLabel(record.association_status)}</span></div><h4>{record.association_status === 'confirmed' ? name(record.character_id) : record.candidate_character_id ? `候选：${name(record.candidate_character_id)}` : '未知说话人'}</h4>{record.overlap && <Alert>重叠发言：同一时间可能有多个说话人，字幕不能单独分配给某一个人。</Alert>}<p>{record.text || '无可用字幕 / 转写文字'}</p><audio controls preload="none" aria-label={`试听声音记录 ${record.id}`} src={`/api${endpoint}/records/${encodeURIComponent(record.id)}/clip`} /><p className="helper">音轨 {record.audio_stream_index} · {record.language} · {record.dub_version} · 关联来源：{identitySourceLabel(record.association_source)}</p><p className="helper identity-scope">作用域内说话人：<code>{record.speaker_id}</code><br />声音记录 ID：<code>{record.id}</code></p>{record.subtitle_ids?.length > 0 && <div className="scene-fact-sources">相关字幕（仅时间关联）：{record.subtitle_ids.map((id, index) => <a key={id} href={subtitleLink(record.asset_id, id, record.start_ms)}>字幕 {index + 1}</a>)}</div>}{record.frame_ids?.length > 0 && <div className="scene-fact-sources">同期原帧（不证明口型 / 出场身份）：{record.frame_ids.map((id, index) => <a key={id} href={`/api/assets/${encodeURIComponent(record.asset_id)}/frames/${encodeURIComponent(id)}`} target="_blank" rel="noreferrer">原帧 {index + 1}</a>)}</div>}
    {record.candidates?.length > 0 && <details><summary>声音候选与匹配距离</summary>{record.candidates.map((candidate) => <p className="helper" key={candidate.character_id}>{name(candidate.character_id)} · 距离 {candidate.distance.toFixed(4)}</p>)}<p className="helper">距离不是身份置信度；未校准或有冲突时保留候选 / 未知。</p></details>}
    <Field label={`确认声音记录 ${record.id} 的角色`}><select value={selected} onChange={(event) => { setSelected(event.target.value); setReviewed(false); }} disabled={busy}><option value="">未知 / 撤销角色关联</option>{profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}</select></Field><label className="check-line"><input type="checkbox" checked={reviewed} onChange={(event) => setReviewed(event.target.checked)} disabled={busy} />我已试听并核对这段声音的说话人</label><button className="button small secondary" disabled={busy || !reviewed} onClick={() => void confirm()}>{busy ? '正在保存…' : selected ? '确认说话人' : '保留未知 / 撤销关联'}</button>{error && <Alert tone="error">{error}</Alert>}<details><summary>音频引用与模型版本</summary><pre className="json-report">{JSON.stringify({ audio_ref: record.audio_ref, model_versions: record.model_versions }, null, 2)}</pre></details>
  </article>;
}

function RegionReview({ region, profiles, assetId, onSeek, onSaved }: { region: IdentityRegion; profiles: CharacterProfile[]; assetId: string; onSeek: (ms: number) => void; onSaved: () => Promise<void> }) {
  const [selected, setSelected] = useState(region.match.character_id || region.match.candidate_character_id || '');
  const [reviewed, setReviewed] = useState(false); const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const name = (id?: string | null) => profiles.find((profile) => profile.id === id)?.name || id || '未知人物';
  const confirm = async () => { setBusy(true); setError(''); try { await post(`/assets/${encodeURIComponent(assetId)}/identity/reviews/${encodeURIComponent(region.id)}`, { status: selected ? 'confirmed' : 'unknown', character_id: selected || null, reason: selected ? '用户核对人物区域身份' : '用户撤销身份，保留未知人物' }); await onSaved(); setReviewed(false); } catch (caught) { setError(message(caught)); } finally { setBusy(false); } };
  return <article className="identity-region" aria-label={`人物区域 ${region.local_id} ${region.id}`}><img src={region.crop_url} alt={`${region.local_id} 的独立人物裁剪`} loading="lazy" /><div><div className="section-row"><strong>{region.local_id}</strong><button className="record-time" onClick={() => onSeek(region.at_ms)}>{formatTime(region.at_ms)}</button></div><p>{identityStatusLabel(region.match.status)}：{region.match.status === 'confirmed' ? name(region.match.character_id) : region.match.candidate_character_id ? `候选 ${name(region.match.candidate_character_id)}` : '未知人物'}</p><p className="helper">关联来源：{identitySourceLabel(region.match.source)}{region.match.distance != null ? ` · CCIP 距离 ${region.match.distance.toFixed(4)}` : ''}{region.match.margin != null ? ` · 候选差距 ${region.match.margin.toFixed(4)}` : ''}</p>{region.quality.reasons?.length ? <p className="warning-text">{region.quality.reasons.join('；')}</p> : null}<Field label={`确认区域 ${region.local_id} ${region.id} 的角色`}><select value={selected} disabled={busy} onChange={(event) => { setSelected(event.target.value); setReviewed(false); }}><option value="">未知 / 撤销角色关联</option>{profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.name}</option>)}</select></Field><label className="check-line"><input type="checkbox" checked={reviewed} disabled={busy} onChange={(event) => setReviewed(event.target.checked)} />已核对当前人物区域</label><button className="button small secondary" disabled={busy || !reviewed} onClick={() => void confirm()}>{busy ? '正在保存…' : selected ? '确认人物身份' : '保留未知人物'}</button>{error && <Alert tone="error">{error}</Alert>}<p className="helper">局部编号仅属于这条画面证据；确认映射不会自动把裁剪加入可信图库。</p></div></article>;
}
