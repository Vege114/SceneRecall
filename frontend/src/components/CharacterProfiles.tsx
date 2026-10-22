import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { ArrowRight, Check, ExternalLink, GitMerge, Images, Pencil, Play, RefreshCw, ScanLine, Search, Trash2, UserRound, UsersRound } from 'lucide-react';
import { api, ApiError, assetLink, duration, formatTime, message, post } from '../api';
import { useApp } from '../context';
import { DEFAULTS, executionDefaults, taskModelLabel, type Asset, type CharacterLibrary, type CharacterProfile, type Estimate, type Job, type KeyframePlan, type TaskModelRole } from '../types';
import { Alert, Empty, Field, Modal, Spinner } from './ui';
import TaskModels, { useTaskModels } from './TaskModels';
import TaskExecution, { saveExecutionDefaults } from './TaskExecution';
import { assetTaskLabel, settleConcurrent } from '../taskHelpers';
import { StageEstimates } from './SceneAnalysis';
import CharacterReferences, { CreateCharacter } from './CharacterReferences';
import CharacterConfirmation from './CharacterConfirmation';

const characterPath = (assetId: string, characterId?: string) => `/assets/${encodeURIComponent(assetId)}/characters${characterId ? `/${encodeURIComponent(characterId)}` : ''}`;
const frameUrl = (assetId: string, frameId: string) => `/api/assets/${encodeURIComponent(assetId)}/frames/${encodeURIComponent(frameId)}`;

function CharacterNames({ profile }: { profile: CharacterProfile }) {
  const sourcedName = (name: string | undefined, languages: string[]) => profile.name_variants?.some((variant) => variant.name === name && variant.source === 'bangumi' && languages.includes(variant.language));
  const convertedChinese = !sourcedName(profile.name_cn, ['zh-Hans', 'zh-Hant']) && profile.name_variants?.some((variant) => variant.name === profile.name_cn && variant.source === 'opencc');
  const transcribedEnglish = !sourcedName(profile.name_en, ['en', 'ja-Latn']) && profile.name_variants?.some((variant) => variant.name === profile.name_en && variant.source === 'kana_transliteration');
  const names = [[convertedChinese ? '中文（汉字转换）' : '中文', profile.name_cn], ['日文', profile.name_ja], [transcribedEnglish ? '罗马字（按假名转写）' : '英文 / 罗马字', profile.name_en]].filter((entry) => entry[1]);
  const aliases = profile.aliases.filter((alias) => ![profile.name, profile.name_cn, profile.name_ja, profile.name_en].includes(alias));
  return <>{names.length > 0 && <dl className="character-names" aria-label={`${profile.name}的多语言姓名`}>{names.map(([language, name]) => <div key={language}><dt>{language}</dt><dd>{name}</dd></div>)}</dl>}{aliases.length > 0 && <p className="character-aliases">又名 {aliases.join('、')}</p>}</>;
}

function ExternalCharacterReference({ profile }: { profile: CharacterProfile }) {
  const source = profile.external_source;
  if (!source) return null;
  return <div className="character-external-reference"><a href={source.url} target="_blank" rel="noreferrer"><ExternalLink size={12} />Bangumi 角色资料</a>{source.image_url && <a className="character-reference-image" href={source.image_url} target="_blank" rel="noreferrer" aria-label={`查看${profile.name}的 Bangumi 参考图`}><img src={source.image_url} alt={`${profile.name}的 Bangumi 公开参考图`} loading="lazy" referrerPolicy="no-referrer" /><span>公开参考图</span></a>}<p>现成资料供身份识别参考，视频出场证据会在分析后补充。</p></div>;
}

function Portrait({ assetId, revision, profile, onOpen }: { assetId: string; revision: number; profile: CharacterProfile; onOpen: (url: string) => void }) {
  const [failed, setFailed] = useState(false);
  const representative = profile.representative;
  const imageVersion = new URLSearchParams({ revision: String(revision), representative: JSON.stringify(representative || null), updated_at: profile.updated_at || '' });
  const url = `/api${characterPath(assetId, profile.id)}/portrait?${imageVersion}`;
  useEffect(() => { setFailed(false); }, [url]);
  return <button type="button" className="character-portrait" aria-label={`查看${profile.name}的当前形象原图`} disabled={!representative} onClick={() => { if (representative) onOpen(frameUrl(representative.asset_id, representative.frame_id)); }}>{representative && !failed ? <img src={url} alt={`${profile.name}的当前形象`} loading="lazy" onError={() => setFailed(true)} /> : <UserRound size={34} strokeWidth={1.3} />}<span>{representative ? '当前形象 · 点击查看原图' : '暂无形象图片'}</span></button>;
}

function CharacterEditModal({ assetId, profile, library, mode, onClose, onSaved }: { assetId: string; profile: CharacterProfile; library: CharacterLibrary; mode: 'edit' | 'merge' | 'delete'; onClose: () => void; onSaved: () => Promise<void> }) {
  const { notify } = useApp();
  const [name, setName] = useState(profile.name);
  const [aliases, setAliases] = useState(profile.aliases.join('、'));
  const [description, setDescription] = useState(profile.description || '');
  const [appearance, setAppearance] = useState((profile.appearance || []).join('\n'));
  const [notes, setNotes] = useState(profile.notes || '');
  const [targetId, setTargetId] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [stale, setStale] = useState(false);
  const [expectedRevision] = useState(library.revision);
  const target = library.profiles.find((candidate) => candidate.id === targetId);
  const aliasList = [...new Set(aliases.split(/[,，、\n]/).map((value) => value.trim()).filter(Boolean))];
  const traits = [...new Set(appearance.split('\n').map((value) => value.trim()).filter(Boolean))];
  const original = { name: profile.name, aliases: profile.aliases, description: profile.description || '', appearance: profile.appearance || [], notes: profile.notes || '' };
  const values = { name: name.trim(), aliases: aliasList, description: description.trim(), appearance: traits, notes: notes.trim() };
  const changes = Object.fromEntries(Object.entries(values).filter(([field, value]) => JSON.stringify(value) !== JSON.stringify(original[field as keyof typeof original])));
  const hasChanges = Object.keys(changes).length > 0;
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (busy || stale || (mode === 'edit' && (!name.trim() || !hasChanges)) || (mode === 'merge' && !targetId)) return;
    if (mode === 'edit' && (aliasList.length > 100 || aliasList.some((alias) => alias.length > 200))) { setError('最多保留 100 个别名，每个不超过 200 字。'); return; }
    if (mode === 'edit' && (traits.length > 64 || traits.some((trait) => trait.length > 2000))) { setError('最多保留 64 条外观特征，每条不超过 2000 字。'); return; }
    setBusy(true); setError('');
    try {
      const path = characterPath(assetId, profile.id);
      if (mode === 'edit') await api(path, { method: 'PATCH', body: JSON.stringify({ ...changes, expected_revision: expectedRevision }) });
      else if (mode === 'delete') await api(`${path}?expected_revision=${expectedRevision}`, { method: 'DELETE' });
      else await post(`${path}/merge`, { target_id: targetId, expected_revision: expectedRevision });
      await onSaved();
      notify(mode === 'edit' ? '角色档案已保存，人物映射与检索展示已更新' : mode === 'merge' ? `已合并到「${target?.name}」` : '角色档案已删除');
      onClose();
    } catch (caught) {
      setError(message(caught));
      if (caught instanceof ApiError && caught.status === 409) setStale(true);
    } finally { setBusy(false); }
  };
  const title = mode === 'edit' ? '编辑角色档案' : mode === 'merge' ? '合并角色档案' : '删除角色档案';
  return <Modal title={title} subtitle={profile.name} onClose={() => { if (!busy) onClose(); }} wide={mode !== 'delete'}><form onSubmit={submit}><div className="modal-body">
    {mode === 'edit' && <><div className="form-grid"><Field label="角色姓名"><input value={name} onChange={(event) => setName(event.target.value)} maxLength={200} required disabled={busy} /></Field><Field label="别名" hint="用顿号或逗号分隔，例如：小夏、夏同学"><input value={aliases} onChange={(event) => setAliases(event.target.value)} disabled={busy} /></Field></div><Field label="外貌与识别特征" hint="描述发型、面部、服装等特征；不同造型可以一并记录。"><textarea rows={5} maxLength={10000} value={description} onChange={(event) => setDescription(event.target.value)} disabled={busy} /></Field><Field label="已积累的外观特征" hint="每行一条，可删除误识别的特征；最多 64 条。"><textarea rows={3} value={appearance} onChange={(event) => setAppearance(event.target.value)} disabled={busy} /></Field><Field label="角色补充说明" hint="填写身份、关系或容易混淆的角色，供后续 AI 识别参考。"><textarea rows={4} maxLength={100000} value={notes} onChange={(event) => setNotes(event.target.value)} disabled={busy} /></Field><Alert tone="info">保存后优先更新角色映射、展示与检索。原始描述仍保留；只有需要重新理解动作或剧情时，才手动分析相关片段。</Alert></>}
    {mode === 'merge' && <><p className="character-modal-copy">如果两份档案记录的是同一个角色，可以把「{profile.name}」并入已有档案。所有出场记录将归入保留的角色。</p><Field label="保留哪一份角色档案"><select value={targetId} onChange={(event) => setTargetId(event.target.value)} required disabled={busy}><option value="">选择要保留的角色</option>{library.profiles.filter((candidate) => candidate.id !== profile.id).map((candidate) => <option key={candidate.id} value={candidate.id}>{candidate.name} · {candidate.appearance_count} 次出场</option>)}</select></Field>{target && <div className="character-merge-preview"><div><strong>{profile.name}</strong><span>{profile.appearance_count} 次出场</span></div><ArrowRight size={20} /><div><strong>{target.name}</strong><span>保留此档案</span></div></div>}<Alert>合并会移除「{profile.name}」这份独立档案，并汇集到所选角色。请核对两者确为同一人。</Alert></>}
    {mode === 'delete' && <><p className="character-modal-copy">删除「{profile.name}」及其人物关联？这会影响同一作品中引用这份档案的所有剧集。</p><Alert>原始画面记录和证据图片仍会保留。已删除的档案不会因后续识别或读取 Bangumi 自动恢复。</Alert></>}
    {error && <Alert tone="error">{error}</Alert>}{stale && <button type="button" className="button secondary" onClick={async () => { setBusy(true); try { await onSaved(); onClose(); } finally { setBusy(false); } }} disabled={busy}>刷新档案后重新操作</button>}
  </div><footer className="modal-footer"><button type="button" className="button ghost" onClick={onClose} disabled={busy}>取消</button><button type="submit" className={`button ${mode === 'delete' ? 'danger' : 'primary'}`} disabled={busy || stale || (mode === 'edit' && (!name.trim() || !hasChanges)) || (mode === 'merge' && !targetId)}>{busy ? <Spinner label="保存中" /> : mode === 'edit' ? <><Check size={16} />保存档案</> : mode === 'merge' ? <><GitMerge size={16} />确认合并</> : <><Trash2 size={16} />确认删除</>}</button></footer></form></Modal>;
}

function ReanalyzeModal({ asset, library, onClose, onReload }: { asset: Asset; library: CharacterLibrary; onClose: () => void; onReload: () => Promise<void> }) {
  const { bootstrap, refresh, notify, jobs } = useApp();
  const defaults = { ...DEFAULTS, ...bootstrap.settings.defaults };
  const model = bootstrap.provider_profiles.find((profile) => profile.id === bootstrap.settings.bindings.vision);
  const preparation = bootstrap.provider_profiles.find((profile) => profile.id === bootstrap.settings.bindings.preparation);
  const analysisStrategy = defaults.analysis_strategy || (preparation ? 'scene-chunks-v1' : 'content-keyframes-v1');
  const usesPreparation = analysisStrategy === 'scene-chunks-v1';
  const modelRoles: TaskModelRole[] = usesPreparation ? ['preparation', 'vision'] : ['vision'];
  const taskModels = useTaskModels(modelRoles);
  const modelOptions = taskModels.overrides ? { model_overrides: taskModels.overrides } : {};
  const usesCodex = model?.provider_type === 'codex_cli' || (usesPreparation && preparation?.provider_type === 'codex_cli');
  const [scope, setScope] = useState<'asset' | 'series'>('asset');
  const [requestLimit, setRequestLimit] = useState(defaults.max_requests);
  const [costLimit, setCostLimit] = useState(defaults.max_cost?.toString() || '');
  const [execution, setExecution] = useState(() => executionDefaults(defaults.execution));
  const [saveDefault, setSaveDefault] = useState(false);
  const [estimates, setEstimates] = useState<Estimate[] | null>(null);
  const [estimateKey, setEstimateKey] = useState('');
  const [busy, setBusy] = useState<'scan' | 'estimate' | 'start' | null>(null);
  const [error, setError] = useState('');
  const [stale, setStale] = useState(false);
  const [plans, setPlans] = useState<Record<string, KeyframePlan>>({});
  const [scanningPlans, setScanningPlans] = useState<Record<string, KeyframePlan>>({});
  const [cancellingScan, setCancellingScan] = useState('');
  const form = useRef<HTMLFormElement>(null);
  const alive = useRef(true);
  const submitted = useRef(false);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const assets = scope === 'series' ? library.assets : [library.assets.find((candidate) => candidate.id === asset.id) || asset];
  const unavailable = assets.filter((candidate) => !candidate.source_available);
  const active = jobs.some((job) => assets.some((candidate) => candidate.id === job.asset_id) && ['queued', 'pending', 'running', 'pausing', 'cancelling'].includes(job.status));
  const maxCost = usesCodex || !costLimit ? null : Number(costLimit);
  const key = JSON.stringify([scope, requestLimit, maxCost, library.revision, defaults, model?.id, preparation?.id, preparation?.max_concurrency, taskModels.overrides, taskModels.sourceKey, execution]);
  const validEstimate = estimates && estimateKey === key;
  const blocked = !model || (usesPreparation && !preparation) || unavailable.length > 0 || active || stale;
  const preparePlans = async () => {
    const ready = { ...plans };
    const outcomes = await settleConcurrent(assets, async (candidate) => {
      if (!alive.current) return;
      let plan = ready[candidate.id];
      if (plan?.status !== 'completed') {
        plan = await post<KeyframePlan>('/keyframe-plans', { asset_id: candidate.id, start_ms: 0, end_ms: null });
        if (!alive.current) return;
        setScanningPlans((current) => ({ ...current, [candidate.id]: plan }));
        while (['queued', 'pending', 'running', 'cancelling', 'pausing'].includes(plan.status) && alive.current) {
          await new Promise((resolve) => window.setTimeout(resolve, 1000));
          if (!alive.current) return;
          plan = await api<KeyframePlan>(`/keyframe-plans/${encodeURIComponent(plan.plan_id || plan.id)}`);
          setScanningPlans((current) => ({ ...current, [candidate.id]: plan }));
        }
        if (!alive.current) return;
        if (plan.status === 'completed' && !plan.frames) plan = await api<KeyframePlan>(`/keyframe-plans/${encodeURIComponent(plan.plan_id || plan.id)}`);
        if (plan.status !== 'completed') throw new Error(plan.error || '关键帧预扫描未完成，请重试。');
        ready[candidate.id] = plan;
        setPlans((current) => ({ ...current, [candidate.id]: plan }));
      }
    });
    const failures = outcomes.flatMap((outcome, index) => outcome.status === 'rejected' ? [`${assetTaskLabel(assets[index])}：${message(outcome.reason)}`] : []);
    if (failures.length) throw new Error(failures.join('；'));
    if (!alive.current) throw new Error('已关闭重新识别窗口。');
    return ready;
  };
  const scan = async () => {
    if (busy || unavailable.length || active || stale) return;
    setBusy('scan'); setError('');
    try { await preparePlans(); }
    catch (caught) { if (alive.current) setError(message(caught)); }
    finally { if (alive.current) { setBusy(null); setScanningPlans({}); } }
  };
  const estimate = async (event: FormEvent) => {
    event.preventDefault();
    if (busy || blocked) return;
    setBusy('estimate'); setError('');
    try {
      const ready = await preparePlans();
      const results = await settleConcurrent(assets, (candidate) => post<Estimate>('/jobs/estimate', { ...modelOptions, execution: { ...execution }, analysis_strategy: analysisStrategy, asset_id: candidate.id, keyframe_plan_id: ready[candidate.id].plan_id || ready[candidate.id].id, stages: ['vision'], start_ms: 0, end_ms: null, force: true, subtitle_fps: defaults.subtitle_fps, subtitle_crop: defaults.subtitle_crop, max_requests: requestLimit, max_cost: maxCost }));
      const failure = results.find((result) => result.status === 'rejected');
      if (failure?.status === 'rejected') throw failure.reason;
      setEstimates(results.flatMap((result) => result.status === 'fulfilled' ? [result.value] : [])); setEstimateKey(key);
    } catch (caught) { setError(message(caught)); } finally { setBusy(null); setScanningPlans({}); }
  };
  const start = async () => {
    if (busy || submitted.current || blocked || !form.current?.reportValidity()) return;
    setBusy('start'); setError('');
    let defaultsSaved = false;
    try {
      const ready = await preparePlans();
      if (saveDefault && usesPreparation) { await saveExecutionDefaults(execution, ['transcript', 'preparation']); defaultsSaved = true; }
      const result = await post<{ jobs: Job[]; revision: number }>(`${characterPath(asset.id)}/reanalyze`, { ...modelOptions, execution: { ...execution }, scope, max_requests: requestLimit, max_cost: maxCost, expected_revision: library.revision, keyframe_plan_ids: Object.fromEntries(assets.map((candidate) => [candidate.id, ready[candidate.id].plan_id || ready[candidate.id].id])) });
      submitted.current = true;
      let refreshError = '';
      try { await refresh(); } catch (error) { refreshError = message(error); }
      notify(`已按新版角色档案创建 ${result.jobs.length} 个重新识别任务`); onClose();
      if (refreshError) notify(`任务已创建，但列表刷新失败：${refreshError}。请在任务队列刷新。`, 'error');
    } catch (caught) {
      if (defaultsSaved) { try { await refresh(); } catch { /* Keep the creation error visible. */ } }
      setError(defaultsSaved ? `默认执行方式已保存，但任务创建失败：${message(caught)}。请重试创建任务。` : message(caught));
      if (caught instanceof ApiError && caught.status === 409) setStale(true);
    } finally { setBusy(null); setScanningPlans({}); }
  };
  const totalRequests = estimates?.reduce((sum, item) => sum + item.estimated_requests, 0) || 0;
  const totalCost = estimates?.every((item) => item.cost_estimate != null) ? estimates.reduce((sum, item) => sum + (item.cost_estimate || 0), 0) : null;
  return <Modal title="按新版档案重新识别" subtitle={`使用当前角色档案 · 修订 ${library.revision}`} onClose={() => { if (!busy) onClose(); }} wide><form ref={form} onSubmit={estimate}><div className="modal-body">
    <div className="character-reanalyze-intro"><RefreshCw size={24} /><div><strong>把你确认的人物，交回给 AI</strong><p>从头重新分析所选视频的画面，优先匹配这些角色档案，并更新人物关联、动作与场景理解。原有分析版本和人工笔记保留。</p></div></div>
    <Field label="重新识别范围"><select value={scope} onChange={(event) => setScope(event.target.value as 'asset' | 'series')} disabled={Boolean(busy)}><option value="asset">当前作品 · {assetTaskLabel(asset)}</option>{library.assets.length > 1 && <option value="series">同一作品全部 {library.assets.length} 个视频</option>}</select></Field>
    <p className="helper character-scope-summary">{assets.length} 个视频 · 共 {duration(assets.reduce((sum, candidate) => sum + candidate.duration_ms, 0))} · {taskModelLabel(model, taskModels.overrides?.vision)}</p>
    <TaskModels roles={modelRoles} value={taskModels.overrides} onChange={taskModels.change} disabled={!!busy} />
    {usesPreparation ? <TaskExecution stages={['transcript', 'preparation']} value={execution} onChange={setExecution} disabled={!!busy} connectionLimits={{ transcript: preparation?.max_concurrency ?? 16, preparation: preparation?.max_concurrency ?? 16 }} saveDefault={saveDefault} onSaveDefaultChange={setSaveDefault} /> : <p className="helper">当前使用逐镜头画面分析，依赖连续人物记忆，始终串行。使用场景分块策略可分别选择台词整理与视觉分块的执行方式。</p>}
    {usesPreparation && !preparation && <Alert>请先在<a href="#/settings" onClick={onClose}>模型与设置</a>中分配分块与简要索引模型。</Alert>}
    {!model && <Alert>请先在<a href="#/settings" onClick={onClose}>模型与设置</a>中分配画面理解模型。</Alert>}{unavailable.length > 0 && <Alert>这些视频的原文件暂不可用：{unavailable.map(assetTaskLabel).join('、')}。请先在各自详情页定位原文件。</Alert>}{active && <Alert>所选作品仍有任务正在进行。请等待任务结束后重新识别。</Alert>}
    <div className="form-grid"><Field label="每个视频最多模型请求数" hint="每个视频独立计数；达到上限后停止，可在任务队列重试。"><input type="number" min={1} max={100000} step={1} value={requestLimit} onChange={(event) => setRequestLimit(Number(event.target.value))} required disabled={Boolean(busy)} /></Field><Field label="每个视频费用上限（USD）" hint={usesCodex ? 'Codex 订阅费用未知，金额上限已停用。' : '留空不限制；费用取决于模型连接的定价。'}><input type="number" min={0.01} step={0.01} value={usesCodex ? '' : costLimit} onChange={(event) => setCostLimit(event.target.value)} placeholder="不限制" disabled={Boolean(busy) || usesCodex} /></Field></div>
    <div className="section-row"><p className="helper">估算用量为可选步骤。可先扫描预览；直接开始时也会自动完成本机关键帧扫描。</p><button type="button" className="button secondary" disabled={Boolean(busy) || unavailable.length > 0 || active || stale} onClick={() => void scan()}>{busy === 'scan' ? <Spinner label="正在扫描关键帧" /> : '扫描关键帧'}</button></div>
    {assets.map((candidate) => { const plan = scanningPlans[candidate.id]; return plan && ['queued', 'pending', 'running', 'cancelling', 'pausing'].includes(plan.status) ? <div className="keyframe-preview" key={candidate.id}><strong>{assetTaskLabel(candidate)}</strong><p className="helper">{plan.message || '正在本机扫描关键帧'} · {Math.round((plan.progress || 0) * 100)}%</p><div className="progress-track" role="progressbar" aria-label={`${assetTaskLabel(candidate)}角色重识别关键帧扫描进度`} aria-valuenow={Math.round((plan.progress || 0) * 100)} aria-valuemin={0} aria-valuemax={100}><div style={{ width: `${Math.min(100, (plan.progress || 0) * 100)}%` }} /></div><button type="button" className="text-button" disabled={cancellingScan === candidate.id || plan.status === 'cancelling'} onClick={async () => { setCancellingScan(candidate.id); try { await post(`/jobs/${encodeURIComponent(plan.job_id)}/cancel`); } catch (error) { setError(message(error)); } finally { setCancellingScan(''); } }}>取消「{assetTaskLabel(candidate)}」扫描</button></div> : null; })}
    {assets.filter((candidate) => plans[candidate.id]?.status === 'completed').map((candidate) => <div className="keyframe-asset" key={candidate.id}><strong>{assetTaskLabel(candidate)} · {plans[candidate.id].frame_count ?? plans[candidate.id].frames?.length ?? 0} 张关键帧</strong><div className="keyframe-grid">{plans[candidate.id].frames?.slice(0, 12).map((frame) => <a key={frame.id} href={frame.thumbnail_url} target="_blank" rel="noreferrer"><img src={frame.thumbnail_url} alt={`${assetTaskLabel(candidate)} ${formatTime(frame.at_ms)} 关键帧`} loading="lazy" /><span>{formatTime(frame.at_ms)}</span></a>)}</div></div>)}
    {validEstimate && <><div className="estimate"><div><span>预计请求总数</span><strong>{totalRequests}</strong></div><div><span>预计总费用</span><strong>{totalCost == null ? '未知' : `$${totalCost.toFixed(4)}`}</strong></div><div><span>档案修订</span><strong>{library.revision}</strong></div></div>{[...new Set(estimates.flatMap((item) => item.warnings))].map((warning) => <Alert key={warning}>{warning}</Alert>)}{estimates.map((entry, index) => <div key={assets[index].id}><strong>{assetTaskLabel(assets[index])}</strong><StageEstimates estimate={entry} /></div>)}{estimates.some((item) => item.estimated_requests > requestLimit) && <Alert>部分视频的预计请求数超过上限，可能只能完成部分分析。</Alert>}</>}
    {error && <Alert tone="error">{error}</Alert>}{stale && <button type="button" className="button secondary" onClick={async () => { await onReload(); onClose(); }}>刷新档案后重新操作</button>}
  </div><footer className="modal-footer"><button className="button secondary" type="submit" disabled={Boolean(busy) || blocked}>{busy === 'estimate' ? <Spinner label="正在估算" /> : validEstimate ? '重新估算' : '估算重新识别用量'}</button><button className="button primary" type="button" disabled={Boolean(busy) || blocked} onClick={start}>{busy === 'start' ? <Spinner label="正在创建任务" /> : <><ScanLine size={16} />开始重新识别</>}</button></footer></form></Modal>;
}

export default function CharacterProfiles({ asset, selectedId, onSeek, onSaved }: { asset: Asset; selectedId?: string; onSeek: (atMs: number) => void; onSaved: () => Promise<void> }) {
  const { jobs, openJob } = useApp();
  const [library, setLibrary] = useState<CharacterLibrary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [expandedId, setExpandedId] = useState(selectedId || '');
  const [editing, setEditing] = useState<{ profile: CharacterProfile; mode: 'edit' | 'merge' | 'delete' } | null>(null);
  const [reanalyzing, setReanalyzing] = useState(false);
  const [references, setReferences] = useState<CharacterProfile | null>(null);
  const [creating, setCreating] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [image, setImage] = useState<string | null>(null);
  const [limit, setLimit] = useState(24);
  const requestSequence = useRef(0);
  const load = useCallback(async () => {
    const sequence = ++requestSequence.current;
    try { const data = await api<CharacterLibrary>(characterPath(asset.id)); if (sequence === requestSequence.current) { setLibrary(data); setError(''); } }
    catch (caught) { if (sequence === requestSequence.current) setError(message(caught)); }
    finally { if (sequence === requestSequence.current) setLoading(false); }
  }, [asset.id]);
  const activeIds = library?.assets.map((candidate) => candidate.id) || [asset.id];
  const running = jobs.some((job) => activeIds.includes(job.asset_id || '') && ['queued', 'pending', 'running'].includes(job.status));
  useEffect(() => { void load(); if (!running) return; const timer = window.setInterval(() => { void load(); }, 5000); return () => window.clearInterval(timer); }, [load, running]);
  useEffect(() => { if (selectedId) { setExpandedId(selectedId); setQuery(''); } }, [selectedId]);
  const saved = async () => { await load(); await onSaved(); };
  const filtered = useMemo(() => {
    const profiles = library?.profiles || [];
    const matching = profiles.filter((profile) => `${profile.name} ${profile.name_cn || ''} ${profile.name_ja || ''} ${profile.name_en || ''} ${profile.aliases.join(' ')} ${profile.description} ${(profile.appearance || []).join(' ')} ${profile.notes}`.toLowerCase().includes(query.trim().toLowerCase()));
    return selectedId ? [...matching].sort((left, right) => Number(right.id === selectedId) - Number(left.id === selectedId)) : matching;
  }, [library, query, selectedId]);
  if (loading) return <div className="character-loading"><Spinner label="正在读取角色档案" /></div>;
  if (!library) return <div className="character-loading"><Alert tone="error">{error || '无法读取角色档案'}</Alert><button className="button secondary" onClick={() => { setLoading(true); void load(); }}>重新加载角色档案</button></div>;
  return <div className="character-library">
    <div className="character-library-heading"><div><span className="eyebrow">CHARACTER ARCHIVE</span><h2>{library.profiles.length} 份角色档案</h2><p>{library.assets.length > 1 ? `同一作品的 ${library.assets.length} 个视频共享档案` : '在这部作品的不同镜头间持续识别'} · 修订 {library.revision}</p></div><div className="button-row">{library.profiles.length > 0 && <button className="button small primary" onClick={() => setConfirming(true)}><Check size={14} />开始角色确认</button>}{library.cast_policy?.mode !== 'bangumi' && asset.anime_profile?.mode !== 'bangumi' && <button className="button small secondary" onClick={() => setCreating(true)}>新建角色档案</button>}<button className="button small secondary" onClick={() => setReanalyzing(true)}><RefreshCw size={14} />按新版档案重新识别</button></div></div>
    <p className="character-library-intro">{library.cast_policy?.mode === 'bangumi' || asset.anime_profile?.mode === 'bangumi' ? '角色档案以 Bangumi 列出的角色为准，画面分析补充这些角色的外观与出场证据。' : '为能确认姓名的剧中人物建立档案，结合画面积累外观与出场证据。'}你可以先添加独立参考图库与声音样本，再核对出场和说话人；修正身份会更新映射与检索展示。</p>
    {running && <Alert tone="info"><Spinner label="分析进行中，角色档案会持续更新" /> <a href="#/jobs">查看任务</a></Alert>}{error && <Alert tone="error">{error}<button className="text-button" onClick={() => { void load(); }}>重试加载</button></Alert>}
    {library.profiles.length > 0 && <div className="character-search"><Search size={16} /><input aria-label="搜索角色档案" placeholder="搜索姓名、别名或外貌特征…" value={query} onChange={(event) => { setQuery(event.target.value); setLimit(24); }} /></div>}
    {filtered.length > 0 ? <><div className="character-grid">{filtered.slice(0, limit).map((profile) => <article className={`character-card ${expandedId === profile.id ? 'selected' : ''}`} key={profile.id} aria-label={`${profile.name}的角色档案`}>
      <Portrait assetId={asset.id} revision={library.revision} profile={profile} onOpen={setImage} /><div className="character-card-body"><div className="character-name-row"><h3>{profile.name}</h3><span className={`character-review ${profile.user_edited ? 'reviewed' : ''}`}>{profile.user_edited ? '人工修订' : profile.external_source ? 'Bangumi 初始资料' : 'AI 识别'}</span></div><CharacterNames profile={profile} /><ExternalCharacterReference profile={profile} /><p className="character-description">{profile.description || (profile.external_source ? 'Bangumi 暂无角色简介。' : '暂无外貌描述，后续识别会继续补充。')}</p>{profile.appearance && profile.appearance.length > 0 && <div className="character-traits">{profile.appearance.slice(0, 6).map((trait, index) => <span key={`${trait}-${index}`}>{trait}</span>)}</div>}{profile.notes && <p className="character-notes"><Pencil size={12} />{profile.notes}</p>}<button className="text-button character-reference-manage" onClick={() => setReferences(profile)} aria-label={`管理${profile.name}的参考图片与声音样本`}>参考图库与声音样本</button><div className="character-actions"><button className="text-button" onClick={() => setExpandedId(expandedId === profile.id ? '' : profile.id)} aria-expanded={expandedId === profile.id} aria-label={`查看${profile.name}的出场记录`}><Images size={14} />{profile.appearance_count} 次出场</button><div><button className="icon-button" aria-label={`编辑${profile.name}的档案`} title="编辑档案" onClick={() => setEditing({ profile, mode: 'edit' })}><Pencil size={15} /></button><button className="icon-button" aria-label={`合并${profile.name}的档案`} title={library.profiles.length < 2 ? '至少需要两份档案才能合并' : '合并档案'} disabled={library.profiles.length < 2} onClick={() => setEditing({ profile, mode: 'merge' })}><GitMerge size={16} /></button><button className="icon-button character-delete" aria-label={`删除${profile.name}的档案`} title="删除档案" onClick={() => setEditing({ profile, mode: 'delete' })}><Trash2 size={15} /></button></div></div></div>
      {expandedId === profile.id && <div className="character-appearances"><h4>出场记录</h4>{profile.appearances.length ? profile.appearances.map((appearance, index) => <div className="character-appearance" key={`${appearance.asset_id}-${appearance.record_id}-${appearance.entity_id}-${index}`}>
        {appearance.frame_ids?.[0] && <button className="character-frame" onClick={() => setImage(frameUrl(appearance.asset_id, appearance.frame_ids[0]))} aria-label={`放大${profile.name}在${appearance.asset_title}的证据画面`}><img src={frameUrl(appearance.asset_id, appearance.frame_ids[0])} alt={`${profile.name}的出场证据`} loading="lazy" /></button>}<div><strong>{appearance.asset_title}</strong><span>{formatTime(appearance.start_ms)} — {formatTime(appearance.end_ms)}</span></div>{appearance.asset_id === asset.id ? <button className="icon-button" aria-label={`播放${profile.name}在${formatTime(appearance.start_ms)}的出场`} onClick={() => onSeek(appearance.start_ms)}><Play size={16} /></button> : <a className="icon-button" aria-label={`前往${appearance.asset_title}查看${profile.name}`} href={assetLink(appearance.asset_id, appearance.start_ms)}><ArrowRight size={16} /></a>}
      </div>) : <p className="helper">当前分析版本还没有此角色的出场记录。</p>}</div>}
    </article>)}</div>{filtered.length > limit && <button className="button secondary load-more" onClick={() => setLimit(limit + 24)}>显示更多角色</button>}</> : <Empty icon={<UsersRound size={29} />} title={query ? '没有找到这个角色' : '角色故事，从第一次登场开始'} action={!query && <button className="button secondary" onClick={() => openJob(asset)} disabled={!asset.source_available || running}>开始画面分析</button>}>{query ? '试试别名或外貌特征。' : '完成画面分析后，这里会汇集角色档案与出场证据。已有作品可重新分析以建立跨镜头人物关联。'}</Empty>}
    {confirming && <CharacterConfirmation asset={asset} onClose={() => setConfirming(false)} onSaved={saved} />}{creating && <CreateCharacter assetId={asset.id} onClose={() => setCreating(false)} onSaved={saved} />}{references && <CharacterReferences assetId={asset.id} profile={references} onClose={() => setReferences(null)} onSaved={saved} />}{editing && <CharacterEditModal assetId={asset.id} profile={editing.profile} library={library} mode={editing.mode} onClose={() => setEditing(null)} onSaved={saved} />}{reanalyzing && <ReanalyzeModal asset={asset} library={library} onClose={() => setReanalyzing(false)} onReload={saved} />}{image && <Modal title="角色出场证据" subtitle="请结合画面与原片，核对角色身份。" onClose={() => setImage(null)} wide><div className="evidence-full"><img src={image} alt="角色出场的完整证据画面" /></div></Modal>}
  </div>;
}
