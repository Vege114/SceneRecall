import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { api, assetLink, formatTime, message, post } from '../api';
import type { Observation, SceneChunk } from '../types';
import { Alert, Field, Spinner } from './ui';

type EntityKind = 'prop' | 'mecha' | 'place' | 'organization_symbol';
type ReferenceView = 'whole_scene' | 'landmark' | 'object_region';
interface Reference { id: string; image_url: string; status: string; view: ReferenceView }
interface WorkEntity {
  concept_id: string; type: EntityKind; name: string; original_name: string; translated_names: string[];
  aliases: string[]; appearance: string; applicability: string; forms: unknown[]; facts: unknown[];
  relations: unknown[]; sources: unknown[]; references: Reference[]; version: number;
}
interface Scope { work_scope_id: string | null; title?: string; work_version?: string; asset_ids?: string[] }
interface Occurrence { record_id: string; asset_id: string; start_ms: number; end_ms: number }
const labels: Record<EntityKind, string> = { prop: '道具', mecha: '机体', place: '场所', organization_symbol: '组织符号' };
const parts = (text: string) => text.split(/[,，、\n]/).map((part) => part.trim()).filter(Boolean);
const blank = { type: 'prop' as EntityKind, name: '', original_name: '', translated_names: [] as string[], aliases: [] as string[], appearance: '', applicability: '', forms: [], facts: [], relations: [], sources: [] };
function jsonArray(text: string, label: string): unknown[] {
  const data = JSON.parse(text || '[]');
  if (!Array.isArray(data)) throw new Error(`${label}须为 JSON 数组`);
  return data;
}
function boxValue(text: string): number[] | null {
  if (!text.trim()) return null;
  const box = JSON.parse(text);
  if (!Array.isArray(box) || box.length !== 4 || box.some((value) => typeof value !== 'number' || !Number.isFinite(value))) throw new Error('区域须为 [x, y, 宽, 高] 四个归一化数值');
  return box;
}

function EntityEditor({ assetId, entity, onSaved, onCancel }: { assetId: string; entity?: WorkEntity; onSaved: () => Promise<void>; onCancel: () => void }) {
  const initial = entity || blank;
  const [type, setType] = useState(initial.type);
  const [name, setName] = useState(initial.name);
  const [original, setOriginal] = useState(initial.original_name);
  const [translated, setTranslated] = useState(initial.translated_names.join('、'));
  const [aliases, setAliases] = useState(initial.aliases.join('、'));
  const [appearance, setAppearance] = useState(initial.appearance);
  const [applicability, setApplicability] = useState(initial.applicability);
  const [advanced, setAdvanced] = useState(Object.fromEntries(['forms', 'facts', 'relations', 'sources'].map((key) => [key, JSON.stringify(initial[key as 'forms'], null, 2)])));
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const data = { type, name: name.trim(), original_name: original.trim(), translated_names: parts(translated), aliases: parts(aliases), appearance, applicability, ...Object.fromEntries(Object.entries(advanced).map(([key, value]) => [key, jsonArray(value, key)])) };
      const path = `/assets/${encodeURIComponent(assetId)}/work-entities`;
      if (entity) await api(`${path}/${encodeURIComponent(entity.concept_id)}`, { method: 'PATCH', body: JSON.stringify({ changes: data, reason }) });
      else await post(path, data);
      await onSaved(); onCancel();
    } catch (error) { setError(message(error)); } finally { setBusy(false); }
  };
  return <form className="work-entity-editor" onSubmit={submit}>
    <h3>{entity ? '编辑作品实体' : '新增作品实体'}</h3>
    {error && <Alert tone="error">{error}</Alert>}
    <div className="form-grid">
      <Field label="实体类型"><select value={type} onChange={(event) => setType(event.target.value as EntityKind)}>{Object.entries(labels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></Field>
      <Field label="正式名称"><input required value={name} onChange={(event) => setName(event.target.value)} /></Field>
      <Field label="原文名称"><input value={original} onChange={(event) => setOriginal(event.target.value)} /></Field>
      <Field label="译名"><input value={translated} onChange={(event) => setTranslated(event.target.value)} /></Field>
      <Field label="别名"><input value={aliases} onChange={(event) => setAliases(event.target.value)} /></Field>
      <Field label="适用阶段与版本"><input value={applicability} onChange={(event) => setApplicability(event.target.value)} /></Field>
    </div>
    <Field label="区分性外观"><textarea value={appearance} onChange={(event) => setAppearance(event.target.value)} /></Field>
    <details><summary>形态、设定事实、关系和出处（JSON）</summary>
      <p className="helper">设定事实须填写来源 ID。功能与归属只用于寻找实体出场，不能证明画面中正在发动功能或当前驾驶者。</p>
      {([['forms', '不同形态', '[{"name":"展开形态","appearance":"外观特征"}]'], ['sources', '资料出处', '[{"id":"source_1","kind":"manual","title":"人工整理","url":"","note":""}]'], ['facts', '设定事实', '[{"id":"fact_1","kind":"function","text":"能停止时间","source_ids":["source_1"],"applicability":""}]'], ['relations', '必要关系', '[{"predicate":"属于","target_concept_id":"概念ID","source_ids":["source_1"]}]']] as const).map(([key, label, example]) => <Field key={key} label={`${label} JSON`} hint={`示例：${example}`}><textarea rows={4} value={advanced[key]} onChange={(event) => setAdvanced({ ...advanced, [key]: event.target.value })} /></Field>)}
    </details>
    {entity && <Field label="修改原因"><input required value={reason} onChange={(event) => setReason(event.target.value)} /></Field>}
    <div className="button-row"><button className="button primary" disabled={busy}>{busy ? <Spinner label="保存中" /> : '保存实体资料'}</button><button type="button" className="button secondary" onClick={onCancel}>取消编辑</button></div>
  </form>;
}

export default function WorkEntities({ assetId, title, records, initialBinding, onSaved }: { assetId: string; title: string; records: (Observation | SceneChunk)[]; initialBinding?: { recordId: string; entityId: string }; onSaved: () => Promise<void> }) {
  const [scope, setScope] = useState<Scope | null>(null);
  const [entities, setEntities] = useState<WorkEntity[]>([]);
  const [selected, setSelected] = useState('');
  const [occurrences, setOccurrences] = useState<Occurrence[]>([]);
  const [edit, setEdit] = useState<'new' | WorkEntity | null>(null);
  const [scopeTitle, setScopeTitle] = useState(title);
  const [version, setVersion] = useState('');
  const [scopeId, setScopeId] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [upload, setUpload] = useState<File | null>(null);
  const [source, setSource] = useState('');
  const [frameId, setFrameId] = useState('');
  const [view, setView] = useState<ReferenceView>('whole_scene');
  const [box, setBox] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [reason, setReason] = useState('');
  const [recordId, setRecordId] = useState('');
  const [localId, setLocalId] = useState('');
  const [bindingStatus, setBindingStatus] = useState('confirmed');
  const [evidence, setEvidence] = useState<string[]>([]);
  const base = `/assets/${encodeURIComponent(assetId)}`;
  const selectedEntity = entities.find((entity) => entity.concept_id === selected);
  const record = records.find((row) => row.id === recordId);
  const frames = [...new Set(records.flatMap((row) => row.evidence_frame_ids || []))];
  useEffect(() => {
    if (!initialBinding) return;
    const initial = records.find((row) => row.id === initialBinding.recordId);
    setRecordId(initialBinding.recordId); setLocalId(initialBinding.entityId); setEvidence(initial?.evidence_frame_ids || []);
  }, [initialBinding]); // Preserve selections while the parent refreshes its records.
  const load = useCallback(async () => {
    const [current, catalog] = await Promise.all([api<Scope>(`${base}/work-entity-scope`), api<{ entities: WorkEntity[] }>(`${base}/work-entities`)]);
    setScope(current); setEntities(catalog.entities);
    setSelected((value) => catalog.entities.some((entity) => entity.concept_id === value) ? value : catalog.entities[0]?.concept_id || '');
  }, [base]);
  useEffect(() => { void load().catch((error) => setError(message(error))); }, [load]);
  useEffect(() => {
    let active = true;
    setOccurrences([]);
    if (selected) void api<{ occurrences: Occurrence[] }>(`${base}/work-entities/${encodeURIComponent(selected)}/occurrences`).then((data) => { if (active) setOccurrences(data.occurrences); }).catch((error) => { if (active) setError(message(error)); });
    return () => { active = false; };
  }, [base, selected, entities]);
  const changed = async () => { await load(); await onSaved(); };
  const mutate = async (action: () => Promise<unknown>) => {
    setBusy(true); setError('');
    try { await action(); await changed(); } catch (error) { setError(message(error)); } finally { setBusy(false); }
  };
  const referencePath = `${base}/work-entities/${encodeURIComponent(selected)}/references`;
  const addUpload = (event: FormEvent) => {
    event.preventDefault();
    if (!upload) return;
    void mutate(async () => {
      const data = new FormData(); data.set('file', upload); data.set('source_url', source); data.set('confirmed', String(confirmed)); data.set('view', view);
      await api(referencePath, { method: 'POST', body: data });
    });
  };
  const addFrame = (event: FormEvent) => { event.preventDefault(); void mutate(() => post(`${referencePath}/from-frame`, { frame_id: frameId, box: boxValue(box), confirmed, view })); };
  const bind = (event: FormEvent) => {
    event.preventDefault();
    void mutate(() => post(`${base}/work-entity-bindings`, { record_id: recordId, entity_id: localId || null, concept_id: bindingStatus === 'unknown' ? null : selected, status: bindingStatus, evidence_frame_ids: evidence, box: boxValue(box), reason }));
  };
  return <section className="work-entities" aria-label="作品图文实体资料库">
    <h2>作品图文设定库</h2>
    <p className="helper">跨集复用道具、机体与场所。当前画面、人工绑定、作品设定与事件判断分别保存；来源链接只作出处。</p>
    {error && <Alert tone="error">{error}</Alert>}
    {!scope ? <Spinner label="读取作品实体资料" /> : !scope.work_scope_id ? <>
      <form onSubmit={(event) => { event.preventDefault(); void mutate(() => post('/work-entity-scopes', { title: scopeTitle, version, asset_ids: [assetId] })); }}>
        <h3>为本集创建作品资料库</h3>
        <Field label="作品资料库名称"><input required value={scopeTitle} onChange={(event) => setScopeTitle(event.target.value)} /></Field>
        <Field label="作品版本"><input value={version} onChange={(event) => setVersion(event.target.value)} /></Field>
        <button className="button primary" disabled={busy}>创建并关联本集</button>
      </form>
      <form onSubmit={(event) => { event.preventDefault(); void mutate(() => api(`${base}/work-entity-scope`, { method: 'PUT', body: JSON.stringify({ work_scope_id: scopeId.trim() }) })); }}>
        <h3>复用已有作品资料库</h3><Field label="已有作品资料库 ID"><input required value={scopeId} onChange={(event) => setScopeId(event.target.value)} /></Field>
        <button className="button secondary" disabled={busy}>按 ID 关联本集</button>
      </form>
    </> : <>
      <div className="work-scope"><strong>{scope.title}</strong> {scope.work_version}<p>共享资料库 ID：<code>{scope.work_scope_id}</code> · {scope.asset_ids?.length || 1} 集</p></div>
      <div className="button-row"><button className="button secondary" onClick={() => setEdit('new')} disabled={busy}>新增实体</button><label className="button secondary">导入资料包<input type="file" accept=".zip" aria-label="导入作品实体资料包" disabled={busy} onChange={(event) => { const file = event.target.files?.[0]; if (file) void mutate(() => { const data = new FormData(); data.set('file', file); return api(`${base}/work-entities/import`, { method: 'POST', body: data }); }); }} /></label></div>
      {edit && <EntityEditor key={edit === 'new' ? 'new' : edit.concept_id} assetId={assetId} entity={edit === 'new' ? undefined : edit} onSaved={changed} onCancel={() => setEdit(null)} />}
      <Field label="当前作品实体"><select value={selected} onChange={(event) => setSelected(event.target.value)}><option value="">选择实体</option>{entities.map((entity) => <option value={entity.concept_id} key={entity.concept_id}>{labels[entity.type]} · {entity.name}</option>)}</select></Field>
      {selectedEntity && <>
        <article className="work-entity-details"><h3>{selectedEntity.name} <small>{labels[selectedEntity.type]} · v{selectedEntity.version}</small></h3><code>{selectedEntity.concept_id}</code><p>{[selectedEntity.original_name, ...selectedEntity.translated_names, ...selectedEntity.aliases].filter(Boolean).join(' · ')}</p><p>{selectedEntity.appearance}</p><p>{selectedEntity.applicability}</p>
          <details><summary>设定事实、形态、关系与出处</summary><pre className="json-report">{JSON.stringify({ forms: selectedEntity.forms, facts: selectedEntity.facts, relations: selectedEntity.relations, sources: selectedEntity.sources }, null, 2)}</pre></details>
          <button className="button secondary" onClick={() => setEdit(selectedEntity)}>编辑实体资料</button>
        </article>
        <Field label="人工操作原因"><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="说明本次核对、纠错或撤销依据" /></Field>
        <h3>多视角参考图</h3>
        <div className="work-reference-list">{selectedEntity.references.map((reference) => <article key={reference.id}><img src={reference.image_url} alt={`${selectedEntity.name}参考图`} loading="lazy" /><p>{reference.status === 'confirmed' ? '可信参考' : reference.status === 'revoked' ? '已撤销' : '候选参考'} · {reference.view}</p><button className="button secondary small" disabled={busy || !reason.trim() || reference.status === 'confirmed'} onClick={() => void mutate(() => api(`${referencePath}/${encodeURIComponent(reference.id)}`, { method: 'PATCH', body: JSON.stringify({ status: 'confirmed', reason }) }))}>确认参考</button><button className="button secondary small" disabled={busy || !reason.trim() || reference.status === 'revoked'} onClick={() => void mutate(() => api(`${referencePath}/${encodeURIComponent(reference.id)}`, { method: 'PATCH', body: JSON.stringify({ status: 'revoked', reason }) }))}>撤销参考</button></article>)}</div>
        <Field label="参考图类型"><select value={view} onChange={(event) => setView(event.target.value as ReferenceView)}><option value="whole_scene">完整场景</option><option value="landmark">场所地标</option><option value="object_region">物体区域</option></select></Field>
        <Field label="局部区域（可选）" hint="从原帧取图及当前绑定共用；填写 [x, y, 宽, 高]，各值为 0–1。驾驶舱与地点优先保留完整场景。"><input value={box} onChange={(event) => setBox(event.target.value)} placeholder="[0.2, 0.1, 0.4, 0.6]" /></Field>
        <label><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />我已核对新增参考属于此实体</label>
        <form onSubmit={addUpload}><Field label="上传参考图"><input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => setUpload(event.target.files?.[0] || null)} /></Field><Field label="参考图出处链接（只记录）"><input value={source} onChange={(event) => setSource(event.target.value)} /></Field><button className="button secondary" disabled={busy || !upload}>添加上传参考</button></form>
        <form onSubmit={addFrame}><Field label="已注册原帧"><select required value={frameId} onChange={(event) => setFrameId(event.target.value)}><option value="">选择当前画面证据帧</option>{frames.map((id) => <option value={id} key={id}>{id}</option>)}</select></Field>{frameId && <img className="work-frame-preview" src={`/api/assets/${encodeURIComponent(assetId)}/frames/${encodeURIComponent(frameId)}`} alt="将加入参考库的原帧" />}<button className="button secondary" disabled={busy || !frameId}>从原帧添加参考</button></form>
        <h3>将当前画面绑定到作品实体</h3>
        <form onSubmit={bind}>
          <Field label="画面观察"><select required value={recordId} onChange={(event) => { const next = records.find((row) => row.id === event.target.value); setRecordId(event.target.value); setLocalId(''); setEvidence(next?.evidence_frame_ids || []); }}><option value="">选择片段</option>{records.filter((row) => row.evidence_frame_ids?.length).map((row) => <option key={row.id} value={row.id}>{formatTime(row.start_ms)} · {row.summary || '画面证据'}</option>)}</select></Field>
          <Field label="镜头局部实体"><select value={localId} onChange={(event) => setLocalId(event.target.value)}><option value="">整个画面 / 场所</option>{record?.entities?.map((entity) => <option key={entity.id} value={entity.id}>{entity.id} · {entity.description || entity.name}</option>)}</select></Field>
          <Field label="绑定状态"><select value={bindingStatus} onChange={(event) => setBindingStatus(event.target.value)}><option value="confirmed">人工确认</option><option value="candidate">仅作为候选</option><option value="unknown">保留未知</option><option value="revoked">撤销绑定</option></select></Field>
          <fieldset><legend>当前绑定证据帧</legend>{record?.evidence_frame_ids.map((id) => <label key={id}><input type="checkbox" checked={evidence.includes(id)} onChange={(event) => setEvidence(event.target.checked ? [...evidence, id] : evidence.filter((value) => value !== id))} />{id}</label>)}</fieldset>
          <button className="button primary" disabled={busy || !recordId || !evidence.length || !reason.trim()}>保存人工绑定</button>
        </form>
        <h3>本集已确认出场</h3><p className="helper">这些链接证明实体出场，设定功能是否发动仍需当前事件证据。</p>
        {occurrences.length ? <ul>{occurrences.map((row) => <li key={row.record_id}><a href={assetLink(row.asset_id, row.start_ms)}>{formatTime(row.start_ms)} — {formatTime(row.end_ms)}</a></li>)}</ul> : <p className="helper">尚无已确认出场。</p>}
        <button className="button secondary" disabled={busy || !reason.trim()} onClick={() => void mutate(() => api(`${base}/work-entities/${encodeURIComponent(selected)}`, { method: 'DELETE', body: JSON.stringify({ reason }) }))}>撤销此实体资料</button>
      </>}
    </>}
  </section>;
}
