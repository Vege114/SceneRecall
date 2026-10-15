import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { api, message, post } from '../api';
import { type CharacterProfile } from '../types';
import { type CharacterReference, type VoiceSample, identitySourceLabel, identityStatusLabel } from '../identityTypes';
import { Alert, Field, Modal, Spinner } from './ui';

export function CreateCharacter({ assetId, onClose, onSaved }: { assetId: string; onClose: () => void; onSaved: () => Promise<void> }) {
  const [name, setName] = useState(''); const [description, setDescription] = useState('');
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const create = async (event: FormEvent) => { event.preventDefault(); if (busy || !name.trim()) return; setBusy(true); setError(''); try { await post(`/assets/${encodeURIComponent(assetId)}/characters`, { name: name.trim(), aliases: [], description: description.trim() }); await onSaved(); onClose(); } catch (caught) { setError(message(caught)); } finally { setBusy(false); } };
  return <Modal title="新建角色档案" subtitle="首次识别前也可以建立可信参考图库" onClose={() => { if (!busy) onClose(); }}><form onSubmit={create}><div className="modal-body"><Field label="角色姓名"><input required maxLength={200} value={name} onChange={(event) => setName(event.target.value)} disabled={busy} /></Field><Field label="角色识别说明"><textarea rows={3} value={description} onChange={(event) => setDescription(event.target.value)} disabled={busy} /></Field><p className="helper">建档后可添加参考图片或声音样本。创建档案本身不会生成出场证据。</p>{error && <Alert tone="error">{error}</Alert>}</div><footer className="modal-footer"><button type="button" className="button ghost" onClick={onClose} disabled={busy}>取消</button><button className="button primary" type="submit" disabled={busy || !name.trim()}>{busy ? <Spinner label="正在建立角色档案" /> : '建立角色档案'}</button></footer></form></Modal>;
}

export default function CharacterReferences({ assetId, profile, onClose, onSaved }: { assetId: string; profile: CharacterProfile; onClose: () => void; onSaved: () => Promise<void> }) {
  const [tab, setTab] = useState<'images' | 'voices'>('images');
  return <Modal title="参考图库与声音样本" subtitle={profile.name} onClose={onClose} wide>
    <div className="modal-body identity-library"><div className="tabs"><button type="button" className={tab === 'images' ? 'active' : ''} onClick={() => setTab('images')}>参考图片</button><button type="button" className={tab === 'voices' ? 'active' : ''} onClick={() => setTab('voices')}>声音样本</button></div>
      {tab === 'images' ? <ReferenceImages assetId={assetId} profile={profile} onSaved={onSaved} /> : <VoiceSamples assetId={assetId} profile={profile} onSaved={onSaved} />}
    </div>
  </Modal>;
}

function ReferenceImages({ assetId, profile, onSaved }: { assetId: string; profile: CharacterProfile; onSaved: () => Promise<void> }) {
  const endpoint = `/assets/${encodeURIComponent(assetId)}/characters/${encodeURIComponent(profile.id)}/references`;
  const [references, setReferences] = useState<CharacterReference[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(''); const [error, setError] = useState('');
  const [mode, setMode] = useState<'upload' | 'external_cached' | 'frame_crop'>('upload');
  const [file, setFile] = useState<File | null>(null); const [sourceUrl, setSourceUrl] = useState('');
  const [frameId, setFrameId] = useState(profile.representative?.asset_id === assetId ? profile.representative.frame_id : '');
  const [box, setBox] = useState([0, 0, 1, 1]);
  const [confirmed, setConfirmed] = useState(false); const [singleCharacter, setSingleCharacter] = useState(false);
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const load = useCallback(async () => { try { const result = await api<{ references: CharacterReference[] }>(endpoint); setReferences(result.references || []); } catch (caught) { setError(message(caught)); } finally { setLoading(false); } }, [endpoint]);
  useEffect(() => { void load(); }, [load]);
  const add = async (event: FormEvent) => {
    event.preventDefault(); if (busy || (mode !== 'frame_crop' && !file)) return;
    if (confirmed && !singleCharacter) { setError('请先确认图片中只有这个角色。'); return; }
    if (mode === 'frame_crop' && (box[0] + box[2] > 1 || box[1] + box[3] > 1)) { setError('裁剪区域不能超出原帧。'); return; }
    setBusy('add'); setError('');
    try {
      if (mode === 'frame_crop') await post(`${endpoint}/crop`, { frame_id: frameId, box, confirmed, single_character: singleCharacter });
      else { const data = new FormData(); data.append('file', file!); data.append('source_kind', mode); data.append('source_url', sourceUrl); data.append('confirmed', String(confirmed)); data.append('single_character', String(singleCharacter)); await api(endpoint, { method: 'POST', body: data }); }
      await load(); await onSaved(); setConfirmed(false); setSingleCharacter(false);
    } catch (caught) { setError(message(caught)); } finally { setBusy(''); }
  };
  const review = async (reference: CharacterReference, status: CharacterReference['status']) => {
    setBusy(reference.id); setError('');
    try { await api(`${endpoint}/${encodeURIComponent(reference.id)}`, { method: 'PATCH', body: JSON.stringify({ status, reason: status === 'revoked' ? '用户撤销错误参考图' : '用户核对角色与单人裁剪', single_character: status === 'confirmed' ? true : reference.quality.single_character }) }); await load(); await onSaved(); }
    catch (caught) { setError(message(caught)); } finally { setBusy(''); }
  };
  return <section aria-label="角色参考图库"><p className="helper">参考图库独立于出场证据。外部图片只帮助识别身份，不能证明角色在当前镜头出现。只有质量通过且人工确认的单角色图片进入可信库。</p>
    {error && <Alert tone="error">{error}</Alert>}
    <form onSubmit={add} className="identity-add-form"><Field label="参考图片来源"><select value={mode} disabled={!!busy} onChange={(event) => setMode(event.target.value as typeof mode)}><option value="upload">人工上传图片</option><option value="external_cached">已缓存的外部参考图</option><option value="frame_crop">当前原片帧裁剪</option></select></Field>
      {mode === 'frame_crop' ? <><Field label="证据帧 ID"><input value={frameId} onChange={(event) => setFrameId(event.target.value)} required /></Field><div className="form-grid four">{['裁剪左侧 X', '裁剪顶部 Y', '裁剪宽度', '裁剪高度'].map((label, index) => <Field label={label} key={label}><input type="number" min={index < 2 ? 0 : 0.01} max={1} step={0.01} value={box[index]} required onChange={(event) => setBox(box.map((value, i) => i === index ? Number(event.target.value) : value))} /></Field>)}</div>{frameId && <img className="identity-frame-preview" src={`/api/assets/${encodeURIComponent(assetId)}/frames/${encodeURIComponent(frameId)}`} alt="待裁剪原帧" />}</> : <><Field label="选择参考图片"><input type="file" accept="image/png,image/jpeg,image/webp" required onChange={(event) => setFile(event.target.files?.[0] || null)} /></Field>{mode === 'external_cached' && <Field label="外部图片来源链接"><input type="url" value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} placeholder="https://…" /></Field>}</>}
      <label className="check-line"><input type="checkbox" checked={singleCharacter} onChange={(event) => setSingleCharacter(event.target.checked)} />我已核对图片 / 裁剪中只有这个角色</label>
      <label className="check-line"><input type="checkbox" checked={confirmed} disabled={!singleCharacter} onChange={(event) => setConfirmed(event.target.checked)} />人工确认身份，质量通过后加入可信库</label>
      <button className="button secondary" type="submit" disabled={!!busy || (mode !== 'frame_crop' && !file) || (confirmed && !singleCharacter)}>{busy === 'add' ? <Spinner label="正在保存参考图" /> : '保存参考图'}</button>
    </form>
    {loading ? <Spinner label="正在读取参考图" /> : !references.length ? <p className="helper">尚无参考图片。首次识别前也可以添加，无需先有出场记录。</p> : <div className="identity-reference-grid">{references.map((reference) => <article className="identity-reference" key={reference.id}><img src={reference.image_url} alt={`${profile.name}的参考图 ${reference.id}`} loading="lazy" /><div><strong>{identitySourceLabel(reference.source.kind)}</strong><span className="tag">{identityStatusLabel(reference.status)} · v{reference.version}</span><p className="helper">{reference.quality.width} × {reference.quality.height} · 清晰度 {reference.quality.sharpness == null ? '未测量' : reference.quality.sharpness.toFixed(1)}</p>{reference.source.url && <a href={reference.source.url} target="_blank" rel="noreferrer">查看来源</a>}{reference.quality.reasons?.length ? <p className="warning-text">质量限制：{reference.quality.reasons.join('；')}</p> : null}
        {reference.status !== 'confirmed' && <label className="check-line"><input type="checkbox" checked={!!checked[reference.id]} onChange={(event) => setChecked({ ...checked, [reference.id]: event.target.checked })} />已核对角色身份与单人图片</label>}
        <div className="button-row">{reference.status !== 'confirmed' && <button className="text-button" disabled={!!busy || !checked[reference.id]} onClick={() => void review(reference, 'confirmed')}>确认参考图</button>}{reference.status !== 'revoked' && <button className="text-button" disabled={!!busy} onClick={() => void review(reference, 'revoked')}>撤销参考图</button>}</div></div></article>)}</div>}
  </section>;
}

export function VoiceSamples({ assetId, profile, onSaved }: { assetId: string; profile: CharacterProfile; onSaved: () => Promise<void> }) {
  const endpoint = `/assets/${encodeURIComponent(assetId)}/characters/${encodeURIComponent(profile.id)}/voice-samples`;
  const [samples, setSamples] = useState<VoiceSample[]>([]); const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(''); const [error, setError] = useState('');
  const [source, setSource] = useState<'original_clip' | 'external_licensed'>('original_clip');
  const [recordId, setRecordId] = useState(''); const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState('ja'); const [dubVersion, setDubVersion] = useState('original');
  const [rightsNote, setRightsNote] = useState(''); const [confirmed, setConfirmed] = useState(false);
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const load = useCallback(async () => { try { const result = await api<{ samples: VoiceSample[] }>(endpoint); setSamples(result.samples || []); } catch (caught) { setError(message(caught)); } finally { setLoading(false); } }, [endpoint]);
  useEffect(() => { void load(); }, [load]);
  const add = async (event: FormEvent) => {
    event.preventDefault(); if (busy) return; setError('');
    if (file && file.size > 10 * 1024 * 1024) { setError('请选取不超过 10 MB 的短声音样本。'); return; }
    setBusy('add');
    try {
      let audioBase64: string | undefined;
      if (source === 'external_licensed' && file) audioBase64 = await new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1]); reader.onerror = () => reject(new Error('无法读取声音文件')); reader.readAsDataURL(file); });
      await post(endpoint, { ...(source === 'original_clip' ? { record_id: recordId } : { audio_base64: audioBase64, filename: file?.name }), source_kind: source, language, dub_version: dubVersion, rights_note: rightsNote, confirmed });
      await load(); await onSaved(); setConfirmed(false);
    } catch (caught) { setError(message(caught)); } finally { setBusy(''); }
  };
  const review = async (sample: VoiceSample, status: VoiceSample['status']) => {
    setBusy(sample.id); setError(''); try { await api(`${endpoint}/${encodeURIComponent(sample.id)}`, { method: 'PATCH', body: JSON.stringify({ status, note: status === 'revoked' ? '用户撤销错误声音样本' : '用户试听并核对说话人身份' }) }); await load(); await onSaved(); } catch (caught) { setError(message(caught)); } finally { setBusy(''); }
  };
  return <section aria-label="角色声音样本"><p className="helper">声音库按作品、语言、配音版本和角色隔离。同一声优不能证明同一角色。单人画面伴随说话仅可提议样本，仍需人工核对说话人。</p>{error && <Alert tone="error">{error}</Alert>}
    <form className="identity-add-form" onSubmit={add}><Field label="声音样本来源"><select value={source} onChange={(event) => setSource(event.target.value as typeof source)}><option value="original_clip">已保存的原片声音记录</option><option value="external_licensed">有权使用的外部样本</option></select></Field>
      {source === 'original_clip' ? <Field label="声音记录 ID" hint="从「视听身份」中复制已保存的声音记录 ID。"><input required value={recordId} onChange={(event) => setRecordId(event.target.value)} /></Field> : <><Field label="选择声音样本" hint="16 kHz 单声道 PCM16 WAV，最多 10 MB。"><input type="file" accept="audio/wav,audio/x-wav,.wav" required onChange={(event) => setFile(event.target.files?.[0] || null)} /></Field><Field label="使用权与来源说明"><textarea rows={2} required value={rightsNote} onChange={(event) => setRightsNote(event.target.value)} /></Field></>}
      <div className="form-grid"><Field label="样本语言"><input required value={language} onChange={(event) => setLanguage(event.target.value)} placeholder="ja / zh / en" /></Field><Field label="样本配音版本"><input required value={dubVersion} onChange={(event) => setDubVersion(event.target.value)} placeholder="original" /></Field></div>
      <label className="check-line"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />我已试听，确认只有这个角色在说话</label><button className="button secondary" type="submit" disabled={!!busy || (source === 'external_licensed' && !file)}>{busy === 'add' ? <Spinner label="正在保存声音样本" /> : '保存声音样本'}</button>
    </form>
    {loading ? <Spinner label="正在读取声音样本" /> : !samples.length ? <p className="helper">暂无声音样本。未确认的提议不会自动进入可信声音库。</p> : <div className="voice-sample-list">{samples.map((sample) => <article className="identity-record" key={sample.id}><strong>{identitySourceLabel(sample.source_kind)}</strong><span className="tag">{identityStatusLabel(sample.status)} · v{sample.version}</span><p className="helper">{sample.language} · {sample.dub_version} · {sample.id}</p><audio controls preload="none" aria-label={`${profile.name}的声音样本 ${sample.id}`} src={`/api${endpoint}/${encodeURIComponent(sample.id)}/audio`} />{sample.status !== 'confirmed' && <label className="check-line"><input type="checkbox" checked={!!checked[sample.id]} onChange={(event) => setChecked({ ...checked, [sample.id]: event.target.checked })} />已试听并确认说话人</label>}<div className="button-row">{sample.status !== 'confirmed' && <button className="text-button" disabled={!!busy || !checked[sample.id]} onClick={() => void review(sample, 'confirmed')}>确认声音样本</button>}{sample.status !== 'revoked' && <button className="text-button" disabled={!!busy} onClick={() => void review(sample, 'revoked')}>撤销声音样本</button>}</div><details><summary>来源与质量记录</summary><pre className="json-report">{JSON.stringify({ provenance: sample.provenance, quality: sample.quality }, null, 2)}</pre></details></article>)}</div>}
  </section>;
}
