import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { api, assetLink, formatTime, message, post } from '../api';
import type { CharacterLibrary } from '../types';
import { Alert, Field, Spinner } from './ui';

export interface KnowledgeTarget { kind: 'concept' | 'character'; concept_id?: string; asset_id?: string; character_scope_id?: string; character_id?: string }
interface Source { source_id: string; snapshot_id: string; title: string; url: string; filename?: string; status: string; text?: string }
interface SourceRef { source_id: string; snapshot_id: string; start: number; end: number }
interface Citation extends Partial<SourceRef> { title?: string; url?: string; quote?: string }
interface Card { card_id: string; kind?: string; predicate?: string; object_ref?: KnowledgeTarget | null; title?: string; text: string; target_ref: KnowledgeTarget; review_status: string; effective_status?: string; sources?: Citation[]; source_refs?: SourceRef[]; applicability?: string; conditions?: string[]; costs?: string[]; exceptions?: string[] }
export interface KnowledgeHit extends Partial<Card> { target_name?: string; has_film_evidence?: boolean; mapping_status?: string; mapping_basis?: string; match_basis?: string; record_ids?: string[] }
interface FilmHit { id?: string; record_id?: string; asset_id: string; start_ms: number; end_ms: number; match_basis?: string }
interface QueryResponse { knowledge_hits: KnowledgeHit[]; film_hits?: FilmHit[]; record_candidates?: FilmHit[]; diagnostics?: Record<string, unknown>; degraded?: string[] }
interface Concept { concept_id: string; name: string; type?: string }
interface ReferenceCandidate { candidate_id: string; concept_id: string; reference_ids: string[]; asset_id: string; record_id: string; frame_id: string; at_ms: number; start_ms: number; end_ms: number; similarity: number; status: string }
interface GraphCandidate { entity_name?: string; id?: string; description?: string; source_id?: string; graph_kind?: string }
interface GraphRelation { source?: string; target?: string; src_id?: string; tgt_id?: string; description?: string }
interface Alignment { upstream_key: string; target_refs: KnowledgeTarget[]; supporting_card_ids: string[]; status: string }
interface BangumiKnowledgeImport { source_count: number; card_count: number; created_sources: number; created_cards: number; warnings?: string[] }
const list = <T,>(value: unknown, key: string): T[] => Array.isArray(value) ? value : value && typeof value === 'object' && Array.isArray((value as Record<string, unknown>)[key]) ? (value as Record<string, T[]>)[key] : [];
const lines = (value: string) => value.split('\n').map((line) => line.trim()).filter(Boolean);
const reviewLabel = (value: string) => ({ candidate: '抽取候选 / 待核对', reviewed: '已核对', rejected: '已拒绝', revoked: '已撤销', stale: '来源或目标已失效', unavailable: '来源或目标已失效', unresolved: '身份待消歧', ambiguous: '身份有歧义' } as Record<string, string>)[value] || value;
const targetValue = (ref?: KnowledgeTarget | null) => ref ? `${ref.kind}:${ref.kind === 'concept' ? ref.concept_id : ref.character_id}` : '';

/** Code-point offsets match Python slicing, including astral Unicode characters. */
export function quoteReference(source: Source, text: string, quote: string): SourceRef {
  const index = text.indexOf(quote);
  if (!quote.trim() || index < 0) throw new Error('引用须与原文完全一致。');
  if (text.indexOf(quote, index + 1) >= 0) throw new Error('这段文字在原文中重复，请扩大引用范围以唯一定位。');
  const start = Array.from(text.slice(0, index)).length;
  return { source_id: source.source_id, snapshot_id: source.snapshot_id, start, end: start + Array.from(quote).length };
}

export function KnowledgeResults({ hits, onAppearances }: { hits: KnowledgeHit[]; onAppearances?: (target: KnowledgeTarget) => void }) {
  return <div className="knowledge-results">{hits.map((hit, index) => <article className="work-entity-details" key={hit.card_id || `${hit.target_name}-${index}`}>
    <h4>{hit.title || hit.target_name || '相关知识'} <span className="tag">资料说明</span></h4>
    {hit.target_name && <p>{hit.target_name}</p>}<p>{hit.text}</p>
    {hit.applicability && <p className="helper">适用阶段：{hit.applicability}</p>}
    {([['conditions', '条件'], ['costs', '代价'], ['exceptions', '例外']] as const).map(([field, label]) => hit[field]?.length ? <p key={field}>{label}：{hit[field]!.join('；')}</p> : null)}
    {hit.review_status && <span className="tag">{reviewLabel(hit.effective_status || hit.review_status)}</span>}
    {hit.mapping_status && !['reviewed', 'mapped'].includes(hit.mapping_status) && <span className="tag">{hit.mapping_status === 'ambiguous' ? '身份待消歧' : '关联待核对'}</span>}
    {hit.mapping_basis === 'reviewed_graph_alignment' && <p className="helper">关联依据：已核对的图谱身份与来源资料卡。</p>}
    {(hit.sources || []).map((source, sourceIndex) => <div key={`${source.source_id}-${sourceIndex}`}><p className="helper">出处：{source.url ? <a href={source.url} target="_blank" rel="noreferrer">{source.title || source.url}</a> : source.title || source.source_id}</p>{source.quote && <blockquote>{source.quote}</blockquote>}</div>)}
    <p className="helper">{hit.has_film_evidence ? '已关联当前影片证据；出场不证明设定正在发生。' : '找到相关设定，尚无已确认影片证据。'}</p>
    {hit.target_ref && onAppearances && <button type="button" className="button small secondary" onClick={() => onAppearances(hit.target_ref!)}>查找{hit.target_ref.kind === 'character' ? '该角色' : '该概念'}出场</button>}
  </article>)}</div>;
}

export default function KnowledgePanel({ assetId, scopeId, entities }: { assetId: string; scopeId: string; entities: Concept[] }) {
  const base = `/work-entity-scopes/${encodeURIComponent(scopeId)}/knowledge`;
  const [open, setOpen] = useState(false);
  const [sources, setSources] = useState<Source[]>([]);
  const [cards, setCards] = useState<Card[]>([]);
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [characters, setCharacters] = useState<CharacterLibrary | null>(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [url, setUrl] = useState('');
  const [sourceId, setSourceId] = useState('');
  const [sourceText, setSourceText] = useState('');
  const [quote, setQuote] = useState('');
  const [refs, setRefs] = useState<SourceRef[]>([]);
  const [target, setTarget] = useState('');
  const [cardTitle, setCardTitle] = useState('');
  const [editingCard, setEditingCard] = useState('');
  const [cardKind, setCardKind] = useState('fact');
  const [predicate, setPredicate] = useState('');
  const [objectTarget, setObjectTarget] = useState('');
  const [text, setText] = useState('');
  const [conditions, setConditions] = useState('');
  const [costs, setCosts] = useState('');
  const [exceptions, setExceptions] = useState('');
  const [applicability, setApplicability] = useState('');
  const [review, setReview] = useState('candidate');
  const [reason, setReason] = useState('');
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('baseline');
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [referenceCandidates, setReferenceCandidates] = useState<ReferenceCandidate[]>([]);
  const [referenceStale, setReferenceStale] = useState(false);
  const [referenceReason, setReferenceReason] = useState('');
  const [graphCandidates, setGraphCandidates] = useState<GraphCandidate[]>([]);
  const [alignments, setAlignments] = useState<Alignment[]>([]);
  const [upstreamKey, setUpstreamKey] = useState('');
  const [alignmentTarget, setAlignmentTarget] = useState('');
  const [alignmentCard, setAlignmentCard] = useState('');
  const [alignmentReason, setAlignmentReason] = useState('');
  const load = useCallback(async () => {
    const [sourceRows, cardRows, current] = await Promise.all([api(`${base}/sources`), api(`${base}/cards`), api<Record<string, unknown>>(`${base}/status`)]);
    setSources(list<Source>(sourceRows, 'sources')); setCards(list<Card>(cardRows, 'cards')); setStatus(current);
  }, [base]);
  useEffect(() => {
    if (!open) return;
    let active = true;
    void load().catch((caught) => { if (active) setError(message(caught)); });
    void api<CharacterLibrary>(`/assets/${encodeURIComponent(assetId)}/characters`).then((value) => { if (active && Array.isArray(value.profiles)) setCharacters(value); }).catch(() => { /* Concepts remain available without a cast library. */ });
    return () => { active = false; };
  }, [open, load, assetId]);
  useEffect(() => {
    setSourceText(''); setQuote('');
    if (!sourceId) return;
    const controller = new AbortController();
    void api<Source & { content?: string }>(`${base}/sources/${encodeURIComponent(sourceId)}`, { signal: controller.signal }).then((value) => { if (!controller.signal.aborted) setSourceText(value.text ?? value.content ?? ''); }).catch((caught) => { if (!controller.signal.aborted) setError(message(caught)); });
    return () => controller.abort();
  }, [sourceId, base]);
  const operate = async (name: string, fn: () => Promise<void>) => {
    setBusy(name); setError(''); setNotice('');
    try { await fn(); } catch (caught) { setError(message(caught)); } finally { setBusy(''); }
  };
  const importSource = (event: FormEvent) => {
    event.preventDefault(); if (!file) return;
    void operate('import', async () => {
      const body = new FormData(); body.set('file', file); body.set('title', title || file.name); body.set('url', url);
      await api(`${base}/sources`, { method: 'POST', body }); await load(); setNotice('原文已保存。选择引用并核对资料卡后再建库。');
    });
  };
  const selectedSource = sources.find((row) => row.source_id === sourceId);
  const targetRef = (selection = target): KnowledgeTarget => {
    if (selection.startsWith('concept:')) return { kind: 'concept', concept_id: selection.slice(8) };
    if (selection.startsWith('character:') && characters) return { kind: 'character', asset_id: assetId, character_scope_id: characters.scope_id, character_id: selection.slice(10) };
    throw new Error('请选择现有概念或角色。');
  };
  const saveCard = (event: FormEvent) => {
    event.preventDefault();
    void operate('card', async () => {
      const sourceRefs = [...refs];
      if (quote.trim() && selectedSource) { const next = quoteReference(selectedSource, sourceText, quote); if (!sourceRefs.some((row) => JSON.stringify(row) === JSON.stringify(next))) sourceRefs.push(next); }
      if (!sourceRefs.length) throw new Error('至少添加一段可追溯的原文引用。');
      const payload = { title: cardTitle, kind: cardKind, predicate: cardKind === 'relation' ? predicate : '', object_ref: cardKind === 'relation' ? targetRef(objectTarget) : null, text, target_ref: targetRef(), source_refs: sourceRefs, conditions: lines(conditions), costs: lines(costs), exceptions: lines(exceptions), applicability, review_status: review, reason };
      if (editingCard) await api(`${base}/cards/${encodeURIComponent(editingCard)}`, { method: 'PATCH', body: JSON.stringify(payload) });
      else await post(`${base}/cards`, payload);
      await load(); setEditingCard(''); setText(''); setCardTitle(''); setRefs([]); setQuote(''); setReview('candidate'); setNotice('资料卡已保存。只有已核对且来源有效的资料会进入检索索引。');
    });
  };
  const search = (event?: FormEvent, explicitTarget?: KnowledgeTarget) => {
    event?.preventDefault(); if (!query.trim() && !explicitTarget) return;
    void operate('query', async () => { setResult(await post<QueryResponse>(`${base}/query`, { query: query.trim() || '出场', mode, filters: { asset_id: assetId }, ...(explicitTarget ? { target_ref: explicitTarget } : {}) })); });
  };
  const filmHits = result?.film_hits || result?.record_candidates || [];
  const loadReferenceCandidates = async () => {
    const response = await api<{ candidates?: ReferenceCandidate[]; stale?: boolean }>(`${base}/reference-candidates`);
    setReferenceCandidates(response.candidates || []); setReferenceStale(!!response.stale);
  };
  const targetOptions = <><option value="">选择内部概念或角色</option><optgroup label="作品概念">{entities.map((row) => <option value={`concept:${row.concept_id}`} key={row.concept_id}>{row.name}</option>)}</optgroup>{characters && <optgroup label="现有角色">{characters.profiles.map((row) => <option value={`character:${row.id}`} key={row.id}>{row.name}</option>)}</optgroup>}</>;
  return <details className="advanced knowledge-panel" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>知识资料与检索 · LightRAG 实验</summary>
    {open && <>
      <p className="helper">导入有出处的文本，关联现有概念或角色。知识说明、台词提及和影片出场分别核对。</p>
      {error && <Alert tone="error">{error}</Alert>}{notice && <Alert tone="success">{notice}</Alert>}
      <div className="button-row"><button type="button" className="button secondary" disabled={!!busy} onClick={() => void operate('status', load)}>刷新建库状态</button><button type="button" className="button secondary" disabled={!!busy} onClick={() => void operate('build', async () => { const job = await post<{ id?: string; job_id?: string }>(`${base}/build`); setNotice(`建库任务已提交：${job.job_id || job.id || '请查看任务队列'}`); await load(); })}>重建知识索引</button><a href="#/jobs">查看任务队列</a><a href="#/settings">配置知识后端</a></div>
      {status && <p className="helper" role="status">知识索引：{String(status.status || status.state || '未建库')}{status.message || status.error ? ` · ${String(status.message || status.error)}` : ''}</p>}
      <section aria-label="复用 Bangumi 资料"><h3>复用 Bangumi 资料</h3><p className="helper">使用这部动画已保存的 Bangumi 档案，整理作品简介、角色说明和资料字段，并为已有角色生成待核对资料卡。此操作不重新联网读取，也不调用模型；重复整理保留已有核对结果。</p>
        <button type="button" className="button secondary" disabled={!!busy} onClick={() => void operate('bangumi', async () => {
          const imported = await post<BangumiKnowledgeImport>(`${base}/bangumi`, { asset_id: assetId });
          await load(); setResult(null);
          setNotice(`Bangumi 资料已整理：新增 ${imported.created_sources} 份来源、${imported.created_cards} 张待核对资料卡；现有 ${imported.source_count} 份来源、${imported.card_count} 张资料卡。${imported.warnings?.length ? ` 提示：${imported.warnings.join('；')}` : ''}`);
        })}>{busy === 'bangumi' ? <Spinner label="整理中" /> : '整理已有 Bangumi 资料'}</button>
        <p className="helper">作品简介可在下方选择并引用。角色卡核对后才参与建库；“主角／配角”等演出标记不代表角色之间的关系，也不证明影片事件。条目未提供的能力或规则仍需补充来源。</p>
      </section>
      <form onSubmit={importSource}><h3>导入来源正文</h3><div className="form-grid"><Field label="文本资料文件"><input type="file" accept=".md,.markdown,.txt,text/plain,text/markdown" required disabled={!!busy} onChange={(event) => setFile(event.target.files?.[0] || null)} /></Field><Field label="资料标题"><input value={title} onChange={(event) => setTitle(event.target.value)} /></Field><Field label="原文出处 URL"><input type="url" value={url} onChange={(event) => setUrl(event.target.value)} /></Field></div><button className="button secondary" disabled={!!busy || !file}>{busy === 'import' ? <Spinner label="导入中" /> : '保存来源正文'}</button></form>
      <form onSubmit={saveCard}><h3>{editingCard ? "编辑资料卡" : "整理资料卡"}</h3><Field label="资料卡类型"><select value={cardKind} onChange={(event) => setCardKind(event.target.value)}><option value="fact">设定事实</option><option value="definition">定义</option><option value="relation">关系</option></select></Field><Field label="资料卡关联对象"><select value={target} onChange={(event) => setTarget(event.target.value)} required>{targetOptions}</select></Field>{cardKind === "relation" && <><Field label="关系谓词"><input value={predicate} onChange={(event) => setPredicate(event.target.value)} required /></Field><Field label="关系目标"><select value={objectTarget} onChange={(event) => setObjectTarget(event.target.value)} required>{targetOptions}</select></Field></>}
        <Field label="来源正文"><select value={sourceId} onChange={(event) => setSourceId(event.target.value)}><option value="">选择来源</option>{sources.filter((row) => row.status !== 'revoked').map((row) => <option key={row.source_id} value={row.source_id}>{row.title}</option>)}</select></Field>
        {sourceText && <details><summary>查看保存的原文</summary><pre className="json-report">{sourceText}</pre></details>}
        <Field label="原文引用" hint="粘贴一段完整、唯一的原文；可为同一卡片添加多个来源。"><textarea value={quote} onChange={(event) => setQuote(event.target.value)} rows={3} /></Field>
        <button type="button" className="button small secondary" disabled={!quote.trim() || !selectedSource} onClick={() => { try { const next = quoteReference(selectedSource!, sourceText, quote); setRefs((old) => [...old.filter((row) => JSON.stringify(row) !== JSON.stringify(next)), next]); setQuote(''); setError(''); } catch (caught) { setError(message(caught)); } }}>添加原文引用</button>
        {refs.length > 0 && <ul>{refs.map((ref, index) => <li key={`${ref.source_id}-${ref.start}-${ref.end}`}>{sources.find((row) => row.source_id === ref.source_id)?.title} · 字符 {ref.start}–{ref.end} <button type="button" className="text-button" onClick={() => setRefs((old) => old.filter((_, position) => position !== index))}>移除引用</button></li>)}</ul>}
        <Field label="资料卡标题"><input value={cardTitle} onChange={(event) => setCardTitle(event.target.value)} /></Field><Field label="定义或设定说明"><textarea required value={text} onChange={(event) => setText(event.target.value)} rows={3} /></Field>
        <div className="form-grid"><Field label="适用阶段"><input value={applicability} onChange={(event) => setApplicability(event.target.value)} /></Field><Field label="条件（每行一条）"><textarea value={conditions} onChange={(event) => setConditions(event.target.value)} /></Field><Field label="代价（每行一条）"><textarea value={costs} onChange={(event) => setCosts(event.target.value)} /></Field><Field label="例外（每行一条）"><textarea value={exceptions} onChange={(event) => setExceptions(event.target.value)} /></Field></div>
        <Field label="资料核对状态"><select value={review} onChange={(event) => setReview(event.target.value)}><option value="candidate">待核对候选</option><option value="reviewed">我已核对原文与内部关联</option></select></Field><button className="button primary" disabled={!!busy || !text.trim() || !target}>{busy === 'card' ? <Spinner label="保存中" /> : '保存资料卡'}</button>{editingCard && <button type="button" className="button secondary" onClick={() => { setEditingCard(''); setText(''); setRefs([]); setCardTitle(''); setQuote(''); setReview('candidate'); }}>取消编辑资料卡</button>}
      </form>
      {cards.length > 0 && <section><h3>已保存资料卡</h3><Field label="资料审核或撤销原因"><input value={reason} onChange={(event) => setReason(event.target.value)} /></Field>{cards.map((card) => <article className="work-entity-details" key={card.card_id}><h4>{card.title || card.text.slice(0, 50)}</h4><p>{card.text}</p><span className="tag">{reviewLabel(card.effective_status || card.review_status)}</span><div className="button-row"><button type="button" className="button small secondary" disabled={!!busy} onClick={() => { setEditingCard(card.card_id); setCardKind(card.kind || 'fact'); setPredicate(card.predicate || ''); setObjectTarget(targetValue(card.object_ref)); setText(card.text); setCardTitle(card.title || ''); setTarget(targetValue(card.target_ref)); setRefs(card.source_refs || []); setQuote(''); setConditions((card.conditions || []).join('\n')); setCosts((card.costs || []).join('\n')); setExceptions((card.exceptions || []).join('\n')); setApplicability(card.applicability || ''); setReview(card.review_status); }}>编辑资料卡</button><button type="button" className="button small secondary" disabled={!!busy || !reason.trim() || card.review_status === 'reviewed'} onClick={() => void operate('review', async () => { await api(`${base}/cards/${encodeURIComponent(card.card_id)}`, { method: 'PATCH', body: JSON.stringify({ review_status: 'reviewed', reason }) }); await load(); })}>核对资料卡</button><button type="button" className="button small secondary" disabled={!!busy || !reason.trim() || card.review_status === 'revoked'} onClick={() => void operate('revoke', async () => { await api(`${base}/cards/${encodeURIComponent(card.card_id)}`, { method: 'DELETE', body: JSON.stringify({ reason }) }); await load(); setResult(null); })}>撤销资料卡</button></div></article>)}</section>}
      {sources.length > 0 && <details><summary>来源状态与撤销</summary>{sources.map((source) => <div key={source.source_id}><span>{source.title} · {source.status === 'revoked' ? '已撤销' : '有效'}</span>{source.url && <> · <a href={source.url} target="_blank" rel="noreferrer">原文出处</a></>}<button type="button" className="text-button" disabled={!!busy || !reason.trim() || source.status === 'revoked'} onClick={() => void operate('source-revoke', async () => { await api(`${base}/sources/${encodeURIComponent(source.source_id)}`, { method: 'DELETE', body: JSON.stringify({ reason }) }); await load(); setResult(null); })}>撤销来源</button></div>)}<p className="helper">撤销原因填写在资料审核区域；撤销后相关资料立即退出检索。</p></details>}
      <details><summary>抽取候选与内部身份核对</summary>
        <p className="helper">抽出的名称只作为候选。选择来源资料卡和内部对象，核对后保存身份关联；关系须核对显示的方向，并选择相应关系资料卡。</p>
        <button type="button" className="button secondary" disabled={!!busy} onClick={() => void operate('graph', async () => { const [graph, mapped] = await Promise.all([api<Record<string, unknown>>(`${base}/graph-candidates`), api(`${base}/alignments`)]); setGraphCandidates([...list<GraphCandidate>(graph, 'entities'), ...list<GraphRelation>(graph, 'relationships').flatMap((edge) => { const source = edge.source || edge.src_id; const target = edge.target || edge.tgt_id; return source && target ? [{ id: source + '->' + target, description: edge.description, graph_kind: 'relation' }] : []; })]); setAlignments(list<Alignment>(mapped, 'alignments')); })}>读取抽取候选</button>
        <form onSubmit={(event) => { event.preventDefault(); void operate('alignment', async () => { await post(`${base}/alignments`, { upstream_key: upstreamKey, target_refs: [targetRef(alignmentTarget)], supporting_card_ids: [alignmentCard], status: 'reviewed', reason: alignmentReason }); const response = await api(`${base}/alignments`); setAlignments(list<Alignment>(response, 'alignments')); setNotice('身份关联已保存；重建知识索引后使用新的映射。'); }); }}>
          <Field label="待核对抽取名称"><select value={upstreamKey} onChange={(event) => setUpstreamKey(event.target.value)} required><option value="">选择派生图中的名称</option>{graphCandidates.map((row, index) => { const key = row.entity_name || row.id || ''; return <option value={key} key={`${key}-${index}`}>{row.graph_kind === 'relation' ? '关系：' : '实体：'}{key}</option>; })}</select></Field>
          {graphCandidates.find((row) => (row.entity_name || row.id) === upstreamKey)?.description && <p>{graphCandidates.find((row) => (row.entity_name || row.id) === upstreamKey)?.description}</p>}
          <Field label="映射到内部对象"><select value={alignmentTarget} onChange={(event) => setAlignmentTarget(event.target.value)} required>{targetOptions}</select></Field>
          <Field label="支持身份映射的资料卡"><select value={alignmentCard} onChange={(event) => setAlignmentCard(event.target.value)} required><option value="">选择已核对资料卡</option>{cards.filter((card) => card.review_status === 'reviewed' && card.effective_status !== 'unavailable').map((card) => <option key={card.card_id} value={card.card_id}>{card.title || card.text.slice(0, 60)}</option>)}</select></Field>
          <Field label="身份核对依据"><input value={alignmentReason} onChange={(event) => setAlignmentReason(event.target.value)} required /></Field><button className="button secondary" disabled={!!busy || !upstreamKey || !alignmentTarget || !alignmentCard || !alignmentReason.trim()}>保存已核对身份关联</button>
        </form>
        {!!alignments.length && <ul>{alignments.map((row, index) => <li key={`${row.upstream_key}-${index}`}>{row.upstream_key} · {reviewLabel(row.status)} · {row.target_refs.length} 个内部对象</li>)}</ul>}
      </details>
      <form onSubmit={(event) => search(event)}><h3>查询知识与影片证据</h3><Field label="知识检索模式"><select value={mode} onChange={(event) => { setMode(event.target.value); setResult(null); }}><option value="baseline">词法知识基线</option><option value="lightrag">LightRAG 实验</option></select></Field><Field label="知识问题"><input value={query} onChange={(event) => setQuery(event.target.value)} required /></Field><button className="button primary" disabled={!!busy || !query.trim()}>{busy === 'query' ? <Spinner label="查询中" /> : '检索知识与片段'}</button></form>
      {result && <section aria-live="polite"><h3>知识命中 · {result.knowledge_hits?.length || 0}</h3>{result.degraded?.length ? <Alert>{result.degraded.join('；')}</Alert> : null}<KnowledgeResults hits={result.knowledge_hits || []} onAppearances={(ref) => search(undefined, ref)} />{!result.knowledge_hits?.length && <p className="helper">没有找到有效知识资料。</p>}<h3>已确认影片关联 · {filmHits.length}</h3>{filmHits.length ? <ul>{filmHits.map((row, index) => <li key={row.id || row.record_id || index}><a href={assetLink(row.asset_id, row.start_ms)}>{formatTime(row.start_ms)} — {formatTime(row.end_ms)}</a> <span className="tag">出场证据 · 非事件确认</span></li>)}</ul> : <p className="helper">没有已确认影片证据；相关设定仍可在上方查看出处。</p>}{result.diagnostics && <details><summary>耗时与调用诊断</summary><pre className="json-report">{JSON.stringify(result.diagnostics, null, 2)}</pre></details>}</section>}
      <details><summary>参考图候选发现 · 无 VLM</summary>
        <p className="helper">复用本作品已有影片帧特征，与已确认参考图比较。相似度结果保持候选，核对原帧后才能保存出场绑定。</p>
        <div className="button-row"><button type="button" className="button secondary" disabled={!!busy} onClick={() => void operate('reference-discovery', async () => { const job = await post<{ id?: string; job_id?: string }>(`${base}/reference-discovery`); setNotice(`参考图发现任务已提交：${job.job_id || job.id || '请查看任务队列'}`); })}>发现参考图候选</button><button type="button" className="button secondary" disabled={!!busy} onClick={() => void operate('reference-candidates', loadReferenceCandidates)}>刷新参考图候选</button></div>
        {referenceStale && <Alert>这些候选的参考图或影片证据已变化，请重新运行发现任务。</Alert>}
        <Field label="参考图出场核对依据"><input value={referenceReason} onChange={(event) => setReferenceReason(event.target.value)} placeholder="说明原帧中哪些特征支持此概念出场" /></Field>
        <div className="work-reference-list">{referenceCandidates.map((candidate) => <article key={candidate.candidate_id}><h4>{entities.find((entity) => entity.concept_id === candidate.concept_id)?.name || candidate.concept_id}</h4>{candidate.reference_ids[0] && <img src={`/api/work-entity-scopes/${encodeURIComponent(scopeId)}/entities/${encodeURIComponent(candidate.concept_id)}/references/${encodeURIComponent(candidate.reference_ids[0])}/image`} alt="已核对概念参考图" loading="lazy" />}<img src={`/api/assets/${encodeURIComponent(candidate.asset_id)}/frames/${encodeURIComponent(candidate.frame_id)}`} alt="候选影片原帧" loading="lazy" /><p>相似度 {candidate.similarity.toFixed(4)} · {formatTime(candidate.at_ms)} · {candidate.status === 'confirmed' ? '已人工确认' : '仅为候选'}</p><a href={assetLink(candidate.asset_id, candidate.at_ms)}>播放候选片段</a><button type="button" className="button small secondary" disabled={!!busy || referenceStale || !referenceReason.trim() || candidate.status === 'confirmed'} onClick={() => void operate('reference-confirm', async () => { await post(`${base}/reference-candidates/${encodeURIComponent(candidate.candidate_id)}/confirm`, { reason: referenceReason }); await loadReferenceCandidates(); setNotice('人工出场绑定已保存，知识召回现在可使用该影片关联。'); })}>核对并确认出场</button></article>)}</div>
      </details>
    </>}
  </details>;
}
