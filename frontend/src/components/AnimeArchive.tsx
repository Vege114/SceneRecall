import { useEffect, useRef, useState } from 'react';
import { ExternalLink } from 'lucide-react';
import { message, post } from '../api';
import type { AnimeProfile, AnimeProfileSelection, BangumiDossier, BangumiSearch, BangumiSubject, ImportInput, Project, Asset } from '../types';
import { Alert, Field, Spinner } from './ui';

type ArchiveStatus = 'waiting' | 'searching' | 'candidates' | 'not_found' | 'error' | 'reading' | 'ready' | 'scratch';
interface ArchiveState {
  signature: string; query: string; status: ArchiveStatus; candidates: BangumiSubject[];
  reference: string; error?: string; dossier?: BangumiDossier; selection?: AnimeProfileSelection;
}
interface ArchiveScope { key: string; signature: string; query: string; existing?: AnimeProfile; preset?: AnimeProfileSelection }
const normalizeTitle = (value: string) => value.normalize('NFKC').toLocaleLowerCase().replace(/[\s\p{P}\p{S}]/gu, '');
function scopeFor(row: ImportInput, project?: Project, sharedBatch = false, assets: Asset[] = []): ArchiveScope | undefined {
  if (row.kind !== 'animation') return;
  const query = row.title.trim();
  const matching = assets.find((asset) => (asset.project_id || null) === (project?.id || null) && normalizeTitle(asset.anime_title || asset.title) === normalizeTitle(query) && (asset.season ?? null) === row.season && asset.anime_profile);
  const legacy = normalizeTitle(query) === normalizeTitle(project?.name || '') ? project?.anime_profile : undefined;
  const saved = matching?.anime_profile || legacy;
  const existing = saved && (!row.anime_profile || JSON.stringify(profileSelection(saved)) === JSON.stringify(row.anime_profile)) ? saved : undefined;
  const identity = sharedBatch ? `batch:${normalizeTitle(query)}` : `item:${row.item_id}`;
  return { key: identity, query, signature: JSON.stringify([identity, query, project?.id || '', existing]), existing, preset: row.anime_profile };
}
function profileSelection(profile: AnimeProfile): AnimeProfileSelection {
  return profile.mode === 'bangumi' ? { mode: 'bangumi', subject_id: profile.dossier.subject.id } : { mode: 'scratch' };
}
const initialState = (scope: ArchiveScope): ArchiveState => ({ signature: scope.signature, query: scope.query, candidates: [], reference: '',
  status: scope.existing?.mode === 'scratch' || scope.preset?.mode === 'scratch' ? 'scratch' : scope.existing || scope.preset ? 'ready' : scope.query ? 'searching' : 'waiting',
  dossier: scope.existing?.mode === 'bangumi' ? scope.existing.dossier : undefined,
  selection: scope.existing ? profileSelection(scope.existing) : scope.preset });

export function useAnimeArchives(items: ImportInput[], project?: Project, sharedBatch = false, assets: Asset[] = []) {
  const [archives, setArchives] = useState<Record<string, ArchiveState>>({});
  const requests = useRef<Record<string, { signature: string; version: number; timer?: number }>>({});
  const sequence = useRef(0);
  const scopes = [...new Map(items.map((row) => scopeFor(row, project, sharedBatch, assets)).filter((scope): scope is ArchiveScope => !!scope).map((scope) => [scope.key, scope])).values()];
  const scopeKey = JSON.stringify(scopes);
  const invalidate = (scope: ArchiveScope) => {
    const prior = requests.current[scope.key];
    if (prior?.timer) window.clearTimeout(prior.timer);
    const request = { signature: scope.signature, version: ++sequence.current };
    requests.current[scope.key] = request;
    return request.version;
  };
  const isCurrent = (scope: ArchiveScope, version: number) => requests.current[scope.key]?.signature === scope.signature && requests.current[scope.key]?.version === version;
  const save = (scope: ArchiveScope, version: number, values: Partial<ArchiveState>) => {
    if (isCurrent(scope, version)) setArchives((current) => ({ ...current, [scope.key]: { ...(current[scope.key]?.signature === scope.signature ? current[scope.key] : initialState(scope)), ...values } }));
  };
  const read = async (scope: ArchiveScope, reference: string, version = invalidate(scope)) => {
    save(scope, version, { status: 'reading', error: '', selection: undefined, dossier: undefined });
    try {
      const dossier = await post<BangumiDossier>('/bangumi/subject', { reference });
      if (dossier.source !== 'bangumi' || !Number.isInteger(dossier.subject?.id)) throw new Error('Bangumi 条目信息不完整，请重试或选择从零建档。');
      save(scope, version, { status: 'ready', dossier, selection: { mode: 'bangumi', subject_id: dossier.subject.id }, error: '' });
    } catch (error) { save(scope, version, { status: 'error', error: message(error) }); }
  };
  const search = async (scope: ArchiveScope, version: number) => {
    save(scope, version, { status: 'searching', error: '', dossier: undefined, selection: undefined, candidates: [] });
    try {
      const result = await post<BangumiSearch>('/bangumi/search', { query: scope.query });
      if (!isCurrent(scope, version)) return;
      if (result.status === 'error') throw new Error(result.error || '暂时无法访问 Bangumi。');
      const candidates = [...new Map((result.candidates || []).map((candidate) => [candidate.id, candidate])).values()];
      const exact = candidates.filter((candidate) => [candidate.name, candidate.name_cn].some((name) => name && normalizeTitle(name) === normalizeTitle(scope.query)));
      if (exact.length === 1) { await read(scope, String(exact[0].id), version); return; }
      save(scope, version, { status: candidates.length ? 'candidates' : 'not_found', candidates });
    } catch (error) { save(scope, version, { status: 'error', error: message(error) }); }
  };
  useEffect(() => {
    const active = new Set(scopes.map((scope) => scope.key));
    for (const [key, request] of Object.entries(requests.current)) {
      if (!active.has(key)) { if (request.timer) window.clearTimeout(request.timer); delete requests.current[key]; }
    }
    setArchives((current) => {
      const next = Object.fromEntries(Object.entries(current).filter(([key]) => active.has(key)));
      for (const scope of scopes) if (next[scope.key]?.signature !== scope.signature) next[scope.key] = initialState(scope);
      return next;
    });
    for (const scope of scopes) {
      if (requests.current[scope.key]?.signature === scope.signature) continue;
      const version = invalidate(scope);
      if (!scope.existing && !scope.preset && scope.query) requests.current[scope.key].timer = window.setTimeout(() => { void search(scope, version); }, 450);
    }
    // Only work identity changes invalidate discovery; subtitle and progress edits do not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);
  useEffect(() => () => {
    for (const request of Object.values(requests.current)) if (request.timer) window.clearTimeout(request.timer);
    requests.current = {};
  }, []);
  const stateFor = (row: ImportInput) => {
    const scope = scopeFor(row, project, sharedBatch, assets);
    if (!scope) return undefined;
    const state = archives[scope.key];
    return state?.signature === scope.signature ? state : initialState(scope);
  };
  return {
    stateFor,
    selectionFor: (row: ImportInput) => stateFor(row)?.selection,
    existingFor: (row: ImportInput) => scopeFor(row, project, sharedBatch, assets)?.existing,
    setReference: (row: ImportInput, reference: string) => {
      const scope = scopeFor(row, project, sharedBatch, assets); if (!scope) return;
      setArchives((current) => ({ ...current, [scope.key]: { ...stateFor(row)!, reference } }));
    },
    read: (row: ImportInput, reference: string) => { const scope = scopeFor(row, project, sharedBatch, assets); if (scope) void read(scope, reference); },
    scratch: (row: ImportInput) => {
      const scope = scopeFor(row, project, sharedBatch, assets); if (!scope) return;
      const version = invalidate(scope); save(scope, version, { status: 'scratch', selection: { mode: 'scratch' }, dossier: undefined, error: '' });
    },
    searchAgain: (row: ImportInput) => { const scope = scopeFor(row, project, sharedBatch, assets); if (scope?.query) void search(scope, invalidate(scope)); },
  };
}

export function AnimeArchivePreview({ profile }: { profile: AnimeProfile }) {
  if (profile.mode === 'scratch') return <p className="anime-scratch">从零建立档案 · 仅为能够确认姓名的人物建档，后续识别补充出场证据。</p>;
  const { subject, characters } = profile.dossier;
  return <div className="anime-archive-preview">
    {subject.image_url && <img src={subject.image_url} alt={`${subject.name_cn || subject.name}的 Bangumi 封面`} loading="lazy" referrerPolicy="no-referrer" />}
    <div><strong>{subject.name_cn || subject.name}</strong>{subject.name_cn && subject.name !== subject.name_cn && <span>{subject.name}</span>}
      <p className="helper">{subject.date ? `${subject.date} · ` : ''}Bangumi 初始档案 · {characters.length} 份现成角色资料</p>
      {subject.summary && <details><summary>作品简介</summary><p className="anime-summary">{subject.summary}</p></details>}
      {characters.length > 0 && <p className="anime-cast">{characters.slice(0, 8).map((character) => character.name_cn || character.name).join('、')}{characters.length > 8 ? '…' : ''}</p>}
      <a href={subject.url} target="_blank" rel="noreferrer"><ExternalLink size={12} />查看 Bangumi 条目</a>
      {profile.dossier.warnings?.length ? <Alert>{profile.dossier.warnings.join('；')}</Alert> : null}
    </div>
  </div>;
}

export default function AnimeArchivePicker({ item, archives, readOnly = false }: { item: ImportInput; archives: ReturnType<typeof useAnimeArchives>; readOnly?: boolean }) {
  const state = archives.stateFor(item);
  if (!state) return null;
  const existing = archives.existingFor(item);
  const loading = ['searching', 'reading'].includes(state.status);
  return <section className="anime-archive-picker" aria-label="动画初始档案" aria-live="polite">
    <h3>动画初始档案</h3><p className="helper">使用 Bangumi 列出的角色档案，后续识别补充外观与出场证据。</p>
    {existing && <p className="helper">沿用这部动画已保存的档案，同一作品的各集共享使用。</p>}
    {readOnly && !existing && <p className="helper">重试沿用这次导入已选择的档案。</p>}
    {loading && <Spinner label={state.status === 'reading' ? '正在读取 Bangumi 作品与角色资料' : `正在 Bangumi 查找「${state.query}」`} />}
    {state.status === 'waiting' && <p className="helper">填写作品名称后，自动查找 Bangumi。</p>}
    {state.status === 'not_found' && <p>没有找到对应的 Bangumi 条目。你可以提供链接，或从零建立档案。</p>}
    {state.status === 'candidates' && <><p>找到相近条目，请选择对应的作品。</p><div className="anime-candidates">{state.candidates.map((candidate) => <button type="button" key={candidate.id} className="anime-candidate" onClick={() => archives.read(item, String(candidate.id))}><strong>使用「{candidate.name_cn || candidate.name}」的条目</strong><span>{candidate.name_cn && candidate.name !== candidate.name_cn ? `${candidate.name} · ` : ''}{candidate.date || `条目 #${candidate.id}`}</span></button>)}</div></>}
    {state.dossier && <AnimeArchivePreview profile={{ mode: 'bangumi', dossier: state.dossier }} />}
    {state.status === 'ready' && !state.dossier && state.selection?.mode === 'bangumi' && <p>已选择 Bangumi 条目 #{state.selection.subject_id}，导入时读取现成档案。</p>}
    {state.status === 'scratch' && <AnimeArchivePreview profile={{ mode: 'scratch' }} />}
    {state.error && <Alert tone="error">{state.error}</Alert>}
    {!existing && !readOnly && <div className="anime-archive-options"><Field label="Bangumi 条目链接" hint="例如：https://bgm.tv/subject/123"><input inputMode="url" value={state.reference} onChange={(event) => archives.setReference(item, event.target.value)} placeholder="粘贴对应作品的 Bangumi 链接" /></Field><div className="button-row"><button type="button" className="button secondary small" disabled={!state.reference.trim() || state.status === 'reading'} onClick={() => archives.read(item, state.reference.trim())}>读取链接</button><button type="button" className="button secondary small" disabled={state.status === 'scratch'} onClick={() => archives.scratch(item)}>从零建档</button>{['scratch', 'error', 'not_found', 'ready'].includes(state.status) && <button type="button" className="text-button" disabled={!state.query || loading} onClick={() => archives.searchAgain(item)}>重新查找</button>}</div></div>}
  </section>;
}
