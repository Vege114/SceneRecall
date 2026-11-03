import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowRight, Check, ExternalLink, Play, RefreshCw, ScanLine, UserRound } from 'lucide-react';
import { api, ApiError, assetLink, formatTime, message, post } from '../api';
import type { Asset } from '../types';
import type { BangumiReferenceCacheResult, CharacterConfirmationBatchResult, CharacterConfirmationCandidate, CharacterConfirmationResult, CharacterConfirmationRole } from '../identityTypes';
import { Alert, Empty, Modal, Spinner } from './ui';

const roleStatus = { pending: '待确认', confirmed: '已确认', skipped: '已跳过' };
const candidateKey = (candidate: CharacterConfirmationCandidate) => `${candidate.asset_id}:${candidate.region_id}`;
const selectionKey = (characterId: string, candidate: CharacterConfirmationCandidate) => `${characterId}:${candidateKey(candidate)}`;
const originalFrame = (candidate: CharacterConfirmationCandidate) => `/api/assets/${encodeURIComponent(candidate.asset_id)}/frames/${encodeURIComponent(candidate.frame_id)}`;

type SaveAction = 'confirm' | 'reject' | 'unclear' | 'skip' | 'resume';
type SaveStatus = 'queued' | 'saving' | 'confirmed' | 'failed' | 'unknown';
interface Selection { characterId: string; characterName: string; candidate: CharacterConfirmationCandidate }
interface SaveItem { key: string; action: SaveAction; selection: Omit<Selection, 'candidate'> & { candidate?: CharacterConfirmationCandidate }; status: SaveStatus; error?: string }
interface SaveJob { action: SaveAction; characterId: string; characterName: string; keys: string[] }
interface ConfirmationSession {
  assetId: string; endpoint: string; selections: Record<string, Selection>; items: Record<string, SaveItem>;
  jobs: SaveJob[]; running: boolean; refreshSequence: number;
}
// A queue accepted by an older dialog must finish before a newer dialog writes.
let confirmationWrites: Promise<void> = Promise.resolve();
const activeSave = (item?: SaveItem) => item?.status === 'queued' || item?.status === 'saving';
const unresolvedSave = (item?: SaveItem) => item?.status === 'failed' || item?.status === 'unknown';
const saveStatus = (item: SaveItem) => item.status === 'queued' ? '排队中' : item.status === 'saving' ? '保存中' : item.status === 'unknown' ? '结果未确认' : item.status === 'failed' ? '保存失败' : item.action === 'confirm' ? '已保存' : item.action === 'reject' ? '已排除' : item.action === 'unclear' ? '已换候选' : item.action === 'skip' ? '已跳过' : '已恢复';
const newSession = (assetId: string, endpoint: string): ConfirmationSession => ({ assetId, endpoint, selections: {}, items: {}, jobs: [], running: false, refreshSequence: 0 });

function ClipPreview({ candidate, onClose }: { candidate: CharacterConfirmationCandidate; onClose: () => void }) {
  const [failed, setFailed] = useState(false);
  const start = Math.max(0, candidate.at_ms / 1000 - 2);
  const end = candidate.at_ms / 1000 + 4;
  return <section className="confirmation-preview" aria-label="候选原片预览"><div className="section-row"><div><strong>{candidate.asset_title}</strong><p className="helper">{formatTime(start * 1000)} — {formatTime(end * 1000)} · 到片段末尾自动暂停</p></div><button type="button" className="text-button" onClick={onClose}>收起预览</button></div>
    {failed ? <Alert>浏览器暂时无法播放这份原片。可查看原始画面，或前往剧集使用原有播放器核对。</Alert> : <video controls autoPlay preload="metadata" aria-label={`${candidate.asset_title}在${formatTime(candidate.at_ms)}的角色确认片段`} src={`/api/assets/${encodeURIComponent(candidate.asset_id)}/media`} onLoadedMetadata={(event) => { event.currentTarget.currentTime = Math.min(start, event.currentTarget.duration || start); }} onPlay={(event) => { if (event.currentTarget.currentTime < start || event.currentTarget.currentTime >= end) event.currentTarget.currentTime = start; }} onTimeUpdate={(event) => { if (event.currentTarget.currentTime >= end && !event.currentTarget.paused) { event.currentTarget.pause(); event.currentTarget.currentTime = end; } }} onError={() => setFailed(true)} />}
    <div className="button-row"><a className="text-button" href={originalFrame(candidate)} target="_blank" rel="noreferrer">查看完整原始画面 <ExternalLink size={12} /></a><a className="text-button" href={assetLink(candidate.asset_id, candidate.at_ms)} target="_blank" rel="noreferrer">前往剧集核对 <ExternalLink size={12} /></a></div>
  </section>;
}

function CandidateCard({ candidate, role, busy, checked, error, status, onCheckedChange, onAvailabilityChange, onDecision, onPreview }: { candidate: CharacterConfirmationCandidate; role: CharacterConfirmationRole; busy: boolean; checked: boolean; error?: string; status?: string; onCheckedChange: (checked: boolean) => void; onAvailabilityChange: (available: boolean) => void; onDecision: (action: 'reject' | 'unclear', candidate: CharacterConfirmationCandidate) => void; onPreview: (candidate: CharacterConfirmationCandidate) => void }) {
  const [failed, setFailed] = useState(false);
  const usable = candidate.quality.accepted !== false && !failed;
  return <article className={`confirmation-candidate${checked ? ' selected' : ''}`} aria-label={`${role.profile.name}在${candidate.asset_title} ${formatTime(candidate.at_ms)}的候选`}>
    <div className="confirmation-crop">{failed ? <p className="helper">裁剪图片暂不可用，请查看原始画面或换一张。</p> : <img src={candidate.crop_url} alt={`${role.profile.name}的待确认正片裁剪 ${formatTime(candidate.at_ms)}`} onLoad={() => onAvailabilityChange(true)} onError={() => { setFailed(true); onAvailabilityChange(false); }} />}</div>
    <div className="confirmation-candidate-body"><span className="tag">视觉候选 · 待核对</span><strong>{candidate.asset_title}</strong><span className="helper">{formatTime(candidate.at_ms)}</span>
      {candidate.candidate_rank != null && candidate.candidate_total != null && <p className="helper">该裁剪的角色候选排名：第 {candidate.candidate_rank} / {candidate.candidate_total} 名（已保存候选）</p>}
      {candidate.distance != null && <p className="helper">匹配距离 {candidate.distance.toFixed(4)}，越小越相似；不是身份置信度。</p>}
      {candidate.candidate_rank != null && candidate.candidate_rank > 1 && <p className="helper">这张裁剪另有排名更前的角色，请结合完整画面核对。</p>}
      <div className="button-row"><button type="button" className="text-button" disabled={candidate.source_available === false} onClick={() => onPreview(candidate)}><Play size={13} />播放前后片段</button><a className="text-button" href={originalFrame(candidate)} target="_blank" rel="noreferrer">原始画面 <ExternalLink size={12} /></a></div>
      {candidate.source_available === false && <p className="helper">原文件暂不可用，可先查看已保存的原始画面。</p>}
      {candidate.recovery_required && <p className="warning-text">上次确认未完成，重试可复用已保存结果。</p>}
      {status && <p className="helper">{status}</p>}
      {error && <p className="warning-text" role="alert">{status === '结果未确认' ? '结果未确认' : '确认失败'}：{error}</p>}
      {!usable && <p className="warning-text">{failed ? '图片加载成功后才能用作识别参考。' : `参考图质量不足${candidate.quality.reasons?.length ? `：${candidate.quality.reasons.join('；')}` : '，请换一张清晰的单人裁剪。'}`}</p>}
      <label className="check-line"><input type="checkbox" checked={checked} disabled={busy || !usable} onChange={(event) => onCheckedChange(event.target.checked)} />已核对：这是{role.profile.name}，且裁剪中只有这个角色</label>
      <div className="confirmation-candidate-actions"><button type="button" className="text-button" disabled={busy} onClick={() => onDecision('reject', candidate)}>不是这个角色</button><button type="button" className="text-button" disabled={busy} onClick={() => onDecision('unclear', candidate)}>看不清／换一张</button></div>
    </div>
  </article>;
}

export default function CharacterConfirmation({ asset, onClose, onSaved, onContinue }: { asset: Asset; onClose: () => void; onSaved: () => Promise<void>; onContinue?: () => Promise<void> }) {
  const endpoint = `/assets/${encodeURIComponent(asset.id)}/identity/confirmation`;
  const [data, setData] = useState<CharacterConfirmationResult | null>(null);
  const [selectedId, setSelectedId] = useState('');
  const [selections, setSelections] = useState<Record<string, Selection>>({});
  const [saveItems, setSaveItems] = useState<Record<string, SaveItem>>({});
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [refreshError, setRefreshError] = useState('');
  const [notice, setNotice] = useState('');
  const [preview, setPreview] = useState<CharacterConfirmationCandidate | null>(null);
  const [cached, setCached] = useState<BangumiReferenceCacheResult | null>(null);
  const alive = useRef(true);
  const sequence = useRef(0);
  const mutation = useRef(false);
  const unavailableCandidates = useRef(new Set<string>());
  const sessionRef = useRef(newSession(asset.id, endpoint));
  if (sessionRef.current.assetId !== asset.id) sessionRef.current = newSession(asset.id, endpoint);
  const current = (session: ConfirmationSession) => alive.current && sessionRef.current === session;
  const publish = (session: ConfirmationSession) => {
    if (!current(session)) return;
    setSelections({ ...session.selections });
    setSaveItems({ ...session.items });
  };
  const refreshAfterSave = (session: ConfirmationSession = sessionRef.current) => {
    const request = ++session.refreshSequence;
    if (current(session)) setRefreshError('');
    // Invalidate even when the dialog has since closed or changed assets.
    void (async () => {
      try { await onSaved(); }
      catch (caught) {
        if (current(session) && request === session.refreshSequence) {
          setRefreshError(`外层列表刷新失败：${message(caught)}`);
        }
      }
    })();
  };
  const apply = useCallback((result: CharacterConfirmationResult, session: ConfirmationSession) => {
    if (!alive.current || sessionRef.current !== session) return;
    // A server queue is a ranking snapshot, not permission to discard newer
    // local checks or candidates already waiting in the serial outbox.
    const protectedCandidates = [...Object.values(session.selections), ...Object.values(session.items)
      .filter((item) => activeSave(item) || unresolvedSave(item)).map((item) => item.selection)];
    const confirmedRegions = new Set(Object.values(session.items)
      .filter((item) => item.action === 'confirm' && item.status === 'confirmed' && item.selection.candidate)
      .map((item) => candidateKey(item.selection.candidate!)));
    const merged = { ...result, roles: result.roles.map((item) => {
      const candidates = new Map<string, CharacterConfirmationCandidate>();
      for (const selection of protectedCandidates) {
        if (selection.characterId === item.character_id && selection.candidate && !confirmedRegions.has(candidateKey(selection.candidate))) {
          candidates.set(candidateKey(selection.candidate), selection.candidate);
        }
      }
      for (const candidate of item.candidates) {
        if (!confirmedRegions.has(candidateKey(candidate)) && !candidates.has(candidateKey(candidate))) candidates.set(candidateKey(candidate), candidate);
      }
      return { ...item, candidates: [...candidates.values()].slice(0, 3) };
    }) };
    setData(merged);
    setSelectedId((id) => merged.roles.some((item) => item.character_id === id) ? id : (merged.roles.find((item) => item.status === 'pending') || merged.roles[0])?.character_id || '');
  }, []);
  const load = useCallback(async () => {
    const session = sessionRef.current;
    if (session.running || session.jobs.length) return;
    const request = ++sequence.current;
    try {
      const result = await api<CharacterConfirmationResult>(endpoint);
      if (request === sequence.current) apply(result, session);
    }
    catch (caught) { if (alive.current && sessionRef.current === session && request === sequence.current) setError(message(caught)); }
    finally { if (alive.current && sessionRef.current === session && request === sequence.current) setLoading(false); }
  }, [endpoint, apply]);
  useEffect(() => {
    alive.current = true;
    mutation.current = false;
    unavailableCandidates.current.clear();
    setLoading(true); setData(null); setCached(null); setSaveItems({}); setSelections({});
    setError(''); setNotice(''); setRefreshError(''); setBusy(''); setPreview(null);
    void load();
    return () => { alive.current = false; sequence.current += 1; };
  }, [load]);
  useEffect(() => {
    const warnBeforeUnload = (event: BeforeUnloadEvent) => {
      const session = sessionRef.current;
      if (!session.running && !session.jobs.length && !Object.keys(session.selections).length) return;
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', warnBeforeUnload);
    return () => window.removeEventListener('beforeunload', warnBeforeUnload);
  }, []);
  const sampling = data?.sampling;
  useEffect(() => {
    if (sampling?.status !== 'running') return;
    const session = sessionRef.current;
    let polling = false;
    const timer = window.setInterval(async () => {
      if (mutation.current || session.running || polling) return;
      polling = true;
      try {
        const result = await api<Pick<CharacterConfirmationResult, 'sampling'>>(`${endpoint}/sampling`);
        if (!alive.current || sessionRef.current !== session || mutation.current || session.running) return;
        setData((value) => value ? { ...value, sampling: result.sampling } : value);
        if (result.sampling?.status !== 'running') void load();
      } catch (caught) { if (alive.current && sessionRef.current === session) setError(`采样进度读取失败：${message(caught)}`); }
      finally { polling = false; }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [endpoint, load, sampling?.status]);
  const role = data?.roles.find((item) => item.character_id === selectedId);
  const readySelections = Object.entries(selections).filter(([key]) => !activeSave(saveItems[key]) && saveItems[key]?.status !== 'confirmed');
  const selected = readySelections.filter(([, selection]) => selection.characterId === selectedId && !unavailableCandidates.current.has(selectionKey(selection.characterId, selection.candidate)));
  const pendingCount = Object.values(saveItems).filter(activeSave).length;
  const failedCount = Object.values(saveItems).filter(unresolvedSave).length;
  const roleSaving = (characterId: string) => Object.values(saveItems).some((item) => item.selection.characterId === characterId && activeSave(item));
  const navigate = (id: string) => { setSelectedId(id); setPreview(null); setNotice(''); };
  const nextRole = (result: CharacterConfirmationResult, from: string) => {
    const index = result.roles.findIndex((item) => item.character_id === from);
    const ordered = [...result.roles.slice(index + 1), ...result.roles.slice(0, index)];
    const next = ordered.find((item) => item.status === 'pending');
    if (next) navigate(next.character_id);
  };
  const candidateBlocked = (characterId: string, candidate: CharacterConfirmationCandidate) => {
    const session = sessionRef.current;
    const key = selectionKey(characterId, candidate);
    return !!busy || Object.values(session.items).some((item) =>
      (activeSave(item) || item.status === 'unknown' || item.action === 'confirm' && item.status === 'confirmed') &&
      (item.selection.candidate ? candidateKey(item.selection.candidate) === candidateKey(candidate) : item.selection.characterId === characterId)) ||
      Object.entries(session.selections).some(([other, selection]) => other !== key && candidateKey(selection.candidate) === candidateKey(candidate));
  };
  const drain = async (session: ConfirmationSession) => {
    if (session.running) return;
    session.running = true;
    sequence.current += 1;
    const previousWrites = confirmationWrites;
    let releaseWrites!: () => void;
    confirmationWrites = new Promise<void>((resolve) => { releaseWrites = resolve; });
    const warnPending = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
    // Protect the captured queue even if its dialog changes assets or unmounts.
    window.addEventListener('beforeunload', warnPending);
    try {
      await previousWrites;
      while (session.jobs.length) {
        const job = session.jobs.shift()!;
        for (const key of job.keys) session.items[key] = { ...session.items[key], status: 'saving', error: undefined };
        publish(session);
        try {
          if (job.action === 'confirm') {
            const candidates = job.keys.map((key) => session.items[key].selection.candidate!);
            const result = await post<CharacterConfirmationBatchResult>(`${session.endpoint}/characters/${encodeURIComponent(job.characterId)}/confirm-batch`, {
              candidates: candidates.map((candidate) => ({ source_asset_id: candidate.asset_id, region_id: candidate.region_id })),
              single_character: true, use_as_reference: true,
            });
            let confirmed = 0;
            let recoverable = false;
            for (const key of job.keys) {
              const item = session.items[key];
              const candidate = item.selection.candidate!;
              const outcome = result.results.find((value) => value.source_asset_id === candidate.asset_id && value.region_id === candidate.region_id);
              if (outcome?.status === 'confirmed') {
                session.items[key] = { ...item, status: 'confirmed' };
                delete session.selections[key];
                confirmed += 1;
              } else {
                const status = outcome?.status === 'failed' ? 'failed' : 'unknown';
                session.items[key] = { ...item, status, error: outcome?.error || '未能取得这张裁剪的保存结果，请重试或重新读取进度。' };
                const selection = { ...item.selection, candidate: { ...candidate, recovery_required: outcome?.recovery_required || candidate.recovery_required } };
                session.items[key].selection = selection;
                recoverable ||= !!outcome?.recovery_required;
                if (!unavailableCandidates.current.has(key)) session.selections[key] = selection as Selection;
                else delete session.selections[key];
              }
            }
            if (confirmed > 0 || recoverable || job.keys.some((key) => session.items[key].status === 'unknown')) refreshAfterSave(session);
            publish(session);
            apply(result.queue, session);
            if (current(session)) {
              if (confirmed > 0) setNotice(`已确认「${job.characterName}」的 ${confirmed} / ${job.keys.length} 张所选裁剪，并加入本作品的可信图库。其他镜头仍需独立匹配或核对。`);
              if (confirmed < job.keys.length) setError('有裁剪尚未确认保存成功。请查看逐项结果，失败项可单独重试。');
            }
          } else {
            // Decision jobs contain one item so every request has an explicit,
            // independent outcome and cannot race a queued confirmation.
            const key = job.keys[0];
            const candidate = session.items[key].selection.candidate;
            const result = await post<CharacterConfirmationResult>(`${session.endpoint}/characters/${encodeURIComponent(job.characterId)}/decision`, {
              action: job.action, ...(candidate ? { source_asset_id: candidate.asset_id, region_id: candidate.region_id } : {}),
            });
            session.items[key] = { ...session.items[key], status: 'confirmed' };
            delete session.selections[key];
            publish(session); apply(result, session); refreshAfterSave(session);
            if (current(session)) {
              if (job.action === 'skip') setNotice('已保存跳过状态，可以稍后继续。');
              else if (job.action === 'reject') setNotice('已记住这个否定结果，不会再为该角色推荐这张候选。');
              else if (job.action === 'unclear') setNotice('已换过这组候选，新的候选会优先展示。');
            }
          }
        } catch (caught) {
          const unknown = !(caught instanceof ApiError) || caught.status === 0 || caught.status >= 500;
          for (const key of job.keys) session.items[key] = { ...session.items[key], status: unknown ? 'unknown' : 'failed', error: message(caught) };
          publish(session);
          // A lost response may follow a durable write. It is not success.
          if (unknown) refreshAfterSave(session);
          if (current(session)) setError(`未能取得本次确认结果：${message(caught)}。所选项已保留，可重试或重新读取确认进度。`);
        }
      }
    } finally {
      session.running = false;
      releaseWrites();
      window.removeEventListener('beforeunload', warnPending);
      publish(session);
    }
  };
  const enqueueConfirmations = (entries: [string, Selection][]) => {
    const session = sessionRef.current;
    if (mutation.current) return;
    const groups = new Map<string, [string, Selection][]>();
    for (const [key, selection] of entries) {
      if (activeSave(session.items[key]) || session.items[key]?.status === 'confirmed' || unavailableCandidates.current.has(key)) continue;
      const conflicts = Object.entries(session.selections).some(([other, value]) => other !== key && value.characterId !== selection.characterId && candidateKey(value.candidate) === candidateKey(selection.candidate)) || Object.values(session.items).some((item) =>
        item.key !== key && (activeSave(item) || item.status === 'unknown' || item.action === 'confirm' && item.status === 'confirmed') &&
        (item.selection.candidate ? candidateKey(item.selection.candidate) === candidateKey(selection.candidate) : item.selection.characterId === selection.characterId));
      if (conflicts) { setError('这张裁剪另有已勾选或尚未核实的操作，请先处理该项，避免给同一裁剪提交冲突身份。'); continue; }
      session.selections[key] = selection;
      const group = groups.get(selection.characterId) || [];
      group.push([key, selection]); groups.set(selection.characterId, group);
    }
    for (const [characterId, group] of groups) {
      // The existing endpoint deliberately caps each role request at three.
      for (let index = 0; index < group.length; index += 3) {
        const chunk = group.slice(index, index + 3);
        for (const [key, selection] of chunk) session.items[key] = { key, action: 'confirm', selection, status: 'queued' };
        session.jobs.push({ action: 'confirm', characterId, characterName: chunk[0][1].characterName, keys: chunk.map(([key]) => key) });
      }
    }
    if (!groups.size) return;
    setError(''); setNotice(''); publish(session); void drain(session);
  };
  const decide = (action: Exclude<SaveAction, 'confirm'>, candidates: CharacterConfirmationCandidate[] = []) => {
    const session = sessionRef.current;
    if (!role || mutation.current || roleSaving(role.character_id)) return;
    if (candidates.some((candidate) => candidateBlocked(role.character_id, candidate))) return;
    const choices = candidates.length ? candidates : [undefined];
    for (const candidate of choices) {
      const key = candidate ? selectionKey(role.character_id, candidate) : `role:${role.character_id}`;
      if (activeSave(session.items[key]) || session.items[key]?.status === 'unknown') continue;
      const selection = { characterId: role.character_id, characterName: role.profile.name, candidate };
      delete session.selections[key];
      session.items[key] = { key, action, selection, status: 'queued' };
      session.jobs.push({ action, characterId: role.character_id, characterName: role.profile.name, keys: [key] });
    }
    if (action === 'skip' || action === 'resume') {
      for (const [key, selection] of Object.entries(session.selections)) if (selection.characterId === role.character_id) delete session.selections[key];
    }
    setError(''); setNotice(''); setPreview(null); publish(session); void drain(session);
  };
  const retry = (item: SaveItem) => {
    const session = sessionRef.current;
    if (mutation.current || !unresolvedSave(session.items[item.key]) || unavailableCandidates.current.has(item.key)) return;
    if (item.action === 'confirm' && item.selection.candidate) {
      const selection = item.selection as Selection;
      enqueueConfirmations([[item.key, selection]]);
    } else {
      if (Object.entries(session.selections).some(([key, value]) => key !== item.key && value.characterId === item.selection.characterId) || Object.values(session.items).some((value) => value.key !== item.key && value.selection.characterId === item.selection.characterId && activeSave(value))) {
        setError('这个角色还有新的勾选或保存操作，请先处理后再重试原决定。'); return;
      }
      session.items[item.key] = { ...item, status: 'queued', error: undefined };
      session.jobs.push({ action: item.action, characterId: item.selection.characterId, characterName: item.selection.characterName, keys: [item.key] });
      setError(''); publish(session); void drain(session);
    }
  };
  const dismissFailure = async (item: SaveItem) => {
    const session = sessionRef.current;
    if (mutation.current || session.running || !unresolvedSave(session.items[item.key])) return;
    mutation.current = true; sequence.current += 1; setBusy('reconcile'); setError('');
    try {
      const result = await api<CharacterConfirmationResult>(session.endpoint);
      if (!current(session)) return;
      // Keep the unknown reservation until a durable read succeeds. A failed
      // read must not silently unlock more writes or the continue button.
      delete session.items[item.key]; delete session.selections[item.key];
      publish(session); apply(result, session); refreshAfterSave(session);
    } catch (caught) { if (current(session)) setError(`未能核对后端进度：${message(caught)}。原操作仍保留，尚未视为保存成功。`); }
    finally { if (current(session)) { mutation.current = false; setBusy(''); } }
  };
  const sample = async () => {
    const session = sessionRef.current;
    if (mutation.current || session.running || Object.keys(session.selections).length || sampling?.status === 'running') return;
    mutation.current = true; sequence.current += 1; setBusy('sample'); setError(''); setNotice('');
    session.selections = {}; publish(session);
    try { apply(await post<CharacterConfirmationResult>(`${endpoint}/sample`, { max_frames: 12 }), session); }
    catch (caught) { if (current(session)) setError(message(caught)); }
    finally { if (current(session)) { mutation.current = false; setBusy(''); } }
  };
  const cacheReferences = async () => {
    const session = sessionRef.current;
    if (mutation.current || session.running || Object.keys(session.selections).length || sampling?.status === 'running') return;
    mutation.current = true; sequence.current += 1; setBusy('cache'); setError(''); setNotice('');
    try {
      const result = await post<BangumiReferenceCacheResult>(`/assets/${encodeURIComponent(asset.id)}/references/cache-bangumi`, { max_images: 20, ...(cached?.next_cursor ? { cursor: cached.next_cursor } : {}) });
      if (!current(session)) return;
      setCached(result);
      await load();
    } catch (caught) { if (current(session)) { setError(message(caught)); setCached(null); } }
    finally { if (current(session)) { mutation.current = false; setBusy(''); } }
  };
  const continueTask = async () => {
    const session = sessionRef.current;
    if (!onContinue || !data || data.summary.pending > 0 || mutation.current || session.running || session.jobs.length || Object.keys(session.selections).length || Object.values(session.items).some(unresolvedSave) || sampling?.status === 'running') return;
    mutation.current = true; setBusy('continue'); setError('');
    try { await onContinue(); }
    catch (caught) { if (current(session)) setError(message(caught)); }
    finally { if (current(session)) { mutation.current = false; setBusy(''); } }
  };
  const close = () => {
    const session = sessionRef.current;
    if (mutation.current) return;
    if (session.running || session.jobs.length) { setError('仍有排队或保存中的操作，请等逐项结果返回后关闭。刷新或离开页面可能中断未发送的操作。'); return; }
    if (Object.keys(session.selections).length) { setError('还有已核对但未保存的裁剪，请先提交，或取消未提交勾选后返回。'); return; }
    onClose();
  };
  return <Modal title={onContinue ? '任务中的角色确认' : '确认角色'} subtitle={`${asset.title} · 将角色档案与正片中的实际形象对应起来 · 提交后按顺序保存`} onClose={close} wide>
    <div className="modal-body character-confirmation">
      <p className="confirmation-intro">先核对少量清晰的单人裁剪，再用作本作品后续集数的识别参考。候选来自已有分析；确认只作用于所选画面。未出场或暂时无法辨认的角色可以跳过。可先跨角色勾选，再集中提交；提交后也可继续核对其他候选。</p>
      {onContinue && <Alert tone="info">任务已开始，目前等待这一步的人工判断。请逐个确认角色，或明确选择暂时跳过；完成后继续后续分析。关闭窗口会保留进度，任务仍在队列中等待。</Alert>}
      {error && <Alert tone="error">{error}<button type="button" className="text-button" disabled={!!busy || pendingCount > 0} onClick={() => { setError(''); void load(); }}>重新读取确认进度</button></Alert>}
      {refreshError && <Alert>{refreshError}<button type="button" className="text-button" onClick={() => refreshAfterSave()}>重试刷新外层列表</button></Alert>}
      {notice && <Alert tone="success">{notice}</Alert>}
      {loading ? <Spinner label="正在读取已有角色候选" /> : data && <>
        <div className="confirmation-progress"><div><strong>已确认 {data.summary.confirmed} / {data.summary.total} 个角色</strong><span>已跳过 {data.summary.skipped} · 待确认 {data.summary.pending}</span></div><progress aria-label="角色确认进度" value={data.summary.confirmed + data.summary.skipped} max={Math.max(1, data.summary.total)} /></div>
        {sampling?.status === 'running' && <div className="confirmation-sampling-progress"><Spinner label={`本地采样中 · ${sampling.completed} / ${sampling.total} 帧`} /><progress aria-label="本地采样进度" value={sampling.completed} max={Math.max(1, sampling.total)} /><p className="helper">可以关闭窗口，稍后回来继续核对。</p></div>}
        <section className="confirmation-save-queue" aria-label="待保存与逐项结果">
          <div className="confirmation-batch-actions"><p className="helper">已核对的勾选会跨角色保留；只有提交并收到保存结果后才计为完成。</p><button type="button" className="button primary" disabled={!!busy || !readySelections.length} onClick={() => enqueueConfirmations(readySelections)}><Check size={14} />提交全部已核对 {readySelections.length} 张</button>{readySelections.length > 0 && <button type="button" className="text-button" disabled={!!busy} onClick={() => {
            const session = sessionRef.current;
            for (const [key] of readySelections) delete session.selections[key];
            publish(session);
          }}>取消未提交勾选</button>}</div>
          {pendingCount > 0 && <p className="helper" role="status">按提交顺序保存中，可以继续切角色、预览和勾选其他裁剪。请保持页面打开；刷新或离开可能中断未发送项。</p>}
          {Object.values(saveItems).length > 0 && <ul className="confirmation-save-results" aria-label="逐项保存结果">{Object.values(saveItems).map((item) => <li key={item.key}>
            <span>{item.selection.characterName}{item.selection.candidate ? ` · ${item.selection.candidate.asset_title} ${formatTime(item.selection.candidate.at_ms)}` : ''} · {saveStatus(item)}</span>
            {unresolvedSave(item) && <><span className="warning-text">{item.error}</span><button type="button" className="text-button" disabled={!!busy || unavailableCandidates.current.has(item.key)} onClick={() => retry(item)}>重试这项</button><button type="button" className="text-button" disabled={!!busy || pendingCount > 0} onClick={() => dismissFailure(item)}>暂不重试并核对进度</button></>}
          </li>)}</ul>}
        </section>
        <details className="confirmation-options"><summary>候选不足？补充参考图与少量采样</summary>
        <div className="confirmation-sampling"><div><strong>候选不足时，少量补充采样</strong><p className="helper">只在「{asset.title}」采样最多 12 帧，使用本地人物检测与图像匹配，不调用 VLM。已有候选可直接核对。</p></div><button type="button" className="button small secondary" disabled={!!busy || pendingCount > 0 || readySelections.length > 0 || sampling?.status === 'running' || !asset.source_available || data.sampling_available === false} onClick={() => void sample()}><ScanLine size={14} />{sampling?.status === 'failed' || sampling?.status === 'interrupted' ? '重试少量采样' : sampling?.status === 'completed' ? '继续少量本地采样' : '补充少量本地采样'}</button></div>
        {data.roles.some((item) => item.profile.external_source?.image_url) && <div className="confirmation-cache"><p className="helper">优先补齐主角参考图，跳过已有完整缓存。每批最多下载 20 张，可继续处理剩余图片；缓存图片仍需核对身份。</p><button type="button" className="text-button" disabled={!!busy || pendingCount > 0 || readySelections.length > 0 || sampling?.status === 'running'} onClick={() => void cacheReferences()}>{cached?.next_cursor ? '继续缓存 Bangumi 参考图' : '缓存 Bangumi 参考图'}</button></div>}
        {cached && <Alert tone="info">Bangumi 参考图：新增 {cached.imported} 张 · 已有 {cached.existing} 张 · 跳过 {cached.skipped} 张。仍需在正片中人工核对身份。{cached.errors?.length > 0 && <ul>{cached.errors.map((item, index) => <li key={`${item.character_id}:${index}`}>{data.roles.find((entry) => entry.character_id === item.character_id)?.profile.name || '角色'}：{item.reason}</li>)}</ul>}</Alert>}
        {cached?.coverage && <p className="helper">来源有图 {cached.coverage.with_image} / {cached.coverage.total} 个角色 · 本地已缓存 {cached.coverage.cached} 个 · 主角已缓存 {cached.coverage.main_cached} / {cached.coverage.main_total} 个 · 剩余待缓存 {cached.coverage.remaining} 个。{cached.time_limited ? '本批已到时间限制，可继续下一批。' : cached.next_cursor ? '本批已到下载数量限制，可继续下一批。' : cached.coverage.remaining ? '本轮已结束，未成功项可再次缓存重试。' : '当前有来源图的角色已处理完成。'}{cached.coverage.revoked > 0 && `另有 ${cached.coverage.revoked} 个角色的来源图已撤销，保留撤销决定。`}</p>}
        {!asset.source_available && <Alert>当前视频原文件暂不可用，恢复原文件后可补充采样。已保存候选仍可核对。</Alert>}
        {data.sampling_available === false && asset.source_available && <Alert>{data.sampling_unavailable_reason || '本地人物检测或图像匹配模型尚未就绪。'}<a href="#/settings" target="_blank" rel="noreferrer">检查本地模型设置</a></Alert>}
        {sampling?.status === 'completed' && <p className="helper">本地采样已完成：{sampling.completed} / {sampling.total} 帧。没有匹配候选不代表角色一定未出场。</p>}
        {sampling && ['failed', 'unavailable', 'interrupted'].includes(sampling.status) && <Alert tone={sampling.status === 'failed' ? 'error' : 'warning'}>{sampling.error || (sampling.status === 'interrupted' ? '上次采样已中断，可重新进行少量采样。' : '本地模型暂不可用，请检查模型设置后重试。')}</Alert>}
        </details>
        {!data.roles.length ? <Empty icon={<UserRound size={28} />} title="先建立角色档案">请先初始化 Bangumi 角色档案，或新建角色，再回来核对正片形象。</Empty> : <>
          <div className="confirmation-role-list" aria-label="选择待核对角色">{data.roles.map((item) => <button type="button" key={item.character_id} className={item.character_id === selectedId ? 'selected' : ''} aria-pressed={item.character_id === selectedId} disabled={!!busy} onClick={() => navigate(item.character_id)}><span>{item.profile.name}</span><small>{roleSaving(item.character_id) ? '等待保存' : roleStatus[item.status]}{Object.values(selections).some((selection) => selection.characterId === item.character_id) ? ' · 已勾选' : ''}</small></button>)}</div>
          {role && <section className="confirmation-review" aria-label={`${role.profile.name}的角色确认`}>
            <aside className="confirmation-profile"><span className="eyebrow">角色档案</span>{role.profile.external_source?.image_url ? <img src={role.profile.external_source.image_url} referrerPolicy="no-referrer" alt={`${role.profile.name}的 Bangumi 档案参考图`} /> : <div className="confirmation-profile-placeholder"><UserRound size={38} /></div>}<h3>{role.profile.name}</h3>{role.profile.name_ja && <p>{role.profile.name_ja}</p>}{role.profile.aliases.length > 0 && <p className="helper">又名 {role.profile.aliases.join('、')}</p>}{role.profile.external_source && <a className="text-button" href={role.profile.external_source.url} target="_blank" rel="noreferrer">Bangumi 角色资料 <ExternalLink size={12} /></a>}<p className="confirmation-description">{role.profile.description || '暂无角色简介，请结合原片核对。'}</p><span className="tag">可信参考 {role.trusted_reference_count} 张</span><p className="helper">身份标签与参考来源会一并保存。同一作品后续集数可复用。</p></aside>
            <div className="confirmation-candidates-area"><div className="confirmation-role-heading"><div><h3>正片里，哪个是{role.profile.name}？</h3><p className="helper">这些是相似度排序的宽泛待核对建议，同一裁剪可出现在多个角色列表。排名不代表身份已确认，请逐张对照原片。</p></div><span className="tag">{roleStatus[role.status]}</span></div>
              {role.matching_reference_count !== undefined && <p className="helper">本地匹配参考 {role.matching_reference_count} 张，仍需通过人物检测。{role.matching_reference_count === 0 && '该角色缺少本地匹配参考，请先缓存或添加正确参考图；增加采样不会补齐参考图。'}</p>}
              {role.status === 'skipped' ? <Empty title="这个角色已暂时跳过" action={<button type="button" className="button secondary" disabled={!!busy || roleSaving(role.character_id)} onClick={() => decide('resume')}>继续确认这个角色</button>}>进度已保留，不要求在当前集数中找到所有角色。</Empty> : <>
                {role.candidates.length ? <><div className="confirmation-batch-actions"><p className="helper">每个角色展示最多 3 张按镜头或时间去重的已保存匹配候选；候选数量不代表整集覆盖。</p><button type="button" className="button primary" disabled={!!busy || !selected.length} onClick={() => enqueueConfirmations(selected)}><Check size={14} />确认所选 {selected.length} 张并用作参考</button></div><div className="confirmation-candidate-grid">{role.candidates.map((candidate) => <CandidateCard key={selectionKey(role.character_id, candidate)} candidate={candidate} role={role} busy={candidateBlocked(role.character_id, candidate)} checked={!!selections[selectionKey(role.character_id, candidate)]} error={saveItems[selectionKey(role.character_id, candidate)]?.error} status={saveItems[selectionKey(role.character_id, candidate)] ? saveStatus(saveItems[selectionKey(role.character_id, candidate)]) : undefined} onCheckedChange={(checked) => {
                  const session = sessionRef.current;
                  if (candidateBlocked(role.character_id, candidate)) return;
                  const key = selectionKey(role.character_id, candidate);
                  if (checked) session.selections[key] = { characterId: role.character_id, characterName: role.profile.name, candidate };
                  else delete session.selections[key];
                  publish(session);
                }} onAvailabilityChange={(available) => {
                  const key = selectionKey(role.character_id, candidate);
                  if (available) unavailableCandidates.current.delete(key);
                  else { unavailableCandidates.current.add(key); delete sessionRef.current.selections[key]; }
                  publish(sessionRef.current);
                }} onDecision={(action, value) => { void decide(action, [value]); }} onPreview={setPreview} />)}</div></> : <Empty title="暂时没有合适候选">{role.matching_reference_count === 0 ? "请先补齐该角色的本地参考图，再匹配已保存帧或补充少量采样。没有候选不代表未出场。" : "已有参考时可匹配已保存帧或补充少量本地采样。没有候选不代表未出场。"}</Empty>}
                <div className="button-row">{role.candidates.length > 0 && <button type="button" className="button secondary" disabled={!!busy || roleSaving(role.character_id) || role.candidates.some((candidate) => candidateBlocked(role.character_id, candidate))} onClick={() => decide('unclear', role.candidates)}><RefreshCw size={14} />看不清／换一组</button>}{role.status !== 'confirmed' && <button type="button" className="button ghost" disabled={!!busy || roleSaving(role.character_id)} onClick={() => decide('skip')}>暂时跳过这个角色</button>}{data.summary.pending > (role.status === 'pending' ? 1 : 0) && <button type="button" className="button ghost" disabled={!!busy} onClick={() => nextRole(data, role.character_id)}>下一个待确认角色 <ArrowRight size={14} /></button>}</div>
              </>}
              {preview && <ClipPreview key={candidateKey(preview)} candidate={preview} onClose={() => setPreview(null)} />}
            </div>
          </section>}
        </>}
      </>}
      {busy && <Spinner label={busy === 'continue' ? '正在继续任务' : busy === 'sample' ? '正在启动少量采样' : busy === 'cache' ? '正在缓存 Bangumi 参考图' : '正在保存确认进度'} />}
    </div>
    <footer className="modal-footer"><p className="helper">{pendingCount > 0 ? `还有 ${pendingCount} 项排队或保存中，请保持页面打开。` : readySelections.length > 0 ? `还有 ${readySelections.length} 张已核对裁剪未保存。` : onContinue ? data?.summary.pending ? `还有 ${data.summary.pending} 个角色等待确认或明确跳过。` : '已保存的确认和跳过结果会用于继续任务。' : '关闭后可从角色档案继续核对。'}</p><div className="button-row"><button type="button" className="button secondary" disabled={!!busy || pendingCount > 0 || readySelections.length > 0} onClick={close}>保存进度并返回</button>{onContinue && <button type="button" className="button primary" disabled={!!busy || pendingCount > 0 || readySelections.length > 0 || failedCount > 0 || loading || !data || data.summary.pending > 0 || sampling?.status === 'running'} onClick={() => void continueTask()}><Play size={15} />继续任务</button>}</div></footer>
  </Modal>;
}
