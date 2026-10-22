import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowRight, Check, ExternalLink, Play, RefreshCw, ScanLine, UserRound } from 'lucide-react';
import { api, assetLink, formatTime, message, post } from '../api';
import type { Asset } from '../types';
import type { BangumiReferenceCacheResult, CharacterConfirmationCandidate, CharacterConfirmationResult, CharacterConfirmationRole } from '../identityTypes';
import { Alert, Empty, Modal, Spinner } from './ui';

const roleStatus = { pending: '待确认', confirmed: '已确认', skipped: '已跳过' };
const candidateKey = (candidate: CharacterConfirmationCandidate) => `${candidate.asset_id}:${candidate.region_id}`;
const originalFrame = (candidate: CharacterConfirmationCandidate) => `/api/assets/${encodeURIComponent(candidate.asset_id)}/frames/${encodeURIComponent(candidate.frame_id)}`;

function ClipPreview({ candidate, onClose }: { candidate: CharacterConfirmationCandidate; onClose: () => void }) {
  const [failed, setFailed] = useState(false);
  const start = Math.max(0, candidate.at_ms / 1000 - 2);
  const end = candidate.at_ms / 1000 + 4;
  return <section className="confirmation-preview" aria-label="候选原片预览"><div className="section-row"><div><strong>{candidate.asset_title}</strong><p className="helper">{formatTime(start * 1000)} — {formatTime(end * 1000)} · 到片段末尾自动暂停</p></div><button type="button" className="text-button" onClick={onClose}>收起预览</button></div>
    {failed ? <Alert>浏览器暂时无法播放这份原片。可查看原始画面，或前往剧集使用原有播放器核对。</Alert> : <video controls autoPlay preload="metadata" aria-label={`${candidate.asset_title}在${formatTime(candidate.at_ms)}的角色确认片段`} src={`/api/assets/${encodeURIComponent(candidate.asset_id)}/media`} onLoadedMetadata={(event) => { event.currentTarget.currentTime = Math.min(start, event.currentTarget.duration || start); }} onPlay={(event) => { if (event.currentTarget.currentTime < start || event.currentTarget.currentTime >= end) event.currentTarget.currentTime = start; }} onTimeUpdate={(event) => { if (event.currentTarget.currentTime >= end && !event.currentTarget.paused) { event.currentTarget.pause(); event.currentTarget.currentTime = end; } }} onError={() => setFailed(true)} />}
    <div className="button-row"><a className="text-button" href={originalFrame(candidate)} target="_blank" rel="noreferrer">查看完整原始画面 <ExternalLink size={12} /></a><a className="text-button" href={assetLink(candidate.asset_id, candidate.at_ms)} target="_blank" rel="noreferrer">前往剧集核对 <ExternalLink size={12} /></a></div>
  </section>;
}

function CandidateCard({ candidate, role, busy, onDecision, onPreview }: { candidate: CharacterConfirmationCandidate; role: CharacterConfirmationRole; busy: boolean; onDecision: (action: 'confirm' | 'reject' | 'unclear', candidate: CharacterConfirmationCandidate) => void; onPreview: (candidate: CharacterConfirmationCandidate) => void }) {
  const [checked, setChecked] = useState(false);
  const [failed, setFailed] = useState(false);
  const usable = candidate.quality.accepted !== false && !failed;
  return <article className="confirmation-candidate" aria-label={`${role.profile.name}在${candidate.asset_title} ${formatTime(candidate.at_ms)}的候选`}>
    <div className="confirmation-crop">{failed ? <p className="helper">裁剪图片暂不可用，请查看原始画面或换一张。</p> : <img src={candidate.crop_url} alt={`${role.profile.name}的待确认正片裁剪 ${formatTime(candidate.at_ms)}`} onError={() => setFailed(true)} />}</div>
    <div className="confirmation-candidate-body"><span className="tag">视觉候选 · 待核对</span><strong>{candidate.asset_title}</strong><span className="helper">{formatTime(candidate.at_ms)}</span>
      <div className="button-row"><button type="button" className="text-button" disabled={candidate.source_available === false} onClick={() => onPreview(candidate)}><Play size={13} />播放前后片段</button><a className="text-button" href={originalFrame(candidate)} target="_blank" rel="noreferrer">原始画面 <ExternalLink size={12} /></a></div>
      {candidate.source_available === false && <p className="helper">原文件暂不可用，可先查看已保存的原始画面。</p>}
      {candidate.recovery_required && <p className="warning-text">上次确认未完成，重试可复用已保存结果。</p>}
      {!usable && <p className="warning-text">{failed ? '图片加载成功后才能用作识别参考。' : `参考图质量不足${candidate.quality.reasons?.length ? `：${candidate.quality.reasons.join('；')}` : '，请换一张清晰的单人裁剪。'}`}</p>}
      <label className="check-line"><input type="checkbox" checked={checked} disabled={busy || !usable} onChange={(event) => setChecked(event.target.checked)} />已核对：这是{role.profile.name}，且裁剪中只有这个角色</label>
      <button type="button" className="button primary" disabled={busy || !checked || !usable} onClick={() => onDecision('confirm', candidate)}><Check size={14} />确认并用作后续参考</button>
      <div className="confirmation-candidate-actions"><button type="button" className="text-button" disabled={busy} onClick={() => onDecision('reject', candidate)}>不是这个角色</button><button type="button" className="text-button" disabled={busy} onClick={() => onDecision('unclear', candidate)}>看不清／换一张</button></div>
    </div>
  </article>;
}

export default function CharacterConfirmation({ asset, onClose, onSaved }: { asset: Asset; onClose: () => void; onSaved: () => Promise<void> }) {
  const endpoint = `/assets/${encodeURIComponent(asset.id)}/identity/confirmation`;
  const [data, setData] = useState<CharacterConfirmationResult | null>(null);
  const [selectedId, setSelectedId] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [preview, setPreview] = useState<CharacterConfirmationCandidate | null>(null);
  const [cached, setCached] = useState<BangumiReferenceCacheResult | null>(null);
  const alive = useRef(true);
  const sequence = useRef(0);
  const mutation = useRef(false);
  const apply = useCallback((result: CharacterConfirmationResult) => {
    if (!alive.current) return;
    setData(result);
    setSelectedId((current) => result.roles.some((role) => role.character_id === current) ? current : (result.roles.find((role) => role.status === 'pending') || result.roles[0])?.character_id || '');
  }, []);
  const load = useCallback(async () => {
    const request = ++sequence.current;
    try { const result = await api<CharacterConfirmationResult>(endpoint); if (request === sequence.current) apply(result); }
    catch (caught) { if (alive.current && request === sequence.current) setError(message(caught)); }
    finally { if (alive.current && request === sequence.current) setLoading(false); }
  }, [endpoint, apply]);
  useEffect(() => { alive.current = true; void load(); return () => { alive.current = false; sequence.current += 1; }; }, [load]);
  const sampling = data?.sampling;
  useEffect(() => {
    if (sampling?.status !== 'running') return;
    let polling = false;
    const timer = window.setInterval(async () => {
      if (mutation.current || polling) return;
      polling = true;
      try {
        const result = await api<Pick<CharacterConfirmationResult, 'sampling'>>(`${endpoint}/sampling`);
        if (!alive.current || mutation.current) return;
        setData((current) => current ? { ...current, sampling: result.sampling } : current);
        if (result.sampling?.status !== 'running') void load();
      } catch (caught) { if (alive.current) setError(`采样进度读取失败：${message(caught)}`); }
      finally { polling = false; }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [endpoint, load, sampling?.status]);
  const role = data?.roles.find((item) => item.character_id === selectedId);
  const navigate = (id: string) => { setSelectedId(id); setPreview(null); setNotice(''); };
  const nextRole = (result: CharacterConfirmationResult, from: string) => {
    const index = result.roles.findIndex((item) => item.character_id === from);
    const ordered = [...result.roles.slice(index + 1), ...result.roles.slice(0, index)];
    const next = ordered.find((item) => item.status === 'pending');
    if (next) { setSelectedId(next.character_id); setPreview(null); }
  };
  const decide = async (action: 'confirm' | 'reject' | 'unclear' | 'skip' | 'resume', candidates: CharacterConfirmationCandidate[] = []) => {
    if (!role || mutation.current) return;
    mutation.current = true; sequence.current += 1; setBusy(action); setError(''); setNotice('');
    const characterId = role.character_id;
    try {
      const decisions = candidates.length ? candidates : [undefined];
      let result: CharacterConfirmationResult | null = null;
      for (const candidate of decisions) {
        result = await post<CharacterConfirmationResult>(`${endpoint}/characters/${encodeURIComponent(characterId)}/decision`, { action, ...(candidate ? { source_asset_id: candidate.asset_id, region_id: candidate.region_id } : {}), ...(action === 'confirm' ? { single_character: true, use_as_reference: true } : {}) });
        apply(result);
      }
      if (!alive.current) return;
      setPreview(null);
      if (action === 'confirm') {
        setNotice(`已确认「${role.profile.name}」的这张正片裁剪，并加入本作品的可信图库。其他镜头仍需独立匹配或核对。`);
        try { await onSaved(); } catch (caught) { if (alive.current) setError(`确认已保存，但外层列表刷新失败：${message(caught)}`); }
      } else if (action === 'skip' && result) { setNotice('已保存跳过状态，可以稍后继续。'); nextRole(result, characterId); }
      else if (action === 'reject') setNotice('已记住这个否定结果，不会再为该角色推荐这张候选。');
      else if (action === 'unclear') setNotice('已换过这组候选，新的候选会优先展示。');
    } catch (caught) { if (alive.current) setError(message(caught)); }
    finally { mutation.current = false; if (alive.current) setBusy(''); }
  };
  const sample = async () => {
    if (mutation.current || sampling?.status === 'running') return;
    mutation.current = true; sequence.current += 1; setBusy('sample'); setError(''); setNotice('');
    try { apply(await post<CharacterConfirmationResult>(`${endpoint}/sample`, { max_frames: 12 })); }
    catch (caught) { if (alive.current) setError(message(caught)); }
    finally { mutation.current = false; if (alive.current) setBusy(''); }
  };
  const cacheReferences = async () => {
    if (mutation.current || sampling?.status === 'running') return;
    mutation.current = true; sequence.current += 1; setBusy('cache'); setError(''); setNotice('');
    try {
      const result = await post<BangumiReferenceCacheResult>(`/assets/${encodeURIComponent(asset.id)}/references/cache-bangumi`, { max_images: 20 });
      if (alive.current) setCached(result);
      await load();
    } catch (caught) { if (alive.current) setError(message(caught)); }
    finally { mutation.current = false; if (alive.current) setBusy(''); }
  };
  const close = () => { if (!mutation.current) onClose(); };
  return <Modal title="开始分析前，确认角色" subtitle="将角色档案与正片中的实际形象对应起来 · 进度自动保存" onClose={close} wide>
    <div className="modal-body character-confirmation">
      <p className="confirmation-intro">先核对少量清晰的单人裁剪，再用作本作品后续集数的识别参考。候选来自已有分析；确认只作用于所选画面。未出场或暂时无法辨认的角色可以跳过。</p>
      {error && <Alert tone="error">{error}<button type="button" className="text-button" disabled={!!busy} onClick={() => { setError(''); void load(); }}>重新读取确认进度</button></Alert>}
      {notice && <Alert tone="success">{notice}</Alert>}
      {loading ? <Spinner label="正在读取已有角色候选" /> : data && <>
        <div className="confirmation-progress"><div><strong>已确认 {data.summary.confirmed} / {data.summary.total} 个角色</strong><span>已跳过 {data.summary.skipped} · 待确认 {data.summary.pending}</span></div><progress aria-label="角色确认进度" value={data.summary.confirmed + data.summary.skipped} max={Math.max(1, data.summary.total)} /></div>
        {sampling?.status === 'running' && <div className="confirmation-sampling-progress"><Spinner label={`本地采样中 · ${sampling.completed} / ${sampling.total} 帧`} /><progress aria-label="本地采样进度" value={sampling.completed} max={Math.max(1, sampling.total)} /><p className="helper">可以关闭窗口，稍后回来继续核对。</p></div>}
        <details className="confirmation-options"><summary>候选不足？补充参考图与少量采样</summary>
        <div className="confirmation-sampling"><div><strong>候选不足时，少量补充采样</strong><p className="helper">只在「{asset.title}」采样最多 12 帧，使用本地人物检测与图像匹配，不调用 VLM。已有候选可直接核对。</p></div><button type="button" className="button small secondary" disabled={!!busy || sampling?.status === 'running' || !asset.source_available || data.sampling_available === false} onClick={() => void sample()}><ScanLine size={14} />{sampling?.status === 'failed' || sampling?.status === 'interrupted' ? '重试少量采样' : sampling?.status === 'completed' ? '继续少量本地采样' : '补充少量本地采样'}</button></div>
        {data.roles.some((item) => item.profile.external_source?.image_url) && <div className="confirmation-cache"><p className="helper">首次采样前，可先缓存 Bangumi 档案参考图用于查找候选。缓存图片不会自动加入可信图库。</p><button type="button" className="text-button" disabled={!!busy || sampling?.status === 'running'} onClick={() => void cacheReferences()}>缓存 Bangumi 参考图</button></div>}
        {cached && <Alert tone="info">Bangumi 参考图：新增 {cached.imported} 张 · 已有 {cached.existing} 张 · 跳过 {cached.skipped} 张。仍需在正片中人工核对身份。{cached.errors?.length > 0 && <ul>{cached.errors.map((item, index) => <li key={`${item.character_id}:${index}`}>{data.roles.find((entry) => entry.character_id === item.character_id)?.profile.name || '角色'}：{item.reason}</li>)}</ul>}</Alert>}
        {!asset.source_available && <Alert>当前视频原文件暂不可用，恢复原文件后可补充采样。已保存候选仍可核对。</Alert>}
        {data.sampling_available === false && asset.source_available && <Alert>{data.sampling_unavailable_reason || '本地人物检测或图像匹配模型尚未就绪。'}<a href="#/settings" target="_blank" rel="noreferrer">检查本地模型设置</a></Alert>}
        {sampling?.status === 'completed' && <p className="helper">本地采样已完成：{sampling.completed} / {sampling.total} 帧。没有匹配候选不代表角色一定未出场。</p>}
        {sampling && ['failed', 'unavailable', 'interrupted'].includes(sampling.status) && <Alert tone={sampling.status === 'failed' ? 'error' : 'warning'}>{sampling.error || (sampling.status === 'interrupted' ? '上次采样已中断，可重新进行少量采样。' : '本地模型暂不可用，请检查模型设置后重试。')}</Alert>}
        </details>
        {!data.roles.length ? <Empty icon={<UserRound size={28} />} title="先建立角色档案">请先初始化 Bangumi 角色档案，或新建角色，再回来核对正片形象。</Empty> : <>
          <div className="confirmation-role-list" aria-label="选择待核对角色">{data.roles.map((item) => <button type="button" key={item.character_id} className={item.character_id === selectedId ? 'selected' : ''} aria-pressed={item.character_id === selectedId} disabled={!!busy} onClick={() => navigate(item.character_id)}><span>{item.profile.name}</span><small>{roleStatus[item.status]}</small></button>)}</div>
          {role && <section className="confirmation-review" aria-label={`${role.profile.name}的角色确认`}>
            <aside className="confirmation-profile"><span className="eyebrow">角色档案</span>{role.profile.external_source?.image_url ? <img src={role.profile.external_source.image_url} referrerPolicy="no-referrer" alt={`${role.profile.name}的 Bangumi 档案参考图`} /> : <div className="confirmation-profile-placeholder"><UserRound size={38} /></div>}<h3>{role.profile.name}</h3>{role.profile.name_ja && <p>{role.profile.name_ja}</p>}{role.profile.aliases.length > 0 && <p className="helper">又名 {role.profile.aliases.join('、')}</p>}{role.profile.external_source && <a className="text-button" href={role.profile.external_source.url} target="_blank" rel="noreferrer">Bangumi 角色资料 <ExternalLink size={12} /></a>}<p className="confirmation-description">{role.profile.description || '暂无角色简介，请结合原片核对。'}</p><span className="tag">可信参考 {role.trusted_reference_count} 张</span><p className="helper">身份标签与参考来源会一并保存。同一作品后续集数可复用。</p></aside>
            <div className="confirmation-candidates-area"><div className="confirmation-role-heading"><div><h3>正片里，哪个是{role.profile.name}？</h3><p className="helper">每张分别核对。不同镜头、不同造型的清晰参考更有帮助。</p></div><span className="tag">{roleStatus[role.status]}</span></div>
              {role.status === 'skipped' ? <Empty title="这个角色已暂时跳过" action={<button type="button" className="button secondary" disabled={!!busy} onClick={() => void decide('resume')}>继续确认这个角色</button>}>进度已保留，不要求在当前集数中找到所有角色。</Empty> : <>
                {role.candidates.length ? <div className="confirmation-candidate-grid">{role.candidates.map((candidate) => <CandidateCard key={`${role.character_id}:${candidateKey(candidate)}`} candidate={candidate} role={role} busy={!!busy} onDecision={(action, value) => { void decide(action, [value]); }} onPreview={setPreview} />)}</div> : <Empty title="暂时没有合适候选">可以补充少量本地采样，或暂时跳过。当前集数可能没有这个角色，也可能尚未采到清晰镜头。</Empty>}
                <div className="button-row">{role.candidates.length > 0 && <button type="button" className="button secondary" disabled={!!busy} onClick={() => void decide('unclear', role.candidates)}><RefreshCw size={14} />看不清／换一组</button>}{role.status !== 'confirmed' && <button type="button" className="button ghost" disabled={!!busy} onClick={() => void decide('skip')}>暂时跳过这个角色</button>}{data.summary.pending > (role.status === 'pending' ? 1 : 0) && <button type="button" className="button ghost" disabled={!!busy} onClick={() => nextRole(data, role.character_id)}>下一个待确认角色 <ArrowRight size={14} /></button>}</div>
              </>}
              {preview && <ClipPreview key={candidateKey(preview)} candidate={preview} onClose={() => setPreview(null)} />}
            </div>
          </section>}
        </>}
      </>}
      {busy && <Spinner label={busy === 'sample' ? '正在启动少量采样' : busy === 'cache' ? '正在缓存 Bangumi 参考图' : '正在保存确认进度'} />}
    </div>
    <footer className="modal-footer"><p className="helper">关闭后可从角色档案继续，确认不是开始分析的必填条件。</p><button type="button" className="button secondary" disabled={!!busy} onClick={close}>保存进度并返回</button></footer>
  </Modal>;
}
