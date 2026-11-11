import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type MouseEvent } from 'react';
import { ArrowRight, Bookmark, Check, Play, Search as SearchIcon, SlidersHorizontal, Sparkles, X } from 'lucide-react';
import { assetLink, formatTime, message, post } from '../api';
import { useApp } from '../context';
import type { SearchResponse, SearchResult, SearchScoreBasis } from '../types';
import { Alert, Empty, PageHeader, Spinner, Thumbnail } from './ui';
import { folderLabel } from './Organization';
import { groupAnime, animeLabel, episodeLabel } from '../anime';
import { DeepenSceneButton, SceneBadges, SceneFacts } from './SceneAnalysis';
import { identitySourceLabel, identityStatusLabel } from '../identityTypes';
import { RetrievalFrames } from './RetrievalEvidence';
import EntityIdentities from './EntityIdentity';
import { PreparationStatus, ResourceStatus } from './PreparationStatus';
import { KnowledgeResults } from './KnowledgePanel';
import SearchTimingDetails from './SearchTiming';
import { searchClickStart, type SearchPresentationTiming } from '../searchTiming';
import './Search.css';

const characterGroundedScoreHint = '人物证据不足会降低分数，最高结果也可能低于100%；分数不代表正确概率。';

function CharacterMatchEvidence({ result }: { result: SearchResult }) {
  const match = result.character_match;
  if (!match || !Number.isFinite(match.required_count) || match.required_count <= 0) return null;
  const hasEvidence = match.status !== 'unverified' && (match.confirmed_count > 0 || match.candidate_count > 0);
  const summary = hasEvidence
    ? `命中帧人物：已确认 ${match.confirmed_count}/${match.required_count}，候选 ${match.candidate_count}/${match.required_count}`
    : '命中帧的指定人物尚未证实（可能漏检）';
  const reason = match.reason?.trim();
  return <div className="search-character-match" aria-label="命中帧人物证据">
    <p>{summary}</p>
    {reason && reason !== summary && <p className="helper">{reason}</p>}
    {hasEvidence && <p className="helper">候选人数不含已确认人物；人物身份不证明动作与空间关系，请结合原片核对。
      {match.frame_id && <> <a href={`/api/assets/${encodeURIComponent(result.asset_id)}/frames/${encodeURIComponent(match.frame_id)}`} target="_blank" rel="noreferrer">查看命中帧人物证据</a></>}
    </p>}
  </div>;
}

export function ResultCard({ result, saved = false, onSaved, scoreBasis }: { result: SearchResult; saved?: boolean; onSaved?: () => void; scoreBasis?: SearchScoreBasis }) {
  const { notify } = useApp();
  const seekMs = result.seek_ms ?? result.start_ms;
  const retrieval = result.evidence_type === 'retrieval_window';
  const scoreHint = (result.match_score_basis || scoreBasis) === 'character_grounded_rrf_v1'
    ? characterGroundedScoreHint : '本次最高候选为100%，随检索范围变化，不代表正确概率';
  const assetUrl = `/api/assets/${encodeURIComponent(result.asset_id)}`;
  const thumbnail = result.hit_frame_id ? `${assetUrl}/frames/${encodeURIComponent(result.hit_frame_id)}`
    : result.thumbnail_url || `${assetUrl}/preview?at_ms=${seekMs}`;
  const evidenceLink = `${assetLink(result.asset_id, seekMs)}${result.kind === 'audio' ? `&audio=${encodeURIComponent(result.record_id || result.id)}` : ''}`;
  const [isSaved, setIsSaved] = useState(saved || result.favorite || false);
  const [busy, setBusy] = useState(false);
  const bookmark = async () => {
    setBusy(true);
    try {
      await post(`/assets/${encodeURIComponent(result.asset_id)}/annotations`, {
        record_id: result.record_id || result.id, favorite: !isSaved,
        ...(!isSaved && result.hit_frame_id ? { preview_frame_id: result.hit_frame_id,
          ...(result.member_ids?.length ? { preview_member_ids: result.member_ids } : {}) } : {}),
      });
      setIsSaved(!isSaved); notify(isSaved ? '已取消收藏' : '片段已加入收藏'); onSaved?.();
    } catch (error) { notify(message(error), 'error'); } finally { setBusy(false); }
  };
  return <article className="result-card"><a className="result-image" href={evidenceLink}><Thumbnail src={thumbnail} title={result.title} /><span className="play-overlay"><Play size={18} fill="currentColor" /></span><span className="timecode">{formatTime(seekMs)}</span></a><div className="result-body"><div className="result-topline"><a href={evidenceLink}><strong>{result.title}</strong></a>{result.episode != null && <span>第 {result.episode} 集</span>}<span className="tag">{result.kind === 'audio' ? '听到的声音' : result.kind === 'subtitle' ? '台词' : result.kind === 'scene' ? '场景' : '画面'}</span>{result.kind === 'scene' && <SceneBadges scene={result} />}{result.is_historical && <span className="tag historical-tag">历史版本</span>}{typeof result.match_score === 'number' && Number.isFinite(result.match_score) && <span className="search-match-score" title={scoreHint}>相对匹配度 {result.match_score.toFixed(1)}%</span>}{result.match_type && <span className={`match-badge ${result.match_type}`}>{result.match_type === 'full' ? '完整匹配' : '部分匹配'}</span>}</div><p className={result.kind === 'subtitle' ? 'result-quote' : 'result-description'}>{result.summary || result.text || (retrieval ? '画面证据' : '暂无描述')}</p>{retrieval && <><p className="helper">直接视觉检索候选，请播放原片核对动作与人物关系。</p><RetrievalFrames assetId={result.asset_id} record={result} /></>}{result.hit_frame_basis === 'character_evidence' && <p className="helper">按人物身份依据定位；动作与位置仍需核对</p>}<CharacterMatchEvidence result={result} /><EntityIdentities entities={result.entities} />{result.kind === 'audio' && <div className="search-audio-evidence"><span className="tag">{identityStatusLabel(result.association_status)}</span><p className="helper">说话人关联：{identitySourceLabel(result.association_source)} · 声音证据不证明角色当前出场或动作。</p><audio controls preload="none" aria-label="试听检索声音证据" src={`/api/assets/${encodeURIComponent(result.asset_id)}/audio/records/${encodeURIComponent(result.record_id || result.id)}/clip`} /></div>}{result.subtitle_text && result.subtitle_text !== result.text && <blockquote>{result.subtitle_text}</blockquote>}{result.kind === 'scene' && <SceneFacts scene={result} assetId={result.asset_id} startMs={result.start_ms} />}{result.match_reason && <p className="match-reason"><Sparkles size={13} />{result.match_reason}</p>}{result.note && <p className="saved-note"><Bookmark size={13} />{result.note}</p>}<div className="result-bottom"><span className="mono">{formatTime(result.start_ms)} — {formatTime(result.end_ms)}</span>{result.version && <span>{result.version}</span>}<span>{result.is_historical ? '历史记录，非当前分析版本' : ['confirmed', 'reviewed', 'user_confirmed'].includes(result.review_status) ? '已复核' : '资料待复核'}</span>{result.kind === 'scene' && result.chunk_id && !result.is_historical && <DeepenSceneButton contentKind={result.content_kind} assetId={result.asset_id} chunkId={result.chunk_id} title={`${result.title} · ${formatTime(result.start_ms)} — ${formatTime(result.end_ms)}`} contextStale={result.context_stale} />}<a href={evidenceLink}>查看原片 <ArrowRight size={14} /></a></div></div><button className={`icon-button result-save ${isSaved ? 'saved' : ''}`} aria-label={isSaved ? '取消收藏片段' : '收藏片段'} title={isSaved ? '取消收藏' : '收藏片段'} onClick={bookmark} disabled={busy}>{isSaved ? <Check size={18} /> : <Bookmark size={18} />}</button></article>;
}

export default function SearchPage() {
  const { assets, bootstrap, projects = [], folders = [] } = useApp();
  const [query, setQuery] = useState('');
  const indexReady = !bootstrap.startup || bootstrap.startup.index.status === 'ready';
  const encoderChecking = !!bootstrap.startup && ['pending', 'running'].includes(bootstrap.startup.encoder.status);
  const scopeParams = () => new URLSearchParams(window.location.hash.split('?')[1] || '');
  const [asset, setAsset] = useState(() => scopeParams().get('asset_id') || '');
  const [anime, setAnime] = useState(() => scopeParams().get('anime_id') || '');
  const [kind, setKind] = useState('');
  const [project, setProject] = useState(() => scopeParams().get('project_id') || '');
  const [folder, setFolder] = useState('');
  const [tag, setTag] = useState('');
  const [character, setCharacter] = useState('');
  const [season, setSeason] = useState('');
  const [episode, setEpisode] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [visualVerification, setVisualVerification] = useState(false);
  const [enhancements, setEnhancements] = useState(false);
  const [knowledgeMode, setKnowledgeMode] = useState('baseline');
  const [minMatchScore, setMinMatchScore] = useState('50');
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [presentationTiming, setPresentationTiming] = useState<SearchPresentationTiming>();
  const buttonClick = useRef<ReturnType<typeof searchClickStart>>();
  const pendingTiming = useRef<{ submittedAt: number; resolvedAt: number; click?: ReturnType<typeof searchClickStart> }>();
  const captureSearchClick = (event: MouseEvent<HTMLButtonElement>) => {
    // Keyboard activation also emits a click, with detail=0. Keep it on the
    // submit boundary rather than reporting it as a physical pointer click.
    const click = event.detail > 0 ? searchClickStart(event.timeStamp) : undefined;
    buttonClick.current = click;
    // A validation failure must not leak this click into a later submission.
    window.setTimeout(() => { if (buttonClick.current === click) buttonClick.current = undefined; }, 0);
  };
  useLayoutEffect(() => {
    const pending = pendingTiming.current;
    if (!response || !pending) { setPresentationTiming(undefined); return; }
    const committedAt = performance.now();
    const timing: SearchPresentationTiming = {
      trigger: pending.click ? 'pointer_click' : 'form_submit',
      ...(pending.click ? { click_time_source: pending.click.source,
        click_to_submit_ms: pending.submittedAt - pending.click.clickedAt,
        click_to_result_commit_ms: committedAt - pending.click.clickedAt } : {}),
      submit_to_result_commit_ms: committedAt - pending.submittedAt,
      submit_to_fetch_ms: response.client_timing ? response.client_timing.fetch_started_at - pending.submittedAt : 0,
      json_to_result_commit_ms: committedAt - pending.resolvedAt, debounce_ms: 0,
    };
    setPresentationTiming(timing);
    // Two frame callbacks provide a paint opportunity, not a guarantee that
    // thumbnails or fonts have finished displaying.
    if (typeof requestAnimationFrame !== 'function') return;
    let frame = requestAnimationFrame(() => {
      frame = requestAnimationFrame(() => {
        const paintedAt = performance.now();
        setPresentationTiming({ ...timing, submit_to_paint_opportunity_ms: paintedAt - pending.submittedAt,
          ...(pending.click ? { click_to_paint_opportunity_ms: paintedAt - pending.click.clickedAt } : {}) });
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [response]);
  const [searched, setSearched] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { const changed = () => { const params = scopeParams(); setProject(params.get('project_id') || ''); setAnime(params.get('anime_id') || ''); setAsset(params.get('asset_id') || ''); setResponse(null); }; window.addEventListener('hashchange', changed); return () => window.removeEventListener('hashchange', changed); }, []);
  const animeGroups = groupAnime(assets.filter((entry) => !project || entry.project_id === project)).filter((group) => group.assets[0].anime_id);
  const scopedAssets = assets.filter((entry) => (!project || entry.project_id === project) && (!anime || entry.anime_id === anime)).sort((a, b) => (a.season ?? 0) - (b.season ?? 0) || (a.episode ?? Infinity) - (b.episode ?? Infinity) || a.title.localeCompare(b.title, 'zh-CN', { numeric: true }) || a.created_at.localeCompare(b.created_at));
  const submit = async (event?: FormEvent) => {
    const submittedAt = performance.now();
    const click = buttonClick.current;
    buttonClick.current = undefined;
    event?.preventDefault(); if (!query.trim() || !indexReady) return;
    const threshold = Number(minMatchScore);
    if (!minMatchScore.trim() || !Number.isFinite(threshold) || threshold < 0 || threshold > 100) {
      setError('相对匹配度阈值须为 0 至 100 的有效数字。');
      return;
    }
    setBusy(true); setError(''); setPresentationTiming(undefined);
    try {
      const result = await post<SearchResponse>('/search', { query: query.trim(), filters: { ...(project ? { project_id: project } : {}), ...(folder ? { folder_id: folder } : {}), ...(tag ? { tag } : {}), ...(anime ? { anime_id: anime } : {}), ...(asset ? { asset_id: asset } : {}), ...(kind ? { kind } : {}), ...(character ? { character } : {}), ...(season ? { season: Number(season) } : {}), ...(episode ? { episode: Number(episode) } : {}) }, min_match_score: threshold, visual_verification: visualVerification, enhancements, ...(knowledgeMode === 'lightrag' ? { knowledge_mode: knowledgeMode } : {}) });
      pendingTiming.current = { submittedAt, resolvedAt: result.client_timing?.json_parsed_at ?? performance.now(), click };
      setResponse(result);
      setSearched(query.trim());
    } catch (error) { setError(message(error)); } finally { setBusy(false); }
  };
  const visualAvailable = bootstrap.visual_encoder?.available === true;
  const hasTextEmbedding = !!bootstrap.settings.bindings.embedding;
  const searchMode = visualAvailable ? hasTextEmbedding ? '视觉 + 文字语义 + 字面检索 · 按已有索引' : '视觉 + 字面检索 · 按已有索引' : hasTextEmbedding ? '语义 + 字面联合检索' : '当前使用字面检索';
  const filtersCount = [asset, anime, kind, character, season, episode, project, folder, tag].filter(Boolean).length;
  return <><PageHeader eyebrow="FIND A MOMENT" title="找回那个镜头">不必记得准确的时间。说出画面、动作，或者那句台词。</PageHeader><PreparationStatus task="index" /><ResourceStatus resource="assets" /><ResourceStatus resource="projects" /><ResourceStatus resource="folders" /><div className="search-panel"><form onSubmit={submit}><div className="main-search"><SearchIcon size={24} strokeWidth={1.6} /><input aria-label="用自然语言搜索镜头或台词" placeholder="例如：一个人站在窗边，背对镜头看向窗外…" value={query} onChange={(e) => setQuery(e.target.value)} required autoComplete="off" />{query && <button type="button" className="icon-button" aria-label="清空搜索" onClick={() => setQuery('')}><X size={17} /></button>}<button type="submit" className="button primary" onClick={captureSearchClick} disabled={busy || !query.trim() || !indexReady}>{busy ? <Spinner label="检索中" /> : <>检索 <ArrowRight size={17} /></>}</button></div><div className="search-scope" aria-label="检索范围"><label>项目<select disabled={busy} value={project} onChange={(e) => { setProject(e.target.value); setAnime(''); setAsset(''); setSeason(''); setEpisode(''); setResponse(null); }}><option value="">全局 · 全部项目</option>{projects.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}</select></label><label>动画 / 季度<select disabled={busy} value={anime} onChange={(e) => { setAnime(e.target.value); setAsset(''); setSeason(''); setEpisode(''); setResponse(null); }}><option value="">全部动画 / 季度</option>{animeGroups.map((group) => <option key={group.id} value={group.id}>{animeLabel(group)}</option>)}</select></label><label>单集 / 视频<select disabled={busy} value={asset} onChange={(e) => { setAsset(e.target.value); setResponse(null); }}><option value="">全部集数</option>{scopedAssets.map((entry) => <option key={entry.id} value={entry.id}>{episodeLabel(entry)}</option>)}</select></label>{(project || anime || asset) && <button type="button" className="text-button" onClick={() => { setProject(''); setAnime(''); setAsset(''); setFolder(''); setTag(''); setKind(''); setCharacter(''); setSeason(''); setEpisode(''); setResponse(null); }}>恢复全局范围</button>}</div><div className="search-match-threshold"><label>最低相对匹配度<span className="search-match-threshold-value"><input type="number" aria-label="最低相对匹配度（%）" aria-describedby="search-match-threshold-hint" min="0" max="100" step="0.1" required disabled={busy} value={minMatchScore} onChange={(event) => setMinMatchScore(event.target.value)} /><span>%</span></span></label><p className="helper" id="search-match-threshold-hint">仅显示严格高于此值的片段。{response?.selection?.score_basis === 'character_grounded_rrf_v1' ? `${characterGroundedScoreHint}分数随检索范围变化。` : '本次最高候选为100%，随检索范围变化，不代表正确概率。'}</p></div><div className="search-options"><button type="button" className={`text-button ${showFilters ? 'active' : ''}`} onClick={() => setShowFilters(!showFilters)}><SlidersHorizontal size={15} />筛选条件 {filtersCount > 0 && <span className="count-badge">{filtersCount}</span>}</button><span><span className={`small-dot ${visualAvailable || hasTextEmbedding ? '' : 'muted'}`} />{searchMode}</span></div>{showFilters && <div className="search-filters"><label>文件夹<select value={folder} onChange={(e) => setFolder(e.target.value)}><option value="">全部文件夹</option>{folders.map((entry) => <option key={entry.id} value={entry.id}>{folderLabel(entry.id, folders)}</option>)}</select></label><label>项目标签<select value={tag} onChange={(e) => setTag(e.target.value)}><option value="">全部标签</option>{[...new Set(projects.flatMap((entry) => entry.tags))].map((value) => <option key={value} value={value}>{value}</option>)}</select></label><label>记录类型<select value={kind} onChange={(e) => setKind(e.target.value)}><option value="">场景、画面与台词</option><option value="scene">场景与剧情</option><option value="visual">画面</option><option value="subtitle">台词</option><option value="audio">说话人与声音</option></select></label><label>角色<input placeholder="已确认的姓名 / 别名" value={character} onChange={(e) => setCharacter(e.target.value)} /></label><label className="number-filter">季<input type="number" min="0" value={season} onChange={(e) => setSeason(e.target.value)} /></label><label className="number-filter">集<input type="number" min="0" value={episode} onChange={(e) => setEpisode(e.target.value)} /></label>{filtersCount > 0 && <button className="text-button" type="button" onClick={() => { setProject(''); setAnime(''); setFolder(''); setTag(''); setAsset(''); setKind(''); setCharacter(''); setSeason(''); setEpisode(''); }}>重置</button>}</div>}<div className="search-enhancements"><label>知识召回<select aria-label="知识召回模式" value={knowledgeMode} disabled={busy} onChange={(event) => { setKnowledgeMode(event.target.value); setResponse(null); }}><option value="baseline">词法知识基线</option><option value="lightrag">LightRAG 实验</option></select></label>{knowledgeMode === 'lightrag' && <p className="helper">选择已建知识索引的作品范围；未配置、多作品或故障时继续使用词法基线。</p>}<label><input type="checkbox" checked={visualVerification} disabled={busy} onChange={(event) => setVisualVerification(event.target.checked)} />核对前 3 个视觉候选</label><p className="helper">最多每候选 4 帧、1 次模型请求、30 秒；成功结果可复用。未复核的视觉候选保留部分匹配。</p><label><input type="checkbox" checked={enhancements} disabled={busy} onChange={(event) => setEnhancements(event.target.checked)} />启用查询理解与结果解释</label></div></form></div>{!encoderChecking && !hasTextEmbedding && !visualAvailable && <div className="subtle-notice"><Sparkles size={15} />配置语义向量模型后，可用相近含义检索。<a href="#/settings">配置模型 <ArrowRight size={13} /></a></div>}{error && <Alert tone="error">{error}</Alert>}{response ? <section className="search-results" aria-live="polite"><div className="section-row"><h3>「{searched}」<span className="muted"> · {response.results.length} 个片段</span></h3></div>{response.selection?.mode === 'threshold' && <p className="helper search-selection-summary">已应用：相对匹配度严格高于 {response.selection.min_match_score}%，按相对匹配度从高到低排列，全部 {response.selection.returned_count} 个片段。{response.selection.score_basis === 'character_grounded_rrf_v1' && <> {characterGroundedScoreHint}</>}</p>}{response.degraded?.length > 0 && <Alert>{response.degraded.join('；')}</Alert>}{response.verification && <p className="helper" role="status">视觉复核：{({ completed: "已完成", cache_hit: "复用已核对结果", timeout: "已超时，保留原候选", failed: "失败，保留原候选", unavailable: "不可用" } as Record<string, string>)[response.verification.status] || response.verification.status}{response.verification.candidate_count != null && ` · ${response.verification.candidate_count} 个候选`}{response.verification.request_count != null && ` · 本次 ${response.verification.request_count} 次请求`}{response.verification.reason && ` · ${response.verification.reason}`}</p>}{!!response.knowledge_hits?.length && <section aria-label="相关知识资料"><h3>相关知识与出处</h3><KnowledgeResults hits={response.knowledge_hits} /></section>}<SearchTimingDetails server={response.search_timing} client={response.client_timing} presentation={presentationTiming} />{response.knowledge_diagnostics && <details><summary>知识检索耗时与调用诊断</summary><pre className="json-report">{JSON.stringify(response.knowledge_diagnostics, null, 2)}</pre></details>}{response.explanation && <div className="search-explanation"><Sparkles size={19} /><p>{response.explanation}</p></div>}{response.results.length > 0 ? <div className="results-list">{response.results.map((result) => <ResultCard key={result.id} result={result} scoreBasis={response.selection?.score_basis} />)}</div> : <Empty icon={<SearchIcon size={29} />} title={response.selection?.mode === 'threshold' ? '没有高于阈值的片段' : '没有找到有证据支持的片段'}>{response.selection?.mode === 'threshold' ? '降低相对匹配度阈值，或调整检索词与范围。尚未分析的画面不会出现在结果中。' : '换一个更具体的动作、缩短台词，或扩大检索范围。尚未分析的画面不会出现在结果中。'}</Empty>}</section> : <div className="search-idle"><div className="eyebrow">A FEW WAYS TO REMEMBER</div><h2>从你记得的细节开始</h2><div className="prompt-grid">{[{ title: '用画面找到片段', text: '穿红色外套的人站在画面左边', icon: '01' }, { title: '用动作找回故事', text: '两个人在雨中拥抱', icon: '02' }, { title: '用台词定位瞬间', text: '输入你记得的一句对白', icon: '03' }].map((prompt) => <button type="button" key={prompt.icon} onClick={() => { setQuery(prompt.icon === '03' ? '' : prompt.text); document.querySelector<HTMLInputElement>('.main-search input')?.focus(); }}><span className="prompt-number">{prompt.icon}</span><h3>{prompt.title}</h3><p>{prompt.text}</p><ArrowRight size={18} /></button>)}</div><p className="search-footnote">每条结果都会带上来源和时间，你可以随时回到原片核对。</p></div>}</>;
}
