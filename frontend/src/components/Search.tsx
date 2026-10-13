import { useState, type FormEvent } from 'react';
import { ArrowRight, Bookmark, Check, Play, Search as SearchIcon, SlidersHorizontal, Sparkles, X } from 'lucide-react';
import { assetLink, formatTime, message, post } from '../api';
import { useApp } from '../context';
import type { SearchResponse, SearchResult } from '../types';
import { Alert, Empty, PageHeader, Spinner, Thumbnail } from './ui';
import { folderLabel } from './Organization';
import { DeepenSceneButton, SceneBadges, SceneFacts } from './SceneAnalysis';

export function ResultCard({ result, saved = false, onSaved }: { result: SearchResult; saved?: boolean; onSaved?: () => void }) {
  const { notify } = useApp();
  const [isSaved, setIsSaved] = useState(saved || result.favorite || false);
  const [busy, setBusy] = useState(false);
  const bookmark = async () => {
    setBusy(true);
    try {
      await post(`/assets/${encodeURIComponent(result.asset_id)}/annotations`, { record_id: result.record_id || result.id, favorite: !isSaved });
      setIsSaved(!isSaved); notify(isSaved ? '已取消收藏' : '片段已加入收藏'); onSaved?.();
    } catch (error) { notify(message(error), 'error'); } finally { setBusy(false); }
  };
  return <article className="result-card"><a className="result-image" href={assetLink(result.asset_id, result.start_ms)}><Thumbnail src={result.thumbnail_url} title={result.title} /><span className="play-overlay"><Play size={18} fill="currentColor" /></span><span className="timecode">{formatTime(result.start_ms)}</span></a><div className="result-body"><div className="result-topline"><a href={assetLink(result.asset_id, result.start_ms)}><strong>{result.title}</strong></a>{result.episode != null && <span>第 {result.episode} 集</span>}<span className="tag">{result.kind === 'subtitle' ? '台词' : result.kind === 'scene' ? '场景' : '画面'}</span>{result.kind === 'scene' && <SceneBadges scene={result} />}{result.is_historical && <span className="tag historical-tag">历史版本</span>}{result.match_type && <span className={`match-badge ${result.match_type}`}>{result.match_type === 'full' ? '完整匹配' : '部分匹配'}</span>}</div><p className={result.kind === 'subtitle' ? 'result-quote' : 'result-description'}>{result.summary || result.text || '暂无描述'}</p>{result.subtitle_text && result.subtitle_text !== result.text && <blockquote>{result.subtitle_text}</blockquote>}{result.kind === 'scene' && <SceneFacts scene={result} assetId={result.asset_id} startMs={result.start_ms} />}{result.match_reason && <p className="match-reason"><Sparkles size={13} />{result.match_reason}</p>}{result.note && <p className="saved-note"><Bookmark size={13} />{result.note}</p>}<div className="result-bottom"><span className="mono">{formatTime(result.start_ms)} — {formatTime(result.end_ms)}</span>{result.version && <span>{result.version}</span>}<span>{result.is_historical ? '历史记录，非当前分析版本' : ['confirmed', 'reviewed', 'user_confirmed'].includes(result.review_status) ? '已复核' : '资料待复核'}</span>{result.kind === 'scene' && result.chunk_id && !result.is_historical && <DeepenSceneButton assetId={result.asset_id} chunkId={result.chunk_id} title={`${result.title} · ${formatTime(result.start_ms)} — ${formatTime(result.end_ms)}`} contextStale={result.context_stale} />}<a href={assetLink(result.asset_id, result.start_ms)}>查看原片 <ArrowRight size={14} /></a></div></div><button className={`icon-button result-save ${isSaved ? 'saved' : ''}`} aria-label={isSaved ? '取消收藏片段' : '收藏片段'} title={isSaved ? '取消收藏' : '收藏片段'} onClick={bookmark} disabled={busy}>{isSaved ? <Check size={18} /> : <Bookmark size={18} />}</button></article>;
}

export default function SearchPage() {
  const { assets, bootstrap, projects = [], folders = [] } = useApp();
  const [query, setQuery] = useState('');
  const [asset, setAsset] = useState('');
  const [kind, setKind] = useState('');
  const [project, setProject] = useState('');
  const [folder, setFolder] = useState('');
  const [tag, setTag] = useState('');
  const [character, setCharacter] = useState('');
  const [season, setSeason] = useState('');
  const [episode, setEpisode] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [searched, setSearched] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const submit = async (event?: FormEvent) => {
    event?.preventDefault(); if (!query.trim()) return;
    setBusy(true); setError('');
    try {
      setResponse(await post<SearchResponse>('/search', { query: query.trim(), filters: { ...(project ? { project_id: project } : {}), ...(folder ? { folder_id: folder } : {}), ...(tag ? { tag } : {}), ...(asset ? { asset_id: asset } : {}), ...(kind ? { kind } : {}), ...(character ? { character } : {}), ...(season ? { season: Number(season) } : {}), ...(episode ? { episode: Number(episode) } : {}) }, limit: 10 }));
      setSearched(query.trim());
    } catch (error) { setError(message(error)); } finally { setBusy(false); }
  };
  const filtersCount = [asset, kind, character, season, episode, project, folder, tag].filter(Boolean).length;
  return <><PageHeader eyebrow="FIND A MOMENT" title="找回那个镜头">不必记得准确的时间。说出画面、动作，或者那句台词。</PageHeader><div className="search-panel"><form onSubmit={submit}><div className="main-search"><SearchIcon size={24} strokeWidth={1.6} /><input aria-label="用自然语言搜索镜头或台词" placeholder="例如：一个人站在窗边，背对镜头看向窗外…" value={query} onChange={(e) => setQuery(e.target.value)} required autoComplete="off" />{query && <button type="button" className="icon-button" aria-label="清空搜索" onClick={() => setQuery('')}><X size={17} /></button>}<button type="submit" className="button primary" disabled={busy || !query.trim()}>{busy ? <Spinner label="检索中" /> : <>检索 <ArrowRight size={17} /></>}</button></div><div className="search-options"><button type="button" className={`text-button ${showFilters ? 'active' : ''}`} onClick={() => setShowFilters(!showFilters)}><SlidersHorizontal size={15} />筛选条件 {filtersCount > 0 && <span className="count-badge">{filtersCount}</span>}</button><span><span className={`small-dot ${bootstrap.settings.bindings.embedding ? '' : 'muted'}`} />{bootstrap.settings.bindings.embedding ? '语义 + 字面联合检索' : '当前使用字面检索'}</span></div>{showFilters && <div className="search-filters"><label>项目<select value={project} onChange={(e) => { setProject(e.target.value); setAsset(''); }}><option value="">全部项目</option>{projects.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}</select></label><label>文件夹<select value={folder} onChange={(e) => setFolder(e.target.value)}><option value="">全部文件夹</option>{folders.map((entry) => <option key={entry.id} value={entry.id}>{folderLabel(entry.id, folders)}</option>)}</select></label><label>项目标签<select value={tag} onChange={(e) => setTag(e.target.value)}><option value="">全部标签</option>{[...new Set(projects.flatMap((entry) => entry.tags))].map((value) => <option key={value} value={value}>{value}</option>)}</select></label><label>作品<select value={asset} onChange={(e) => setAsset(e.target.value)}><option value="">全部作品</option>{assets.map((asset) => <option key={asset.id} value={asset.id}>{asset.title}</option>)}</select></label><label>记录类型<select value={kind} onChange={(e) => setKind(e.target.value)}><option value="">场景、画面与台词</option><option value="scene">场景与剧情</option><option value="visual">画面</option><option value="subtitle">台词</option></select></label><label>角色<input placeholder="已确认的姓名 / 别名" value={character} onChange={(e) => setCharacter(e.target.value)} /></label><label className="number-filter">季<input type="number" min="0" value={season} onChange={(e) => setSeason(e.target.value)} /></label><label className="number-filter">集<input type="number" min="0" value={episode} onChange={(e) => setEpisode(e.target.value)} /></label>{filtersCount > 0 && <button className="text-button" type="button" onClick={() => { setProject(''); setFolder(''); setTag(''); setAsset(''); setKind(''); setCharacter(''); setSeason(''); setEpisode(''); }}>重置</button>}</div>}</form></div>{!bootstrap.settings.bindings.embedding && <div className="subtle-notice"><Sparkles size={15} />配置语义向量模型后，可用相近含义检索。<a href="#/settings">配置模型 <ArrowRight size={13} /></a></div>}{error && <Alert tone="error">{error}</Alert>}{response ? <section className="search-results" aria-live="polite"><div className="section-row"><h3>「{searched}」<span className="muted"> · {response.results.length} 个片段</span></h3></div>{response.degraded?.length > 0 && <Alert>{response.degraded.join('；')}</Alert>}{response.explanation && <div className="search-explanation"><Sparkles size={19} /><p>{response.explanation}</p></div>}{response.results.length > 0 ? <div className="results-list">{response.results.map((result) => <ResultCard key={result.id} result={result} />)}</div> : <Empty icon={<SearchIcon size={29} />} title="没有找到有证据支持的片段">换一个更具体的动作、缩短台词，或扩大检索范围。尚未分析的画面不会出现在结果中。</Empty>}</section> : <div className="search-idle"><div className="eyebrow">A FEW WAYS TO REMEMBER</div><h2>从你记得的细节开始</h2><div className="prompt-grid">{[{ title: '用画面找到片段', text: '穿红色外套的人站在画面左边', icon: '01' }, { title: '用动作找回故事', text: '两个人在雨中拥抱', icon: '02' }, { title: '用台词定位瞬间', text: '输入你记得的一句对白', icon: '03' }].map((prompt) => <button type="button" key={prompt.icon} onClick={() => { setQuery(prompt.icon === '03' ? '' : prompt.text); document.querySelector<HTMLInputElement>('.main-search input')?.focus(); }}><span className="prompt-number">{prompt.icon}</span><h3>{prompt.title}</h3><p>{prompt.text}</p><ArrowRight size={18} /></button>)}</div><p className="search-footnote">每条结果都会带上来源和时间，你可以随时回到原片核对。</p></div>}</>;
}
