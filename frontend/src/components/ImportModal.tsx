import { useEffect, useRef, useState, type FormEvent } from 'react';
import { ArrowRight, FileVideo, Plus, ScanLine, Subtitles, Trash2 } from 'lucide-react';
import { api, post, message } from '../api';
import { useApp } from '../context';
import type { Asset, ImportInput, Job, SubtitleMode, SubtitleDetection } from '../types';
import { compareAnimeImports, inferAnimeFilename } from '../importFilenames';
import { pairSubtitles } from '../importSubtitles';
import { importItemProgress, importProgressMessage } from '../importProgress';
import { groupAnime, animeLabel } from '../anime';
import { localFileDirectory, pickLocalFiles } from '../lib/localFiles';
import { Alert, Field, Modal, Spinner, Status } from './ui';
import SubtitleTrackSelect from './SubtitleTrackSelect';
import AnimeArchivePicker, { useAnimeArchives } from './AnimeArchive';

const cleanPath = (value: string) => {
  if (value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value)) return value;
  const trimmed = value.trim();
  return /^(["']).*\1$/s.test(trimmed) ? trimmed.slice(1, -1) : trimmed;
};
const makeItem = (path = ''): ImportInput => { const guessed = inferAnimeFilename(path); return { item_id: crypto.randomUUID(), video_path: cleanPath(path), title: guessed.title || cleanPath(path).split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') || '', kind: 'animation', series: '', season: guessed.season, episode: guessed.episode, version: '', subtitle_mode: 'auto', subtitle_path: null, subtitle_offset_ms: 0 }; };
export default function ImportModal({ onClose, projectId, retryJobId }: { onClose: () => void; projectId?: string; retryJobId?: string }) {
  const { refresh, notify, bootstrap, assets, projects = [], openJobs } = useApp();
  const [items, setItems] = useState<ImportInput[]>(() => {
    const target = projects.find((entry) => entry.id === projectId); const row = makeItem();
    return [target?.kind === 'animation' ? { ...row, title: target.name, series: target.name } : row];
  });
  const [selected, setSelected] = useState(0);
  const [project, setProject] = useState(() => { const target = projects.find((entry) => entry.id === projectId); return target ? projectId || '' : ''; });
  const [bulkSeason, setBulkSeason] = useState('1');
  const [firstEpisode, setFirstEpisode] = useState('1');
  const [paths, setPaths] = useState('');
  const [offsetTexts, setOffsetTexts] = useState<Record<string, string>>({});
  const [picking, setPicking] = useState(false);
  const [subtitleFiles, setSubtitleFiles] = useState<string[]>([]);
  const [tracks, setTracks] = useState<Record<string, SubtitleDetection>>({});
  const [probeErrors, setProbeErrors] = useState<Record<string, string>>({});
  const [probing, setProbing] = useState<Record<string, boolean>>({});
  const probeVersions = useRef<Record<string, number>>({});
  const detectedPaths = useRef<Record<string, string>>({});
  const manualSubtitles = useRef(new Set<string>());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const request = useRef({ key: '', id: '' });
  const titleEdited = useRef(false);
  const episodeOverrides = useRef(new Set<string>());
  const seasonOverrides = useRef(new Set<string>());
  const mounted = useRef(true);
  const item = items[selected] || items[0];
  const targetProject = projects.find((entry) => entry.id === project);
  const fresh = !job && !retryJobId;
  const archives = useAnimeArchives(items, targetProject, fresh, assets);
  useEffect(() => {
    if (!retryJobId && targetProject?.kind === 'animation') setItems((current) => current.map((row) => !row.video_path && !row.title ? { ...row, kind: 'animation', title: targetProject.name, series: targetProject.name } : row));
  }, [targetProject?.id, targetProject?.kind, targetProject?.name, retryJobId]);
  const status = job?.items?.find((value) => value.item_id === item?.item_id);
  const running = !!job && ['queued', 'running', 'pending', 'pausing', 'cancelling'].includes(job.status);
  const locked = running || !!status?.asset_id;
  useEffect(() => {
    if (!retryJobId) return;
    let current = true; setBusy(true);
    api<{ job: Job; items: ImportInput[]; project_id?: string }>(`/imports/${encodeURIComponent(retryJobId)}`).then((value) => { if (current) { value.items.filter((row) => row.subtitle_mode !== 'auto').forEach((row) => manualSubtitles.current.add(row.item_id)); setItems(value.items); setSelected(Math.max(0, value.items.findIndex((input) => value.job.items?.find((result) => result.item_id === input.item_id)?.status === 'failed'))); setProject(value.project_id || ''); setJob(value.job); } }).catch((error) => { if (current) setError(message(error)); }).finally(() => { if (current) setBusy(false); });
    return () => { current = false; };
  }, [retryJobId]);
  useEffect(() => {
    if (!running || !job) return;
    let active = true;
    const timer = window.setInterval(() => { void api<Job>(`/jobs/${encodeURIComponent(job.id)}`).then((value) => { if (!active) return; setJob(value); if (!['queued', 'running', 'pending', 'pausing', 'cancelling'].includes(value.status)) void refresh(); }).catch((error) => { if (active) setError(message(error)); }); }, 1500);
    return () => { active = false; window.clearInterval(timer); };
  }, [job?.id, running, refresh]);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; for (const id of Object.keys(probeVersions.current)) probeVersions.current[id] += 1; }; }, []);
  const removeItem = () => {
    probeVersions.current[item.item_id] = (probeVersions.current[item.item_id] || 0) + 1;
    delete detectedPaths.current[item.item_id];
    setProbing((current) => ({ ...current, [item.item_id]: false }));
    setItems((current) => current.filter((row) => row.item_id !== item.item_id)); setSelected(0);
  };
  const update = (values: Partial<ImportInput>) => { setItems((current) => current.map((row) => row.item_id === item.item_id ? { ...row, ...values } : row)); setError(''); };
  const changeTitle = (title: string) => {
    titleEdited.current = true;
    setItems((current) => current.map((row) => ({ ...row, title, ...(title !== row.title ? { anime_profile: undefined } : {}) })));
    setError('');
  };
  const changePath = (value: string) => {
    const path = cleanPath(value); probeVersions.current[item.item_id] = (probeVersions.current[item.item_id] || 0) + 1;
    delete detectedPaths.current[item.item_id]; manualSubtitles.current.delete(item.item_id);
    setTracks((current) => { const next = { ...current }; delete next[item.item_id]; return next; });
    setProbeErrors((current) => ({ ...current, [item.item_id]: '' })); setProbing((current) => ({ ...current, [item.item_id]: false }));
    const guessed = inferAnimeFilename(path);
    setItems((current) => {
      const title = fresh && !titleEdited.current && !current[0]?.title && /\.(?:mkv|mp4|m4v|avi|mov|webm|ts|m2ts)$/i.test(path) ? guessed.title || path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') || '' : undefined;
      return current.map((row) => ({ ...row, ...(title !== undefined ? { title } : {}), ...(row.item_id === item.item_id ? { video_path: value, subtitle_stream_index: null, subtitle_mode: 'auto' as const, subtitle_path: null, ...(fresh && !episodeOverrides.current.has(row.item_id) ? { episode: guessed.episode } : {}), ...(fresh && !seasonOverrides.current.has(row.item_id) ? { season: guessed.season } : {}) } : {}) }));
    }); setError('');
  };
  const addPaths = (values: string[]) => {
    setItems((current) => {
      const present = current.filter((row) => cleanPath(row.video_path)); const seen = new Set(present.map((row) => cleanPath(row.video_path)));
      const added = values.map(cleanPath).filter((value) => value && !seen.has(value) && !!seen.add(value)).map(makeItem);
      const rows = [...present, ...added].sort(compareAnimeImports);
      const title = current[0]?.title || (!titleEdited.current ? rows[0]?.title || targetProject?.name || '' : '');
      return rows.length ? rows.map((row) => ({ ...row, title, kind: 'animation', series: targetProject?.name || '' })) : [{ ...makeItem(), title, series: targetProject?.name || '' }];
    }); setSelected(0); setPaths('');
  };
  const probe = async (row: ImportInput) => {
    const id = row.item_id; const path = cleanPath(row.video_path); const version = (probeVersions.current[id] || 0) + 1;
    probeVersions.current[id] = version; detectedPaths.current[id] = path; setProbing((current) => ({ ...current, [id]: true })); setProbeErrors((current) => ({ ...current, [id]: '' }));
    try {
      const result = await post<SubtitleDetection>('/media/subtitle-detection', { video_path: path });
      if (probeVersions.current[id] !== version) return;
      setTracks((current) => ({ ...current, [id]: result }));
      if (!manualSubtitles.current.has(id)) setItems((current) => current.map((value) => value.item_id === id && cleanPath(value.video_path) === path ? { ...value, subtitle_mode: result.recommended_mode, subtitle_path: result.recommended_subtitle_path, subtitle_stream_index: result.recommended_mode === 'container' ? result.recommended_stream_index : null } : value));
    }
    catch (error) { if (probeVersions.current[id] === version) setProbeErrors((current) => ({ ...current, [id]: message(error) })); }
    finally { if (probeVersions.current[id] === version) setProbing((current) => ({ ...current, [id]: false })); }
  };
  const detectionKey = JSON.stringify(items.map((row) => [row.item_id, cleanPath(row.video_path), !!job?.items?.find((value) => value.item_id === row.item_id)?.asset_id]));
  useEffect(() => {
    if (running || busy) return;
    const timer = window.setTimeout(() => {
      const pending = items.filter((row) => cleanPath(row.video_path) && detectedPaths.current[row.item_id] !== cleanPath(row.video_path) && (!job || job.items?.some((value) => value.item_id === row.item_id && value.status === 'failed' && !value.asset_id)));
      pending.forEach((row) => { detectedPaths.current[row.item_id] = cleanPath(row.video_path); });
      const worker = async () => { let row; while (mounted.current && (row = pending.shift())) if (detectedPaths.current[row.item_id] === cleanPath(row.video_path)) await probe(row); };
      void Promise.all(Array.from({ length: Math.min(3, pending.length) }, worker));
    }, 400);
    return () => window.clearTimeout(timer);
  }, [detectionKey, running, busy]);
  const changeMode = (mode: SubtitleMode) => {
    manualSubtitles.current.add(item.item_id);
    const candidates = tracks[item.item_id]?.external_candidates || [];
    const matched = pairSubtitles([item], candidates.map((candidate) => candidate.path), tracks)[item.item_id];
    update({ subtitle_mode: mode, subtitle_stream_index: mode === 'container' ? tracks[item.item_id]?.recommended_stream_index : null, subtitle_path: mode === 'external' ? item.subtitle_path || matched || null : null, subtitle_offset_ms: mode === 'embedded' ? 0 : item.subtitle_offset_ms });
  };
  const chooseFiles = async (kind: 'video' | 'subtitle', replace = false) => {
    setPicking(true); setError('');
    try {
      const values = await pickLocalFiles(kind, !replace, localFileDirectory(cleanPath(item.video_path)));
      if (!values.length || !mounted.current) return;
      if (kind === 'video') { if (replace) changePath(values[0]); else addPaths(values); return; }
      setSubtitleFiles((current) => [...new Set([...current, ...values])]);
      const editable = items.filter((row) => !job?.items?.find((value) => value.item_id === row.item_id)?.asset_id);
      const assignments = values.length === 1 ? { [item.item_id]: values[0] } : pairSubtitles(editable, values, tracks);
      Object.keys(assignments).forEach((id) => manualSubtitles.current.add(id));
      setItems((current) => current.map((row) => assignments[row.item_id] ? { ...row, subtitle_mode: 'external', subtitle_path: assignments[row.item_id], subtitle_stream_index: null } : row));
      const used = new Set(Object.values(assignments));
      notify(values.length === 1 ? '已选择外挂字幕' : `已为 ${used.size} 个视频匹配字幕${values.length > used.size ? `；${values.length - used.size} 个文件请在字幕列表中手动选择` : ''}`);
    } catch (error) { setError(message(error)); } finally { setPicking(false); }
  };
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const selectedItems = job ? items.filter((row) => job.items?.some((result) => result.item_id === row.item_id && result.status === 'failed')) : items;
      if (!selectedItems.length) throw new Error('没有待重试的失败文件。');
      const normalized = selectedItems.map((row) => ({ ...row, anime_profile: row.kind === 'animation' ? job && row.anime_profile ? row.anime_profile : archives.selectionFor(row) : undefined, video_path: cleanPath(row.video_path), title: row.title.trim(), subtitle_path: row.subtitle_mode === 'external' ? cleanPath(row.subtitle_path || '') : null, subtitle_stream_index: row.subtitle_mode === 'container' ? row.subtitle_stream_index : null, subtitle_offset_ms: row.subtitle_mode === 'embedded' ? 0 : Math.round(Number(offsetTexts[row.item_id] ?? row.subtitle_offset_ms / 1000) * 1000) }));
      for (const row of normalized) {
        if (!Number.isFinite(row.subtitle_offset_ms)) throw new Error(`「${row.title}」的字幕时间偏移无效。`);
        if (!row.video_path) throw new Error('请为每一项选择视频文件。');
        if (row.kind === 'animation' && !row.anime_profile) throw new Error(`请先为「${row.title || row.video_path}」选择 Bangumi 档案或从零建档。`);
        if (row.subtitle_mode === 'external' && !row.subtitle_path) throw new Error(`请为「${row.video_path.split(/[\\/]/).pop()}」选择外挂字幕文件。`);
        if (row.subtitle_mode === 'embedded' && !bootstrap.settings.bindings.subtitle) throw new Error('请先为画面字幕 OCR 绑定字幕视觉模型。');
      }
      const payload = { project_id: project || null, items: normalized };
      const key = JSON.stringify(payload); if (request.current.key !== key) request.current = { key, id: crypto.randomUUID() };
      const result = await post<Job>(job ? `/imports/${encodeURIComponent(job.id)}/retry` : '/imports', job ? { items: normalized } : { ...payload, request_id: request.current.id });
      setItems((current) => current.map((row) => normalized.find((input) => input.item_id === row.item_id) || row));
      setJob(result); await refresh(); notify(job ? '失败文件已重新排队' : '视频已加入导入队列');
    } catch (error) { setError(message(error)); } finally { setBusy(false); }
  };
  const analyze = async () => { setBusy(true); try { const all = await api<Asset[]>('/assets'); const ids = new Set(job?.items?.filter((result) => result.asset_id).map((result) => result.asset_id)); const imported = all.filter((asset) => ids.has(asset.id)); if (imported.length) { onClose(); openJobs?.(imported); } else setError('这批视频已不在资料库，可到回收站查看或恢复。'); } catch (error) { setError(message(error)); } finally { setBusy(false); } };
  const result = tracks[item?.item_id]; const usingTracks = item?.subtitle_mode === 'auto' || item?.subtitle_mode === 'container';
  const subtitleChoices = [...new Set([...(result?.external_candidates || []).map((candidate) => candidate.path), ...subtitleFiles, ...(item?.subtitle_path ? [item.subtitle_path] : [])])];
  const detecting = Object.values(probing).some(Boolean);
  const unsupported = items.some((row) => (!job || job.items?.some((value) => value.item_id === row.item_id && value.status === 'failed' && !value.asset_id)) && ['auto', 'container'].includes(row.subtitle_mode) && !!tracks[row.item_id]?.tracks.length && !tracks[row.item_id].tracks.some((track) => track.supported));
  const failed = job?.items?.filter((entry) => entry.status === 'failed').length || 0;
  const unresolved = items.filter((row) => (!job || job.items?.some((entry) => entry.item_id === row.item_id && entry.status === 'failed')) && row.kind === 'animation' && !(job && row.anime_profile) && !archives.selectionFor(row)).length;
  return <Modal title={job ? '批量导入进度' : '导入本机视频'} subtitle="同一部动画一次选择多集，逐集设置集号与字幕，原视频保留在原位置。" onClose={onClose} wide><form onSubmit={submit}>
    <div className="modal-body">
      <>
        <div className="file-entry"><FileVideo size={27} /><div><strong>连接本地视频文件</strong><p>不会复制或上传整部视频。重复导入会建立独立档案，导入后可单独选择分析。</p></div></div>
        {!job && <><div className="button-row"><button type="button" className="button primary" disabled={picking || busy} onClick={() => void chooseFiles('video')}><Plus size={16} />选择多个视频</button>{picking && <Spinner label="请在系统文件窗口中选择" />}</div><p className="helper">打开系统文件选择窗口，可按住 Command 或 Shift 多选。</p><details className="advanced"><summary>粘贴多个绝对路径</summary><Field label="每行一个视频路径"><textarea value={paths} onChange={(event) => setPaths(event.target.value)} rows={3} /></Field><button type="button" className="button secondary small" disabled={!paths.trim() || picking || busy} onClick={() => addPaths(paths.split(/\r?\n/))}>添加路径列表</button></details></>}
        <Field label="归属项目"><select value={project} disabled={!!job || !!retryJobId || busy} onChange={(event) => { setProject(event.target.value); const target = projects.find((entry) => entry.id === event.target.value); setItems((current) => current.map((row) => ({ ...row, kind: 'animation', anime_profile: undefined, ...(target ? { series: target.name } : { series: '' }) }))); }}><option value="">未归类</option>{projects.map((entry) => <option value={entry.id} key={entry.id}>{entry.name}</option>)}</select></Field>
        {fresh && groupAnime(assets.filter((entry) => (entry.project_id || '') === project)).length > 0 && <Field label="已有动画 / 季度" hint="选择后沿用该动画名称、季度与档案；新动画可直接填写作品名称。"><select value="" onChange={(event) => { const group = groupAnime(assets).find((entry) => entry.id === event.target.value); if (!group) return; titleEdited.current = true; setItems((current) => current.map((row) => ({ ...row, title: group.title, season: group.season, anime_profile: undefined }))); }}><option value="">新建动画，或选择已有动画…</option>{groupAnime(assets.filter((entry) => (entry.project_id || '') === project)).map((group) => <option key={group.id} value={group.id}>{animeLabel(group)}</option>)}</select></Field>}{fresh && item && <fieldset disabled={busy}><Field label="作品名称" hint="填写具体动画或季度的名称；同一项目中的 TV 与剧场版分别命名。所有选入文件共用这个名称。"><input value={items[0].title} onChange={(event) => changeTitle(event.target.value)} required /></Field><AnimeArchivePicker item={items[0]} archives={archives} /></fieldset>}
        {fresh && items.length > 1 && <details className="advanced"><summary>批量设置季号与连续集号</summary><p className="helper">已按文件名识别集号并排序；按左侧文件顺序编号，应用后仍可逐集修改。</p><div className="form-grid"><Field label="批量季号"><input type="number" min="0" step="1" value={bulkSeason} onChange={(event) => setBulkSeason(event.target.value)} /></Field><Field label="起始集号"><input type="number" min="0" step="1" value={firstEpisode} onChange={(event) => setFirstEpisode(event.target.value)} /></Field></div><button type="button" className="button secondary small" disabled={!bulkSeason || !firstEpisode || !Number.isInteger(Number(bulkSeason)) || !Number.isInteger(Number(firstEpisode)) || Number(bulkSeason) < 0 || Number(firstEpisode) < 0} onClick={() => setItems((current) => current.map((row, index) => { episodeOverrides.current.add(row.item_id); seasonOverrides.current.add(row.item_id); return { ...row, season: Number(bulkSeason), episode: Number(firstEpisode) + index }; }))}>应用连续集号</button></details>}
        {job && <div className="import-progress" aria-live="polite"><Status status={job.status} /><span>{job.items?.filter((entry) => ['completed', 'complete'].includes(entry.status)).length || 0} / {items.length} 个视频已导入{failed ? ` · ${failed} 个失败` : ''}</span>{running && <Spinner label={importProgressMessage(job)} />}<a href="#/jobs" onClick={onClose}>在任务队列查看</a></div>}
        <div className="import-layout"><div className="import-items" aria-label="待导入视频">{items.map((row, index) => { const state = job?.items?.find((entry) => entry.item_id === row.item_id); return <button type="button" className={`import-row ${selected === index ? 'active' : ''}`} key={row.item_id} onClick={() => setSelected(index)}><strong>{fresh ? row.episode !== null ? `${row.season !== null ? `第 ${row.season} 季 · ` : ''}第 ${row.episode} 集` : `视频 ${index + 1} · 集号待确认` : row.title || `视频 ${index + 1}`}</strong><span>{row.video_path.split(/[\\/]/).pop() || '请选择文件'}</span>{state && <><Status status={state.status} />{job && importItemProgress(job, state) && <span>{importItemProgress(job, state)}</span>}</>}</button>; })}</div>{item && <div className="import-editor"><div className="section-row"><h3>第 {selected + 1} 项 · 文件设置</h3>{fresh && items.length > 1 && <button type="button" className="text-button" aria-label="移除当前视频" disabled={busy || picking} onClick={removeItem}><Trash2 size={14} />移除</button>}</div>
          {status?.error && <Alert tone="error">{status.error}</Alert>}{status?.asset_id && status.status === 'failed' && <Alert>视频资料已保存，可重试完成剩余处理。原文件配置保持不变。</Alert>}
          {status?.warnings?.map((warning) => <Alert key={warning}>{warning}</Alert>)}
          <fieldset disabled={busy || locked || picking}>
            <div className="selected-local-file"><strong>{item.video_path.split(/[\\/]/).pop() || '尚未选择视频'}</strong><button type="button" className="button secondary small" onClick={() => void chooseFiles('video', true)}>{item.video_path ? '更换视频文件' : '选择视频文件'}</button></div>
            <details className="advanced"><summary>查看或粘贴视频路径</summary><Field label="视频绝对路径"><input autoComplete="off" value={item.video_path} onChange={(event) => changePath(event.target.value)} placeholder="/Users/你的名字/Movies/第一集.mkv" /></Field></details>
            {!fresh && <><Field label="作品名称"><input value={item.title} disabled required /></Field>{item.kind === 'animation' && <AnimeArchivePicker item={item} archives={archives} readOnly={!!item.anime_profile} />}</>}
            <div className="form-grid"><Field label="季（可选）"><input type="number" min="0" step="1" value={item.season ?? ''} onChange={(event) => { seasonOverrides.current.add(item.item_id); update({ season: event.target.value ? Number(event.target.value) : null }); }} /></Field><Field label="集（可选）"><input type="number" min="0" step="1" value={item.episode ?? ''} onChange={(event) => { episodeOverrides.current.add(item.item_id); update({ episode: event.target.value ? Number(event.target.value) : null }); }} /></Field></div>
            <Field label="视频版本（可选）"><input value={item.version} onChange={(event) => update({ version: event.target.value })} placeholder="例如：1080p、导演剪辑版" /></Field>
            <div className="section-row"><span className="field-label section-label">这集的字幕来源</span><button type="button" className="text-button" disabled={!cleanPath(item.video_path) || probing[item.item_id]} onClick={() => { manualSubtitles.current.delete(item.item_id); void probe(item); }}>重新自动检测</button></div>
            {probing[item.item_id] ? <Spinner label="正在检测字幕轨与同目录外挂字幕" /> : result && <p className="helper" aria-live="polite">{result.reason} 可随时手动切换。</p>}
            {result?.warnings?.map((warning) => <Alert key={warning}>{warning}</Alert>)}
            {probeErrors[item.item_id] && <Alert tone="error">{probeErrors[item.item_id]}</Alert>}
            <div className="choice-grid subtitle-source-choices"><button type="button" className={`choice ${usingTracks ? 'selected' : ''}`} aria-pressed={usingTracks} onClick={() => changeMode('container')}><Subtitles size={19} /><strong>自动提取字幕轨</strong><span>在本机提取文本，无需模型</span></button><button type="button" className={`choice ${item.subtitle_mode === 'external' ? 'selected' : ''}`} aria-pressed={item.subtitle_mode === 'external'} onClick={() => changeMode('external')}><Subtitles size={19} /><strong>外挂字幕文件</strong><span>SRT / VTT / ASS</span></button><button type="button" className={`choice ${item.subtitle_mode === 'embedded' ? 'selected' : ''}`} aria-pressed={item.subtitle_mode === 'embedded'} onClick={() => changeMode('embedded')}><ScanLine size={19} /><strong>画面字幕 OCR</strong><span>仅用于烧录在画面中的文字</span></button></div>
            {usingTracks && <div className="subtitle-track-panel">{result ? <SubtitleTrackSelect result={result} value={item.subtitle_stream_index ?? null} allowAuto onChange={(index) => { manualSubtitles.current.add(item.item_id); update({ subtitle_stream_index: index, subtitle_mode: 'container' }); }} /> : <p className="helper">选入视频后自动检测字幕，文本字幕无需配置模型。</p>}</div>}
            {item.subtitle_mode === 'external' && <><button className="button secondary small" type="button" onClick={() => void chooseFiles('subtitle')}>选择本机字幕文件（可多选）</button><p className="helper">多选字幕按文件名与集号匹配到这批视频；匹配不明确时逐集选择。</p><Field label="这集的外挂字幕" hint={item.subtitle_path || '请通过系统窗口选择字幕，或从检测到的文件中选择。'}><select value={item.subtitle_path || ''} onChange={(event) => { manualSubtitles.current.add(item.item_id); update({ subtitle_path: event.target.value || null }); }}><option value="">请选择字幕文件</option>{subtitleChoices.map((path) => <option key={path} value={path}>{result?.external_candidates.find((candidate) => candidate.path === path)?.relative_path || path}</option>)}</select></Field></>}
            {item.subtitle_mode !== 'embedded' && <Field label="时间偏移（秒）" hint="正值表示延后，仅作用于当前文件"><input type="number" step="0.001" value={offsetTexts[item.item_id] ?? item.subtitle_offset_ms / 1000} onChange={(event) => { setOffsetTexts((current) => ({ ...current, [item.item_id]: event.target.value })); if (Number.isFinite(Number(event.target.value))) update({ subtitle_offset_ms: Math.round(Number(event.target.value) * 1000) }); }} required /></Field>}
            {item.subtitle_mode === 'embedded' && !bootstrap.settings.bindings.subtitle && <Alert>请先在<a href="#/settings" onClick={onClose}>模型与设置</a>中绑定字幕视觉模型，再导入画面烧录字幕视频。</Alert>}
          </fieldset>
        </div>}</div>
      </>{error && <Alert tone="error">{error}</Alert>}
    </div>
    <footer className="modal-footer"><span>{unresolved ? `${unresolved} 个动画待选择初始档案` : job ? '成功项保留；修正失败文件后可单独重试' : `${items.length} 个视频 · 不自动调用分析模型`}</span>{job && !running && job.items?.some((entry) => entry.asset_id) && openJobs && <button type="button" className="button secondary" disabled={busy || picking} onClick={() => void analyze()}>分析这批</button>}{(!job || failed > 0) && <button type="submit" className="button primary" disabled={busy || picking || running || detecting || !!unsupported || unresolved > 0 || (item?.subtitle_mode === 'embedded' && !bootstrap.settings.bindings.subtitle)}>{busy ? <Spinner label="正在提交" /> : <>{job ? '重试失败文件' : '加入资料库'}<ArrowRight size={16} /></>}</button>}</footer>
  </form></Modal>;
}
