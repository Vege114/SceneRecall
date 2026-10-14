import { useEffect, useRef, useState, type FormEvent } from 'react';
import { ArrowRight, FileVideo, Plus, ScanLine, Subtitles, Trash2 } from 'lucide-react';
import { api, post, message } from '../api';
import { useApp } from '../context';
import type { Asset, AssetKind, ImportInput, Job, SubtitleMode, SubtitleTracks } from '../types';
import { Alert, Field, Modal, Spinner, Status } from './ui';
import SubtitleTrackSelect from './SubtitleTrackSelect';
import FileBrowser from './FileBrowser';
import AnimeArchivePicker, { useAnimeArchives } from './AnimeArchive';

const cleanPath = (value: string) => value.trim().replace(/^['"]|['"]$/g, '');
const makeItem = (path = ''): ImportInput => ({ item_id: crypto.randomUUID(), video_path: cleanPath(path), title: cleanPath(path).split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') || '', kind: 'movie', series: '', season: null, episode: null, version: '', subtitle_mode: 'auto', subtitle_path: null, subtitle_offset_ms: 0 });
export default function ImportModal({ onClose, projectId, retryJobId }: { onClose: () => void; projectId?: string; retryJobId?: string }) {
  const { refresh, notify, bootstrap, projects = [], openJobs } = useApp();
  const [items, setItems] = useState<ImportInput[]>(() => {
    const target = projects.find((entry) => entry.id === projectId); const row = makeItem();
    return [target ? { ...row, kind: target.kind, series: target.name } : row];
  });
  const [selected, setSelected] = useState(0);
  const [project, setProject] = useState(projectId || '');
  const [bulkSeason, setBulkSeason] = useState('1');
  const [firstEpisode, setFirstEpisode] = useState('1');
  const [paths, setPaths] = useState('');
  const [offsetTexts, setOffsetTexts] = useState<Record<string, string>>({});
  const [browser, setBrowser] = useState<'video' | 'subtitle' | null>(null);
  const [tracks, setTracks] = useState<Record<string, SubtitleTracks>>({});
  const [probeErrors, setProbeErrors] = useState<Record<string, string>>({});
  const [probing, setProbing] = useState('');
  const probeVersions = useRef<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const request = useRef({ key: '', id: '' });
  const item = items[selected] || items[0];
  const targetProject = projects.find((entry) => entry.id === project);
  const archives = useAnimeArchives(items, targetProject);
  useEffect(() => {
    if (!retryJobId && targetProject) setItems((current) => current.map((row) => !row.video_path && !row.title ? { ...row, kind: targetProject.kind, series: targetProject.name } : row));
  }, [targetProject?.id, targetProject?.kind, targetProject?.name, retryJobId]);
  const status = job?.items?.find((value) => value.item_id === item?.item_id);
  const running = !!job && ['queued', 'running', 'pending', 'pausing', 'cancelling'].includes(job.status);
  const locked = running || !!status?.asset_id;
  useEffect(() => {
    if (!retryJobId) return;
    let current = true; setBusy(true);
    api<{ job: Job; items: ImportInput[]; project_id?: string }>(`/imports/${encodeURIComponent(retryJobId)}`).then((value) => { if (current) { setItems(value.items); setSelected(Math.max(0, value.items.findIndex((input) => value.job.items?.find((result) => result.item_id === input.item_id)?.status === 'failed'))); setProject(value.project_id || ''); setJob(value.job); } }).catch((error) => { if (current) setError(message(error)); }).finally(() => { if (current) setBusy(false); });
    return () => { current = false; };
  }, [retryJobId]);
  useEffect(() => {
    if (!running || !job) return;
    let active = true;
    const timer = window.setInterval(() => { void api<Job>(`/jobs/${encodeURIComponent(job.id)}`).then((value) => { if (!active) return; setJob(value); if (!['queued', 'running', 'pending', 'pausing', 'cancelling'].includes(value.status)) void refresh(); }).catch((error) => { if (active) setError(message(error)); }); }, 1500);
    return () => { active = false; window.clearInterval(timer); };
  }, [job?.id, running, refresh]);
  useEffect(() => () => { for (const id of Object.keys(probeVersions.current)) probeVersions.current[id] += 1; }, []);
  const update = (values: Partial<ImportInput>) => { setItems((current) => current.map((row) => row.item_id === item.item_id ? { ...row, ...values, ...((values.kind !== undefined && values.kind !== row.kind) || (!job && values.title !== undefined && values.title !== row.title && targetProject?.kind !== 'animation') ? { anime_profile: undefined } : {}) } : row)); setError(''); };
  const changePath = (value: string) => {
    const path = cleanPath(value); probeVersions.current[item.item_id] = (probeVersions.current[item.item_id] || 0) + 1;
    setTracks((current) => { const next = { ...current }; delete next[item.item_id]; return next; });
    setProbeErrors((current) => ({ ...current, [item.item_id]: '' })); setProbing('');
    const oldStem = cleanPath(item.video_path).split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') || '';
    update({ video_path: value, ...(item.title === oldStem || !item.title ? { title: path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') || '' } : {}), subtitle_stream_index: null, ...(item.subtitle_mode === 'container' ? { subtitle_mode: 'auto' } : {}) });
  };
  const addPaths = (values: string[]) => {
    setItems((current) => {
      const present = current.filter((row) => cleanPath(row.video_path)); const seen = new Set(present.map((row) => cleanPath(row.video_path)));
      const added = values.map(cleanPath).filter((value) => value && !seen.has(value) && !!seen.add(value)).map((value) => { const row = makeItem(value); const target = projects.find((entry) => entry.id === project); return target ? { ...row, kind: target.kind, series: target.name } : row; });
      return [...present, ...added].length ? [...present, ...added].sort((left, right) => (left.video_path.split(/[\\/]/).pop() || '').localeCompare(right.video_path.split(/[\\/]/).pop() || '', 'zh-CN', { numeric: true, sensitivity: 'base' }) || left.video_path.localeCompare(right.video_path, 'zh-CN', { numeric: true })) : [makeItem()];
    }); setSelected(0); setPaths(''); setBrowser(null);
  };
  const probe = async () => {
    const id = item.item_id; const version = (probeVersions.current[id] || 0) + 1; probeVersions.current[id] = version; setProbing(id); setProbeErrors((current) => ({ ...current, [id]: '' }));
    try { const result = await post<SubtitleTracks>('/media/subtitle-tracks', { video_path: cleanPath(item.video_path) }); if (probeVersions.current[id] === version) setTracks((current) => ({ ...current, [id]: result })); }
    catch (error) { if (probeVersions.current[id] === version) setProbeErrors((current) => ({ ...current, [id]: message(error) })); }
    finally { if (probeVersions.current[id] === version) setProbing(''); }
  };
  const changeMode = (mode: SubtitleMode) => update({ subtitle_mode: mode, subtitle_stream_index: null, subtitle_path: mode === 'external' ? item.subtitle_path : null, subtitle_offset_ms: mode === 'embedded' ? 0 : item.subtitle_offset_ms });
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const selectedItems = job ? items.filter((row) => job.items?.some((result) => result.item_id === row.item_id && result.status === 'failed')) : items;
      if (!selectedItems.length) throw new Error('没有待重试的失败文件。');
      const normalized = selectedItems.map((row) => ({ ...row, anime_profile: row.kind === 'animation' ? archives.selectionFor(row) : undefined, video_path: cleanPath(row.video_path), title: row.title.trim(), subtitle_path: row.subtitle_mode === 'external' ? cleanPath(row.subtitle_path || '') : null, subtitle_stream_index: row.subtitle_mode === 'container' ? row.subtitle_stream_index : null, subtitle_offset_ms: row.subtitle_mode === 'embedded' ? 0 : Math.round(Number(offsetTexts[row.item_id] ?? row.subtitle_offset_ms / 1000) * 1000) }));
      for (const row of normalized) {
        if (!Number.isFinite(row.subtitle_offset_ms)) throw new Error(`「${row.title}」的字幕时间偏移无效。`);
        if (!row.video_path) throw new Error('请为每一项选择视频文件。');
        if (row.kind === 'animation' && !row.anime_profile) throw new Error(`请先为「${row.title || row.video_path}」选择 Bangumi 档案或从零建档。`);
        if (row.subtitle_mode === 'external' && !row.subtitle_path) throw new Error(`「${row.title || row.video_path}」缺少外挂字幕路径。`);
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
  const unsupported = usingTracks && !!result?.tracks.length && !result.tracks.some((track) => track.supported);
  const failed = job?.items?.filter((entry) => entry.status === 'failed').length || 0;
  const unresolved = items.filter((row) => (!job || job.items?.some((entry) => entry.item_id === row.item_id && entry.status === 'failed')) && row.kind === 'animation' && !archives.selectionFor(row)).length;
  return <Modal title={job ? '批量导入进度' : '导入本机视频'} subtitle="一次选择多集，逐集设置字幕，原视频保留在原位置。" onClose={onClose} wide><form onSubmit={submit}>
    <div className="modal-body">
      {browser ? <FileBrowser kind={browser} onClose={() => setBrowser(null)} onSelect={(values) => { if (browser === 'video') addPaths(values); else { update({ subtitle_path: values[0] }); setBrowser(null); } }} /> : <>
        <div className="file-entry"><FileVideo size={27} /><div><strong>连接本地视频文件</strong><p>不会复制或上传整部视频。导入完成后可单独选择分析。</p></div></div>
        {!job && <><div className="button-row"><button type="button" className="button primary" onClick={() => setBrowser('video')}><Plus size={16} />选择多个视频</button></div><details className="advanced"><summary>粘贴多个绝对路径</summary><Field label="每行一个视频路径"><textarea value={paths} onChange={(event) => setPaths(event.target.value)} rows={3} /></Field><button type="button" className="button secondary small" disabled={!paths.trim()} onClick={() => addPaths(paths.split(/\r?\n/))}>添加路径列表</button></details></>}
        <Field label="归属项目"><select value={project} disabled={!!job} onChange={(event) => { setProject(event.target.value); const target = projects.find((entry) => entry.id === event.target.value); setItems((current) => current.map((row) => ({ ...row, anime_profile: undefined, ...(target ? { kind: target.kind, series: target.name } : { series: '' }) }))); }}><option value="">未归类</option>{projects.map((entry) => <option value={entry.id} key={entry.id}>{entry.name}</option>)}</select></Field>
        {!job && items.length > 1 && <details className="advanced"><summary>批量设置季号与连续集号</summary><p className="helper">按左侧文件顺序编号；应用后仍可逐集修改。</p><div className="form-grid"><Field label="批量季号"><input type="number" min="0" step="1" value={bulkSeason} onChange={(event) => setBulkSeason(event.target.value)} /></Field><Field label="起始集号"><input type="number" min="0" step="1" value={firstEpisode} onChange={(event) => setFirstEpisode(event.target.value)} /></Field></div><button type="button" className="button secondary small" disabled={!bulkSeason || !firstEpisode || !Number.isInteger(Number(bulkSeason)) || !Number.isInteger(Number(firstEpisode)) || Number(bulkSeason) < 0 || Number(firstEpisode) < 0} onClick={() => setItems((current) => current.map((row, index) => ({ ...row, season: Number(bulkSeason), episode: Number(firstEpisode) + index })))}>应用连续集号</button></details>}
        {job && <div className="import-progress" aria-live="polite"><Status status={job.status} /><span>{job.items?.filter((entry) => !!entry.asset_id).length || 0} / {items.length} 个视频已导入{failed ? ` · ${failed} 个失败` : ''}</span>{running && <Spinner label="正在本机校验与提取字幕" />}<a href="#/jobs" onClick={onClose}>在任务队列查看</a></div>}
        <div className="import-layout"><div className="import-items" aria-label="待导入视频">{items.map((row, index) => { const state = job?.items?.find((entry) => entry.item_id === row.item_id); return <button type="button" className={`import-row ${selected === index ? 'active' : ''}`} key={row.item_id} onClick={() => setSelected(index)}><strong>{row.title || `视频 ${index + 1}`}</strong><span>{row.video_path.split(/[\\/]/).pop() || '请选择文件'}</span>{state && <Status status={state.status} />}</button>; })}</div>{item && <div className="import-editor"><div className="section-row"><h3>第 {selected + 1} 项 · 文件设置</h3>{!job && items.length > 1 && <button type="button" className="text-button" aria-label="移除当前视频" onClick={() => { setItems((current) => current.filter((row) => row.item_id !== item.item_id)); setSelected(0); }}><Trash2 size={14} />移除</button>}</div>
          {status?.error && <Alert tone="error">{status.error}</Alert>}{status?.asset_id && status.status === 'failed' && <Alert>视频资料已保存，可重试完成剩余处理。原文件配置保持不变。</Alert>}
          <fieldset disabled={busy || locked}>
            <Field label="视频绝对路径" hint="也可在 Finder 中复制路径后粘贴。"><input autoComplete="off" value={item.video_path} onChange={(event) => changePath(event.target.value)} placeholder="/Users/你的名字/Movies/第一集.mkv" required /></Field>
            <div className="form-grid"><Field label="作品名称"><input value={item.title} onChange={(event) => update({ title: event.target.value })} required /></Field><Field label="作品类型"><select value={item.kind} disabled={!!job && !!item.anime_profile} onChange={(event) => update({ kind: event.target.value as AssetKind })}><option value="movie">电影</option><option value="animation">动画</option><option value="series">剧集</option></select></Field></div>
            {item.kind === 'animation' && <AnimeArchivePicker item={item} archives={archives} readOnly={!!job && !!item.anime_profile} />}
            <div className="form-grid"><Field label="季（可选）"><input type="number" min="0" value={item.season ?? ''} onChange={(event) => update({ season: event.target.value ? Number(event.target.value) : null })} /></Field><Field label="集（可选）"><input type="number" min="0" value={item.episode ?? ''} onChange={(event) => update({ episode: event.target.value ? Number(event.target.value) : null })} /></Field></div>
            <Field label="视频版本（可选）"><input value={item.version} onChange={(event) => update({ version: event.target.value })} placeholder="例如：1080p、导演剪辑版" /></Field>
            <div className="field-label section-label">这集的字幕来源</div><div className="choice-grid subtitle-source-choices"><button type="button" className={`choice ${usingTracks ? 'selected' : ''}`} aria-pressed={usingTracks} onClick={() => changeMode('auto')}><Subtitles size={19} /><strong>自动提取字幕轨</strong><span>在本机提取文本，无需模型</span></button><button type="button" className={`choice ${item.subtitle_mode === 'external' ? 'selected' : ''}`} aria-pressed={item.subtitle_mode === 'external'} onClick={() => changeMode('external')}><Subtitles size={19} /><strong>外挂字幕文件</strong><span>SRT / VTT / ASS</span></button><button type="button" className={`choice ${item.subtitle_mode === 'embedded' ? 'selected' : ''}`} aria-pressed={item.subtitle_mode === 'embedded'} onClick={() => changeMode('embedded')}><ScanLine size={19} /><strong>画面字幕 OCR</strong><span>仅用于烧录在画面中的文字</span></button></div>
            {usingTracks && <div className="subtitle-track-panel"><div className="section-row"><p className="helper">每集独立选择字幕轨，轨道编号不跨集套用。</p><button type="button" className="button secondary small" disabled={!cleanPath(item.video_path) || !!probing} onClick={() => void probe()}>{probing === item.item_id ? <Spinner label="正在探测字幕轨" /> : '探测字幕轨'}</button></div>{result ? <SubtitleTrackSelect result={result} value={item.subtitle_stream_index ?? null} allowAuto onChange={(index) => update({ subtitle_stream_index: index, subtitle_mode: index === null ? 'auto' : 'container' })} /> : <p className="helper">无字幕轨时先加入资料库，之后可配置画面 OCR。</p>}{probeErrors[item.item_id] && <Alert tone="error">{probeErrors[item.item_id]}</Alert>}</div>}
            {item.subtitle_mode === 'external' && <><Field label="字幕文件绝对路径 · 必填"><input required value={item.subtitle_path || ''} onChange={(event) => update({ subtitle_path: event.target.value })} /></Field><button className="text-button" type="button" onClick={() => setBrowser('subtitle')}>选择本机字幕文件</button></>}
            {item.subtitle_mode !== 'embedded' && <Field label="时间偏移（秒）" hint="正值表示延后，仅作用于当前文件"><input type="number" step="0.001" value={offsetTexts[item.item_id] ?? item.subtitle_offset_ms / 1000} onChange={(event) => { setOffsetTexts((current) => ({ ...current, [item.item_id]: event.target.value })); if (Number.isFinite(Number(event.target.value))) update({ subtitle_offset_ms: Math.round(Number(event.target.value) * 1000) }); }} required /></Field>}
            {item.subtitle_mode === 'embedded' && !bootstrap.settings.bindings.subtitle && <Alert>请先在<a href="#/settings" onClick={onClose}>模型与设置</a>中绑定字幕视觉模型，再导入画面烧录字幕视频。</Alert>}
          </fieldset>
        </div>}</div>
      </>}{error && <Alert tone="error">{error}</Alert>}
    </div>
    {!browser && <footer className="modal-footer"><span>{unresolved ? `${unresolved} 个动画待选择初始档案` : job ? '成功项保留；修正失败文件后可单独重试' : `${items.length} 个视频 · 不自动调用分析模型`}</span>{job && !running && job.items?.some((entry) => entry.asset_id) && openJobs && <button type="button" className="button secondary" disabled={busy} onClick={() => void analyze()}>分析这批</button>}{(!job || failed > 0) && <button type="submit" className="button primary" disabled={busy || running || !!probing || !!unsupported || unresolved > 0 || (item?.subtitle_mode === 'embedded' && !bootstrap.settings.bindings.subtitle)}>{busy ? <Spinner label="正在提交" /> : <>{job ? '重试失败文件' : '加入资料库'}<ArrowRight size={16} /></>}</button>}</footer>}
  </form></Modal>;
}
