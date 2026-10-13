import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Folder, FileVideo, Check } from 'lucide-react';
import { post, message } from '../api';
import { Alert, Spinner } from './ui';

interface Listing { directory: string; parent: string | null; entries: { name: string; path: string; type: 'directory' | 'file' }[]; next_offset: number | null }
export default function FileBrowser({ kind = 'video', onSelect, onClose }: { kind?: 'video' | 'subtitle'; onSelect: (paths: string[]) => void; onClose: () => void }) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [path, setPath] = useState('');
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const version = useRef(0);
  const load = async (directory?: string, offset = 0) => {
    const request = ++version.current; setBusy(true); setError('');
    try {
      const value = await post<Listing>('/local-files/list', { ...(directory ? { path: directory } : {}), kind, offset, limit: 200 });
      if (version.current !== request) return;
      setListing((previous) => offset && previous?.directory === value.directory ? { ...value, entries: [...previous.entries, ...value.entries] } : value);
      setPath(value.directory);
    } catch (error) { if (request === version.current) setError(message(error)); }
    finally { if (request === version.current) setBusy(false); }
  };
  useEffect(() => { void load(); return () => { version.current += 1; }; }, []);
  const selectDirectory = async () => {
    if (!listing) return;
    const request = ++version.current; const directory = listing.directory; let entries = [...listing.entries]; let offset = listing.next_offset;
    setBusy(true); setError('');
    try {
      while (offset != null) {
        const page = await post<Listing>('/local-files/list', { path: directory, kind, offset, limit: 200 });
        if (version.current !== request) return;
        entries = [...entries, ...page.entries];
        if (page.next_offset != null && page.next_offset <= offset) throw new Error('目录分页已变化，请重新打开目录后全选。');
        offset = page.next_offset;
      }
      setListing({ ...listing, entries, next_offset: null });
      setSelected((current) => [...new Set([...current, ...entries.filter((entry) => entry.type === 'file').map((entry) => entry.path)])]);
    } catch (error) { if (version.current === request) setError(message(error)); }
    finally { if (version.current === request) setBusy(false); }
  };
  const files = listing?.entries.filter((entry) => entry.type === 'file') || [];
  return <section className="file-browser" aria-label="本机文件选择器">
    <div className="section-row"><h3>{kind === 'video' ? '选择本机视频' : '选择外挂字幕'}</h3><button type="button" className="text-button" onClick={onClose}>返回导入设置</button></div>
    <p className="helper">浏览运行 SceneRecall 的这台电脑。可跨目录勾选；视频保留在原位置。</p>
    <div className="browser-location"><button type="button" className="icon-button" aria-label="上级目录" disabled={!listing?.parent || busy} onClick={() => void load(listing?.parent || undefined)}><ArrowUp size={18} /></button><input aria-label="浏览目录路径" value={path} onChange={(event) => setPath(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void load(path); } }} /><button className="button secondary small" type="button" disabled={busy} onClick={() => void load(path)}>打开目录</button></div>
    {error && <Alert tone="error">{error}</Alert>}
    <div className="section-row"><span className="helper">已选择 {selected.length} 个文件</span>{kind === 'video' && <button type="button" className="text-button" disabled={busy || (!files.length && listing?.next_offset == null)} onClick={() => void selectDirectory()}>全选当前目录视频</button>}</div>
    <div className="browser-entries">{listing?.entries.map((entry) => entry.type === 'directory' ? <button className="browser-entry" type="button" key={`${entry.path}:${entry.name}`} onClick={() => void load(entry.path)} disabled={busy}><Folder size={18} /><span>{entry.name}</span></button> : <label className="browser-entry" key={`${entry.path}:${entry.name}`}><input type={kind === 'subtitle' ? 'radio' : 'checkbox'} name="local-file" checked={selected.includes(entry.path)} onChange={(event) => setSelected((current) => kind === 'subtitle' ? [entry.path] : event.target.checked ? [...current, entry.path] : current.filter((item) => item !== entry.path))} /><FileVideo size={17} /><span>{entry.name}</span></label>)}{!busy && !listing?.entries.length && !error && <p className="helper">此目录没有可选择的{kind === 'video' ? '视频' : '字幕'}文件。</p>}</div>
    {busy && <Spinner label="正在读取目录" />}{listing?.next_offset != null && <button type="button" className="button secondary small" disabled={busy} onClick={() => void load(listing.directory, listing.next_offset!)}>显示更多文件</button>}
    {selected.length > 0 && <details><summary>查看已选择的 {selected.length} 个文件</summary><div className="browser-selected">{selected.map((value) => <div key={value}><span>{value}</span><button type="button" className="text-button" aria-label={`移除 ${value}`} onClick={() => setSelected((current) => current.filter((item) => item !== value))}>移除</button></div>)}</div></details>}
    <div className="form-actions"><button type="button" className="button primary" disabled={!selected.length || busy} onClick={() => onSelect(selected)}><Check size={16} />使用所选文件</button></div>
  </section>;
}
