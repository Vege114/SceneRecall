import { useCallback, useEffect, useState } from 'react';
import { ArchiveRestore, RefreshCw, Trash2 } from 'lucide-react';
import { api, friendlyDate, message, post } from '../api';
import { useApp } from '../context';
import type { TrashEntry } from '../types';
import { DeletionContents, SourceDeletionResults } from './DeleteModal';
import { Alert, Empty, Modal, PageHeader, Spinner } from './ui';

export default function Trash() {
  const { refresh, notify } = useApp();
  const [entries, setEntries] = useState<TrashEntry[]>([]); const [loading, setLoading] = useState(true); const [error, setError] = useState('');
  const [restoring, setRestoring] = useState<TrashEntry | null>(null); const [busy, setBusy] = useState(false); const [restoreError, setRestoreError] = useState('');
  const load = useCallback(async () => { setLoading(true); setError(''); try { setEntries(await api<TrashEntry[]>('/trash')); } catch (error) { setError(message(error)); } finally { setLoading(false); } }, []);
  useEffect(() => { void load(); }, [load]);
  const restore = async () => { if (!restoring || busy) return; setBusy(true); setRestoreError(''); let restored = false; try { await post(`/trash/${encodeURIComponent(restoring.id)}/restore`); restored = true; setEntries((current) => current.filter((entry) => entry.id !== restoring.id)); setRestoring(null); notify(restoring.delete_source ? '应用资料已恢复；已删除的源视频需重新定位才能播放和分析' : '已恢复到资料库，分析记录与收藏一并恢复'); await Promise.all([refresh(), load()]); } catch (error) { if (restored) setError(`资料已恢复，刷新未完成：${message(error)}`); else setRestoreError(message(error)); } finally { setBusy(false); } };
  return <><PageHeader eyebrow="RECENTLY REMOVED" title="回收站" action={<button className="button secondary" disabled={loading} onClick={() => void load()}><RefreshCw size={16} className={loading ? 'spin' : ''} />刷新</button>}>按删除批次恢复视频、项目和文件夹，分析记录与收藏随视频一并恢复。</PageHeader>
    {error && <Alert tone="error">{error}</Alert>}
    {loading && entries.length === 0 ? <Spinner label="正在读取回收站" /> : !error && entries.length === 0 ? <Empty icon={<Trash2 size={29} />} title="回收站是空的" action={<a href="#/library" className="button secondary">返回资料库</a>}>删除的视频、项目和文件夹会保留在这里，供你恢复。</Empty> : <div className="trash-list">{entries.map((entry) => <article className="trash-card" key={entry.id}><div className="section-row"><div><h2>{entry.title}</h2><p className="helper">{friendlyDate(entry.deleted_at)} · {entry.kind === 'assets' ? '视频' : entry.kind === 'project' ? '项目' : '文件夹'}删除</p></div><button className="button secondary small" onClick={() => { setRestoreError(''); setRestoring(entry); }}><ArchiveRestore size={16} />恢复</button></div><DeletionContents scope={entry} /><SourceDeletionResults entry={entry} /></article>)}</div>}
    {restoring && <Modal title="恢复这批资料" onClose={() => { if (!busy) setRestoring(null); }}><div className="modal-body"><h3>{restoring.title}</h3><DeletionContents scope={restoring} /><p className="helper">恢复整批资料及原分类关系。如果分类关系已被修改，将提示冲突并保留回收项；不会覆盖后续整理，也不会自动重启历史分析任务。</p>{restoring.delete_source && <Alert>本次只恢复应用内资料。已永久删除的源视频无法恢复，视频会离线，需要重新定位原片才能播放和分析。</Alert>}{restoreError && <Alert tone="error">{restoreError}</Alert>}</div><footer className="modal-footer"><button className="button secondary" disabled={busy} onClick={() => setRestoring(null)}>取消</button><button className="button primary" disabled={busy} onClick={() => void restore()}>{busy ? <Spinner label="恢复中" /> : '确认恢复'}</button></footer></Modal>}
  </>;
}
