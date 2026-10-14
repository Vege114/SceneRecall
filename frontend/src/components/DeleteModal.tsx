import { useEffect, useRef, useState } from 'react';
import { Trash2 } from 'lucide-react';
import { api, ApiError, message, post } from '../api';
import { useApp } from '../context';
import type { DeletionPreview, DeletionScope, DeletionTarget, TrashEntry } from '../types';
import { Alert, Modal, Spinner } from './ui';

export function DeletionContents({ scope }: { scope: DeletionScope }) {
  return <div className="deletion-scope"><p className="deletion-counts"><strong>{scope.asset_count} 个视频</strong><span>{scope.project_count} 个项目</span><span>{scope.folder_count} 个文件夹</span></p>{(scope.assets.length > 0 || scope.projects.length > 0 || scope.folders.length > 0) && <details><summary>查看完整范围</summary><ul>{scope.folders.map((entry) => <li key={`folder-${entry.id}`}>文件夹：{entry.name}</li>)}{scope.projects.map((entry) => <li key={`project-${entry.id}`}>项目：{entry.name}</li>)}{scope.assets.map((entry) => <li key={`asset-${entry.id}`}>视频：{entry.title}</li>)}</ul></details>}</div>;
}

export function SourceDeletionResults({ entry }: { entry: TrashEntry }) {
  if (!entry.delete_source) return <p className="helper">原视频文件保留在原位置。</p>;
  const results = entry.source_results || []; const failed = results.filter((item) => item.status !== 'deleted');
  return <><Alert tone={failed.length ? 'error' : 'warning'}>原视频永久删除：已删除 {results.filter((item) => item.status === 'deleted').length} 个文件{failed.length > 0 && `，${failed.length} 个未完成删除`}。回收站只能恢复应用内资料，已删除的源文件无法恢复。</Alert>{failed.length > 0 && <ul className="source-failures">{failed.map((item, index) => <li key={index}><strong>{item.asset_ids.map((id) => entry.assets.find((asset) => asset.id === id)?.title || id).join('、')}</strong>：{item.message || (item.status === 'pending' ? '源文件删除尚未完成' : '源文件未能删除')}</li>)}</ul>}</>;
}

export default function DeleteModal({ target, onClose, onDeleted }: { target: DeletionTarget; onClose: () => void; onDeleted: () => void }) {
  const { refresh, notify } = useApp();
  const [deleteContents, setDeleteContents] = useState(false);
  const [deleteSource, setDeleteSource] = useState(false); const [sourceConfirmed, setSourceConfirmed] = useState(false);
  const [result, setResult] = useState<TrashEntry | null>(null);
  const [preview, setPreview] = useState<DeletionPreview | null>(null);
  const [loading, setLoading] = useState(true); const [busy, setBusy] = useState(false);
  const [error, setError] = useState(''); const [revision, setRevision] = useState(0);
  const request = useRef(0);
  const targetKey = JSON.stringify(target);
  useEffect(() => {
    const current = ++request.current; setLoading(true); setPreview(null); setSourceConfirmed(false);
    const payload = target.kind === 'assets' ? { kind: target.kind, asset_ids: target.asset_ids, delete_source: deleteSource } : { kind: target.kind, id: target.id, delete_contents: deleteContents, delete_source: deleteSource };
    void post<DeletionPreview>('/deletions/preview', payload).then((value) => { if (current === request.current) setPreview(value); }).catch((error) => { if (current === request.current) setError(message(error)); }).finally(() => { if (current === request.current) setLoading(false); });
    return () => { request.current++; };
  }, [targetKey, deleteContents, deleteSource, revision]);
  const sourceBlocked = !!(deleteSource && preview?.source_files.some((file) => file.status !== 'ready'));
  const remove = async () => {
    if (!preview || loading || preview.blocked_jobs.length || busy || result || sourceBlocked || (deleteSource && !sourceConfirmed)) return;
    setBusy(true); setError('');
    try {
      const query = `delete_source=${deleteSource}&expected_token=${encodeURIComponent(preview.token)}`;
      const response = target.kind === 'assets' ? target.asset_ids.length === 1 ? await api<{ trash: TrashEntry }>(`/assets/${encodeURIComponent(target.asset_ids[0])}?${query}`, { method: 'DELETE' }) : await post<{ trash: TrashEntry }>('/assets/delete', { asset_ids: target.asset_ids, delete_source: deleteSource, expected_token: preview.token }) : await api<{ trash: TrashEntry }>(`/${target.kind === 'project' ? 'projects' : 'folders'}/${encodeURIComponent(target.id)}?delete_contents=${deleteContents}&${query}`, { method: 'DELETE' });
      setResult(response.trash);
      const incomplete = response.trash.delete_source && response.trash.source_results?.some((item) => item.status !== 'deleted');
      await refresh();
      if (incomplete) notify('应用内资料已移入回收站，部分源文件未删除，请查看详情', 'error');
      else { notify(deleteSource ? '应用资料已移入回收站，源视频已永久删除' : '已移入回收站，可在回收站恢复'); onDeleted(); }
    } catch (error) {
      setError(message(error));
      if (error instanceof ApiError && error.status === 409) { setPreview(null); setRevision((value) => value + 1); }
    } finally { setBusy(false); }
  };
  const close = () => { if (!busy) { if (result) onDeleted(); else onClose(); } };
  if (result) return <Modal title="应用资料已移入回收站" onClose={close} wide><div className="modal-body"><DeletionContents scope={result} /><SourceDeletionResults entry={result} />{error && <Alert tone="error">资料已删除，刷新未完成：{error}</Alert>}<a className="text-button" href="#/trash" onClick={close}>查看回收站</a></div><footer className="modal-footer"><button className="button secondary" onClick={close}>完成</button></footer></Modal>;
  return <Modal title={target.kind === 'assets' ? `删除 ${target.asset_ids.length} 个视频` : `删除${target.kind === 'project' ? '项目' : '文件夹'}「${target.name}」`} subtitle="先核对范围与源文件处理方式，再确认删除。" onClose={close} wide>
    <div className="modal-body deletion-body">
      {target.kind !== 'assets' && <fieldset disabled={busy}><label className="check-line"><input type="checkbox" checked={deleteContents} onChange={(event) => { setDeleteContents(event.target.checked); setDeleteSource(false); setError(''); }} />同时删除内部视频{target.kind === 'folder' && '及下级项目、文件夹'}</label><p className="helper">{target.kind === 'project' ? (deleteContents ? '项目及其全部视频一起移入回收站。' : '仅删除项目分类，视频保留在资料库并变为未归类。') : (deleteContents ? '包含这个文件夹的所有下级文件夹、项目及视频。' : '仅删除这个文件夹，内部项目及下级文件夹移至上一级；视频资料保留。')}</p></fieldset>}
      {(target.kind === 'assets' || deleteContents) && <fieldset className="delete-source-choices" disabled={busy}><legend className="section-label">如何处理视频</legend><label className={`delete-choice ${!deleteSource ? 'selected' : ''}`}><input type="radio" name="source-deletion" checked={!deleteSource} onChange={() => { setDeleteSource(false); setError(''); }} /><span><strong>仅删除应用内影片资料</strong><small>资料移入回收站，原视频文件保留。</small></span></label><label className={`delete-choice ${deleteSource ? 'selected' : ''}`}><input type="radio" name="source-deletion" checked={deleteSource} disabled={!loading && preview?.asset_count === 0} onChange={() => { setDeleteSource(true); setError(''); }} /><span><strong>同时永久删除源视频文件</strong><small>资料仍可恢复，源视频文件无法通过回收站恢复。</small></span></label></fieldset>}
      {loading ? <Spinner label="正在核对删除范围" /> : preview ? <><DeletionContents scope={preview} />{preview.asset_count > 0 && <Alert>所列视频及其分析记录、字幕、人工注释和收藏将从资料库与检索中移除，恢复视频时一并恢复。历史任务记录保留。</Alert>}{deleteSource ? <section className="source-deletion-preview" aria-label="永久删除的源视频文件"><h3>将永久删除 {preview.source_file_count} 个源视频文件</h3><p className="helper">仅处理下方列出的源视频路径，不删除外挂字幕或整个目录。共享、离线、变化或不安全的文件会阻止这次删除。</p><ul>{preview.source_files.map((file) => <li key={file.path}><code>{file.path}</code><span className={file.status === 'ready' ? 'helper' : 'warning-text'}>{({ ready: '可删除', missing: '文件不可用', changed: '文件已变化', shared: '被其他视频共用', unsafe: '无法安全删除' })[file.status]}{file.message && `：${file.message}`}</span></li>)}</ul>{sourceBlocked ? <Alert tone="error">存在无法安全删除的源文件，请处理以上问题或改为仅删除应用内资料。</Alert> : <label className="check-line source-confirm"><input type="checkbox" checked={sourceConfirmed} disabled={busy} onChange={(event) => setSourceConfirmed(event.target.checked)} />我已核对以上 {preview.source_file_count} 个文件，了解源视频将被永久删除且无法恢复。</label>}</section> : <p className="helper">原视频文件保留在原位置。</p>}{preview.blocked_jobs.length > 0 && <Alert tone="error"><p>还有 {preview.blocked_jobs.length} 个相关任务未停止，暂时不能删除。请先到<a href="#/jobs" onClick={onClose}>任务队列</a>暂停或取消相关任务，并等待当前步骤结束，再重新核对。</p><ul>{preview.blocked_jobs.map((job) => <li key={job.id}>{job.message || job.id}</li>)}</ul></Alert>}</> : null}
      {error && <Alert tone="error">{error}{preview && ' 删除范围已重新核对，请确认后再提交。'}</Alert>}
      {(!preview || preview.blocked_jobs.length > 0 || sourceBlocked || error) && !loading && <button type="button" className="button secondary small" disabled={busy} onClick={() => { setError(''); setRevision((value) => value + 1); }}>重新核对范围</button>}
    </div>
    <footer className="modal-footer"><button type="button" className="button secondary" disabled={busy} onClick={onClose}>取消</button><button type="button" className="button danger" disabled={loading || busy || !preview || preview.blocked_jobs.length > 0 || sourceBlocked || (deleteSource && !sourceConfirmed)} onClick={() => void remove()}>{busy ? <Spinner label="正在删除" /> : <><Trash2 size={16} />{deleteSource ? '确认删除资料与源文件' : '确认移入回收站'}</>}</button></footer>
  </Modal>;
}
