import { useState, type FormEvent } from 'react';
import { FolderPlus, Plus, Pencil, Trash2 } from 'lucide-react';
import { api, post, message } from '../api';
import { useApp } from '../context';
import type { AssetKind, Folder, Project } from '../types';
import { Alert, Field, Modal, Spinner } from './ui';
import DeleteModal from './DeleteModal';

export function folderLabel(id: string, folders: Folder[]): string {
  const names: string[] = []; const seen = new Set<string>(); let current: string | null = id;
  while (current && !seen.has(current)) { seen.add(current); const folder = folders.find((entry) => entry.id === current); if (!folder) break; names.unshift(folder.name); current = folder.parent_id; }
  return names.join(' / ');
}
export function inFolder(id: string | null | undefined, parent: string, folders: Folder[]): boolean {
  const seen = new Set<string>(); let current = id;
  while (current && !seen.has(current)) { if (current === parent) return true; seen.add(current); current = folders.find((entry) => entry.id === current)?.parent_id; }
  return false;
}
export default function Organization({ projectId, folderId, onProject, onFolder }: { projectId: string; folderId: string; onProject: (id: string) => void; onFolder: (id: string) => void }) {
  const { projects = [], folders = [] } = useApp();
  const [editor, setEditor] = useState<{ type: 'project' | 'folder'; entity?: Project | Folder } | null>(null);
  const [deleting, setDeleting] = useState<{ type: 'project' | 'folder'; id: string; name: string } | null>(null);
  const selected = projects.find((entry) => entry.id === projectId); const folder = folders.find((entry) => entry.id === folderId);
  return <section className="organization" aria-label="项目与文件夹管理">
    <div className="section-row"><div><h2>剧集与项目</h2><p className="helper">一部动画一个项目；用文件夹和标签整理多部作品。</p></div><div className="button-row"><button type="button" className="button secondary small" onClick={() => setEditor({ type: 'folder' })}><FolderPlus size={15} />新建文件夹</button><button type="button" className="button secondary small" onClick={() => setEditor({ type: 'project' })}><Plus size={15} />新建项目</button></div></div>
    <div className="organization-controls"><Field label="文件夹"><select value={folderId} onChange={(event) => { onFolder(event.target.value); onProject('all'); }}><option value="">全部文件夹</option>{folders.map((entry) => <option key={entry.id} value={entry.id}>{folderLabel(entry.id, folders)}</option>)}</select></Field><Field label="项目"><select value={projectId} onChange={(event) => onProject(event.target.value)}><option value="all">全部项目</option><option value="unassigned">未归类视频</option>{projects.filter((entry) => !folderId || inFolder(entry.folder_id, folderId, folders)).map((entry) => <option key={entry.id} value={entry.id}>{entry.name} · {entry.asset_count} 个视频</option>)}</select></Field></div>
    {(selected || folder) && <div className="section-row"><div className="project-tags"><strong>{selected?.name || (folder && folderLabel(folder.id, folders))}</strong>{selected?.tags.map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div><div className="button-row">{selected && <><button className="text-button" onClick={() => setEditor({ type: 'project', entity: selected })}><Pencil size={14} />编辑项目与标签</button><button className="text-button" onClick={() => { setDeleting({ type: 'project', id: selected.id, name: selected.name }); }}><Trash2 size={14} />删除项目</button></>}{folder && <><button className="text-button" onClick={() => setEditor({ type: 'folder', entity: folder })}>编辑文件夹</button><button className="text-button" onClick={() => { setDeleting({ type: 'folder', id: folder.id, name: folder.name }); }}>删除文件夹</button></>}</div></div>}
    {editor && <OrganizationEditor type={editor.type} entity={editor.entity} folderId={folderId} onClose={() => setEditor(null)} onSaved={(id) => { if (editor.type === 'project') onProject(id); else onFolder(id); setEditor(null); }} />}
    {deleting && <DeleteModal target={{ kind: deleting.type, id: deleting.id, name: deleting.name }} onClose={() => setDeleting(null)} onDeleted={() => { if (deleting.type === 'project') onProject('all'); else { onFolder(''); onProject('all'); } setDeleting(null); }} />}
  </section>;
}
function OrganizationEditor({ type, entity, folderId, onClose, onSaved }: { type: 'project' | 'folder'; entity?: Project | Folder; folderId: string; onClose: () => void; onSaved: (id: string) => void }) {
  const { folders = [], refresh, notify } = useApp(); const project = type === 'project' ? entity as Project | undefined : undefined;
  const [name, setName] = useState(entity?.name || ''); const [kind, setKind] = useState<AssetKind>(project?.kind || 'animation'); const [parent, setParent] = useState(entity ? (type === 'project' ? (entity as Project).folder_id : (entity as Folder).parent_id) || '' : folderId);
  const [tags, setTags] = useState(project?.tags.join('，') || ''); const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const submit = async (event: FormEvent) => { event.preventDefault(); setBusy(true); setError(''); try { const payload = type === 'project' ? { name: name.trim(), kind, folder_id: parent || null, tags: [...new Set(tags.split(/[,，\n]/).map((tag) => tag.trim()).filter(Boolean))] } : { name: name.trim(), parent_id: parent || null }; const path = `/${type === 'project' ? 'projects' : 'folders'}`; const value = entity ? await api<Project | Folder>(`${path}/${encodeURIComponent(entity.id)}`, { method: 'PATCH', body: JSON.stringify(payload) }) : await post<Project | Folder>(path, payload); await refresh(); notify('分类已保存'); onSaved(value.id); } catch (error) { setError(message(error)); } finally { setBusy(false); } };
  return <Modal title={`${entity ? '编辑' : '新建'}${type === 'project' ? '项目' : '文件夹'}`} onClose={onClose}><form onSubmit={submit}><div className="modal-body"><Field label={type === 'project' ? '项目名称' : '文件夹名称'}><input value={name} onChange={(event) => setName(event.target.value)} maxLength={200} required /></Field><Field label={type === 'project' ? '所在文件夹' : '上级文件夹'}><select value={parent} onChange={(event) => setParent(event.target.value)}><option value="">资料库根目录</option>{folders.filter((entry) => type !== 'folder' || !entity || !inFolder(entry.id, entity.id, folders)).map((entry) => <option key={entry.id} value={entry.id}>{folderLabel(entry.id, folders)}</option>)}</select></Field>{type === 'project' && <><Field label="项目类型"><select value={kind} onChange={(event) => setKind(event.target.value as AssetKind)}><option value="animation">动画</option><option value="series">剧集</option><option value="movie">电影</option></select></Field><Field label="项目标签" hint="用中文或英文逗号分隔，可添加多个标签"><input value={tags} onChange={(event) => setTags(event.target.value)} placeholder="待看，影评素材，科幻" /></Field></>}{error && <Alert tone="error">{error}</Alert>}</div><footer className="modal-footer"><button type="submit" className="button primary" disabled={busy || !name.trim()}>{busy ? <Spinner label="保存中" /> : '保存分类'}</button></footer></form></Modal>;
}
