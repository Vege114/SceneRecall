import { useState } from 'react';
import type { CharacterConfirmationRole } from '../identityTypes';

interface Props {
  roles: CharacterConfirmationRole[];
  hasDraftCrops: boolean;
  busy: boolean;
  onCancel: () => void;
  onSave: (characterIds: string[]) => void;
}

export default function RecognitionSelection({ roles, hasDraftCrops, busy, onCancel, onSave }: Props) {
  const [selected, setSelected] = useState(() => new Set(roles.filter((role) => role.recognition_enabled !== false).map((role) => role.character_id)));
  const [query, setQuery] = useState('');
  const term = query.trim().toLocaleLowerCase();
  const visible = roles.filter(({ profile }) => [profile.name, profile.name_cn, profile.name_ja, profile.name_en, ...profile.aliases]
    .some((name) => name?.toLocaleLowerCase().includes(term)));

  function toggle(id: string) {
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return <section className="recognition-selection" aria-label="选择识别角色">
    <div className="section-row"><h3>选择识别角色</h3><span className="helper" aria-live="polite">已勾选 {selected.size} / {roles.length}</span></div>
    <p className="helper">同一作品的各集共用这份名单。勾选需要标注和识别的角色，清空名单可跳过角色识别。</p>
    {hasDraftCrops && <p className="helper">移出名单的角色会清除尚未提交的裁剪勾选。</p>}
    <div className="recognition-selection-tools">
      <input type="search" aria-label="搜索角色姓名或别名" placeholder="搜索角色姓名或别名" value={query} onChange={(event) => setQuery(event.target.value)} />
      <button type="button" className="text-button" disabled={busy || selected.size === roles.length} onClick={() => setSelected(new Set(roles.map((role) => role.character_id)))}>全选角色</button>
      <button type="button" className="text-button" disabled={busy || !selected.size} onClick={() => setSelected(new Set())}>清空名单</button>
    </div>
    <div className="recognition-selection-list">{visible.map(({ character_id, profile }) => <label key={character_id} className="check-line">
      <input type="checkbox" checked={selected.has(character_id)} disabled={busy} onChange={() => toggle(character_id)} />
      <span>{profile.name}{profile.name_ja && profile.name_ja !== profile.name && <small lang="ja">{profile.name_ja}</small>}</span>
    </label>)}{!visible.length && <p className="helper">没有匹配的角色。</p>}</div>
    <div className="button-row"><button type="button" className="button secondary" disabled={busy} onClick={onCancel}>取消修改</button>
      <button type="button" className="button primary" disabled={busy} onClick={() => onSave([...selected])}>保存识别名单</button></div>
  </section>;
}
