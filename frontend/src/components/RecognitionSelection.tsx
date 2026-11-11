import { useEffect, useState } from 'react';
import { UserRound } from 'lucide-react';
import type { CharacterConfirmationRole } from '../identityTypes';

interface Props {
  roles: CharacterConfirmationRole[];
  hasDraftCrops: boolean;
  busy: boolean;
  onCancel: () => void;
  onSave: (characterIds: string[]) => void;
  onDraftChange?: (dirty: boolean) => void;
  selectedId?: string;
  onReview?: (characterId: string) => void;
}

const statusLabels = { pending: '待核对', confirmed: '已确认', skipped: '已跳过' };

export default function RecognitionSelection({ roles, hasDraftCrops, busy, onCancel, onSave, onDraftChange, selectedId, onReview }: Props) {
  const savedIds = roles.filter((role) => role.recognition_enabled !== false).map((role) => role.character_id);
  const [selected, setSelected] = useState(() => new Set(savedIds));
  const [query, setQuery] = useState('');
  const [failedImages, setFailedImages] = useState(new Set<string>());
  const term = query.trim().toLocaleLowerCase();
  const visible = roles.filter(({ profile }) => [profile.name, profile.name_cn, profile.name_ja, profile.name_en, ...profile.aliases]
    .some((name) => name?.toLocaleLowerCase().includes(term)));

  const dirty = savedIds.length !== selected.size || savedIds.some((id) => !selected.has(id));
  useEffect(() => { onDraftChange?.(dirty); }, [dirty, onDraftChange]);
  const update = (next: Set<string>) => {
    if (busy) return;
    setSelected(next);
  };
  const toggle = (id: string) => { const next = new Set(selected); if (next.has(id)) next.delete(id); else next.add(id); update(next); };
  const selectVisible = (checked: boolean) => {
    const next = new Set(selected);
    for (const role of visible) { if (checked) next.add(role.character_id); else next.delete(role.character_id); }
    update(next);
  };

  return <section className="recognition-selection" aria-label="选择识别角色">
    <div className="section-row"><div><h3>本作品的识别名单</h3><p className="helper">同一作品的各集共用这份名单。直接在带图角色卡勾选；筛选和批选只影响当前显示的角色，保存后对整个作品生效。</p></div><span className="helper" aria-live="polite">已勾选 {selected.size} / {roles.length}{dirty ? ' · 尚未保存' : ' · 已保存'}</span></div>
    {hasDraftCrops && <p className="helper">移出名单会清除该角色尚未提交的裁剪勾选；已经确认、否定的记录会保留。</p>}
    <div className="recognition-selection-tools">
      <input type="search" aria-label="搜索角色姓名或别名" placeholder="搜索角色姓名或别名" value={query} disabled={busy} onChange={(event) => setQuery(event.target.value)} />
      <button type="button" className="text-button" disabled={busy || !visible.length || visible.every((role) => selected.has(role.character_id))} onClick={() => selectVisible(true)}>全选当前显示</button>
      <button type="button" className="text-button" disabled={busy || !visible.some((role) => selected.has(role.character_id))} onClick={() => selectVisible(false)}>取消当前显示</button><span className="helper">当前显示 {visible.length} 个角色</span>
    </div>
    <div className="recognition-selection-list">{visible.map((role) => {
      const { character_id, profile } = role;
      const image = role.reference_preview_url || profile.external_source?.image_url;
      return <article key={character_id} className={`recognition-role-card${selected.has(character_id) ? ' target-selected' : ''}${selectedId === character_id ? ' reviewing' : ''}`}>
        {image && !failedImages.has(character_id) ? <img src={image} referrerPolicy="no-referrer" alt={`${profile.name}的角色头像`} onError={() => setFailedImages((previous) => new Set(previous).add(character_id))} /> : <div className="recognition-avatar-placeholder" aria-label={`${profile.name}暂无头像`}><UserRound size={24} /></div>}
        <div className="recognition-role-content"><label className="check-line"><input type="checkbox" aria-label={`识别${profile.name}`} checked={selected.has(character_id)} disabled={busy} onChange={() => toggle(character_id)} /><strong>{profile.name}</strong></label>
          {profile.name_ja && profile.name_ja !== profile.name && <small lang="ja">{profile.name_ja}</small>}
          <span className="helper">本地参考 {role.matching_reference_count ?? role.trusted_reference_count} 张 · 候选 {role.available_count ?? role.candidates.length} 张</span>
          {role.reference_detection_status && <span className="helper">检测可用 {role.usable_matching_reference_count ?? 0} 张 · {({ missing: '缺少本地参考', usable: '已有单人参考', not_checked: '参考图尚未检测', failed: '参考图未通过人物检测', model_unavailable: '检测模型不可用' })[role.reference_detection_status]}</span>}
          <span className="helper">{statusLabels[role.status]}{role.coverage?.reason ? ` · ${role.coverage.reason}` : ''}</span>
          {onReview && <button type="button" className="text-button" disabled={busy || role.recognition_enabled === false} aria-pressed={selectedId === character_id} onClick={() => onReview(character_id)}>核对{profile.name}候选</button>}
        </div>
      </article>;
    })}{!visible.length && <p className="helper">没有匹配的角色。</p>}</div>
    <div className="button-row"><button type="button" className="button secondary" disabled={busy || !dirty} onClick={() => { update(new Set(savedIds)); onCancel(); }}>取消修改</button>
      <button type="button" className="button primary" disabled={busy} onClick={() => onSave([...selected])}>保存名单并生成候选</button></div>
    <p className="helper">先保存识别名单，再缓存所选角色参考图，并在作品内跨集有界补充候选。候选需要人工核对；清空名单保存后跳过生成。</p>
  </section>;
}
