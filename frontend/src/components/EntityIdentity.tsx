import { UsersRound } from 'lucide-react';
import type { Entity } from '../types';
import { identitySourceLabel, identityStatusLabel } from '../identityTypes';

export function entityIdentityState(entity: Entity): string | undefined {
  return entity.binding_status || entity.identity_binding?.status || entity.association_status;
}

export function entityDisplayName(entity: Entity): string {
  const state = entityIdentityState(entity);
  const visible = entity.description || entity.appearance || entity.id || '未知人物';
  if (state === 'candidate' || (!state && entity.candidate_character_id && !entity.character_id)) {
    return `候选：${entity.candidate_character_name || entity.character_name || entity.candidate_character_id || visible}`;
  }
  if (state === 'unknown' || state === 'revoked') return `未知人物 · ${visible}`;
  return entity.character_name || entity.name || entity.description || entity.appearance || entity.id || '未知对象';
}

export default function EntityIdentities({ entities, onCharacter, onWorkEntity, correctedName }: {
  entities?: Entity[];
  onCharacter?: (id: string) => void;
  onWorkEntity?: (id: string) => void;
  correctedName?: (entity: Entity) => string | undefined;
}) {
  if (!entities?.length) return null;
  return <div className="entity-tags" aria-label="画面人物与实体身份">{entities.map((entity, index) => {
    const corrected = correctedName?.(entity);
    const state = corrected ? 'human_confirmed' : entityIdentityState(entity);
    const candidate = state === 'candidate' || (!state && !!entity.candidate_character_id && !entity.character_id);
    const characterId = candidate ? entity.candidate_character_id : state === 'unknown' || state === 'revoked' ? null : entity.character_id;
    const label = corrected || entityDisplayName(entity);
    const source = corrected ? 'human' : entity.identity_binding?.source || entity.association_source;
    return <span key={entity.id || entity.entity_id || entity.local_id || index} title={[entity.description, entity.position, ...(entity.candidate_character_aliases || [])].filter(Boolean).join(' · ')}>
      {characterId && onCharacter && <button className="entity-character-link" onClick={() => onCharacter(characterId)} aria-label={`查看${candidate ? '候选' : ''}${candidate ? entity.candidate_character_name || characterId : entity.character_name || entity.name || '此角色'}的角色档案`}><UsersRound size={12} /></button>}
      {label}
      {!entity.character_id && entity.id && onWorkEntity && <button className="entity-character-link" aria-label={`关联作品实体 ${entity.id}`} onClick={() => onWorkEntity(entity.id!)}>关联设定</button>}
      {entity.position && <small>{entity.position}</small>}
      {state && <small>{identityStatusLabel(state)}{source && ` · ${identitySourceLabel(source)}`}</small>}
    </span>;
  })}</div>;
}
