import type { Asset } from './types';

export interface AnimeGroup { id: string; title: string; season: number | null; projectId: string | null; assets: Asset[] }

export function groupAnime(assets: Asset[]): AnimeGroup[] {
  const groups = new Map<string, AnimeGroup>();
  for (const asset of assets) {
    // The server supplies the identity for current libraries. Older clients
    // without that projection keep their existing single-video navigation.
    const id = asset.anime_id || asset.id;
    const group = groups.get(id) || { id, title: asset.anime_title || asset.title, season: asset.season ?? null, projectId: asset.project_id || null, assets: [] };
    group.assets.push(asset); groups.set(id, group);
  }
  for (const group of groups.values()) group.assets.sort((a, b) => (a.episode ?? Infinity) - (b.episode ?? Infinity) || a.title.localeCompare(b.title, 'zh-CN', { numeric: true }) || a.created_at.localeCompare(b.created_at));
  return [...groups.values()];
}

export function animeLabel(group: AnimeGroup): string { return `${group.title}${group.season !== null ? ` · 第 ${group.season} 季` : ''}`; }
export function episodeLabel(asset: Asset): string { return `${asset.episode != null ? `第 ${asset.episode} 集 · ` : ''}${asset.title}${asset.version ? ` · ${asset.version}` : ''}`; }
export function searchLink(filters: { project_id?: string | null; anime_id?: string; asset_id?: string }): string {
  const params = new URLSearchParams(Object.entries(filters).filter((entry): entry is [string, string] => !!entry[1]));
  return `#/search${params.size ? `?${params}` : ''}`;
}
