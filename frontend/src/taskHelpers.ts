import type { Asset } from './types';

export function assetTaskLabel(asset: Asset): string {
  const episode = [asset.season != null ? `第 ${asset.season} 季` : '', asset.episode != null ? `第 ${asset.episode} 集` : ''].filter(Boolean).join(' · ');
  const fallback = asset.episode == null ? asset.source_filename || `视频 ${asset.id.slice(-8)}` : '';
  const detail = [episode, fallback].filter(Boolean).join(' · ');
  return [asset.anime_title || asset.title, detail, asset.version].filter(Boolean).join(' · ');
}

// Limit browser requests independently of the backend's local/model worker limits.
export async function settleConcurrent<T, R>(items: readonly T[], run: (item: T, index: number) => Promise<R>, limit = 4): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const index = next++;
      try { results[index] = { status: 'fulfilled', value: await run(items[index], index) }; }
      catch (reason) { results[index] = { status: 'rejected', reason }; }
    }
  };
  await Promise.all(Array.from({ length: Math.min(items.length, limit) }, worker));
  return results;
}
