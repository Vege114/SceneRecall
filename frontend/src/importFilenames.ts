import type { ImportInput } from './types';

const filename = (path: string) => path.trim().replace(/^['"]|['"]$/g, '').split(/[\\/]/).pop() || '';
const qualityNumbers = new Set([480, 576, 720, 1080, 1440, 2160, 4320]);
const chineseNumber = (value: string): number | null => {
  if (/^\d+$/.test(value)) return Number(value);
  const digits: Record<string, number> = { 零: 0, 〇: 0, 一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9 };
  let total = 0; let current = 0;
  for (const char of value) {
    if (char in digits) current = digits[char];
    else if (char === '十' || char === '百' || char === '千') { total += (current || 1) * ({ 十: 10, 百: 100, 千: 1000 }[char]); current = 0; }
    else return null;
  }
  return total + current;
};

/** Read episode markers before release metadata, which often contains larger numbers. */
export function inferAnimeFilename(path: string): { title: string; season: number | null; episode: number | null } {
  const stem = filename(path).replace(/\.(?:mkv|mp4|m4v|avi|mov|webm|ts|m2ts)$/i, '').normalize('NFKC');
  let season: number | null = null; let episode: number | null = null; let marker = '';
  const seasonal = stem.match(/(?:^|[\s._\-[\](])S(\d{1,3})[\s._-]*E(\d{1,4})(?:v\d+)?(?=$|[^\d])/i);
  const explicit = stem.match(/(?:^|[\s._\-[\](])(?:EP?|Episode)[\s._-]*(\d{1,4})(?:v\d+)?(?=$|[^\d])/i);
  const chinese = stem.match(/第\s*([\d零〇一二两三四五六七八九十百千]+)\s*[集话話]/);
  if (seasonal) { season = Number(seasonal[1]); episode = Number(seasonal[2]); marker = seasonal[0].replace(/^[\s._\-[\](]+/, ''); }
  else if (explicit) { episode = Number(explicit[1]); marker = explicit[0].replace(/^[\s._\-[\](]+/, ''); }
  else if (chinese) { episode = chineseNumber(chinese[1]); marker = chinese[0]; }
  else {
    // Bare bracketed and trailing numbers are common in fansub releases. Avoid
    // resolution values and years; checksums/codecs do not match these patterns.
    const candidates = [...stem.matchAll(/[[(]\s*(\d{1,4})(?:v\d+)?\s*[\])]/gi)];
    const bracket = candidates.find((match) => !qualityNumbers.has(Number(match[1])) && !(Number(match[1]) >= 1900 && Number(match[1]) <= 2099));
    const metadataFree = stem.replace(/\[[^\]]*\]|\([^)]*\)/g, ' ').trim();
    const trailing = metadataFree.match(/(?:^|[\s._-])(\d{1,4})(?:v\d+)?\s*$/i);
    const bare = bracket || (trailing && !qualityNumbers.has(Number(trailing[1])) && !(Number(trailing[1]) >= 1900 && Number(trailing[1]) <= 2099) ? trailing : null);
    if (bare) { episode = Number(bare[1]); marker = bare[0]; }
  }
  const chineseSeason = stem.match(/第\s*([\d零〇一二两三四五六七八九十百千]+)\s*季/);
  if (season === null && chineseSeason) season = chineseNumber(chineseSeason[1]);
  const title = stem.replace(marker || /$^/, ' ').replace(/\[[^\]]*\]/g, ' ')
    .replace(/\((?:[^)]*(?:\d{3,4}p|x26[45]|HEVC|AVC|FLAC|AAC|Blu-?ray|BDRip)[^)]*)\)/gi, ' ')
    .replace(/(?:^|[\s._-])(?:\d{3,4}p|x26[45]|HEVC|AVC|FLAC|AAC|BDRip|Blu-?ray)(?=$|[\s._-])/gi, ' ')
    .replace(/[._]+/g, ' ').replace(/^[\s-]+|[\s-]+$/g, '').replace(/\s+/g, ' ');
  return { title: title || (episode === null ? stem : ''), season, episode };
}

export function compareAnimeImports(left: Pick<ImportInput, 'video_path' | 'season' | 'episode'>, right: Pick<ImportInput, 'video_path' | 'season' | 'episode'>): number {
  const season = (left.season ?? 0) - (right.season ?? 0);
  const episode = (left.episode ?? Infinity) - (right.episode ?? Infinity);
  return season || (Number.isNaN(episode) ? 0 : episode) || filename(left.video_path).localeCompare(filename(right.video_path), 'zh-CN', { numeric: true, sensitivity: 'base' }) || left.video_path.localeCompare(right.video_path, 'zh-CN', { numeric: true });
}
