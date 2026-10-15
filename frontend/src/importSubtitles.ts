import { inferAnimeFilename } from './importFilenames';
import type { ImportInput, SubtitleDetection } from './types';

const stem = (path: string) => (path.split(/[\\/]/).pop() || '').normalize('NFKC').toLowerCase().replace(/\.[^.]+$/, '');
const subtitleVideoName = (path: string) => path.replace(/\.(?:srt|vtt|ass|ssa)$/i, '').replace(/(?:[._ -](?:sc|tc|chs|cht|zh|zho|chi|zh-cn|zh-tw|en|eng|ja|jpn|简体|繁体|中文|英文))+$/i, '') + '.mkv';
const titleKey = (value: string) => value.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');

function matchScore(row: ImportInput, path: string, detection?: SubtitleDetection): number {
  const known = detection?.external_candidates.find((candidate) => candidate.path === path);
  if (known && known.match !== 'unmatched') return 400 + known.score;
  const video = stem(row.video_path); const subtitle = stem(path);
  if (subtitle === video || stem(subtitleVideoName(path)) === video) return 300;
  const years = (name: string) => [...name.matchAll(/(?:^|[^\d])((?:19|20)\d{2})(?=$|[^\d])/g)].map((match) => match[1]).join(',');
  if (years(video) && years(subtitle) && years(video) !== years(subtitle)) return 0;
  const a = inferAnimeFilename(row.video_path); const b = inferAnimeFilename(subtitleVideoName(path));
  if (a.episode === null || b.episode === null || a.episode !== b.episode) return 0;
  if (a.season !== null && b.season !== null && a.season !== b.season) return 0;
  const titleA = titleKey(a.title); const titleB = titleKey(b.title);
  if (titleA && titleB && titleA !== titleB) return 0;
  return titleA && titleB ? 200 : 100;
}

/** Match by identity, never the system dialog's selection order. Ties remain user choices. */
export function pairSubtitles(rows: ImportInput[], paths: string[], detections: Record<string, SubtitleDetection> = {}): Record<string, string> {
  const choices = new Map<string, { row: ImportInput; score: number }[]>();
  for (const path of [...new Set(paths)]) {
    const scores = rows.map((row) => ({ row, score: matchScore(row, path, detections[row.item_id]) })).filter((value) => value.score > 0);
    const best = Math.max(0, ...scores.map((value) => value.score));
    choices.set(path, scores.filter((value) => value.score === best));
  }
  const assignments: Record<string, string> = {};
  for (const row of rows) {
    const matches = [...choices].filter(([, values]) => values.length === 1 && values[0].row.item_id === row.item_id);
    const best = Math.max(0, ...matches.map(([, values]) => values[0].score));
    const winners = matches.filter(([, values]) => values[0].score === best);
    if (winners.length === 1) assignments[row.item_id] = winners[0][0];
  }
  return assignments;
}
