import { describe, expect, it } from 'vitest';
import { compareAnimeImports, inferAnimeFilename } from '../src/importFilenames';

describe('Anime import filename metadata', () => {
  it('finds bracketed episode numbers without confusing release metadata with episodes', () => {
    expect(inferAnimeFilename('/anime/[DMG&MH&VCB-Studio] Shoujo Kageki Revue Starlight [01][Ma10p_1080p][x265_flac_aac].mkv'))
      .toEqual({ title: 'Shoujo Kageki Revue Starlight', season: null, episode: 1 });
    expect(inferAnimeFilename('[Group] Anime [12v2][1080p][A91F5C20].mkv')).toMatchObject({ title: 'Anime', episode: 12 });
    expect(inferAnimeFilename('[Group] Anime [1080][x265].mkv')).toMatchObject({ title: 'Anime', episode: null });
  });

  it('recognizes season, episode and Chinese markers while leaving film titles unnumbered', () => {
    expect(inferAnimeFilename('Anime.S02E03.1080p.mkv')).toEqual({ title: 'Anime', season: 2, episode: 3 });
    expect(inferAnimeFilename('[Group] Anime [S02E03][1080p].mkv')).toEqual({ title: 'Anime', season: 2, episode: 3 });
    expect(inferAnimeFilename('[Group] Anime [E02].mkv')).toEqual({ title: 'Anime', season: null, episode: 2 });
    expect(inferAnimeFilename('Anime - EP07.mkv')).toMatchObject({ title: 'Anime', episode: 7 });
    expect(inferAnimeFilename('动画 第二季 第十二话.mkv')).toMatchObject({ season: 2, episode: 12 });
    expect(inferAnimeFilename('动画剧场版 2024.mkv')).toEqual({ title: '动画剧场版 2024', season: null, episode: null });
  });

  it('sorts episodes before release group names and naturally sorts unknown episodes', () => {
    const row = (path: string) => ({ video_path: path, ...inferAnimeFilename(path) });
    const known = ['[A] Anime [10].mkv', '[Z] Anime [02].mkv', '[B] Anime [01].mkv'].map(row).sort(compareAnimeImports);
    expect(known.map((entry) => entry.episode)).toEqual([1, 2, 10]);
    const unknown = ['SP10.mkv', 'SP2.mkv'].map(row).sort(compareAnimeImports);
    expect(unknown.map((entry) => entry.video_path)).toEqual(['SP2.mkv', 'SP10.mkv']);
  });
});
