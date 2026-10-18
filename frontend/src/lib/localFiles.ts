import { post } from '../api';

export type LocalFileKind = 'video' | 'subtitle' | 'backup';

export function localFileDirectory(path: string): string | undefined {
  // C:\video.mkv must start at C:\, rather than the drive-relative C:.
  const directory = path.match(/^(.*)[\\/][^\\/]*$/s)?.[1];
  if (directory === undefined) return undefined;
  if (!directory) return path[0];
  return /^[A-Za-z]:$/.test(directory) ? directory + path[directory.length] : directory;
}

export async function pickLocalFiles(kind: LocalFileKind, multiple = false, directory?: string): Promise<string[]> {
  const result = await post<{ paths: string[]; cancelled: boolean }>('/local-files/pick', { kind, multiple, ...(directory ? { directory } : {}) });
  return result.cancelled ? [] : result.paths;
}
