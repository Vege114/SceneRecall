import { post } from '../api';

export type LocalFileKind = 'video' | 'subtitle' | 'backup';

export async function pickLocalFiles(kind: LocalFileKind, multiple = false, directory?: string): Promise<string[]> {
  const result = await post<{ paths: string[]; cancelled: boolean }>('/local-files/pick', { kind, multiple, ...(directory ? { directory } : {}) });
  return result.cancelled ? [] : result.paths;
}
