import { createContext, useContext } from 'react';
import type { Asset, Bootstrap, Job, Project, Folder } from './types';

export interface AppContextValue {
  bootstrap: Bootstrap;
  assets: Asset[];
  projects?: Project[];
  folders?: Folder[];
  jobs: Job[];
  refresh: () => Promise<void>;
  refreshJobs: (afterMutation?: boolean) => Promise<void>;
  notify: (message: string, kind?: 'success' | 'error') => void;
  openImport: (projectId?: string) => void;
  openJobs?: (assets: Asset[]) => void;
  openJob: (asset: Asset) => void;
}
export const AppContext = createContext<AppContextValue | null>(null);
export function useApp(): AppContextValue {
  const app = useContext(AppContext);
  if (!app) throw new Error('App context unavailable');
  return app;
}
