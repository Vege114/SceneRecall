import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { ArrowRight, Bookmark, Check, ChevronRight, Clapperboard, Film, HardDrive, KeyRound, Menu, Plus, Search, Settings2, ShieldCheck, Workflow, Trash2, X } from 'lucide-react';
import { api, ApiError, message, post } from './api';
import { AppContext, type LibraryResource, type ResourceStates } from './context';
import type { Asset, Bootstrap, Job, Project, Folder } from './types';
import Library from './components/Library';
import SearchPage from './components/Search';
import Jobs from './components/Jobs';
import Collection from './components/Collection';
import Settings from './components/Settings';
import Trash from './components/Trash';
import AssetDetail from './components/AssetDetail';
import ImportModal from './components/ImportModal';
import JobModal from './components/JobModal';
import { Alert, Spinner } from './components/ui';

type FollowupRefresh = { after: Promise<void>; promise: Promise<void> };
const initialResources = (): ResourceStates => ({ assets: { loaded: false, loading: true }, projects: { loaded: false, loading: true }, folders: { loaded: false, loading: true }, jobs: { loaded: false, loading: true } });
function afterCurrentRefresh(current: Promise<void>, queued: { current: FollowupRefresh | null }, start: () => Promise<void>): Promise<void> {
  if (queued.current?.after === current) return queued.current.promise;
  const pending = current.catch(() => undefined).then(start).finally(() => {
    if (queued.current?.promise === pending) queued.current = null;
  });
  queued.current = { after: current, promise: pending };
  return pending;
}

function useRoute() {
  const [hash, setHash] = useState(window.location.hash || '#/library');
  useEffect(() => { const changed = () => { setHash(window.location.hash || '#/library'); window.scrollTo({ top: 0 }); }; window.addEventListener('hashchange', changed); return () => window.removeEventListener('hashchange', changed); }, []);
  const raw = hash.replace(/^#\/?/, '');
  const [path, search] = raw.split('?');
  const [page, id] = path.split('/');
  const params = new URLSearchParams(search || '');
  const atMs = Number(params.get('t') || 0);
  return { page: page || 'library', id: id ? decodeURIComponent(id) : '', atMs: Number.isFinite(atMs) ? Math.max(0, atMs) : 0, cueId: params.get('cue') || undefined, chunkId: params.get('chunk') || undefined, audioId: params.get('audio') || undefined };
}

function SessionGate({ onConnected, initialError }: { onConnected: () => Promise<void>; initialError: string }) {
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(initialError);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError('');
    try { await post('/session', { token: token.trim() }); setToken(''); await onConnected(); } catch (error) { setError(message(error)); } finally { setBusy(false); }
  };
  return <div className="session-screen"><div className="session-decoration"><div className="brand-mark"><Clapperboard size={25} /></div><span>SceneRecall</span><h1>每一个镜头，<br />都值得被记住。</h1><p>你的本地影视记忆库</p><div className="session-lines" /></div><div className="session-content"><div className="session-card"><span className="eyebrow">WELCOME TO YOUR ARCHIVE</span><h2>连接本地资料库</h2><p>使用启动终端显示的专属链接打开应用，或在下面填写本机会话令牌。</p><form onSubmit={submit}><label className="field"><span className="field-label">本机会话令牌</span><div className="token-input"><KeyRound size={18} /><input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} required placeholder="粘贴启动时显示的 token" /></div></label>{error && <Alert tone="error">{error}</Alert>}<button className="button primary" type="submit" disabled={busy || !token.trim()}>{busy ? <Spinner label="正在连接" /> : <>打开资料库 <ArrowRight size={18} /></>}</button></form><div className="session-security"><ShieldCheck size={17} />这不是 AI API Key，仅用于保护本地服务。</div></div></div></div>;
}

export default function App() {
  const route = useRoute();
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [resources, setResources] = useState(initialResources);
  const [auth, setAuth] = useState<'loading' | 'ready' | 'required' | 'error'>('loading');
  const [connectionError, setConnectionError] = useState('');
  const [showImport, setShowImport] = useState(false);
  const [importProject, setImportProject] = useState<string>();
  const [projects, setProjects] = useState<Project[]>([]);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [jobAssets, setJobAssets] = useState<Asset[] | null>(null);
  const [jobAsset, setJobAsset] = useState<Asset | null>(null);
  const [toast, setToast] = useState<{ message: string; kind: 'success' | 'error'; key: number } | null>(null);
  const [mobileNav, setMobileNav] = useState(false);
  const bootstrapReady = useRef(false);
  const requestEpoch = useRef(0);
  const bootstrapFlight = useRef<Promise<void> | null>(null);
  const libraryFlight = useRef<Promise<void> | null>(null);
  const resourceFlights = useRef<Partial<Record<LibraryResource, Promise<void>>>>({});
  const resourceRevisions = useRef<Partial<Record<LibraryResource, number>>>({});
  const jobsFlight = useRef<Promise<void> | null>(null);
  const bootstrapFollowup = useRef<FollowupRefresh | null>(null);
  const libraryFollowup = useRef<FollowupRefresh | null>(null);
  const jobsFollowup = useRef<FollowupRefresh | null>(null);
  const bootstrapRevision = useRef(0);
  const libraryRevision = useRef(0);
  const jobsRevision = useRef(0);
  const refreshBootstrap = useCallback((afterMutation = false): Promise<void> => {
    const epoch = requestEpoch.current;
    if (afterMutation) bootstrapRevision.current += 1;
    if (bootstrapFlight.current) return afterMutation
      ? afterCurrentRefresh(bootstrapFlight.current, bootstrapFollowup, () => requestEpoch.current === epoch ? refreshBootstrap() : Promise.resolve())
      : bootstrapFlight.current;
    const revision = bootstrapRevision.current;
    const pending = api<Bootstrap>('/bootstrap').then((data) => {
      if (requestEpoch.current === epoch && bootstrapRevision.current === revision) { setBootstrap(data); bootstrapReady.current = true; }
    }).finally(() => { if (bootstrapFlight.current === pending) bootstrapFlight.current = null; });
    bootstrapFlight.current = pending;
    return pending;
  }, []);
  const refreshLibraryResource = useCallback((resource: LibraryResource): Promise<void> => {
    const epoch = requestEpoch.current;
    const revision = libraryRevision.current;
    const flight = resourceFlights.current[resource];
    if (flight) return resourceRevisions.current[resource] === revision ? flight
      : flight.catch(() => undefined).then(() => requestEpoch.current === epoch ? refreshLibraryResource(resource) : undefined);
    const current = () => requestEpoch.current === epoch && libraryRevision.current === revision;
    setResources((state) => ({ ...state, [resource]: { ...state[resource], loading: true, error: undefined } }));
    const pending = api<Asset[] | Project[] | Folder[]>(`/${resource}`).then((data) => {
      if (!current()) return;
      if (resource === 'assets') setAssets(data as Asset[]);
      else if (resource === 'projects') setProjects(data as Project[]);
      else setFolders(data as Folder[]);
      setResources((state) => ({ ...state, [resource]: { loaded: true, loading: false } }));
    }).catch((error) => {
      if (current()) setResources((state) => ({ ...state, [resource]: { ...state[resource], loading: false, error: message(error) } }));
      throw error;
    }).finally(() => { if (resourceFlights.current[resource] === pending) delete resourceFlights.current[resource]; });
    resourceFlights.current[resource] = pending;
    resourceRevisions.current[resource] = revision;
    return pending;
  }, []);
  const refreshLibrary = useCallback((afterMutation = false): Promise<void> => {
    const epoch = requestEpoch.current;
    if (afterMutation) libraryRevision.current += 1;
    if (libraryFlight.current) return afterMutation
      ? afterCurrentRefresh(libraryFlight.current, libraryFollowup, () => requestEpoch.current === epoch ? refreshLibrary() : Promise.resolve())
      : libraryFlight.current;
    const pending = Promise.allSettled([
      refreshLibraryResource('assets'), refreshLibraryResource('projects'), refreshLibraryResource('folders'),
    ]).then((results) => {
      const failure = results.find((result) => result.status === 'rejected');
      if (failure?.status === 'rejected') throw failure.reason;
    }).finally(() => { if (libraryFlight.current === pending) libraryFlight.current = null; });
    libraryFlight.current = pending;
    return pending;
  }, [refreshLibraryResource]);
  const refreshJobs = useCallback((afterMutation = false): Promise<void> => {
    const epoch = requestEpoch.current;
    if (afterMutation) jobsRevision.current += 1;
    if (jobsFlight.current) return afterMutation
      ? afterCurrentRefresh(jobsFlight.current, jobsFollowup, () => requestEpoch.current === epoch ? refreshJobs() : Promise.resolve())
      : jobsFlight.current;
    const revision = jobsRevision.current;
    const current = () => requestEpoch.current === epoch && jobsRevision.current === revision;
    setResources((state) => ({ ...state, jobs: { ...state.jobs, loading: true, error: undefined } }));
    // Every refresh uses this one request and publishes its result immediately.
    // A slow bootstrap can neither delay nor overwrite a newer jobs response.
    const pending = api<Job[]>('/jobs').then((data) => {
      if (current()) { setJobs(data); setResources((state) => ({ ...state, jobs: { loaded: true, loading: false } })); }
    }).catch((error) => {
      if (current()) setResources((state) => ({ ...state, jobs: { ...state.jobs, loading: false, error: message(error) } }));
      throw error;
    }).finally(() => { if (jobsFlight.current === pending) jobsFlight.current = null; });
    jobsFlight.current = pending;
    return pending;
  }, []);
  const refresh = useCallback(async () => {
    const epoch = requestEpoch.current;
    const initial = !bootstrapReady.current;
    // Explicit callers have just changed server state. A GET started before
    // that operation is dirty: discard its snapshot and follow it with one
    // fresh request. Polls still merge with whichever request is in flight.
    const pendingBootstrap = refreshBootstrap(!initial);
    if (initial) {
      // Only the authenticated, lightweight snapshot gates entry. Lists publish
      // independently and handle their errors inside their own page sections.
      await pendingBootstrap;
      if (requestEpoch.current !== epoch || !bootstrapReady.current) return;
      setAuth('ready'); setConnectionError('');
      void refreshLibrary().catch(() => undefined);
      void refreshJobs().catch(() => undefined);
      return;
    }
    void pendingBootstrap.catch(() => undefined);
    await Promise.all([refreshLibrary(true), refreshJobs(true)]);
    if (requestEpoch.current === epoch) { setAuth('ready'); setConnectionError(''); }
  }, [refreshBootstrap, refreshLibrary, refreshJobs]);
  const retryResource = useCallback((resource: LibraryResource | 'jobs') => resource === 'jobs' ? refreshJobs() : refreshLibraryResource(resource), [refreshJobs, refreshLibraryResource]);
  const retryStartup = useCallback(async () => { await post('/startup/retry', {}); await refreshBootstrap(true); }, [refreshBootstrap]);
  const preparing = !!bootstrap?.startup && Object.values(bootstrap.startup).some((task) => ['pending', 'running'].includes(task.status));
  const notify = useCallback((message: string, kind: 'success' | 'error' = 'success') => setToast({ message, kind, key: Date.now() }), []);
  useEffect(() => {
    const init = async () => {
      const url = new URL(window.location.href);
      const token = url.searchParams.get('token');
      if (token) { url.searchParams.delete('token'); window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`); }
      try { if (token) await post('/session', { token }); await refresh(); }
      catch (error) { setAuth(error instanceof ApiError && error.status === 401 ? 'required' : 'error'); setConnectionError(error instanceof ApiError && error.status === 401 ? '' : message(error)); }
    };
    void init();
    const invalidate = () => { requestEpoch.current += 1; bootstrapReady.current = false; bootstrapFlight.current = null; libraryFlight.current = null; resourceFlights.current = {}; jobsFlight.current = null; bootstrapFollowup.current = null; libraryFollowup.current = null; jobsFollowup.current = null; };
    const expired = () => { invalidate(); setBootstrap(null); setAssets([]); setProjects([]); setFolders([]); setJobs([]); setResources(initialResources()); setAuth('required'); }; window.addEventListener('session-expired', expired);
    return () => { window.removeEventListener('session-expired', expired); invalidate(); };
  }, [refresh]);
  useEffect(() => {
    if (auth !== 'ready') return;
    let generation = 0;
    let jobsTimer: number | undefined;
    let dataTimer: number | undefined;
    const active = (current: number) => generation === current && !document.hidden;
    const pollJobs = async (current: number) => {
      if (!active(current)) return;
      try { await refreshJobs(); } catch { /* Retry after the next polling delay. */ }
      if (active(current)) jobsTimer = window.setTimeout(() => void pollJobs(current), 4000);
    };
    const pollData = async (current: number) => {
      // Queue controls need only lightweight job snapshots while local analysis
      // is busy. Keep aggregate projections off this page's polling path.
      if (!active(current) || route.page === 'jobs') return;
      // Jobs have their own fast loop; the slower library loop never requests
      // another jobs snapshot, and waits for its aggregate request to settle.
      try { await Promise.all([refreshBootstrap(), refreshLibrary()]); } catch { /* Keep the last loaded library visible. */ }
      if (active(current)) dataTimer = window.setTimeout(() => void pollData(current), 20000);
    };
    const stopTimers = () => { generation += 1; window.clearTimeout(jobsTimer); window.clearTimeout(dataTimer); };
    const resume = () => {
      stopTimers();
      if (!document.hidden) { void pollJobs(generation); void pollData(generation); }
    };
    if (!document.hidden) {
      const current = generation;
      jobsTimer = window.setTimeout(() => void pollJobs(current), 4000);
      if (route.page !== 'jobs') dataTimer = window.setTimeout(() => void pollData(current), 20000);
    }
    document.addEventListener('visibilitychange', resume);
    return () => { stopTimers(); document.removeEventListener('visibilitychange', resume); };
  }, [auth, route.page, refreshBootstrap, refreshLibrary, refreshJobs]);
  useEffect(() => {
    if (auth !== 'ready' || !preparing) return;
    let generation = 0;
    let timer: number | undefined;
    const poll = async (current: number) => {
      if (current !== generation || document.hidden) return;
      try { await refreshBootstrap(); } catch { /* Keep browsing and retry the readiness snapshot. */ }
      if (current === generation && !document.hidden) timer = window.setTimeout(() => void poll(current), 2000);
    };
    const resume = () => { generation += 1; window.clearTimeout(timer); if (!document.hidden) void poll(generation); };
    if (!document.hidden) timer = window.setTimeout(() => void poll(generation), 2000);
    document.addEventListener('visibilitychange', resume);
    return () => { generation += 1; window.clearTimeout(timer); document.removeEventListener('visibilitychange', resume); };
  }, [auth, preparing, refreshBootstrap]);
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(null), toast.kind === 'error' ? 7000 : 4500); return () => window.clearTimeout(timer); }, [toast]);
  useEffect(() => { setMobileNav(false); }, [route.page]);
  if (auth === 'loading') return <div className="app-loading"><div className="brand-mark"><Clapperboard size={25} /></div><h1>SceneRecall</h1><Spinner label="正在打开本地资料库" /></div>;
  if (auth === 'required') return <SessionGate onConnected={refresh} initialError={connectionError} />;
  if (auth === 'error' || !bootstrap) return <div className="app-loading"><div className="brand-mark"><Clapperboard size={25} /></div><h1>无法连接本地服务</h1><p>{connectionError}</p><button className="button primary" onClick={() => { setAuth('loading'); void refresh().catch((error) => { setConnectionError(message(error)); setAuth(error instanceof ApiError && error.status === 401 ? 'required' : 'error'); }); }}>重新连接</button><p className="helper">请先按照项目 README 启动 SceneRecall，再打开终端提供的地址。</p></div>;
  const nav = [{ id: 'library', label: '我的资料库', icon: Film }, { id: 'search', label: '镜头检索', icon: Search }, { id: 'collections', label: '片段收藏', icon: Bookmark }, { id: 'jobs', label: '任务队列', icon: Workflow }, { id: 'trash', label: '回收站', icon: Trash2 }];
  const activeJobs = jobs.filter((job) => ['running', 'queued', 'pending', 'waiting_input'].includes(job.status)).length;
  const currentName = route.page === 'settings' ? '模型与设置' : route.page === 'asset' ? '作品详情' : nav.find((item) => item.id === route.page)?.label || '我的资料库';
  return <AppContext.Provider value={{ bootstrap, assets, jobs, projects, folders, resources, retryResource, retryStartup, refresh, refreshJobs, notify, openImport: (projectId) => { setImportProject(projectId); setShowImport(true); }, openJob: setJobAsset, openJobs: setJobAssets }}><div className="app-shell">{mobileNav && <div className="mobile-scrim" onClick={() => setMobileNav(false)} />}<aside className={`sidebar ${mobileNav ? 'open' : ''}`}><a className="brand" href="#/library"><div className="brand-mark"><Clapperboard size={24} strokeWidth={1.6} /></div><div><strong>SceneRecall</strong><span>你的影视记忆库</span></div></a><div className="sidebar-label">WORKSPACE</div><nav aria-label="主要导航">{nav.map(({ id, label, icon: Icon }) => <a className={route.page === id || (id === 'library' && route.page === 'asset') ? 'active' : ''} href={`#/${id}`} key={id}><Icon size={19} strokeWidth={1.6} /><span>{label}</span>{id === 'library' && <small>{resources.assets.loaded ? assets.length : '…'}</small>}{id === 'jobs' && activeJobs > 0 && <small className="active-count">{activeJobs}</small>}</a>)}</nav><div className="sidebar-divider" /><a className={`sidebar-settings ${route.page === 'settings' ? 'active' : ''}`} href="#/settings"><Settings2 size={19} strokeWidth={1.6} /><span>模型与设置</span></a><div className="sidebar-bottom"><div className="local-card"><HardDrive size={19} /><div><strong>本地工作空间</strong><span><span className="online-dot" />资料保存在此电脑</span></div></div><div className="sidebar-footer"><span>SCENERECALL</span><span>v{bootstrap.version}</span></div></div></aside><div className="main-shell"><header className="topbar"><div><button className="icon-button menu-toggle" aria-label="打开导航" onClick={() => setMobileNav(!mobileNav)}><Menu size={21} /></button><span className="breadcrumb">工作空间 <ChevronRight size={14} /><strong>{currentName}</strong></span></div><div className="topbar-right"><span className="local-indicator"><span className="online-dot" />本地服务已连接</span><span className="topbar-separator" /><button className="icon-button" title="导入作品" aria-label="导入作品" onClick={() => setShowImport(true)}><Plus size={20} /></button><span className="user-avatar">我</span></div></header><main className={`main-content ${route.page === 'asset' ? 'detail-page' : ''}`}>{route.page === 'search' ? <SearchPage /> : route.page === 'jobs' ? <Jobs /> : route.page === 'collections' ? <Collection /> : route.page === 'settings' ? <Settings /> : route.page === 'trash' ? <Trash /> : route.page === 'asset' && route.id ? <AssetDetail key={route.id} id={route.id} initialTime={route.atMs} initialCue={route.cueId} initialChunk={route.chunkId} initialAudio={route.audioId} /> : <Library />}</main><footer className="main-footer"><span>每个片段，都有出处。</span><span>LOCAL FIRST. YOUR MEMORY, YOUR MODELS.</span></footer></div></div>{showImport && <ImportModal projectId={importProject} onClose={() => { setShowImport(false); setImportProject(undefined); }} />}{jobAssets && <JobModal assets={jobAssets} onClose={() => setJobAssets(null)} />}{jobAsset && <JobModal asset={jobAsset} onClose={() => setJobAsset(null)} />}{toast && <div className={`toast toast-${toast.kind}`} role={toast.kind === 'error' ? 'alert' : 'status'}>{toast.kind === 'success' ? <Check size={18} /> : <X size={18} />}<span>{toast.message}</span><button className="icon-button" onClick={() => setToast(null)} aria-label="关闭提示"><X size={15} /></button></div>}</AppContext.Provider>;
}
