import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';
import { DEFAULTS, type Asset, type Bootstrap, type Project } from '../src/types';

const bootstrap: Bootstrap = { library_path: '/tmp/startup', settings: { defaults: DEFAULTS, bindings: {} },
  provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' };
const preparing: Bootstrap = { ...bootstrap, visual_encoder: { available: false }, startup: {
  index: { status: 'running' }, statistics: { status: 'pending' }, encoder: { status: 'pending' },
} };
const ready: Bootstrap = { ...preparing, stats: { records: 42 }, startup: {
  index: { status: 'ready' }, statistics: { status: 'ready' }, encoder: { status: 'ready' },
} };
const asset: Asset = { id: 'episode', work_id: 'work', title: '已载入的视频', kind: 'animation', duration_ms: 120000,
  width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false,
  created_at: '2026-10-20T00:00:00Z', status: 'ready' };
const project: Project = { id: 'project', name: '已载入的项目', kind: 'animation', folder_id: null, tags: [], asset_count: 0, created_at: '2026-10-20T00:00:00Z', updated_at: '2026-10-20T00:00:00Z' };
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
function requests(handler: (url: string, count: number, options?: RequestInit) => unknown | Promise<unknown>) {
  const counts: Record<string, number> = {};
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const value = await handler(url, counts[url] = (counts[url] || 0) + 1, options);
    return value instanceof Response ? value : new Response(JSON.stringify(value), { status: 200 });
  });
  vi.stubGlobal('fetch', fetch);
  return counts;
}
async function mount() { await act(async () => { render(<App />); }); }
async function advance(ms: number) { await act(async () => { await vi.advanceTimersByTimeAsync(ms); }); }
async function navigate(page: string) { await act(async () => { window.history.replaceState({}, '', `/#/${page}`); window.dispatchEvent(new Event('hashchange')); }); }
let hidden = false;
beforeEach(() => {
  vi.useFakeTimers(); hidden = false;
  vi.stubGlobal('scrollTo', vi.fn());
  vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden);
  window.history.replaceState({}, '', '/#/library');
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Progressive startup', () => {
  it('opens the shell before any list finishes, publishes videos independently, and keeps navigation available', async () => {
    const assets = deferred<Asset[]>(); const projects = deferred<Project[]>(); const jobs = deferred<never[]>();
    requests((url) => url === '/api/bootstrap' ? preparing : url === '/api/assets' ? assets.promise
      : url === '/api/projects' ? projects.promise : url === '/api/jobs' ? jobs.promise : []);
    await mount();
    expect(screen.getByRole('navigation', { name: '主要导航' })).toBeTruthy();
    expect(screen.queryByText('正在打开本地资料库')).toBeNull();
    expect(screen.getByText('正在加载作品列表')).toBeTruthy();
    expect(screen.getByText('正在加载项目列表')).toBeTruthy();
    expect(screen.queryByText('你的第一部作品，从这里开始')).toBeNull();
    expect(screen.getByText(/正在准备资料统计/)).toBeTruthy();
    const statistics = screen.getByText('可检索片段').parentElement!;
    expect(within(statistics).getByText('—')).toBeTruthy();
    await act(async () => { assets.resolve([asset]); });
    expect(screen.getByRole('heading', { name: asset.title })).toBeTruthy();
    expect(screen.getByText('正在加载项目列表')).toBeTruthy();
    await navigate('jobs');
    expect(screen.getByText('正在加载任务列表')).toBeTruthy();
    expect(screen.queryByText('还没有分析任务')).toBeNull();
    await navigate('settings');
    expect(screen.getByRole('heading', { name: '模型与设置' })).toBeTruthy();
    expect(screen.getByText('正在检查本机视觉编码器')).toBeTruthy();
    await act(async () => { projects.resolve([project]); jobs.resolve([]); });
  });

  it('keeps a failed project list local and retries it without waiting for a slow video list', async () => {
    const assets = deferred<Asset[]>();
    const counts = requests((url, count) => {
      if (url === '/api/bootstrap') return ready;
      if (url === '/api/assets') return assets.promise;
      if (url === '/api/projects') return count === 1 ? new Response(JSON.stringify({ detail: '项目暂时无法读取' }), { status: 500 }) : [project];
      return [];
    });
    await mount();
    expect(screen.queryByRole('heading', { name: '无法连接本地服务' })).toBeNull();
    expect(screen.getByText(/项目列表加载失败/)).toBeTruthy();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '重试项目列表' })); });
    expect(counts['/api/projects']).toBe(2);
    expect(counts['/api/assets']).toBe(1);
    expect(screen.getByRole('option', { name: `${project.name} · 0 个视频` })).toBeTruthy();
    expect(screen.getByText('正在加载作品列表')).toBeTruthy();
    await act(async () => { assets.resolve([asset]); });
    expect(screen.getByRole('heading', { name: asset.title })).toBeTruthy();
    expect(screen.queryByText(/项目列表加载失败/)).toBeNull();
  });

  it('never represents an initial video load failure as an empty library and allows retry', async () => {
    const counts = requests((url, count) => url === '/api/bootstrap' ? ready
      : url === '/api/assets' && count === 1 ? new Response(JSON.stringify({ detail: '磁盘暂忙' }), { status: 500 }) : []);
    await mount();
    expect(screen.queryByText('你的第一部作品，从这里开始')).toBeNull();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '重试作品列表' })); });
    expect(screen.getByText('你的第一部作品，从这里开始')).toBeTruthy();
    expect(counts['/api/bootstrap']).toBe(1);
  });

  it('polls lightweight preparation state on the queue page, pauses in hidden tabs, and stops when ready', async () => {
    window.history.replaceState({}, '', '/#/jobs');
    let complete = false;
    const counts = requests((url) => url === '/api/bootstrap' ? complete ? ready : preparing : []);
    await mount(); await advance(2000);
    expect(counts['/api/bootstrap']).toBe(2);
    expect(counts['/api/assets']).toBe(1);
    await act(async () => { hidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    await advance(20000);
    expect(counts['/api/bootstrap']).toBe(2);
    complete = true;
    await act(async () => { hidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    expect(counts['/api/bootstrap']).toBe(3);
    await advance(20000);
    expect(counts['/api/bootstrap']).toBe(3);
    expect(counts['/api/assets']).toBe(1);
  });

  it('keeps search input while indexing, retries an indexing failure, and enables search when preparation finishes', async () => {
    window.history.replaceState({}, '', '/#/search');
    let snapshot = preparing;
    const counts = requests((url, _count, options) => {
      if (url === '/api/bootstrap') return snapshot;
      if (url === '/api/startup/retry') { expect(options?.method).toBe('POST'); snapshot = ready; return {}; }
      if (url === '/api/search') return { results: [], degraded: [], explanation: '' };
      return [];
    });
    await mount();
    const input = screen.getByRole('textbox', { name: '用自然语言搜索镜头或台词' });
    fireEvent.change(input, { target: { value: '雨中拥抱' } });
    const search = screen.getByRole('button', { name: '检索' }) as HTMLButtonElement;
    expect(search.disabled).toBe(true);
    fireEvent.submit(input.closest('form')!);
    expect(counts['/api/search']).toBeUndefined();
    expect(screen.getByText(/正在准备检索索引/)).toBeTruthy();
    expect(screen.queryByText(/配置语义向量模型后/)).toBeNull();
    snapshot = { ...ready, startup: { ...ready.startup!, index: { status: 'error', message: '索引文件暂时无法访问' } } };
    await advance(2000);
    expect(screen.getByText(/检索索引准备失败/)).toBeTruthy();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '重试检索索引' })); });
    expect(counts['/api/startup/retry']).toBe(1);
    expect((input as HTMLInputElement).value).toBe('雨中拥抱');
    expect(search.disabled).toBe(false);
    await act(async () => { fireEvent.click(search); });
    expect(counts['/api/search']).toBe(1);
  });

  it('still requires an authenticated bootstrap before opening the shell or starting list requests', async () => {
    const counts = requests(() => new Response(JSON.stringify({ detail: 'Unauthorized' }), { status: 401 }));
    await mount();
    expect(screen.getByRole('heading', { name: '连接本地资料库' })).toBeTruthy();
    expect(screen.queryByRole('navigation', { name: '主要导航' })).toBeNull();
    expect(counts['/api/assets']).toBeUndefined();
  });
});
