import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';
import { DEFAULTS, type Bootstrap, type Job } from '../src/types';

// Exercise the real App, API requests, context updates and polling. Keep the
// library view small so assertions inspect delivered state rather than cards.
vi.mock('../src/components/Library', async () => {
  const { useApp } = await import('../src/context');
  const { post } = await import('../src/api');
  const { useState } = await import('react');
  return { default: function LibraryProbe() {
    const { bootstrap, assets, projects, jobs, refresh } = useApp();
    const [refreshed, setRefreshed] = useState(0);
    return <><h1>测试资料库</h1><p aria-label="当前任务">{jobs.map((job) => job.status).join(',')}</p>
      <p aria-label="可选模型">{bootstrap.provider_profiles.map((profile) => profile.name).join(',')}</p>
      <p aria-label="当前视频">{assets.map((asset) => asset.title).join(',')}</p>
      <p aria-label="当前项目">{projects?.map((project) => project.name).join(',')}</p>
      <button onClick={() => { void refresh().then(() => setRefreshed((value) => value + 1)).catch(() => undefined); }}>刷新列表</button>
      <button onClick={() => { void post('/profiles', { name: '新模型' }).then(refresh).then(() => setRefreshed((value) => value + 1)).catch(() => undefined); }}>保存模型</button>
      <p aria-label="完成刷新次数">{refreshed}</p></>;
  } };
});

const bootstrap: Bootstrap = { library_path: '/tmp/app-refresh', settings: { defaults: DEFAULTS, bindings: {} },
  provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' };
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((done, failed) => { resolve = done; reject = failed; });
  return { promise, resolve, reject };
}
function requests(handler: (url: string, count: number, options?: RequestInit) => unknown | Promise<unknown>) {
  const counts: Record<string, number> = {};
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const value = await handler(url, counts[url] = (counts[url] || 0) + 1, options);
    return new Response(JSON.stringify(value), { status: 200 });
  });
  vi.stubGlobal('fetch', fetch);
  return counts;
}
function initial(url: string) { return url === '/api/bootstrap' ? bootstrap : url === '/api/jobs' ? [{ id: 'job', status: 'queued' }] : []; }
async function mount() { await act(async () => { render(<App />); }); }
async function advance(ms: number) { await act(async () => { await vi.advanceTimersByTimeAsync(ms); }); }
let hidden = false;
beforeEach(() => {
  vi.useFakeTimers();
  hidden = false;
  vi.stubGlobal('scrollTo', vi.fn());
  vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden);
  window.history.replaceState({}, '', '/#/library');
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('App refresh and polling under slow local API responses', () => {
  it('keeps jobs polling on the queue page without aggregate polling, then resumes library polling after leaving', async () => {
    window.history.replaceState({}, '', '/#/jobs');
    const counts = requests((url) => url === '/api/jobs' ? [] : initial(url));
    await mount();
    // First-load authentication and cached library state still load normally.
    expect(counts['/api/bootstrap']).toBe(1);
    expect(counts['/api/assets']).toBe(1);
    await advance(60000);
    expect(counts['/api/jobs']).toBe(16);
    expect(counts['/api/bootstrap']).toBe(1);
    expect(counts['/api/assets']).toBe(1);
    await act(async () => { hidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    await act(async () => { hidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    expect(counts['/api/jobs']).toBe(17);
    expect(counts['/api/bootstrap']).toBe(1);
    await act(async () => { window.history.replaceState({}, '', '/#/library'); window.dispatchEvent(new Event('hashchange')); });
    await advance(20000);
    expect(counts['/api/bootstrap']).toBe(2);
    expect(counts['/api/assets']).toBe(2);
  });

  it('refreshes only jobs after queue controls and manual refresh, following a GET started before cancellation with a fresh snapshot', async () => {
    window.history.replaceState({}, '', '/#/jobs');
    const oldJobs = deferred<Job[]>();
    const jobs: Job[] = [1, 2].map((number) => ({ id: `job-${number}`, batch_id: 'batch', type: 'analysis',
      status: 'queued', progress: 0, completed: 0, total: 1, created_at: '2026-10-18T00:00:00Z' }));
    let cancelled = false;
    let batchCancelled = false;
    const counts = requests((url, count, options) => {
      if (options?.method === 'POST' && url === '/api/jobs/job-1/cancel') { cancelled = true; return {}; }
      if (options?.method === 'POST' && url === '/api/jobs/batches/batch/cancel') { batchCancelled = true; return { failures: [] }; }
      if (url === '/api/jobs') return count === 2 ? oldJobs.promise : jobs.map((job, index) => ({ ...job,
        status: batchCancelled || (cancelled && index === 0) ? 'cancelled' : 'queued' }));
      return initial(url);
    });
    await mount();
    await advance(4000); // A poll now holds the old queued snapshot.
    await act(async () => { fireEvent.click(within(screen.getAllByRole('article')[0]).getByRole('button', { name: '取消' })); });
    expect(cancelled).toBe(true);
    expect(counts['/api/jobs']).toBe(2);
    await act(async () => { oldJobs.resolve(jobs); });
    expect(counts['/api/jobs']).toBe(3);
    expect(within(screen.getAllByRole('article')[0]).getByText('已取消')).toBeTruthy();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '取消整批' })); });
    expect(batchCancelled).toBe(true);
    expect(counts['/api/jobs']).toBe(4);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '刷新' })); });
    expect(counts['/api/jobs']).toBe(5);
    for (const endpoint of ['/api/bootstrap', '/api/assets', '/api/projects', '/api/folders']) expect(counts[endpoint]).toBe(1);
  });

  it('publishes jobs independently of a slow bootstrap and cannot restore an older full-refresh jobs snapshot', async () => {
    const slowBootstrap = deferred<Bootstrap>();
    const counts = requests((url, count) => url === '/api/bootstrap' && count > 1 ? slowBootstrap.promise
      : url === '/api/jobs' ? [{ id: 'job', status: count < 7 ? 'running' : 'completed' }] : initial(url));
    await mount();
    await advance(20000);
    expect(counts['/api/bootstrap']).toBe(2);
    expect(screen.getByLabelText('当前任务').textContent).toBe('running');
    await advance(4000);
    expect(screen.getByLabelText('当前任务').textContent).toBe('completed');
    await advance(40000);
    expect(counts['/api/bootstrap']).toBe(2);
    await act(async () => { slowBootstrap.resolve({ ...bootstrap, stats: { records: 15000 } }); });
    expect(screen.getByLabelText('当前任务').textContent).toBe('completed');
  });

  it('finishes a post-action list refresh without waiting for aggregate bootstrap statistics', async () => {
    const slowBootstrap = deferred<Bootstrap>();
    const counts = requests((url, count) => url === '/api/bootstrap' && count > 1 ? slowBootstrap.promise : initial(url));
    await mount();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '刷新列表' })); });
    expect(screen.getByLabelText('完成刷新次数').textContent).toBe('1');
    expect(counts['/api/bootstrap']).toBe(2);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '刷新列表' })); });
    expect(screen.getByLabelText('完成刷新次数').textContent).toBe('2');
    expect(counts['/api/bootstrap']).toBe(2);
  });

  it('shares one slow jobs request across the fast poll and whole-library refresh without queuing intervals', async () => {
    const slowJobs = deferred<unknown>();
    const counts = requests((url, count) => url === '/api/jobs' && count === 2 ? slowJobs.promise
      : url === '/api/jobs' && count > 2 ? [{ id: 'job', status: 'completed' }] : initial(url));
    await mount();
    await advance(4000);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '刷新列表' })); });
    await advance(60000);
    expect(counts['/api/jobs']).toBe(2);
    await act(async () => { slowJobs.resolve([{ id: 'job', status: 'completed' }]); });
    expect(screen.getByLabelText('当前任务').textContent).toBe('completed');
    await advance(3999);
    expect(counts['/api/jobs']).toBe(3);
    await advance(1);
    expect(counts['/api/jobs']).toBe(4);
  });

  it('suspends both expensive and jobs polling in hidden tabs, then refreshes once on return', async () => {
    const counts = requests((url) => initial(url));
    await mount();
    await act(async () => { hidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    await advance(120000);
    expect(counts['/api/jobs']).toBe(1);
    expect(counts['/api/bootstrap']).toBe(1);
    expect(counts['/api/assets']).toBe(1);
    await act(async () => { hidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    expect(counts['/api/jobs']).toBe(2);
    expect(counts['/api/bootstrap']).toBe(2);
    expect(counts['/api/assets']).toBe(2);
    await advance(4000);
    expect(counts['/api/jobs']).toBe(3);
  });

  it('keeps slow library requests single-flight even if a sibling endpoint has already failed', async () => {
    const slowAssets = deferred<unknown>();
    const counts = requests((url, count) => {
      if (count > 1 && url === '/api/assets') return slowAssets.promise;
      if (count > 1 && url === '/api/projects') throw new Error('temporary failure');
      return initial(url);
    });
    await mount();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '刷新列表' })); });
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '刷新列表' })); });
    expect(counts['/api/assets']).toBe(2);
    expect(counts['/api/projects']).toBe(2);
    await act(async () => { slowAssets.resolve([]); });
  });

  it('discards GETs started before a mutation and follows them with fresh profiles, lists and jobs', async () => {
    const profileSnapshot = (name: string): Bootstrap => ({ ...bootstrap, provider_profiles: [
      { id: 'model', name, model: 'model', base_url: 'http://localhost', capabilities: [], secret_mode: 'session' },
    ] });
    const oldBootstrap = deferred<Bootstrap>();
    const freshBootstrap = deferred<Bootstrap>();
    const oldAssets = deferred<unknown>();
    const oldProjects = deferred<unknown>();
    const oldJobs = deferred<unknown>();
    let saved = false;
    const counts = requests((url, count, options) => {
      if (url === '/api/profiles' && options?.method === 'POST') { saved = true; return { id: 'model' }; }
      if (url === '/api/bootstrap') return count === 1 ? profileSnapshot('初始模型') : count === 2 ? oldBootstrap.promise : freshBootstrap.promise;
      if (url === '/api/assets' && count === 2) return oldAssets.promise;
      if (url === '/api/projects' && count === 2) return oldProjects.promise;
      if (url === '/api/jobs' && count === 2) return oldJobs.promise;
      if (saved && url === '/api/assets') return [{ id: 'new-asset', title: '新视频' }];
      if (saved && url === '/api/projects') return [{ id: 'new-project', name: '新项目' }];
      if (saved && url === '/api/jobs') return [{ id: 'new-job', status: 'completed' }];
      return initial(url);
    });
    await mount();
    await advance(20000); // Old jobs and aggregate/list GETs are already in flight.
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '保存模型' })); });
    expect(saved).toBe(true);
    expect(counts['/api/bootstrap']).toBe(2);
    expect(counts['/api/assets']).toBe(2);
    expect(screen.getByLabelText('完成刷新次数').textContent).toBe('0');
    await act(async () => {
      oldBootstrap.resolve(profileSnapshot('操作前的旧模型'));
      oldAssets.resolve([{ id: 'old-asset', title: '操作前的旧视频' }]);
      oldProjects.resolve([{ id: 'old-project', name: '操作前的旧项目' }]);
      oldJobs.resolve([{ id: 'old-job', status: 'running' }]);
    });
    expect(counts['/api/bootstrap']).toBe(3);
    expect(counts['/api/assets']).toBe(3);
    expect(counts['/api/projects']).toBe(3);
    expect(counts['/api/jobs']).toBe(3);
    expect(screen.getByLabelText('完成刷新次数').textContent).toBe('1');
    expect(screen.getByLabelText('当前视频').textContent).toBe('新视频');
    expect(screen.getByLabelText('当前项目').textContent).toBe('新项目');
    expect(screen.getByLabelText('当前任务').textContent).toBe('completed');
    expect(screen.getByLabelText('可选模型').textContent).toBe('初始模型');
    expect(screen.queryByText('操作前的旧模型')).toBeNull();
    await act(async () => { freshBootstrap.resolve(profileSnapshot('保存后的新模型')); });
    expect(screen.getByLabelText('可选模型').textContent).toBe('保存后的新模型');
  });

  it('queues another fresh bootstrap when a later mutation occurs after the first follow-up GET has started', async () => {
    const first = deferred<Bootstrap>();
    const second = deferred<Bootstrap>();
    const counts = requests((url, count) => url === '/api/bootstrap' && count === 2 ? first.promise
      : url === '/api/bootstrap' && count === 3 ? second.promise
      : url === '/api/bootstrap' && count === 4 ? { ...bootstrap, version: 'latest' } : initial(url));
    await mount();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '刷新列表' })); });
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '刷新列表' })); });
    expect(counts['/api/bootstrap']).toBe(2);
    await act(async () => { first.resolve(bootstrap); });
    expect(counts['/api/bootstrap']).toBe(3);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '刷新列表' })); });
    await act(async () => { second.resolve(bootstrap); });
    expect(counts['/api/bootstrap']).toBe(4);
    expect(screen.getByText('vlatest')).toBeTruthy();
  });
});
