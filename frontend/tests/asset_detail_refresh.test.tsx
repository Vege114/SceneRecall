import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AssetDetail from '../src/components/AssetDetail';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type AssetDetail as Detail, type Job } from '../src/types';

vi.mock('../src/components/AudioIdentity', async () => {
  const { post } = await import('../src/api');
  return { default: function IdentityEditor({ asset, onSaved }: { asset: Asset; onSaved: () => Promise<void> }) {
    return <button onClick={() => { void post(`/assets/${asset.id}/audio/confirm`).then(onSaved); }}>保存身份测试</button>;
  } };
});

const asset: Asset = { id: 'episode', work_id: 'work', title: 'Current episode', kind: 'movie', duration_ms: 12000,
  width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '', status: 'ready' };
function detail(title = asset.title, id = asset.id): Detail {
  return { asset: { ...asset, id, title }, observations: [], scenes: [], subtitles: [], annotations: [], runs: [] };
}
function context(running = true): AppContextValue {
  return { bootstrap: { library_path: '/fixture', provider_profiles: [], settings: { defaults: DEFAULTS, bindings: {} }, stats: {}, tools: {}, version: 'test' },
    assets: [asset], jobs: running ? [{ id: 'job', asset_id: asset.id, status: 'running' } as Job] : [],
    refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
function requests(handler: (url: string, count: number, options?: RequestInit) => unknown | Promise<unknown>) {
  const counts: Record<string, number> = {};
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    const value = await handler(url, counts[url] = (counts[url] || 0) + 1, options);
    return new Response(JSON.stringify(value), { status: 200 });
  });
  vi.stubGlobal('fetch', fetch);
  return { counts, fetch };
}
async function advance(ms: number) { await act(async () => { await vi.advanceTimersByTimeAsync(ms); }); }
async function mount(value = context()) {
  let view!: ReturnType<typeof render>;
  await act(async () => { view = render(<AppContext.Provider value={value}><AssetDetail id={asset.id} /></AppContext.Provider>); });
  return view;
}
let hidden = false;
beforeEach(() => { vi.useFakeTimers(); hidden = false; vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden); });
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Asset detail refresh under slow server projections', () => {
  it('keeps one slow GET in flight and waits five seconds after completion before polling again', async () => {
    const first = deferred<Detail>();
    const { counts } = requests((_url, count) => count === 1 ? first.promise : detail('Updated episode'));
    await mount();
    await advance(60000);
    expect(counts['/api/assets/episode']).toBe(1);
    expect(screen.getByText('正在读取作品资料')).toBeTruthy();
    await act(async () => { first.resolve(detail()); });
    expect(screen.getByRole('heading', { name: asset.title })).toBeTruthy();
    await advance(4999);
    expect(counts['/api/assets/episode']).toBe(1);
    await advance(1);
    expect(counts['/api/assets/episode']).toBe(2);
    expect(screen.getByRole('heading', { name: 'Updated episode' })).toBeTruthy();
  });

  it('does not duplicate a pending GET when job state or tab visibility changes', async () => {
    const first = deferred<Detail>();
    const { counts } = requests((_url, count) => count === 1 ? first.promise : detail());
    const value = context();
    const view = await mount(value);
    await act(async () => { view.rerender(<AppContext.Provider value={{ ...value, jobs: [] }}><AssetDetail id={asset.id} /></AppContext.Provider>); });
    hidden = true;
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); });
    await advance(30000);
    hidden = false;
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); });
    expect(counts['/api/assets/episode']).toBe(1);
    await act(async () => { first.resolve(detail()); });
    await advance(30000);
    expect(counts['/api/assets/episode']).toBe(1); // No active job remains.
  });

  it('suspends hidden-tab polling and cancels the active request on unmount', async () => {
    const second = deferred<Detail>();
    const { counts, fetch } = requests((_url, count) => count === 1 ? detail() : second.promise);
    const view = await mount();
    hidden = true;
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); });
    await advance(60000);
    expect(counts['/api/assets/episode']).toBe(1);
    hidden = false;
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); });
    expect(counts['/api/assets/episode']).toBe(2);
    const signal = fetch.mock.calls[1][1]?.signal;
    view.unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => { second.resolve(detail('Response after unmount')); });
    await advance(30000);
    expect(counts['/api/assets/episode']).toBe(2);
  });

  it('aborts the previous asset GET and ignores its late response after navigation', async () => {
    const first = deferred<Detail>();
    const { fetch } = requests((url) => url === '/api/assets/episode' ? first.promise : detail('Second episode', 'second'));
    const value = context();
    const view = await mount(value);
    await act(async () => { view.rerender(<AppContext.Provider value={value}><AssetDetail id="second" /></AppContext.Provider>); });
    expect(fetch.mock.calls[0][1]?.signal?.aborted).toBe(true);
    expect(screen.getByRole('heading', { name: 'Second episode' })).toBeTruthy();
    await act(async () => { first.resolve(detail('Stale first episode')); });
    expect(screen.queryByRole('heading', { name: 'Stale first episode' })).toBeNull();
    expect(screen.getByRole('heading', { name: 'Second episode' })).toBeTruthy();
  });

  it('discards a pre-mutation snapshot and waits for a fresh detail before completing save refresh', async () => {
    const stale = deferred<Detail>();
    const fresh = deferred<Detail>();
    const { counts } = requests((url, count, options) => options?.method === 'POST' ? {}
      : count === 1 ? detail() : count === 2 ? stale.promise : fresh.promise);
    const value = context();
    await mount(value);
    await advance(5000);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '视听身份' })); });
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '保存身份测试' })); });
    expect(counts['/api/assets/episode']).toBe(2);
    await act(async () => { stale.resolve(detail('Pre-mutation snapshot')); });
    expect(counts['/api/assets/episode']).toBe(3);
    expect(screen.queryByRole('heading', { name: 'Pre-mutation snapshot' })).toBeNull();
    expect(value.refresh).not.toHaveBeenCalled();
    await act(async () => { fresh.resolve(detail('Post-mutation snapshot')); });
    expect(screen.getByRole('heading', { name: 'Post-mutation snapshot' })).toBeTruthy();
    expect(value.refresh).toHaveBeenCalledOnce();
  });
});
