import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CharacterProfiles from '../src/components/CharacterProfiles';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type CharacterLibrary, type Job } from '../src/types';

vi.mock('../src/components/CharacterConfirmation', () => ({ default: ({ onSaved, onClose }: { onSaved: () => Promise<void>; onClose: () => void }) =>
  <div role="dialog"><button onClick={() => { void onSaved().catch(() => undefined); }}>确认示例</button><button onClick={onClose}>返回档案</button></div> }));
const asset: Asset = { id: 'episode', title: '第一集', kind: 'animation', duration_ms: 10000, width: 640, height: 360,
  subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '', status: 'ready' };
const library = (revision: number): CharacterLibrary => ({ scope_id: 'work', revision, assets: [asset], profiles: [
  { id: 'c1', name: '小夏', aliases: [], description: '', notes: '', appearance_count: revision - 1, appearances: [] },
] });
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((done) => { resolve = done; }); return { promise, resolve }; }
function setup(handler: (count: number) => CharacterLibrary | Promise<CharacterLibrary>, jobs: Job[] = []) {
  let reads = 0;
  const fetch = vi.fn(async (_url: string, _options?: RequestInit) => new Response(JSON.stringify(await handler(++reads))));
  vi.stubGlobal('fetch', fetch);
  const onSaved = vi.fn().mockResolvedValue(undefined);
  const context: AppContextValue = { bootstrap: { library_path: '/fixture', settings: { defaults: DEFAULTS, bindings: {} }, provider_profiles: [], stats: {}, tools: {}, version: 'test' },
    assets: [asset], jobs, refresh: vi.fn(), refreshJobs: vi.fn(), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
  const view = render(<AppContext.Provider value={context}><CharacterProfiles asset={asset} onSeek={vi.fn()} onSaved={onSaved} /></AppContext.Provider>);
  return { fetch, onSaved, view };
}
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Character catalog refresh after confirmation', () => {
  it('defers catalog and parent refresh until close, combining two confirmations into one read', async () => {
    const fresh = deferred<CharacterLibrary>();
    const { fetch, onSaved } = setup((count) => count === 1 ? library(1) : fresh.promise);
    fireEvent.click(await screen.findByRole('button', { name: '开始角色确认' }));
    fireEvent.click(screen.getByRole('button', { name: '确认示例' }));
    fireEvent.click(screen.getByRole('button', { name: '确认示例' }));
    expect(fetch).toHaveBeenCalledOnce();
    expect(onSaved).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: '返回档案' }));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(fetch).toHaveBeenCalledTimes(2); expect(onSaved).toHaveBeenCalledOnce();
    expect(screen.getByText(/修订 1/)).toBeTruthy();
    await act(async () => { fresh.resolve(library(3)); });
    expect(screen.getByText(/修订 3/)).toBeTruthy();
    expect(screen.getByText('2 次出场')).toBeTruthy();
    expect(fetch).toHaveBeenCalledTimes(2); expect(onSaved).toHaveBeenCalledOnce();
  });

  it('pauses analysis catalog polling while confirming and resumes it after closing', async () => {
    vi.useFakeTimers();
    let mounted!: ReturnType<typeof setup>;
    await act(async () => { mounted = setup((count) => library(count), [{ id: 'job', type: 'analysis', status: 'running',
      progress: 0.5, completed: 1, total: 2, asset_id: asset.id, created_at: '' }]); });
    fireEvent.click(screen.getByRole('button', { name: '开始角色确认' }));
    fireEvent.click(screen.getByRole('button', { name: '确认示例' }));
    fireEvent.click(screen.getByRole('button', { name: '确认示例' }));
    await act(async () => { await vi.advanceTimersByTimeAsync(30000); });
    expect(mounted.fetch).toHaveBeenCalledOnce(); expect(mounted.onSaved).not.toHaveBeenCalled();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '返回档案' })); });
    expect(mounted.fetch).toHaveBeenCalledTimes(2); expect(mounted.onSaved).toHaveBeenCalledOnce();
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(mounted.fetch).toHaveBeenCalledTimes(3);
  });

  it('keeps confirmation saved when outer refresh fails and retries only the refresh', async () => {
    const { fetch, onSaved } = setup((count) => library(count));
    onSaved.mockRejectedValueOnce(new Error('详情暂不可用'));
    fireEvent.click(await screen.findByRole('button', { name: '开始角色确认' }));
    fireEvent.click(screen.getByRole('button', { name: '确认示例' }));
    fireEvent.click(screen.getByRole('button', { name: '返回档案' }));
    expect(await screen.findByText('角色确认已保存，但档案或详情刷新失败：详情暂不可用')).toBeTruthy();
    expect(screen.getByText(/修订 2/)).toBeTruthy();
    expect(screen.queryByRole('dialog')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: '重试刷新档案与详情' }));
    await waitFor(() => expect(screen.getByText(/修订 3/)).toBeTruthy());
    expect(screen.queryByText(/角色确认已保存，但档案或详情刷新失败/)).toBeNull();
    expect(fetch).toHaveBeenCalledTimes(3); expect(onSaved).toHaveBeenCalledTimes(2);
  });

  it('does not reload unchanged profiles after closing without a confirmation', async () => {
    const { fetch, onSaved } = setup(() => library(1));
    fireEvent.click(await screen.findByRole('button', { name: '开始角色确认' }));
    fireEvent.click(screen.getByRole('button', { name: '返回档案' }));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(fetch).toHaveBeenCalledOnce(); expect(onSaved).not.toHaveBeenCalled();
  });

  it('keeps a saved refresh running when the wizard closes and cancels it when the catalog unmounts', async () => {
    const pending = deferred<CharacterLibrary>();
    const { fetch, view, onSaved } = setup((count) => count === 1 ? library(1) : pending.promise);
    fireEvent.click(await screen.findByRole('button', { name: '开始角色确认' }));
    fireEvent.click(screen.getByRole('button', { name: '确认示例' }));
    fireEvent.click(screen.getByRole('button', { name: '返回档案' }));
    const signal = fetch.mock.calls[1][1]?.signal;
    expect(signal?.aborted).toBe(false); expect(onSaved).toHaveBeenCalledOnce();
    view.unmount(); expect(signal?.aborted).toBe(true);
    await act(async () => { pending.resolve(library(2)); });
    expect(fetch).toHaveBeenCalledTimes(2); expect(onSaved).toHaveBeenCalledOnce();
  });
});
