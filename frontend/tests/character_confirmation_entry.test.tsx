import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type Job } from '../src/types';
import type { CharacterConfirmationResult } from '../src/identityTypes';
import Jobs from '../src/components/Jobs';

const first: Asset = { id: 'first', work_id: 'work', title: '第一集', kind: 'animation', duration_ms: 120000,
  width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '', status: 'ready' };
const second: Asset = { ...first, id: 'second', title: '第二集' };
const waitingJob = (asset: Asset): Job => ({ id: `job-${asset.id}`, asset_id: asset.id, type: 'analysis', status: 'waiting_input',
  progress: 0, completed: 0, total: 1, created_at: '', stage: 'character_confirmation', analysis_strategy: 'scene-chunks-v1',
  input_required: { kind: 'character_confirmation', asset_id: asset.id },
  workflow_progress: { keyframes: { status: 'completed', completed: 4, total: 4 },
    character_confirmation: { status: 'waiting_input', completed: 0, total: 1 }, analysis: { status: 'pending' } } });
const queue = (asset = first): CharacterConfirmationResult => ({ asset_id: asset.id, scope_id: 'work',
  summary: { total: 1, pending: 1, confirmed: 0, skipped: 0, candidates: 0 }, sampling: null,
  roles: [{ character_id: 'character', profile: { id: 'character', name: '小夏', aliases: [], description: '', notes: '' },
    status: 'pending', trusted_reference_count: 0, candidates: [], available_count: 0 }] });

function setup(jobs = [waitingJob(first)]) {
  const context: AppContextValue = { bootstrap: { library_path: '/tmp', settings: { defaults: DEFAULTS, bindings: {} },
    provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' },
    assets: [first, second], jobs, refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined),
    notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
  render(<AppContext.Provider value={context}><Jobs /></AppContext.Provider>);
  return context;
}
function mockApi(handler: (url: string, init?: RequestInit) => unknown) {
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    const response = handler(url, init);
    return response instanceof Response ? response : new Response(JSON.stringify(response));
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.clearAllMocks(); });

it('keeps keyframes and role decisions in the created task and requires a saved decision before continuing', async () => {
  const state = queue();
  const fetch = mockApi((url, init) => {
    if (url.endsWith('/decision')) {
      expect(JSON.parse(init!.body as string)).toEqual({ action: 'skip' });
      state.summary.pending = 0; state.summary.skipped = 1; state.roles[0].status = 'skipped';
    }
    return url.endsWith('/continue') ? { id: 'job-first', status: 'queued' } : state;
  });
  const app = setup();
  fireEvent.click(screen.getByRole('button', { name: '进行中', exact: true }));
  expect(screen.getByText('等待角色确认')).toBeTruthy();
  const flow = within(screen.getByRole('list', { name: '完整任务流程' }));
  expect(within(flow.getByLabelText('关键帧划分流程阶段')).getByText('已完成')).toBeTruthy();
  expect(within(flow.getByLabelText('角色确认流程阶段')).getByText('等待人工确认')).toBeTruthy();
  expect(within(flow.getByLabelText('分析与索引流程阶段')).getByText('待开始')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '确认角色并继续' }));
  await screen.findByText('暂时没有合适候选');
  expect(screen.getByRole('dialog', { name: '任务中的角色确认' })).toBeTruthy();
  expect((screen.getByRole('button', { name: '继续任务' }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: '暂时跳过这个角色' }));
  await screen.findByText('这个角色已暂时跳过');
  await waitFor(() => expect(app.refresh).toHaveBeenCalledOnce());
  expect(fetch.mock.calls.some(([url]) => url.endsWith('/continue'))).toBe(false);
  fireEvent.click(screen.getByRole('button', { name: '继续任务' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  const continuation = fetch.mock.calls.find(([url]) => url === '/api/jobs/job-first/continue');
  expect(continuation?.[1]?.method).toBe('POST');
  expect(JSON.parse(continuation![1]!.body as string)).toEqual({});
  expect(app.refreshJobs).toHaveBeenCalledWith(true);
});

it('closing role confirmation preserves the waiting task without implicitly skipping or continuing', async () => {
  const fetch = mockApi(() => queue()); setup();
  fireEvent.click(screen.getByRole('button', { name: '确认角色并继续' }));
  await screen.findByText('暂时没有合适候选');
  fireEvent.click(screen.getByRole('button', { name: '保存进度并返回' }));
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(screen.getByRole('button', { name: '确认角色并继续' })).toBeTruthy();
  expect(fetch.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
  fireEvent.click(screen.getByRole('button', { name: '暂停', exact: true }));
  await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/jobs/job-first/pause')).toBe(true));
});

it('keeps batch confirmation per episode and includes waiting tasks in batch controls', async () => {
  const fetch = mockApi((url) => url.includes('/assets/second/') ? queue(second) : queue(first));
  setup([waitingJob(first), waitingJob(second)].map((job) => ({ ...job, batch_id: 'batch' })));
  const batch = within(screen.getByLabelText('批量分析进度'));
  expect(batch.getByText(/等待角色确认 2 集/)).toBeTruthy();
  expect(batch.getByRole('button', { name: '暂停整批' })).toBeTruthy();
  expect(batch.getByRole('button', { name: '取消整批' })).toBeTruthy();
  fireEvent.click(screen.getAllByRole('button', { name: '确认角色并继续' })[1]);
  await screen.findByText('暂时没有合适候选');
  expect(fetch.mock.calls[0][0]).toBe('/api/assets/second/identity/confirmation');
  fireEvent.click(screen.getByRole('button', { name: '保存进度并返回' }));
  fireEvent.click(batch.getByRole('button', { name: '确认下一集角色' }));
  await screen.findByText('暂时没有合适候选');
  expect(fetch.mock.calls[1][0]).toBe('/api/assets/first/identity/confirmation');
  expect(fetch.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
});

it('keeps continuation errors visible and allows rereading changed shared character decisions', async () => {
  const state = queue(); state.summary.pending = 0; state.summary.skipped = 1; state.roles[0].status = 'skipped';
  const fetch = mockApi((url) => url.endsWith('/continue') ? new Response(JSON.stringify({ detail: '仍有角色待确认，请重新读取确认进度' }), { status: 409 }) : state);
  setup();
  fireEvent.click(screen.getByRole('button', { name: '确认角色并继续' }));
  await screen.findByText('这个角色已暂时跳过');
  fireEvent.click(screen.getByRole('button', { name: '继续任务' }));
  await screen.findByText('仍有角色待确认，请重新读取确认进度');
  expect(screen.getByRole('dialog')).toBeTruthy();
  state.summary.pending = 1; state.summary.skipped = 0; state.roles[0].status = 'pending';
  fireEvent.click(screen.getByRole('button', { name: '重新读取确认进度' }));
  await screen.findByText('暂时没有合适候选');
  expect((screen.getByRole('button', { name: '继续任务' }) as HTMLButtonElement).disabled).toBe(true);
  expect(fetch.mock.calls.filter(([url]) => url.endsWith('/continue'))).toHaveLength(1);
});
