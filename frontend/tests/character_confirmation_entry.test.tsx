import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset } from '../src/types';
import JobModal from '../src/components/JobModal';

vi.mock('../src/components/CharacterConfirmation', () => ({
  default: ({ asset, onClose, onSaved }: { asset: Asset; onClose: () => void; onSaved: () => Promise<void> }) =>
    <div role="dialog" aria-label="角色确认"><span>{asset.id}</span><button onClick={onClose}>返回分析设置</button><button onClick={() => void onSaved()}>保存示例</button></div>,
}));

const first: Asset = { id: 'first', work_id: 'work', title: '第一集', kind: 'animation', duration_ms: 120000,
  width: 640, height: 360, subtitle_mode: 'external', source_available: false, source_changed: false, created_at: '', status: 'ready' };
const second: Asset = { ...first, id: 'second', title: '第二集' };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.clearAllMocks(); });

it('opens existing role confirmation for a selected episode without submitting jobs and preserves analysis settings', async () => {
  const fetch = vi.fn(async () => new Response(JSON.stringify({})));
  vi.stubGlobal('fetch', fetch);
  const refresh = vi.fn().mockResolvedValue(undefined);
  const context: AppContextValue = { bootstrap: { library_path: '/tmp', settings: { defaults: DEFAULTS, bindings: {} },
    provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' },
  assets: [first, second], jobs: [], refresh, refreshJobs: vi.fn(), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
  render(<AppContext.Provider value={context}><JobModal assets={[first, second]} onClose={vi.fn()} /></AppContext.Provider>);
  fireEvent.change(screen.getByLabelText('处理策略'), { target: { value: 'retrieval-first-v1' } });
  fireEvent.change(screen.getByLabelText(/开始时间（秒）/), { target: { value: '12' } });
  fireEvent.change(screen.getByLabelText(/角色确认的起始剧集/), { target: { value: 'second' } });
  fireEvent.click(screen.getByRole('button', { name: '开始角色确认' }));
  expect(screen.getAllByRole('dialog')).toHaveLength(1);
  expect(screen.getByText('second')).toBeTruthy();
  expect(fetch).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: '保存示例' }));
  await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
  fireEvent.click(screen.getByRole('button', { name: '返回分析设置' }));
  expect((screen.getByLabelText('处理策略') as HTMLSelectElement).value).toBe('retrieval-first-v1');
  expect((screen.getByLabelText(/开始时间（秒）/) as HTMLInputElement).value).toBe('12');
  expect(screen.getAllByRole('dialog')).toHaveLength(1);
  expect(fetch).not.toHaveBeenCalled();
});
