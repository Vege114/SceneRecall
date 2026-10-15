import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AnimeCredits from '../src/components/AnimeCredits';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('reads saved credits locally and fetches source only on request, keeping unspecified scope separate', async () => {
  const fetch = vi.fn(async (_url: string, options?: RequestInit) => new Response(JSON.stringify(options?.method === 'POST' ? {
    status: 'completed', fetched_at: '2026-10-15T01:00:00Z', url: 'https://bgm.tv/subject/100/persons', warnings: [],
    rows: [
      { person_id: 1, name: '本集分镜', position: '分镜', episodes: '1,3', scope: 'episode', url: 'https://bgm.tv/person/1' },
      { person_id: 2, name: '作品监督', position: '监督', episodes: '', scope: 'subject', url: 'https://bgm.tv/person/2' },
    ],
  } : { status: 'missing', rows: [], warnings: [] }), { status: 200 }));
  vi.stubGlobal('fetch', fetch);
  render(<AnimeCredits assetId="episode-1" />);
  await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
  expect(fetch.mock.calls[0][1]?.method).not.toBe('POST');
  await userEvent.click(screen.getByRole('button', { name: '读取演职员资料' }));
  await screen.findByText('本集分镜');
  expect(screen.getByText('本集演职员 · 1 条')).toBeTruthy();
  expect(screen.getByText('作品级人员，来源未注明参与集数 · 1 条')).toBeTruthy();
  expect(fetch).toHaveBeenCalledTimes(2);
});

it('preserves the saved table and shows a source error after failed refresh', async () => {
  vi.stubGlobal('fetch', vi.fn(async (_url: string, options?: RequestInit) => options?.method === 'POST'
    ? new Response(JSON.stringify({ detail: '来源网络失败' }), { status: 400 })
    : new Response(JSON.stringify({ status: 'completed', rows: [{ person_id: 1, name: '现成画师', position: '分镜', episodes: '1', scope: 'episode', url: 'https://bgm.tv/person/1' }], warnings: [] }), { status: 200 })));
  render(<AnimeCredits assetId="episode-1" />);
  await screen.findByText('现成画师');
  await userEvent.click(screen.getByRole('button', { name: '更新演职员资料' }));
  await screen.findByText('来源网络失败');
  expect(screen.getByText('现成画师')).toBeTruthy();
});
