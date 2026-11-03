import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AudioIdentity from '../src/components/AudioIdentity';
import type { Asset } from '../src/types';

const asset: Asset = { id: 'episode', work_id: 'work', title: '测试动画', kind: 'animation', duration_ms: 120000,
  width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '', status: 'ready' };

function configure(effective?: 'retain' | 'discard') {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    let result: unknown = {};
    if (url.endsWith('/identity/settings')) {
      result = options?.method === 'PUT' ? JSON.parse(options.body as string)
        : { enabled: true, model: 'ccip-caformer-5_fp32', top_k: 5, calibration: null,
          ...(effective ? { crop_retention: 'default', crop_retention_effective: effective } : {}) };
    } else if (url.endsWith('/audio/settings')) {
      result = { settings: { enabled: false }, availability: { reasons: [], diarization: {}, embedding: {}, asr: {}, sync: {} } };
    } else if (url.endsWith('/audio/tracks')) result = { tracks: [] };
    else if (url.endsWith('/audio/records')) result = { status: 'disabled', records: [] };
    else if (url.endsWith('/identity/reviews')) result = { status: 'completed', regions: [] };
    else if (url.endsWith('/characters')) result = { profiles: [] };
    return new Response(JSON.stringify(result), { status: 200 });
  });
  vi.stubGlobal('fetch', fetch);
  render(<AudioIdentity asset={asset} onSeek={vi.fn()} onSaved={vi.fn()} />);
  return fetch;
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it('saves explicit discard without forwarding read-only effective policy or starting recognition', async () => {
  const user = userEvent.setup(); const fetch = configure('retain');
  await user.selectOptions(await screen.findByLabelText(/^普通人物 PNG 保留策略/), 'discard');
  await user.click(screen.getByRole('button', { name: '保存视觉身份配置' }));
  await screen.findByText('视觉身份配置已保存。');
  const call = fetch.mock.calls.find(([url, options]) => url.endsWith('/identity/settings') && options?.method === 'PUT');
  expect(JSON.parse(call![1]!.body as string)).toEqual({ enabled: true, model: 'ccip-caformer-5_fp32', top_k: 5,
    calibration: null, crop_retention: 'discard' });
  expect(fetch.mock.calls.some(([url]) => url.endsWith('/identity/run'))).toBe(false);
});

it('keeps an older configuration on the deployment default until explicitly changed', async () => {
  const user = userEvent.setup(); const fetch = configure();
  const selector = await screen.findByLabelText(/^普通人物 PNG 保留策略/) as HTMLSelectElement;
  expect(selector.value).toBe('default');
  await user.click(screen.getByRole('button', { name: '保存视觉身份配置' }));
  await waitFor(() => expect(fetch.mock.calls.some(([url, options]) => url.endsWith('/identity/settings') && options?.method === 'PUT')).toBe(true));
  const call = fetch.mock.calls.find(([url, options]) => url.endsWith('/identity/settings') && options?.method === 'PUT');
  expect(JSON.parse(call![1]!.body as string).crop_retention).toBe('default');
});
