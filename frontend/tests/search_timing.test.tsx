import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS } from '../src/types';
import SearchPage from '../src/components/Search';
import { api } from '../src/api';
import { searchClickStart, searchTransportTiming, type SearchTiming } from '../src/searchTiming';

const trace: SearchTiming = { schema: 'search-timing-v1', request_id: 'request-test', total_ms: 10,
  boundary: 'test', attributes: { result_count: 0 }, spans: [
    { id: 0, parent_id: null, name: 'http.application', start_ms: 0, end_ms: 10, elapsed_ms: 10, status: 'ok', attributes: {} },
    { id: 1, parent_id: 0, name: 'recall.parallel', start_ms: 1, end_ms: 8, elapsed_ms: 7, status: 'ok', attributes: {} },
  ] };
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('search timing boundaries', () => {
  it('normalizes event clocks and labels the fallback instead of mixing epoch and monotonic timestamps', () => {
    expect(searchClickStart(90, 100, 1000000)).toEqual({ clickedAt: 90, source: 'event_timestamp' });
    expect(searchClickStart(1000090, 100, 1000000)).toEqual({ clickedAt: 90, source: 'event_timestamp' });
    expect(searchClickStart(0, 100, 1000000)).toEqual({ clickedAt: 100, source: 'click_handler' });
    expect(searchClickStart(110, 100, 1000000)).toEqual({ clickedAt: 100, source: 'click_handler' });
  });

  it('keeps browser duration residual separate from server spans and tolerates old servers', () => {
    const timed = searchTransportTiming(new Response('{}', { headers: { 'X-Search-Timing': JSON.stringify(trace) } }), 100, 118, 123);
    expect(timed.search_timing?.request_id).toBe('request-test');
    expect(timed.client_timing.request_to_json_ms).toBe(23);
    expect(timed.client_timing.client_server_remainder_ms).toBe(13);
    expect(timed.client_timing.headers_to_json_ms).toBe(5);
    expect(searchTransportTiming(new Response('{}'), 100, 118, 123).client_timing.client_server_remainder_ms).toBeNull();
    expect(searchTransportTiming(new Response('{}', { headers: { 'X-Search-Timing': 'broken' } }), 100, 118, 123).search_timing).toBeUndefined();
  });

  it('reads only search response timing headers without changing other endpoint data', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ results: [], degraded: [] }), {
      headers: { 'X-Search-Timing': JSON.stringify(trace) },
    })));
    const result = await api<{ search_timing?: SearchTiming }>('/search', { method: 'POST', body: '{}' });
    expect(result.search_timing?.request_id).toBe('request-test');
    const other = await api<{ search_timing?: SearchTiming }>('/assets');
    expect(other.search_timing).toBeUndefined();
  });

  it('measures submit to committed results and shows a privacy-safe machine record', async () => {
    const value = { bootstrap: { library_path: '/fixtures', settings: { defaults: DEFAULTS, bindings: {} },
      provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' },
      assets: [], jobs: [], refresh: vi.fn(), refreshJobs: vi.fn(), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn(),
    } as unknown as AppContextValue;
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ results: [], degraded: [] }), {
      headers: { 'X-Search-Timing': JSON.stringify(trace) },
    })));
    render(<AppContext.Provider value={value}><SearchPage /></AppContext.Provider>);
    const user = userEvent.setup();
    await user.type(screen.getByRole('textbox', { name: '用自然语言搜索镜头或台词' }), 'PRIVATE QUERY');
    await user.click(screen.getByRole('button', { name: '检索' }));
    await waitFor(() => expect(screen.getByLabelText('搜索耗时机器记录')).toBeTruthy());
    const record = JSON.parse(screen.getByLabelText('搜索耗时机器记录').textContent || '{}');
    expect(record.server.request_id).toBe('request-test');
    expect(record.presentation.submit_to_result_commit_ms).toBeGreaterThanOrEqual(0);
    expect(record.presentation.trigger).toBe('pointer_click');
    expect(record.presentation.click_to_submit_ms).toBeGreaterThanOrEqual(0);
    expect(record.presentation.click_to_result_commit_ms).toBeCloseTo(
      record.presentation.click_to_submit_ms + record.presentation.submit_to_result_commit_ms, 5);
    expect(record.presentation.debounce_ms).toBe(0);
    expect(JSON.stringify(record)).not.toContain('PRIVATE QUERY');
    expect(screen.getByText(/请求到 JSON 的剩余项/)).toBeTruthy();
    expect(screen.getByText('点击到结果 DOM 可用')).toBeTruthy();

    // Enter on the focused search button generates a synthetic click (detail=0).
    // Its request must not reuse the previous pointer-click measurement.
    await user.keyboard('{Enter}');
    await waitFor(() => {
      const keyboardRecord = JSON.parse(screen.getByLabelText('搜索耗时机器记录').textContent || '{}');
      expect(keyboardRecord.presentation.trigger).toBe('form_submit');
      expect(keyboardRecord.presentation.click_to_submit_ms).toBeUndefined();
      expect(keyboardRecord.presentation.click_to_result_commit_ms).toBeUndefined();
    });
    expect(screen.queryByText('点击到结果 DOM 可用')).toBeNull();
  });
});
