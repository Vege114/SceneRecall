import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContext, type AppContextValue } from '../src/context';
import SearchPage, { ResultCard } from '../src/components/Search';
import { DEFAULTS, type SearchResponse, type SearchResult } from '../src/types';

const baseResult: SearchResult = {
  id: 'one', record_id: 'one', asset_id: 'episode', work_id: 'work', title: '测试片段',
  kind: 'subtitle', start_ms: 1000, end_ms: 2000, text: '测试台词',
  source: 'subtitle', review_status: 'reviewed', score: 0.0164,
};
const context = { bootstrap: { library_path: '/fixtures', settings: { defaults: DEFAULTS, bindings: {} },
  provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' },
  assets: [], jobs: [], refresh: vi.fn(), refreshJobs: vi.fn(), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn(),
} as unknown as AppContextValue;
const selection = (threshold: number, returned: number): SearchResponse['selection'] => ({
  mode: 'threshold', min_match_score: threshold, score_basis: 'relative_rrf_v1', candidate_count: 30, returned_count: returned,
});
const mount = () => render(<AppContext.Provider value={context}><SearchPage /></AppContext.Provider>);
const mockResponse = (response: SearchResponse) => {
  const fetch = vi.fn(async () => new Response(JSON.stringify(response), { headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetch);
  return fetch;
};
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('relative search match threshold', () => {
  it('defaults to 50 and renders every returned result in server order without a ten-result limit', async () => {
    const results = Array.from({ length: 12 }, (_, index) => ({ ...baseResult,
      id: `result-${index}`, record_id: `result-${index}`, title: `片段 ${index + 1}`,
      score: index * 0.001, match_score: 100 - index, match_score_basis: 'relative_rrf_v1' as const,
    }));
    const fetch = mockResponse({ results, degraded: [], selection: selection(50, results.length) });
    const user = userEvent.setup(); mount();
    const threshold = screen.getByRole('spinbutton', { name: '最低相对匹配度（%）' }) as HTMLInputElement;
    expect(threshold.value).toBe('50');
    expect(threshold.min).toBe('0'); expect(threshold.max).toBe('100');
    expect(screen.getByText(/本次最高候选为100%，随检索范围变化，不代表正确概率/)).toBeTruthy();
    await user.type(screen.getByRole('textbox', { name: '用自然语言搜索镜头或台词' }), '测试台词');
    await user.click(screen.getByRole('button', { name: /^检索/ }));
    await screen.findByText(/已应用：相对匹配度严格高于 50%/);
    const request = JSON.parse((fetch.mock.calls[0] as unknown as [string, RequestInit])[1].body as string);
    expect(request.min_match_score).toBe(50);
    expect(request).not.toHaveProperty('limit');
    expect(screen.getAllByRole('article')).toHaveLength(12);
    expect(screen.getAllByText(/^相对匹配度 \d+\.\d%$/).map((element) => element.textContent))
      .toEqual(results.map((result) => `相对匹配度 ${result.match_score.toFixed(1)}%`));
    expect(screen.getByText(/按相对匹配度从高到低排列，全部 12 个片段/)).toBeTruthy();
    expect(screen.queryByText(/人物证据不足会降低分数/)).toBeNull();

    // Editing the next request must not relabel results from the previous one.
    await user.clear(threshold); await user.type(threshold, '90');
    expect(screen.getByText(/已应用：相对匹配度严格高于 50%/)).toBeTruthy();
    expect(fetch).toHaveBeenCalledOnce();
  });

  it('explains character-grounded scores can stay below 100 without presenting them as confirmed identity', async () => {
    mockResponse({ results: [{ ...baseResult, match_score: 82.5, character_match: {
      status: 'unverified', required_count: 2, confirmed_count: 0, candidate_count: 0,
    } }], degraded: [], selection: { ...selection(75, 1)!, score_basis: 'character_grounded_rrf_v1' } });
    const user = userEvent.setup(); mount();
    const threshold = screen.getByRole('spinbutton', { name: '最低相对匹配度（%）' });
    await user.clear(threshold); await user.type(threshold, '75');
    await user.type(screen.getByRole('textbox', { name: '用自然语言搜索镜头或台词' }), '华恋和小光站在东京塔上');
    await user.click(screen.getByRole('button', { name: /^检索/ }));
    const badge = await screen.findByText('相对匹配度 82.5%');
    const explanation = '人物证据不足会降低分数，最高结果也可能低于100%；分数不代表正确概率。';
    expect(badge.getAttribute('title')).toBe(explanation);
    expect(screen.getAllByText(/人物证据不足会降低分数，最高结果也可能低于100%；分数不代表正确概率/)).toHaveLength(2);
    expect(screen.getByText(/已应用：相对匹配度严格高于 75%/).textContent).toContain(explanation);
    expect(screen.getByText(/仅显示严格高于此值的片段/).textContent).toContain('不代表正确概率');
    expect(screen.queryByText(/本次最高候选为100%/)).toBeNull();
    expect(screen.getByText('命中帧的指定人物尚未证实（可能漏检）')).toBeTruthy();
  });

  it.each(['0', '66.5', '100'])('sends an editable finite threshold %s and explains an empty threshold result', async (value) => {
    const fetch = mockResponse({ results: [], degraded: [], selection: selection(Number(value), 0) });
    const user = userEvent.setup(); mount();
    const threshold = screen.getByRole('spinbutton', { name: '最低相对匹配度（%）' });
    await user.clear(threshold); await user.type(threshold, value);
    await user.type(screen.getByRole('textbox', { name: '用自然语言搜索镜头或台词' }), '画面');
    await user.click(screen.getByRole('button', { name: /^检索/ }));
    await screen.findByText('没有高于阈值的片段');
    const request = JSON.parse((fetch.mock.calls[0] as unknown as [string, RequestInit])[1].body as string);
    expect(request.min_match_score).toBe(Number(value));
    expect(screen.getByText(/降低相对匹配度阈值/)).toBeTruthy();
  });

  it.each(['', '-1', '101'])('rejects invalid threshold %s even if form validity is bypassed', async (value) => {
    const fetch = mockResponse({ results: [], degraded: [] }); mount();
    const query = screen.getByRole('textbox', { name: '用自然语言搜索镜头或台词' });
    fireEvent.change(query, { target: { value: '画面' } });
    fireEvent.change(screen.getByRole('spinbutton', { name: '最低相对匹配度（%）' }), { target: { value } });
    fireEvent.submit(query.closest('form')!);
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('0 至 100 的有效数字'));
    expect(fetch).not.toHaveBeenCalled();
  });

  it('never presents the legacy raw score or a missing score as a percentage', () => {
    const { rerender } = render(<AppContext.Provider value={context}><ResultCard result={baseResult} /></AppContext.Provider>);
    expect(screen.queryByText(/相对匹配度/)).toBeNull();
    rerender(<AppContext.Provider value={context}><ResultCard result={{ ...baseResult, match_score: 81.234 }} /></AppContext.Provider>);
    expect(screen.getByText('相对匹配度 81.2%')).toBeTruthy();
    rerender(<AppContext.Provider value={context}><ResultCard result={{ ...baseResult, match_score: Number.NaN }} /></AppContext.Provider>);
    expect(screen.queryByText(/相对匹配度/)).toBeNull();
  });
});
