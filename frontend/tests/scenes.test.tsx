import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type Estimate, type Job, type Profile, type SceneChunk, type SearchResult } from '../src/types';
import JobModal from '../src/components/JobModal';
import SearchPage, { ResultCard } from '../src/components/Search';
import Settings from '../src/components/Settings';
import Jobs from '../src/components/Jobs';
import AssetDetail from '../src/components/AssetDetail';

const asset: Asset = { id: 'episode-one', work_id: 'project-one', title: '雨中的约定', kind: 'animation', duration_ms: 120000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const profile: Profile = { id: 'shared', name: '共享模型', base_url: 'https://example.invalid/v1', model: 'test-model', capabilities: ['preparation', 'vision'], secret_mode: 'none' };
const scene: SceneChunk = { id: 'stable-scene', chunk_id: 'chunk-one', start_ms: 1000, end_ms: 10000, summary: '两个人在站台交谈。', understanding_level: 'initial', needs_deep: true, evidence_frame_ids: ['original-one'], visual_facts: [{ text: '两人面对面站立。', frame_ids: ['original-one'] }], dialogue_facts: [{ text: '台词提到有人离开。', cue_ids: ['cue-one'] }], inferences: [{ text: '可能在讨论告别。', cue_ids: ['cue-one'], frame_ids: ['original-one'] }] };
const result: SearchResult = { ...scene, id: 'indexed-scene', record_id: scene.id, asset_id: asset.id, work_id: asset.work_id, title: asset.title, kind: 'scene', text: scene.summary, source: 'model', review_status: 'needs_review' };
const estimate: Estimate = { duration_ms: 9000, estimated_frames: 15, estimated_requests: 4, estimated_requests_min: 2, estimated_requests_max: 4, cost_estimate: null, warnings: [], stage_estimates: { transcript: { requests_min: 1, requests_max: 1 }, preparation: { requests_min: 1, requests_max: 1 }, vision: { requests_min: 0, requests_max: 2 }, context_refresh: { requests_min: 0, requests_max: 2 } } };
function context(): AppContextValue {
  return { bootstrap: { library_path: '/tmp/scenes-ui', settings: { defaults: DEFAULTS, bindings: { preparation: profile.id, vision: profile.id } }, provider_profiles: [profile], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test' }, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function mount(ui: ReactNode, value = context()) { return render(<AppContext.Provider value={value}>{ui}</AppContext.Provider>); }
function mockFetch(handler: (url: string, options?: RequestInit) => unknown) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => { const value = handler(url, options); return value instanceof Response ? value : new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } }); });
  vi.stubGlobal('fetch', fetch); return fetch;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Scene preparation and budgets', () => {
  it.each(['openai_compatible', 'codex_cli'] as const)('assigns preparation and vision to the same %s connection', async (providerType) => {
    const user = userEvent.setup(); const value = context();
    value.bootstrap.provider_profiles = [{ ...profile, provider_type: providerType }];
    value.bootstrap.settings.bindings = {};
    const fetch = mockFetch((_url, options) => JSON.parse(options!.body as string));
    mount(<Settings />, value);
    await user.selectOptions(screen.getByLabelText('分配分块与简要索引模型'), profile.id);
    await user.selectOptions(screen.getByLabelText('分配画面理解模型'), profile.id);
    await user.click(screen.getByRole('button', { name: '保存分配' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string).bindings).toEqual({ preparation: profile.id, vision: profile.id });
  });

  it('requires preparation before starting the new strategy and includes staged request ranges', async () => {
    const user = userEvent.setup(); const value = context(); delete value.bootstrap.settings.bindings.preparation;
    const fetch = mockFetch((url) => url === '/api/keyframe-plans' ? { id: 'reused-plan', status: 'completed', frames: [], frame_count: 112 } : estimate);
    mount(<JobModal asset={asset} onClose={vi.fn()} />, value);
    await user.click(screen.getByRole('button', { name: '扫描关键帧' }));
    await screen.findByRole('button', { name: '重新扫描关键帧' });
    await user.click(screen.getByRole('button', { name: '估算用量' }));
    await screen.findByRole('button', { name: '重新估算' });
    expect((screen.getByRole('button', { name: /开始分析/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByLabelText('分阶段请求估算').textContent).toContain('0–2 次');
    expect(within(screen.getByLabelText('分阶段请求估算')).getByText('更新连续理解')).toBeTruthy();
    expect(screen.getByText(/预计 2 批视觉分块/)).toBeTruthy();
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string)).toMatchObject({ analysis_strategy: 'scene-chunks-v1', keyframe_plan_id: 'reused-plan' });
  });

  it('disables dollar budgets when the preparation connection uses Codex', async () => {
    const value = context(); value.bootstrap.provider_profiles.push({ ...profile, id: 'codex-prep', provider_type: 'codex_cli', model: '', capabilities: ['preparation'] }); value.bootstrap.settings.bindings.preparation = 'codex-prep'; value.bootstrap.settings.defaults = { ...DEFAULTS, max_cost: 10 };
    const fetch = mockFetch(() => ({})); mount(<JobModal asset={asset} onClose={vi.fn()} />, value);
    expect((screen.getByLabelText(/费用上限（USD/) as HTMLInputElement).disabled).toBe(true);
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe('Explicit scene deepening and evidence', () => {
  it('searches existing scenes without model analysis, then requires a current estimate before budgeted deepening', async () => {
    const user = userEvent.setup(); const fetch = mockFetch((url) => url === '/api/search' ? { results: [result], degraded: [] } : url.endsWith('/estimate') ? estimate : { id: 'deep-job' });
    mount(<SearchPage />);
    await user.type(screen.getByRole('textbox', { name: /用自然语言搜索/ }), '站台告别');
    await user.click(screen.getByRole('button', { name: /^检索/ }));
    await screen.findByText(scene.summary);
    expect(screen.getByText('初步理解')).toBeTruthy();
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/search']);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    const dialog = screen.getByRole('dialog');
    expect(fetch).toHaveBeenCalledOnce();
    const start = within(dialog).getByRole('button', { name: '开始深入分析' }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    await user.click(within(dialog).getByRole('button', { name: '估算用量' }));
    await waitFor(() => expect(start.disabled).toBe(false));
    const maxRequests = within(dialog).getByLabelText(/最多模型请求数/);
    await user.clear(maxRequests); await user.type(maxRequests, '6');
    expect(start.disabled).toBe(true);
    await user.click(within(dialog).getByRole('button', { name: '估算用量' }));
    await waitFor(() => expect(start.disabled).toBe(false));
    await user.click(start);
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetch.mock.calls.at(-1)?.[0]).toBe('/api/assets/episode-one/chunks/chunk-one/deepen');
    expect(JSON.parse(fetch.mock.calls.at(-1)![1]!.body as string)).toEqual({ max_requests: 6, max_cost: null });
  });

  it('shows current scene facts separately from dialogue and inference and bookmarks its stable record', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => ({}));
    mount(<ResultCard result={{ ...result, understanding_level: 'deep', context_stale: true, stale_reasons: ['此前角色已修订'] }} />);
    expect(screen.getByText('深入理解')).toBeTruthy(); expect(screen.getByText('上下文待更新')).toBeTruthy(); expect(fetch).not.toHaveBeenCalled();
    await user.click(screen.getByText('查看事实与来源'));
    expect(screen.getByText('画面事实')).toBeTruthy(); expect(screen.getByText('台词事实')).toBeTruthy(); expect(screen.getByText('综合推断')).toBeTruthy();
    expect(screen.getAllByRole('link', { name: '原帧 1' })[0].getAttribute('href')).toBe('/api/assets/episode-one/frames/original-one');
    expect(screen.getAllByRole('link', { name: '查看台词 1' })[0].getAttribute('href')).toBe('#/asset/episode-one?t=1000&cue=cue-one');
    expect(screen.queryByText('cue-one')).toBeNull();
    await user.click(screen.getByRole('button', { name: '收藏片段' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toEqual({ record_id: 'stable-scene', favorite: true });
  });

  it('renders scenes in asset details, resolves subtitle references and exposes manual deepening', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => ({ asset, scenes: [scene], observations: [], subtitles: [{ id: 'cue-one', start_ms: 1200, end_ms: 2200, text: '他已经走了。', source: 'external', language: 'zh', review_status: 'confirmed' }], annotations: [], runs: [] }));
    mount(<AssetDetail id={asset.id} />);
    await screen.findByText(scene.summary);
    expect(screen.getByRole('button', { name: '深入分析此片段' })).toBeTruthy();
    await user.click(screen.getByText('查看事实与来源'));
    expect(screen.getAllByText(/「他已经走了。」/).length).toBeGreaterThan(0);
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/assets/episode-one']);
  });

  it('shows representatives first and reveals 24 remaining original frames per click without duplicates', async () => {
    const user = userEvent.setup();
    const frameIds = Array.from({ length: 60 }, (_, index) => `frame-${index}`);
    const longScene = { ...scene, frame_ids: frameIds, evidence_frame_ids: frameIds, representative_frame_ids: ['frame-20', 'frame-0', 'frame-59', 'frame-20'] };
    const legacyIds = Array.from({ length: 10 }, (_, index) => `legacy-${index}`);
    mockFetch(() => ({ asset, scenes: [longScene], observations: [{ id: 'legacy-observation', summary: '保留的旧式画面记录', start_ms: 10000, end_ms: 11000, entities: [], events: [], uncertainties: [], evidence_frame_ids: legacyIds }], subtitles: [], annotations: [], runs: [] }));
    mount(<AssetDetail id={asset.id} />);
    const gallery = await screen.findByLabelText('场景原始关键帧');
    const images = () => within(gallery).getAllByRole('img');
    expect(images().length).toBeLessThanOrEqual(8);
    expect(images().map((image) => image.getAttribute('src'))).toEqual(['frame-20', 'frame-0', 'frame-59'].map((id) => `/api/assets/episode-one/frames/${id}`));
    expect(screen.getAllByAltText('模型使用的证据画面')).toHaveLength(10);
    await user.click(within(gallery).getByRole('button', { name: /显示更多原帧/ }));
    expect(images()).toHaveLength(27);
    await user.click(within(gallery).getByRole('button', { name: /显示更多原帧/ }));
    expect(images()).toHaveLength(51);
    await user.click(within(gallery).getByRole('button', { name: /显示更多原帧/ }));
    expect(images()).toHaveLength(60);
    expect(new Set(images().map((image) => image.getAttribute('src'))).size).toBe(60);
    expect(within(gallery).queryByRole('button', { name: /显示更多原帧/ })).toBeNull();
    await user.click(within(gallery).getAllByRole('button', { name: '放大证据画面' })[0]);
    const imageDialog = screen.getByRole('dialog', { name: '证据画面' });
    expect(within(imageDialog).getByRole('img').getAttribute('src')).toBe('/api/assets/episode-one/frames/frame-20');
  });

  it('opens a linked subtitle beyond the first page at its real time without additional requests', async () => {
    const cues = Array.from({ length: 45 }, (_, index) => ({ id: `cue-${index}`, text: `台词 ${index}`, start_ms: index * 1000, end_ms: index * 1000 + 800, source: 'external', language: 'zh', review_status: 'confirmed' }));
    const fetch = mockFetch(() => ({ asset, scenes: [scene], observations: [], subtitles: cues, annotations: [], runs: [] }));
    mount(<AssetDetail id={asset.id} initialTime={1000} initialCue="cue-44" />);
    await waitFor(() => expect(screen.getByRole('button', { name: /台词 45/ }).className).toBe('active'));
    const cue = document.getElementById('subtitle-cue-44');
    expect(cue?.className).toContain('active');
    expect(cue?.textContent).toContain('00:00:44');
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/assets/episode-one']);
  });

  it('links possible similar scenes without merging evidence and reveals a linked scene beyond the first page', async () => {
    const user = userEvent.setup(); const value = context();
    const scenes: SceneChunk[] = Array.from({ length: 45 }, (_, index) => ({ id: `scene-${index}`, chunk_id: `scene-${index}`, summary: `独立场景 ${index}`, start_ms: index * 2000, end_ms: index * 2000 + 1500, evidence_frame_ids: [], understanding_level: 'initial' }));
    scenes[0].scene_key = '雨夜站台'; scenes[0].related_chunk_ids = ['scene-44', 'scene-44', 'scene-0', 'missing-scene'];
    scenes[44].scene_key = '雨夜站台'; scenes[44].related_chunk_ids = ['scene-0'];
    const fetch = mockFetch(() => ({ asset, scenes, observations: [], subtitles: [], annotations: [], runs: [] }));
    const view = mount(<AssetDetail id={asset.id} />, value);
    const navigation = await screen.findByRole('navigation', { name: '可能相似的场景' });
    expect(within(navigation).getByText('雨夜站台')).toBeTruthy();
    expect(within(navigation).getByText(/相似性尚未确认为同一地点、人物或事件/)).toBeTruthy();
    const links = within(navigation).getAllByRole('link'); expect(links).toHaveLength(1);
    expect(links[0].getAttribute('href')).toBe('#/asset/episode-one?t=88000&chunk=scene-44');
    expect(document.getElementById('scene-scene-44')).toBeNull();
    await user.click(links[0]);
    view.rerender(<AppContext.Provider value={value}><AssetDetail id={asset.id} initialTime={88000} initialChunk="scene-44" /></AppContext.Provider>);
    await waitFor(() => expect(document.getElementById('scene-scene-44')?.className).toContain('active'));
    expect(document.getElementById('scene-scene-0')?.textContent).toContain('独立场景 0');
    expect(document.getElementById('scene-scene-44')?.textContent).toContain('独立场景 44');
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/assets/episode-one']);
  });

  it('keeps an estimate error visible without starting a task and hides historical scene actions', async () => {
    const user = userEvent.setup(); const fetch = mockFetch(() => new Response(JSON.stringify({ detail: '该片段已被新方案替换' }), { status: 409 }));
    const view = mount(<ResultCard result={{ ...result, is_historical: true }} />);
    expect(screen.queryByRole('button', { name: '深入分析此片段' })).toBeNull();
    view.unmount(); mount(<ResultCard result={result} />);
    await user.click(screen.getByRole('button', { name: '深入分析此片段' }));
    await user.click(screen.getByRole('button', { name: '估算用量' }));
    await screen.findByText('该片段已被新方案替换');
    expect((screen.getByRole('button', { name: '开始深入分析' }) as HTMLButtonElement).disabled).toBe(true);
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/assets/episode-one/chunks/chunk-one/deepen/estimate']);
  });

  it('shows stage progress, elapsed time, requests and cache reuse for resumed scene jobs', () => {
    const value = context(); const job: Job = { id: 'scene-job', type: 'scene_deepen', status: 'paused', progress: 0.5, completed: 1, total: 2, created_at: asset.created_at, asset_id: asset.id, stage: 'deep_analysis', stage_metrics: { transcript: { request_count: 1, elapsed_ms: 1250, cache_hits: 2 }, preparation: { request_count: 4, elapsed_ms: 8000, cache_hits: 3 }, vision: { request_count: 1, elapsed_ms: 2000, cache_hits: 1, completed: 1, total: 2, usage: { total_tokens: 50 } }, context_refresh: { request_count: 1, elapsed_ms: 120, cache_hits: 0 } } }; value.jobs = [job];
    mount(<Jobs />, value);
    const metrics = screen.getByLabelText('分阶段用量');
    expect(within(metrics).getByText('台词整理')).toBeTruthy(); expect(within(metrics).getByText('视觉分块')).toBeTruthy(); expect(within(metrics).getByText('深入理解')).toBeTruthy();
    expect(within(metrics).getByText('耗时 1.3 秒')).toBeTruthy(); expect(within(metrics).getByText('复用 3 次')).toBeTruthy(); expect(within(metrics).getByText('1 / 2 · 请求 1 次')).toBeTruthy();
    expect(within(metrics).getByText('更新连续理解')).toBeTruthy();
    expect(within(metrics).queryByText('待开始')).toBeNull();
    expect(within(metrics).queryByText(/待分块确定/)).toBeNull();
  });

  it('shows reported stage progress and statuses while keeping unknown chunk totals explicit', () => {
    const value = context();
    const job: Job = { id: 'staged-job', type: 'analysis', status: 'running', progress: 0.2, completed: 2, total: 4, created_at: asset.created_at, asset_id: asset.id, stage: 'preparation', stage_progress: { transcript: { completed: 1, total: 1, status: 'completed' }, preparation: { completed: 2, total: 4, status: 'running' }, deep_analysis: { completed: 0, total: null, status: 'pending' }, context_refresh: { completed: 0, status: 'pending' } }, stage_metrics: { transcript: { request_count: 1, elapsed_ms: 500, cache_hits: 1, completed: 99, total: 99 }, preparation: { request_count: 2, elapsed_ms: 1200, cache_hits: 0 } } };
    value.jobs = [job]; const view = mount(<Jobs />, value);
    const transcript = screen.getByLabelText('台词整理阶段');
    expect(within(transcript).getByText('已完成 1 / 1')).toBeTruthy();
    expect(within(transcript).getByText('已完成')).toBeTruthy();
    expect(within(transcript).queryByText(/99/)).toBeNull();
    expect(within(transcript).getByText('请求 1 次')).toBeTruthy();
    const preparation = screen.getByLabelText('视觉分块阶段');
    expect(within(preparation).getByText('已完成 2 / 4')).toBeTruthy();
    expect(within(preparation).getByText('进行中')).toBeTruthy();
    const deep = screen.getByLabelText('深入理解阶段');
    expect(within(deep).getByText('已完成 0 · 待分块确定')).toBeTruthy();
    expect(within(deep).getByText('待开始')).toBeTruthy();
    expect(within(deep).queryByText(/请求/)).toBeNull();
    expect(within(screen.getByLabelText('更新连续理解阶段')).getByText('已完成 0 · 待分块确定')).toBeTruthy();
    const finished = { ...value, jobs: [{ ...job, stage_progress: { ...job.stage_progress, deep_analysis: { completed: 0, total: 0, status: 'completed' as const } } }] };
    view.rerender(<AppContext.Provider value={finished}><Jobs /></AppContext.Provider>);
    expect(within(screen.getByLabelText('深入理解阶段')).getByText('已完成 0 / 0')).toBeTruthy();
    expect(within(screen.getByLabelText('深入理解阶段')).getByText('已完成')).toBeTruthy();
  });
});
