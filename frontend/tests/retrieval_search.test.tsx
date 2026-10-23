import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type Job, type SearchResult } from '../src/types';
import SearchPage, { ResultCard } from '../src/components/Search';
import JobModal from '../src/components/JobModal';
import Jobs from '../src/components/Jobs';
import AssetDetail from '../src/components/AssetDetail';
import { RetrievalFrames } from '../src/components/RetrievalEvidence';

const asset: Asset = { id: 'episode', work_id: 'work', title: '测试动画', kind: 'animation', duration_ms: 120000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-10-20T00:00:00Z', status: 'ready' };
const result: SearchResult = { id: 'window', record_id: 'window', asset_id: asset.id, work_id: asset.work_id, title: asset.title, kind: 'visual', start_ms: 1000, end_ms: 6000, text: '', summary: '', source: 'retrieval-first-v1', review_status: 'unreviewed', evidence_type: 'retrieval_window', seek_ms: 3000, hit_frame_id: 'frame_hit', match_type: 'partial', member_ids: ['window', 'neighbor'], ordered_frames: [{ id: 'frame_start', at_ms: 1000 }, { id: 'frame_hit', at_ms: 3000 }] };
function context(): AppContextValue {
  return { bootstrap: { library_path: '/fixtures', settings: { defaults: DEFAULTS, bindings: {} }, provider_profiles: [], stats: {}, tools: { ffmpeg: true, ffprobe: true }, version: 'test', visual_encoder: { available: true, model: 'local-test' } }, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), refreshJobs: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn() };
}
function mount(ui: ReactNode, value = context()) { return render(<AppContext.Provider value={value}>{ui}</AppContext.Provider>); }
function mockFetch(handler: (url: string, options?: RequestInit) => unknown) {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => new Response(JSON.stringify(await handler(url, options)), { status: 200, headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetch); return fetch;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); window.history.replaceState({}, '', '/'); });

describe('Direct visual retrieval evidence', () => {
  it('limits a dense window preview to eight ordered frames including the hit, and expands every original frame', async () => {
    const user = userEvent.setup(); const onSeek = vi.fn();
    const frames = Array.from({ length: 68 }, (_, index) => ({ id: `dense_${index}`, at_ms: 1000 + index * 73, pts: index, time_base: [73, 1000] as [number, number] }));
    const original = JSON.stringify(frames);
    render(<RetrievalFrames assetId="episode" record={{ evidence_type: 'retrieval_window', ordered_frames: frames, hit_frame_id: 'dense_13' }} onSeek={onSeek} />);
    let links = screen.getAllByRole('link', { name: /跳到证据画面/ });
    expect(links).toHaveLength(8);
    const previewTimes = links.map((link) => Number(link.getAttribute('href')!.split('t=')[1]));
    expect(previewTimes).toEqual([...previewTimes].sort((a, b) => a - b));
    expect(previewTimes).toContain(frames[0].at_ms);
    expect(previewTimes).toContain(frames[67].at_ms);
    expect(previewTimes).toContain(frames[13].at_ms);
    await user.click(links.find((link) => link.getAttribute('aria-current') === 'true')!);
    expect(onSeek).toHaveBeenLastCalledWith(frames[13].at_ms);
    const expand = screen.getByRole('button', { name: '展开全部 68 帧' });
    expect(expand.getAttribute('aria-expanded')).toBe('false');
    await user.click(expand);
    links = screen.getAllByRole('link', { name: /跳到证据画面/ });
    expect(links.map((link) => link.getAttribute('href'))).toEqual(frames.map((frame) => `#/asset/episode?t=${frame.at_ms}`));
    expect(screen.getByRole('button', { name: '收起' }).getAttribute('aria-expanded')).toBe('true');
    await user.click(links[52]);
    expect(onSeek).toHaveBeenLastCalledWith(frames[52].at_ms);
    await user.click(screen.getByRole('button', { name: '收起' }));
    expect(screen.getAllByRole('link', { name: /跳到证据画面/ })).toHaveLength(8);
    expect(JSON.stringify(frames)).toBe(original);
  });

  it('starts a changed evidence window collapsed instead of carrying an expanded list into the next result', async () => {
    const user = userEvent.setup();
    const frames = Array.from({ length: 12 }, (_, index) => ({ id: `f_${index}`, at_ms: index * 100 }));
    const { rerender } = render(<RetrievalFrames assetId="episode" record={{ evidence_type: 'retrieval_window', ordered_frames: frames, hit_frame_id: 'f_4' }} />);
    await user.click(screen.getByRole('button', { name: '展开全部 12 帧' }));
    rerender(<RetrievalFrames assetId="episode" record={{ evidence_type: 'retrieval_window', ordered_frames: frames.map((frame) => ({ ...frame, id: `new_${frame.id}` })), hit_frame_id: 'new_f_3' }} />);
    expect(screen.getAllByRole('link', { name: /跳到证据画面/ })).toHaveLength(8);
    expect(screen.getByRole('button', { name: '展开全部 12 帧' }).getAttribute('aria-expanded')).toBe('false');
    expect(screen.getAllByRole('link', { name: /跳到证据画面/ }).find((link) => link.getAttribute('aria-current') === 'true')?.getAttribute('href')).toBe('#/asset/episode?t=300');
  });

  it('opens the hit frame time, keeps ordered source evidence and partial status without inventing a caption', async () => {
    const fetch = mockFetch(() => ({})); const user = userEvent.setup();
    mount(<ResultCard result={result} />);
    expect(screen.getByText('画面证据')).toBeTruthy();
    expect(screen.queryByText('暂无描述')).toBeNull();
    expect(screen.getByText('部分匹配')).toBeTruthy();
    expect(screen.getByRole('link', { name: '查看原片' }).getAttribute('href')).toBe('#/asset/episode?t=3000');
    expect(screen.getByRole('link', { name: '查看命中原帧' }).getAttribute('href')).toBe('/api/assets/episode/frames/frame_hit');
    expect(screen.getAllByRole('link', { name: /跳到证据画面/ }).map((link) => link.getAttribute('href'))).toEqual(['#/asset/episode?t=1000', '#/asset/episode?t=3000']);
    expect(screen.queryByRole('button', { name: /展开全部/ })).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: '收藏片段' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toEqual({ record_id: 'window', favorite: true });
  });

  it('preserves a zero hit timestamp instead of falling back to the window start', () => {
    mount(<ResultCard result={{ ...result, seek_ms: 0 }} />);
    expect(screen.getByRole('link', { name: '查看原片' }).getAttribute('href')).toBe('#/asset/episode?t=0');
  });

  it('shows uncertain candidate names separately from reliable bindings and suppresses revoked model names', () => {
    mount(<ResultCard result={{ ...result, entities: [
      { id: 'e1', type: 'character', character_name: '旧猜测', candidate_character_id: 'karen', candidate_character_name: '爱城华恋', candidate_character_aliases: ['华恋'], binding_status: 'candidate', association_source: 'ccip_uncalibrated' },
      { id: 'e2', type: 'character', character_id: 'hikari', character_name: '神乐光', binding_status: 'auto_bound', identity_binding: { status: 'auto_bound', source: 'identity-binding-v2' } },
      { id: 'e3', type: 'character', character_name: '错误姓名', description: '背对镜头的人物', binding_status: 'revoked' },
    ] }} />);
    const identities = screen.getByLabelText('画面人物与实体身份');
    expect(within(identities).getByText('候选：爱城华恋')).toBeTruthy();
    expect(within(identities).queryByText('爱城华恋', { exact: true })).toBeNull();
    expect(within(identities).getByText(/候选 · 待确认 · 未校准的图像候选/)).toBeTruthy();
    expect(within(identities).getByText('神乐光', { exact: true })).toBeTruthy();
    expect(within(identities).getByText(/自动证据绑定/)).toBeTruthy();
    expect(within(identities).getByText('未知人物 · 背对镜头的人物')).toBeTruthy();
    expect(within(identities).queryByText('错误姓名')).toBeNull();
    expect(within(identities).queryByText('旧猜测')).toBeNull();
  });

  it('labels a supported event onset separately from the original visual hit', () => {
    mount(<ResultCard result={{ ...result, seek_ms: 2000, seek_basis: 'local_event', event_evidence: {
      record_id: 'window', action: '转身', start_ms: 2000, end_ms: 3000, evidence_frame_ids: ['frame_start', 'frame_hit'],
    } }} />);
    expect(screen.getByRole('link', { name: '查看原片' }).getAttribute('href')).toBe('#/asset/episode?t=2000');
    expect(screen.getByRole('link', { name: '查看命中原帧' }).getAttribute('href')).toBe('/api/assets/episode/frames/frame_hit');
    expect(screen.getByText(/按局部事件证据定位：转身/)).toBeTruthy();
    expect(screen.getByText('部分匹配')).toBeTruthy();
  });

  it('advertises visual retrieval without a text embedding binding and only sends a search request', async () => {
    const fetch = mockFetch(() => ({ results: [result], degraded: [] })); const user = userEvent.setup();
    mount(<SearchPage />);
    expect(screen.getByText('视觉 + 字面检索 · 按已有索引')).toBeTruthy();
    expect(screen.queryByText('当前使用字面检索')).toBeNull();
    await user.type(screen.getByRole('textbox', { name: '用自然语言搜索镜头或台词' }), '红色物体');
    await user.click(screen.getByRole('button', { name: /^检索/ }));
    await screen.findByText('画面证据');
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/search']);
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({ visual_verification: false, enhancements: false });
    expect(screen.queryByRole('button', { name: '深入分析此片段' })).toBeNull();
  });

  it('only requests bounded visual verification after explicit selection and reports fallback honestly', async () => {
    const fetch = mockFetch(() => ({ results: [result], degraded: [], verification: { status: 'timeout', request_count: null, deadline_ms: 30000 } }));
    const user = userEvent.setup(); mount(<SearchPage />);
    const verify = screen.getByRole('checkbox', { name: '核对前 3 个视觉候选' }) as HTMLInputElement;
    expect(verify.checked).toBe(false);
    expect((screen.getByRole('checkbox', { name: '启用查询理解与结果解释' }) as HTMLInputElement).checked).toBe(false);
    await user.click(verify);
    await user.type(screen.getByRole('textbox', { name: '用自然语言搜索镜头或台词' }), '将物品递给另一人');
    await user.click(screen.getByRole('button', { name: /^检索/ }));
    await screen.findByText('视觉复核：已超时，保留原候选');
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({ visual_verification: true, enhancements: false });
    expect(screen.getByText('部分匹配')).toBeTruthy();
    expect(screen.queryByText('完整匹配')).toBeNull();
  });

  it('shows retrieval windows in the original-video view and seeks each evidence frame', async () => {
    mockFetch(() => ({ asset, observations: [], scenes: [], subtitles: [], annotations: [], runs: [], retrieval_windows: [result] }));
    mount(<AssetDetail id={asset.id} />);
    await screen.findByText('画面索引 · 待核对');
    const video = screen.getByLabelText(`${asset.title} 原片播放器`) as HTMLVideoElement;
    fireEvent.click(screen.getByRole('link', { name: '跳到证据画面 00:00:03' }));
    fireEvent.loadedMetadata(video);
    expect(video.currentTime).toBe(3);
    expect(screen.queryByRole('button', { name: '深入分析此片段' })).toBeNull();
  });

  it('adds local identity to existing windows without starting sampling, encoding or a VLM job and prevents duplicate submissions', async () => {
    let finish!: (value: unknown) => void;
    const pending = new Promise((resolve) => { finish = resolve; });
    const fetch = mockFetch((_url, options) => options?.method === 'POST' ? pending : ({ asset, observations: [], scenes: [], subtitles: [], annotations: [], runs: [], retrieval_windows: [result] }));
    const value = context(); const user = userEvent.setup();
    mount(<AssetDetail id={asset.id} />, value);
    const supplement = await screen.findByRole('button', { name: '补充人物匹配' });
    await user.dblClick(supplement);
    const mutations = fetch.mock.calls.filter(([, options]) => options?.method === 'POST');
    expect(mutations).toHaveLength(1);
    expect(mutations[0][0]).toBe('/api/assets/episode/retrieval/identity');
    expect(JSON.parse(mutations[0][1]!.body as string)).toEqual({});
    finish({ id: 'identity-job', status: 'queued' });
    await waitFor(() => expect(value.refresh).toHaveBeenCalledOnce());
    expect((screen.getByRole('button', { name: '人物匹配已提交' }) as HTMLButtonElement).disabled).toBe(true);
    expect(value.notify).toHaveBeenCalledWith('人物匹配任务已加入队列；复用已有画面，不调用 VLM。');
  });

  it('keeps a successfully submitted identity job locked when refreshing the task list fails', async () => {
    const fetch = mockFetch((_url, options) => options?.method === 'POST' ? { id: 'queued-identity' } : ({ asset, observations: [], scenes: [], subtitles: [], annotations: [], runs: [], retrieval_windows: [result] }));
    const value = context(); value.refresh = vi.fn().mockRejectedValue(new Error('本机连接中断'));
    const user = userEvent.setup(); mount(<AssetDetail id={asset.id} />, value);
    await user.click(await screen.findByRole('button', { name: '补充人物匹配' }));
    await waitFor(() => expect(value.notify).toHaveBeenCalledWith(expect.stringContaining('任务已创建，但队列刷新失败'), 'error'));
    expect((screen.getByRole('button', { name: '人物匹配已提交' }) as HTMLButtonElement).disabled).toBe(true);
    expect(fetch.mock.calls.filter(([, options]) => options?.method === 'POST')).toHaveLength(1);
  });

  it('does not offer identity supplementation without existing retrieval windows', async () => {
    mockFetch(() => ({ asset, observations: [], scenes: [], subtitles: [], annotations: [], runs: [], retrieval_windows: [] }));
    mount(<AssetDetail id={asset.id} />);
    await screen.findByRole('heading', { name: asset.title });
    expect(screen.queryByRole('button', { name: '补充人物匹配' })).toBeNull();
  });

  it('disables identity supplementation during an asset job', async () => {
    mockFetch(() => ({ asset, observations: [], scenes: [], subtitles: [], annotations: [], runs: [], retrieval_windows: [result] }));
    const value = context(); value.jobs = [{ id: 'existing', asset_id: asset.id, status: 'running' } as Job];
    mount(<AssetDetail id={asset.id} />, value);
    expect((await screen.findByRole('button', { name: '补充人物匹配' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('resolves event participants through current local identity status instead of stale names', async () => {
    mockFetch(() => ({ asset, observations: [], scenes: [], subtitles: [], annotations: [], runs: [], retrieval_windows: [{
      ...result, entities: [{ id: 'e1', type: 'character', candidate_character_id: 'karen', candidate_character_name: '爱城华恋', binding_status: 'candidate' }],
      events: [{ action: '转身', actor_ids: ['e1'], actor_names: ['已撤销的旧实名'] }],
    }] }));
    const user = userEvent.setup(); mount(<AssetDetail id={asset.id} />);
    await user.click(await screen.findByText('动作与空间关系'));
    expect(screen.getByText('候选：爱城华恋 · 转身')).toBeTruthy();
    expect(screen.queryByText('已撤销的旧实名')).toBeNull();
    expect(screen.getByRole('button', { name: '查看候选爱城华恋的角色档案' })).toBeTruthy();
  });
});

describe('Explicit retrieval core experiments', () => {
  it('keeps the current strategy default and submits a 60-second core experiment without VLM or prescan', async () => {
    const fetch = mockFetch(() => ({ id: 'core-job' })); const user = userEvent.setup(); const close = vi.fn();
    mount(<JobModal asset={asset} onClose={close} />);
    expect((screen.getByLabelText('处理策略') as HTMLSelectElement).value).toBe('scene-chunks-v1');
    expect((screen.getByRole('button', { name: '开始任务' }) as HTMLButtonElement).disabled).toBe(true);
    await user.selectOptions(screen.getByLabelText('处理策略'), 'retrieval-first-v1');
    expect(screen.queryByLabelText('本次任务模型设置')).toBeNull();
    expect(screen.queryByRole('button', { name: '扫描关键帧' })).toBeNull();
    expect((screen.getByRole('checkbox', { name: '使用 VLM 生成局部描述与事件' }) as HTMLInputElement).checked).toBe(false);
    expect(screen.getByText(/人物匹配仍按作品识别设置执行/)).toBeTruthy();
    expect(fetch).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: '先试运行前 60 秒' }));
    await user.click(screen.getByRole('button', { name: '开始任务' }));
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/jobs']);
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({ analysis_strategy: 'retrieval-first-v1', retrieval_mode: 'core', start_ms: 0, end_ms: 60000, stages: ['vision'], max_cost: null });
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string).model_overrides).toEqual({});
  });

  it('explains an unavailable local encoder before starting an experiment', async () => {
    const value = context(); value.bootstrap.visual_encoder = { available: false, reason: '未安装本机权重' };
    const user = userEvent.setup(); mount(<JobModal asset={asset} onClose={vi.fn()} />, value);
    await user.selectOptions(screen.getByLabelText('处理策略'), 'retrieval-first-v1');
    expect(screen.getByText(/本机视觉编码器尚不可用：未安装本机权重/)).toBeTruthy();
    expect((screen.getByRole('button', { name: '开始任务' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('requires only a vision binding when VLM is checked and allows no-VLM indexing when unchecked', async () => {
    const fetch = mockFetch(() => ({})); const user = userEvent.setup();
    mount(<JobModal asset={asset} onClose={vi.fn()} />);
    await user.selectOptions(screen.getByLabelText('处理策略'), 'retrieval-first-v1');
    const toggle = screen.getByRole('checkbox', { name: '使用 VLM 生成局部描述与事件' });
    await user.click(toggle);
    expect(screen.getByLabelText('本次任务模型设置')).toBeTruthy();
    expect(screen.getByText(/请为画面理解选择可用连接/)).toBeTruthy();
    expect((screen.getByRole('button', { name: '开始任务' }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(toggle);
    expect(screen.queryByLabelText('本次任务模型设置')).toBeNull();
    expect((screen.getByRole('button', { name: '开始任务' }) as HTMLButtonElement).disabled).toBe(false);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('submits complete mode only after the VLM checkbox is explicitly checked', async () => {
    const value = context(); const user = userEvent.setup(); const close = vi.fn();
    value.bootstrap.settings.bindings = { vision: 'vision-api' };
    value.bootstrap.provider_profiles = [{ id: 'vision-api', name: '局部模型', provider_type: 'openai_compatible',
      model: 'vision-test', base_url: 'http://localhost:9999/v1', capabilities: ['vision'], secret_mode: 'none' }];
    const fetch = mockFetch(() => ({ id: 'complete-job' }));
    mount(<JobModal asset={asset} onClose={close} />, value);
    await user.selectOptions(screen.getByLabelText('处理策略'), 'retrieval-first-v1');
    await user.click(screen.getByRole('checkbox', { name: '使用 VLM 生成局部描述与事件' }));
    expect(screen.queryByRole('button', { name: '扫描关键帧' })).toBeNull();
    await user.click(screen.getByRole('button', { name: '开始任务' }));
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/jobs']);
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({
      analysis_strategy: 'retrieval-first-v1', retrieval_mode: 'complete', stages: ['vision'], end_ms: null,
    });
  });

  it('submits both complete episode ranges without VLM after switching the checkbox off', async () => {
    const user = userEvent.setup(); const close = vi.fn(); const fetch = mockFetch(() => ({}));
    mount(<JobModal assets={[asset, { ...asset, id: 'episode_two', duration_ms: 180000 }]} onClose={close} />);
    await user.selectOptions(screen.getByLabelText('处理策略'), 'retrieval-first-v1');
    const toggle = screen.getByRole('checkbox', { name: '使用 VLM 生成局部描述与事件' });
    await user.click(toggle); await user.click(toggle);
    await user.click(screen.getByRole('button', { name: '处理全片' }));
    await user.click(screen.getByRole('button', { name: '开始任务' }));
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/jobs/batch']);
    const payload = JSON.parse(fetch.mock.calls[0][1]!.body as string);
    expect(payload.items).toHaveLength(2);
    for (const item of payload.items) {
      expect(item).toMatchObject({ retrieval_mode: 'core', analysis_strategy: 'retrieval-first-v1', start_ms: 0, end_ms: null, stages: ['vision'], max_cost: null });
      expect(item.model_overrides).toEqual({});
    }
  });

  it('does not present core_ready as completion of the default full workflow', () => {
    const value = context();
    value.jobs = [{ id: 'core', batch_id: 'core_batch', type: 'analysis', asset_id: asset.id, analysis_strategy: 'retrieval-first-v1', retrieval_mode: 'core', status: 'completed', progress: 1, completed: 2, total: 2, created_at: asset.created_at, stage: 'final_publish', retrieval: { status: 'core_ready', default_completed_at: null, first_searchable_at: asset.created_at, coverage: { start_ms: 0, end_ms: 60000, total_units: 2, processed_units: 2, published_units: 2 }, capabilities: { sampling: 'completed', visual_index: 'completed', subtitle_index: 'completed', local_records: 'not_requested', identity: 'not_requested', final_publish: 'completed' } } } as Job];
    mount(<Jobs />, value);
    const card = screen.getByText('核心检索就绪').closest('article')!;
    expect(within(card).getByText('核心检索实验完成 · 默认完整流程未完成')).toBeTruthy();
    expect(within(card).getByText('已发布 2 / 2 个检索窗口 · 已处理 2 个')).toBeTruthy();
    expect(within(card).getByText(/局部事件：本次未启用/)).toBeTruthy();
    expect(within(card).queryByText('默认完整流程已完成')).toBeNull();
    expect(within(card).getByText('发布检索索引')).toBeTruthy();
    const batch = screen.getByRole('region', { name: '批量分析进度' });
    expect(batch.textContent).toContain('任务完成 1 / 1');
    expect(batch.textContent).toContain('未启用局部事件与身份，不等于默认完整流程');
    expect(batch.textContent).toContain('不等于默认完整流程');
  });

  it.each([
    ['completed', '已完成'], ['running', '已启用（处理中）'], ['disabled', '已关闭'],
  ] as const)('reports actual new-core batch identity capability %s while preserving the legacy contract wording', (identity, label) => {
    const value = context();
    value.jobs = [
      { id: 'new-core', batch_id: 'modern-batch', type: 'analysis', asset_id: asset.id, status: 'completed', progress: 1, completed: 2, total: 2, created_at: asset.created_at,
        retrieval: { contract_version: 2, status: 'core_ready', default_completed_at: null, capabilities: { local_records: 'not_requested', identity } } },
      { id: 'legacy-core', batch_id: 'legacy-batch', type: 'analysis', asset_id: asset.id, status: 'completed', progress: 1, completed: 2, total: 2, created_at: asset.created_at,
        retrieval: { status: 'core_ready', default_completed_at: null, capabilities: { local_records: 'not_requested', identity: 'not_requested' } } },
    ] as Job[];
    mount(<Jobs />, value);
    const [modern, legacy] = screen.getAllByRole('region', { name: '批量分析进度' });
    expect(modern.textContent).toContain(`局部事件未启用；本地人物匹配：1 集${label}`);
    expect(modern.textContent).not.toContain('未启用局部事件与身份');
    expect(modern.textContent).toContain('不等于默认完整流程');
    expect(legacy.textContent).toContain('未启用局部事件与身份，不等于默认完整流程');
  });

  it('shows local identity completed for the new no-VLM contract without implying events were generated', () => {
    const value = context();
    value.jobs = [{ id: 'new-core', type: 'analysis', asset_id: asset.id, status: 'completed', progress: 1, completed: 2, total: 2, created_at: asset.created_at,
      retrieval: { contract_version: 2, status: 'core_ready', default_completed_at: null, capabilities: { local_records: 'not_requested', identity: 'completed' } } } as Job];
    mount(<Jobs />, value);
    expect(screen.getByText(/局部事件：本次未启用 · 角色身份：已完成/)).toBeTruthy();
    expect(screen.queryByText('默认完整流程已完成')).toBeNull();
  });

  it('reports a disabled local identity setting in Chinese and preserves its reason', () => {
    const value = context();
    value.jobs = [{ id: 'disabled-core', type: 'analysis', asset_id: asset.id, status: 'completed', progress: 1, completed: 2, total: 2, created_at: asset.created_at,
      retrieval: { contract_version: 2, status: 'core_ready', default_completed_at: null, capabilities: { identity: 'disabled' }, identity: { status: 'disabled', reason: '视觉身份识别未启用' } } } as Job];
    mount(<Jobs />, value);
    expect(screen.getByText('角色身份：已关闭')).toBeTruthy();
    expect(screen.getByText('本地人物匹配：视觉身份识别未启用')).toBeTruthy();
    expect(screen.queryByText('角色身份：已完成')).toBeNull();
  });
});
