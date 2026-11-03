import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import KnowledgePanel, { KnowledgeResults, quoteReference } from '../src/components/KnowledgePanel';
import KnowledgeSettings from '../src/components/KnowledgeSettings';
import type { Profile } from '../src/types';

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
const source = { source_id: 'source_one', snapshot_id: 'snapshot_one', title: '原作设定', url: 'https://example.test/source', status: 'active' };
const base = '/api/work-entity-scopes/work_one/knowledge';
function setup() {
  const fetch = vi.fn(async (input: string, options?: RequestInit) => {
    let value: unknown = {};
    if (input === `${base}/sources`) value = { sources: [source] };
    if (input === `${base}/sources/source_one`) value = { ...source, text: '前言😀。时间暂停需要付出代价。' };
    if (input === `${base}/cards`) value = { cards: [] };
    if (input === `${base}/status`) value = { status: 'not_built' };
    if (input.endsWith('/characters')) value = { scope_id: 'cast_one', profiles: [{ id: 'char_one', name: '小光' }] };
    if (input === `${base}/query`) value = { knowledge_hits: [{ card_id: 'card_one', title: '时间停止规则', text: '可以暂停时间', review_status: 'reviewed', target_ref: { kind: 'concept', concept_id: 'concept_one' }, sources: [{ ...source, quote: '时间暂停需要付出代价。' }] }], film_hits: [], diagnostics: { keyword_ms: 12, calls: [] } };
    if (input === '/api/knowledge/settings') value = options?.method === 'PUT' ? JSON.parse(options.body as string) : { enabled: false, text_profile_id: null, embedding_profile_id: null, embedding_dim: 384, runtime: { version: '1.5.7' }, preflight_signature: 'read-only' };
    if (input === '/api/knowledge/check') value = { status: 'ready', message: 'embedding 维度已确认' };
    return new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch); return fetch;
}
function mountPanel() {
  render(<KnowledgePanel assetId="asset_one" scopeId="work_one" entities={[{ concept_id: 'concept_one', name: '时间暂停', type: 'ability' }]} />);
  const details = screen.getByText('知识资料与检索 · LightRAG 实验').closest('details')!;
  details.open = true;
  fireEvent(details, new Event('toggle'));
}

describe('Knowledge evidence workflow', () => {
  it('anchors exact source quotes using code-point offsets and rejects ambiguous text', () => {
    expect(quoteReference(source, '前😀后规则', '后规则')).toEqual({ source_id: 'source_one', snapshot_id: 'snapshot_one', start: 2, end: 5 });
    expect(() => quoteReference(source, '规则和规则', '规则')).toThrow('重复');
    expect(() => quoteReference(source, '原文', '不存在')).toThrow('完全一致');
  });

  it('keeps knowledge-only results traceable without creating a film link', () => {
    render(<KnowledgeResults hits={[{ card_id: 'card_one', title: '规则', text: '设定说明', sources: [{ title: '原作', url: source.url, quote: '原文证据' }] }]} />);
    expect(screen.getByText('找到相关设定，尚无已确认影片证据。')).toBeTruthy();
    expect(screen.getByRole('link', { name: '原作' }).getAttribute('href')).toBe(source.url);
    expect(screen.queryByRole('link', { name: /00:/ })).toBeNull();
  });

  it('saves reviewed abstract knowledge with exact source and stable concept identity', async () => {
    const fetch = setup(); const user = userEvent.setup(); mountPanel();
    await screen.findByText('整理资料卡');
    await user.selectOptions(screen.getByLabelText('资料卡关联对象'), 'concept:concept_one');
    await user.selectOptions(screen.getByLabelText('来源正文'), 'source_one');
    await waitFor(() => expect(screen.getByText('前言😀。时间暂停需要付出代价。')).toBeTruthy());
    await user.type(screen.getByLabelText(/^原文引用/), '时间暂停需要付出代价。');
    await user.type(screen.getByLabelText('定义或设定说明'), '暂停时间需要消耗资源');
    await user.type(screen.getByLabelText('代价（每行一条）'), '消耗资源');
    await user.selectOptions(screen.getByLabelText('资料核对状态'), 'reviewed');
    await user.click(screen.getByRole('button', { name: '保存资料卡' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url, options]) => url === `${base}/cards` && options?.method === 'POST')).toBe(true));
    const request = fetch.mock.calls.find(([url, options]) => url === `${base}/cards` && options?.method === 'POST')!;
    expect(JSON.parse(request[1]!.body as string)).toMatchObject({ target_ref: { kind: 'concept', concept_id: 'concept_one' }, source_refs: [{ source_id: 'source_one', snapshot_id: 'snapshot_one', start: 4, end: 15 }], costs: ['消耗资源'], review_status: 'reviewed' });
    expect(fetch.mock.calls.every(([url]) => url.startsWith('/api/'))).toBe(true);
    expect(fetch.mock.calls.some(([url]) => url.includes('bindings'))).toBe(false);
  });

  it('switches query mode and preserves knowledge without film evidence', async () => {
    const fetch = setup(); const user = userEvent.setup(); mountPanel(); await screen.findByText('整理资料卡');
    await user.selectOptions(screen.getByLabelText('知识检索模式'), 'lightrag');
    await user.type(screen.getByLabelText('知识问题'), '时间停止要付出什么代价');
    await user.click(screen.getByRole('button', { name: '检索知识与片段' }));
    await screen.findByText('时间停止规则');
    expect(screen.getByText('没有已确认影片证据；相关设定仍可在上方查看出处。')).toBeTruthy();
    const request = fetch.mock.calls.find(([url]) => url === `${base}/query`)!;
    expect(JSON.parse(request[1]!.body as string)).toMatchObject({ mode: 'lightrag', filters: { asset_id: 'asset_one' } });
  });

  it('reuses the saved Bangumi archive and reloads candidate cards without approving or building them', async () => {
    const fallback = setup(); let imported = false;
    const fetch = vi.fn(async (input: string, options?: RequestInit) => {
      let value: unknown;
      if (input === `${base}/bangumi`) {
        imported = true;
        value = { source_count: 3, card_count: 2, created_sources: 2, created_cards: 1, warnings: ['部分角色没有简介'] };
      } else if (input === `${base}/cards` && imported) {
        value = { cards: [
          { card_id: 'reviewed_one', title: '已核对的学校资料', text: '保留原有核对结果', target_ref: { kind: 'concept', concept_id: 'concept_one' }, review_status: 'reviewed' },
          { card_id: 'bangumi_one', title: '小光的 Bangumi 简介', text: '来源里的角色说明', target_ref: { kind: 'character', asset_id: 'asset_one', character_scope_id: 'cast_one', character_id: 'char_one' }, review_status: 'candidate' },
        ] };
      } else return fallback(input, options);
      return new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } });
    });
    vi.stubGlobal('fetch', fetch);
    const user = userEvent.setup(); mountPanel();
    await screen.findByText('整理资料卡');
    await user.click(screen.getByRole('button', { name: '整理已有 Bangumi 资料' }));
    await screen.findByText(/新增 2 份来源、1 张待核对资料卡/);
    expect(screen.getByText(/部分角色没有简介/)).toBeTruthy();
    expect(screen.getByText('保留原有核对结果')).toBeTruthy();
    expect(screen.getByText('小光的 Bangumi 简介')).toBeTruthy();
    expect(screen.getByText('抽取候选 / 待核对')).toBeTruthy();
    const request = fetch.mock.calls.find(([url]) => url === `${base}/bangumi`)!;
    expect(request[1]?.method).toBe('POST');
    expect(JSON.parse(request[1]!.body as string)).toEqual({ asset_id: 'asset_one' });
    expect(fetch.mock.calls.some(([url, options]) => url.endsWith('/build') || url.includes('/bangumi/subject') || options?.method === 'PATCH')).toBe(false);
  });

  it('shows an unavailable Bangumi archive without changing cards or starting a model task', async () => {
    const fallback = setup();
    const fetch = vi.fn(async (input: string, options?: RequestInit) => input === `${base}/bangumi`
      ? new Response(JSON.stringify({ detail: '这部动画尚未保存 Bangumi 档案' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
      : fallback(input, options));
    vi.stubGlobal('fetch', fetch);
    const user = userEvent.setup(); mountPanel(); await screen.findByText('整理资料卡');
    await user.click(screen.getByRole('button', { name: '整理已有 Bangumi 资料' }));
    await screen.findByText('这部动画尚未保存 Bangumi 档案');
    expect((screen.getByRole('button', { name: '整理已有 Bangumi 资料' }) as HTMLButtonElement).disabled).toBe(false);
    expect(fetch.mock.calls.filter(([, options]) => options?.method && options.method !== 'GET')).toHaveLength(1);
  });

  it('uses local multilingual embeddings by default and preserves an optional API configuration', async () => {
    const fetch = setup(); const user = userEvent.setup();
    const profile = { id: 'codex', name: 'Codex 连接', model: '', provider_type: 'codex_cli', capabilities: ['query'] } as Profile;
    render(<KnowledgeSettings profiles={[profile]} />);
    await screen.findByLabelText('知识向量运行方式');
    expect((screen.getByLabelText('知识向量运行方式') as HTMLSelectElement).value).toBe('local');
    await user.type(screen.getByLabelText(/^本地 embedding 模型目录/), 'D:/models/knowledge');
    await user.selectOptions(screen.getByLabelText('知识文本模型连接'), 'codex');
    await user.type(screen.getByLabelText(/^知识文本模型 ID/), 'gpt-test');
    await user.click(screen.getByRole('button', { name: '保存知识后端设置' }));
    await user.click(screen.getByRole('button', { name: '检查知识后端连接' }));
    await screen.findByText('知识后端检查通过');
    const request = fetch.mock.calls.find(([url, options]) => url === '/api/knowledge/settings' && options?.method === 'PUT')!;
    expect(JSON.parse(request[1]!.body as string)).toMatchObject({ text_profile_id: 'codex', text_model: 'gpt-test', embedding_backend: 'local', local_model_dir: 'D:/models/knowledge', embedding_profile_id: null, query_timeout_s: 60 });
    expect(JSON.parse(request[1]!.body as string)).not.toHaveProperty('embedding_dim');
    expect(JSON.parse(request[1]!.body as string)).not.toHaveProperty('preflight_signature');
    await user.selectOptions(screen.getByLabelText('知识向量运行方式'), 'api');
    expect(screen.getByText('尚无 embedding 连接。请先添加声明“语义向量”能力的 API 连接。')).toBeTruthy();
  });

  it('saves an explicit API thinking mode and clears it when switching to Codex', async () => {
    const fetch = setup(); const user = userEvent.setup();
    const profiles = [
      { id: 'api', name: 'DeepSeek 连接', model: 'deepseek-flash', provider_type: 'openai_compatible', capabilities: ['query'] },
      { id: 'codex', name: 'Codex 连接', model: 'codex-model', provider_type: 'codex_cli', capabilities: ['query'] },
    ] as Profile[];
    render(<KnowledgeSettings profiles={profiles} />);
    await screen.findByLabelText('知识文本模型连接');
    await user.selectOptions(screen.getByLabelText('知识文本模型连接'), 'api');
    await user.selectOptions(screen.getByLabelText(/^知识文本 API 思考模式/), 'disabled');
    await user.click(screen.getByRole('button', { name: '保存知识后端设置' }));
    await screen.findByText('知识后端已关闭，检索继续使用词法基线。');
    let requests = fetch.mock.calls.filter(([url, options]) => url === '/api/knowledge/settings' && options?.method === 'PUT');
    expect(JSON.parse(requests[0][1]!.body as string).text_thinking).toBe('disabled');
    await user.selectOptions(screen.getByLabelText('知识文本模型连接'), 'codex');
    expect(screen.queryByLabelText(/^知识文本 API 思考模式/)).toBeNull();
    await user.click(screen.getByRole('button', { name: '保存知识后端设置' }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url, options]) => url === '/api/knowledge/settings' && options?.method === 'PUT')).toHaveLength(2));
    requests = fetch.mock.calls.filter(([url, options]) => url === '/api/knowledge/settings' && options?.method === 'PUT');
    expect(JSON.parse(requests[1][1]!.body as string).text_thinking).toBeNull();
  });
});
