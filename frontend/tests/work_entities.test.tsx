import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import WorkEntities from '../src/components/WorkEntities';
import type { Observation } from '../src/types';

const scope = { work_scope_id: 'workscope_one', title: '测试作品', work_version: 'TV', asset_ids: ['asset'] };
const concept = { concept_id: 'concept_one', type: 'prop', name: '时钟', original_name: '時計', aliases: ['怀表'], translated_names: [], appearance: '金色表壳', applicability: '第一季', forms: [], facts: [], relations: [], sources: [], version: 1, references: [{ id: 'ref_one', image_url: '/api/reference.png', status: 'candidate', view: 'object_region' }] };
const record: Observation = { id: 'window_one', window_id: 'window_one', shot_id: 'shot', start_ms: 1000, end_ms: 4000, summary: '金色物品', entities: [{ id: 'O1', description: '金色物品' }], events: [], uncertainties: [], evidence_frame_ids: ['frame_one', 'frame_two'], run_id: 'run' };
function setup(initialScope: object = scope) {
  let currentScope = initialScope;
  const fetch = vi.fn(async (url: string, options?: RequestInit) => {
    if (url === '/api/work-entity-scopes' && options?.method === 'POST') currentScope = scope;
    if (url.endsWith('/work-entity-scope') && options?.method === 'PUT') currentScope = scope;
    const value = url.endsWith('/work-entity-scope') ? currentScope : url.endsWith('/work-entities') ? { entities: [concept] } : url.endsWith('/occurrences') ? { occurrences: [{ record_id: 'window_one', asset_id: 'asset', start_ms: 1000, end_ms: 4000 }] } : {};
    return new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
function mount(initialBinding?: { recordId: string; entityId: string }) {
  return render(<WorkEntities assetId="asset" title="测试作品" records={[record]} initialBinding={initialBinding} onSaved={vi.fn().mockResolvedValue(undefined)} />);
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Work entity local workflows', () => {
  it('creates an explicitly scoped dossier for this asset without guessing an anime identifier', async () => {
    const fetch = setup({ work_scope_id: null }); const user = userEvent.setup(); mount();
    await screen.findByText('为本集创建作品资料库');
    await user.type(screen.getByLabelText('作品版本'), 'TV');
    await user.click(screen.getByRole('button', { name: '创建并关联本集' }));
    await screen.findByText('workscope_one');
    const creation = fetch.mock.calls.find(([url]) => url === '/api/work-entity-scopes')!;
    expect(JSON.parse(creation[1]!.body as string)).toEqual({ title: '测试作品', version: 'TV', asset_ids: ['asset'] });
  });

  it('requires explicit existing scope ID before sharing another episode dossier', async () => {
    const fetch = setup({ work_scope_id: null }); const user = userEvent.setup(); mount();
    await screen.findByText('复用已有作品资料库');
    await user.type(screen.getByLabelText('已有作品资料库 ID'), 'workscope_one');
    await user.click(screen.getByRole('button', { name: '按 ID 关联本集' }));
    await screen.findByText('workscope_one');
    const attach = fetch.mock.calls.find(([url, options]) => url.endsWith('/work-entity-scope') && options?.method === 'PUT')!;
    expect(JSON.parse(attach[1]!.body as string)).toEqual({ work_scope_id: 'workscope_one' });
  });

  it('saves names, aliases, appearance and sourced facts without downloading source links', async () => {
    const fetch = setup(); const user = userEvent.setup(); mount();
    await screen.findByText('编辑实体资料');
    await user.click(screen.getByRole('button', { name: '新增实体' }));
    await user.selectOptions(screen.getByLabelText('实体类型'), 'mecha');
    await user.type(screen.getByLabelText('正式名称'), '银翼');
    await user.type(screen.getByLabelText('原文名称'), 'Silver Wing');
    await user.type(screen.getByLabelText('别名'), '一号机、白色机体');
    await user.type(screen.getByLabelText('区分性外观'), '双翼和蓝色胸灯');
    await user.click(screen.getByText('形态、设定事实、关系和出处（JSON）'));
    const source = screen.getByLabelText(/^资料出处 JSON/);
    await user.clear(source); await user.paste('[{"id":"s1","kind":"official","url":"https://example.test/source"}]');
    const facts = screen.getByLabelText(/^设定事实 JSON/);
    await user.clear(facts); await user.paste('[{"id":"f1","kind":"function","text":"能飞行","source_ids":["s1"]}]');
    await user.click(screen.getByRole('button', { name: '保存实体资料' }));
    await waitFor(() => expect(screen.queryByText('新增作品实体')).toBeNull());
    const creation = fetch.mock.calls.find(([url, options]) => url.endsWith('/work-entities') && options?.method === 'POST')!;
    expect(JSON.parse(creation[1]!.body as string)).toMatchObject({ type: 'mecha', name: '银翼', aliases: ['一号机', '白色机体'], facts: [{ source_ids: ['s1'] }], sources: [{ url: 'https://example.test/source' }] });
    expect(fetch.mock.calls.every(([url]) => url.startsWith('/api/'))).toBe(true);
  });

  it('adds registered frame references, requires an explicit confirmation and preserves reference revocation', async () => {
    const fetch = setup(); const user = userEvent.setup(); mount();
    await screen.findByText('多视角参考图');
    expect((screen.getByRole('button', { name: '确认参考' }) as HTMLButtonElement).disabled).toBe(true);
    await user.selectOptions(screen.getByLabelText('已注册原帧'), 'frame_one');
    await user.selectOptions(screen.getByLabelText('参考图类型'), 'object_region');
    await user.click(screen.getByLabelText(/^局部区域/)); await user.paste('[0.1,0.2,0.4,0.5]');
    await user.click(screen.getByRole('button', { name: '从原帧添加参考' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.endsWith('/references/from-frame'))).toBe(true));
    const addition = fetch.mock.calls.find(([url]) => url.endsWith('/references/from-frame'))!;
    expect(JSON.parse(addition[1]!.body as string)).toEqual({ frame_id: 'frame_one', view: 'object_region', box: [.1, .2, .4, .5], confirmed: false });
    await user.type(screen.getByLabelText('人工操作原因'), '核对原片发现错误');
    await user.click(screen.getByRole('button', { name: '撤销参考' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url, options]) => url.endsWith('/references/ref_one') && options?.method === 'PATCH')).toBe(true));
    const revoke = fetch.mock.calls.find(([url]) => url.endsWith('/references/ref_one'))!;
    expect(JSON.parse(revoke[1]!.body as string)).toEqual({ status: 'revoked', reason: '核对原片发现错误' });
  });

  it('uploads user-selected bytes and does not infer trust from a supplied source URL', async () => {
    const fetch = setup(); const user = userEvent.setup(); mount();
    await screen.findByText('多视角参考图');
    await user.upload(screen.getByLabelText('上传参考图'), new File(['png'], 'clock.png', { type: 'image/png' }));
    await user.type(screen.getByLabelText('参考图出处链接（只记录）'), 'https://example.test/image');
    await user.click(screen.getByRole('button', { name: '添加上传参考' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.endsWith('/references'))).toBe(true));
    const upload = fetch.mock.calls.find(([url]) => url.endsWith('/references'))!;
    const data = upload[1]!.body as FormData;
    expect(data.get('file')).toBeInstanceOf(File);
    expect(data.get('source_url')).toBe('https://example.test/image');
    expect(data.get('confirmed')).toBe('false');
  });

  it('binds the selected local entity with source frames and shows appearances separately from current events', async () => {
    const fetch = setup(); const user = userEvent.setup(); mount({ recordId: 'window_one', entityId: 'O1' });
    await screen.findByText('本集已确认出场');
    expect((screen.getByLabelText('画面观察') as HTMLSelectElement).value).toBe('window_one');
    expect((screen.getByLabelText('镜头局部实体') as HTMLSelectElement).value).toBe('O1');
    await user.type(screen.getByLabelText('人工操作原因'), '当前原帧中有金色表壳');
    await user.click(screen.getByRole('button', { name: '保存人工绑定' }));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.endsWith('/work-entity-bindings'))).toBe(true));
    const binding = fetch.mock.calls.find(([url]) => url.endsWith('/work-entity-bindings'))!;
    expect(JSON.parse(binding[1]!.body as string)).toEqual({ record_id: 'window_one', entity_id: 'O1', concept_id: 'concept_one', status: 'confirmed', evidence_frame_ids: ['frame_one', 'frame_two'], box: null, reason: '当前原帧中有金色表壳' });
    expect(screen.getByText(/设定功能是否发动仍需当前事件证据/)).toBeTruthy();
    expect(screen.getByRole('link', { name: '00:00:01 — 00:00:04' }).getAttribute('href')).toBe('#/asset/asset?t=1000');
  });
});
