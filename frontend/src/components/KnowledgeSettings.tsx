import { useEffect, useState, type FormEvent } from 'react';
import { api, message, post } from '../api';
import type { Profile } from '../types';
import { Alert, Field, Spinner } from './ui';

interface Configuration {
  enabled: boolean;
  text_profile_id: string;
  text_model: string;
  text_thinking: '' | 'disabled' | 'enabled';
  reasoning_effort: string;
  embedding_profile_id: string;
  embedding_backend: 'local' | 'api';
  local_model_dir: string;
  query_timeout_s: number;
  python: string;
}
const initial: Configuration = { enabled: false, text_profile_id: '', text_model: '', text_thinking: '', reasoning_effort: '', embedding_profile_id: '', embedding_backend: 'local', local_model_dir: '', query_timeout_s: 60, python: '' };
function configuration(value: Partial<Configuration>): Configuration {
  const result = { ...initial };
  for (const key of Object.keys(initial) as (keyof Configuration)[]) {
    const field = value[key];
    if (field != null) Object.assign(result, { [key]: field });
  }
  return result;
}

export function KnowledgeSettingsSection({ profiles }: { profiles: Profile[] }) {
  const [open, setOpen] = useState(false);
  return <details className="advanced" onToggle={(event) => setOpen(event.currentTarget.open)}><summary>作品知识库后端 · LightRAG</summary>{open && <KnowledgeSettings profiles={profiles} />}</details>;
}

export default function KnowledgeSettings({ profiles }: { profiles: Profile[] }) {
  const [config, setConfig] = useState(initial);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [check, setCheck] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    let active = true;
    void api<Partial<Configuration>>('/knowledge/settings').then((value) => { if (active) setConfig(configuration(value)); }).catch((caught) => { if (active) setError(message(caught)); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);
  const textProfiles = profiles.filter((profile) => profile.capabilities.includes('query'));
  const textProfile = textProfiles.find((profile) => profile.id === config.text_profile_id);
  const embeddings = profiles.filter((profile) => profile.provider_type !== 'codex_cli' && profile.capabilities.includes('embedding'));
  const save = async (event: FormEvent) => {
    event.preventDefault(); setBusy('save'); setError(''); setNotice(''); setCheck(null);
    try { const value = await api<Configuration>('/knowledge/settings', { method: 'PUT', body: JSON.stringify({ ...config, text_profile_id: config.text_profile_id || null, embedding_profile_id: config.embedding_profile_id || null, reasoning_effort: config.reasoning_effort || null, text_thinking: textProfile?.provider_type === 'codex_cli' ? null : config.text_thinking || null }) }); setConfig(configuration(value)); setNotice(config.enabled ? '知识后端设置已保存；连接检查通过后可在作品设定库建库。' : '知识后端已关闭，检索继续使用词法基线。'); } catch (caught) { setError(message(caught)); } finally { setBusy(''); }
  };
  const test = async () => {
    setBusy('check'); setError(''); setNotice('');
    try { setCheck(await post<Record<string, unknown>>('/knowledge/check')); } catch (caught) { setError(message(caught)); } finally { setBusy(''); }
  };
  return <section className="data-card knowledge-settings" aria-label="知识后端设置">
    <h2>作品知识库 · LightRAG 实验</h2><p>文本抽取和查询处理复用已有连接；默认使用本地多语言文本向量。</p>
    {loading ? <Spinner label="读取知识后端设置" /> : <form onSubmit={save}>
      <label className="check-line"><input type="checkbox" checked={config.enabled} onChange={(event) => setConfig({ ...config, enabled: event.target.checked })} />启用 LightRAG 知识检索</label>
      <div className="form-grid"><Field label="知识文本模型连接"><select value={config.text_profile_id} onChange={(event) => { const profile = profiles.find((row) => row.id === event.target.value); setConfig({ ...config, text_profile_id: event.target.value, text_model: profile?.model || '', text_thinking: profile?.provider_type === 'codex_cli' ? '' : config.text_thinking }); }}><option value="">选择具有查询能力的连接</option>{textProfiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}</select></Field><Field label="知识文本模型 ID" hint="填写明确的可用模型 ID，便于索引和实验复现。"><input value={config.text_model} onChange={(event) => setConfig({ ...config, text_model: event.target.value })} required={config.enabled} /></Field><Field label="知识文本思考等级"><select value={config.reasoning_effort || ''} onChange={(event) => setConfig({ ...config, reasoning_effort: event.target.value })}><option value="">沿用连接默认</option>{['minimal', 'low', 'medium', 'high', 'xhigh', 'max'].map((value) => <option value={value} key={value}>{value}</option>)}</select></Field>{textProfile?.provider_type === 'openai_compatible' && <Field label="知识文本 API 思考模式" hint="服务默认不发送 thinking 参数；仅为支持该参数的 API 显式选择。"><select value={config.text_thinking} onChange={(event) => setConfig({ ...config, text_thinking: event.target.value as Configuration['text_thinking'] })}><option value="">服务默认</option><option value="disabled">关闭思考</option><option value="enabled">启用思考</option></select></Field>}<Field label="知识向量运行方式"><select value={config.embedding_backend} onChange={(event) => setConfig({ ...config, embedding_backend: event.target.value as Configuration['embedding_backend'] })}><option value="local">本地多语言 embedding（CPU）</option><option value="api">已有 embedding API 连接</option></select></Field>{config.embedding_backend === 'local' ? <Field label="本地 embedding 模型目录" hint="通过知识运行环境安装脚本显式下载模型，再填写模型目录；推理时只读取本地文件。"><input value={config.local_model_dir} onChange={(event) => setConfig({ ...config, local_model_dir: event.target.value })} /></Field> : <Field label="知识 embedding 连接"><select value={config.embedding_profile_id} onChange={(event) => setConfig({ ...config, embedding_profile_id: event.target.value })}><option value="">选择 embedding API 连接</option>{embeddings.map((profile) => <option key={profile.id} value={profile.id}>{profile.name} · {profile.model}</option>)}</select></Field>}<Field label="知识查询超时（秒）"><input type="number" min="1" max="600" value={config.query_timeout_s} onChange={(event) => setConfig({ ...config, query_timeout_s: Number(event.target.value) })} /></Field></div>
      {config.embedding_backend === 'api' && !embeddings.length && <Alert>尚无 embedding 连接。请先添加声明“语义向量”能力的 API 连接。</Alert>}
      <details><summary>独立运行环境</summary><Field label="知识 worker Python 路径" hint="填写安装脚本输出的独立 Python 3.12 解释器路径。"><input value={config.python} onChange={(event) => setConfig({ ...config, python: event.target.value })} placeholder=".cache\knowledge-runtime\Scripts\python.exe" /></Field><p className="helper">固定 LightRAG 1.5.7。图和向量会占用内存，关闭后端可释放工作进程。</p></details>
      <div className="button-row"><button className="button primary" disabled={!!busy}>{busy === 'save' ? <Spinner label="保存中" /> : '保存知识后端设置'}</button><button type="button" className="button secondary" disabled={!!busy} onClick={() => void test()}>{busy === 'check' ? <Spinner label="检查中" /> : '检查知识后端连接'}</button><button type="button" className="button secondary" disabled={!!busy} onClick={async () => { setBusy('close'); setError(''); try { await post('/knowledge/close'); setNotice('知识工作进程已停止；资料和索引保留。'); } catch (caught) { setError(message(caught)); } finally { setBusy(''); } }}>停止知识工作进程</button></div>
      <p className="helper">本地向量模型固定为多语言 MiniLM-L12-v2（CPU、384 维）。连接检查使用已保存设置，可能产生少量文本模型用量；选择 API 向量时另有 embedding 用量。</p>
    </form>}
    {notice && <Alert tone="success">{notice}</Alert>}{error && <Alert tone="error">{error}</Alert>}{check && <div role="status"><p>{check.ok === true || check.status === 'ready' ? '知识后端检查通过' : '知识后端检查结果'}</p>{check.message ? <p>{String(check.message)}</p> : null}<details><summary>模型与调用诊断</summary><pre className="json-report">{JSON.stringify(check, null, 2)}</pre></details></div>}
  </section>;
}
