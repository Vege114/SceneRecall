import { useCallback, useEffect, useState } from 'react';
import { api, message } from '../api';
import { useApp } from '../context';
import { profileModelLabel, type CodexModelCatalog, type TaskModelOverride, type TaskModelOverrides, type TaskModelRole } from '../types';
import { Alert, Field, Spinner } from './ui';

const roleLabels: Record<TaskModelRole, string> = { preparation: '分块与简要索引', vision: '画面理解', subtitle: '画面字幕 OCR' };
const effortLabels: Record<string, string> = { none: '不思考', minimal: '最少', low: '低', medium: '中', high: '高', xhigh: '很高', max: '最高', ultra: '极高' };
const effortLabel = (effort: string) => effortLabels[effort] ? `${effortLabels[effort]}（${effort}）` : effort;

export function taskModelOptions(overrides?: TaskModelOverrides, includeSubtitle = true) {
  const values = Object.fromEntries(Object.entries(overrides || {}).filter(([role]) => includeSubtitle || role !== 'subtitle'));
  return Object.keys(values).length ? { model_overrides: values as TaskModelOverrides } : {};
}

export function useTaskModels(roles: TaskModelRole[]) {
  const { bootstrap } = useApp();
  const [selections, setSelections] = useState<TaskModelOverrides>({});
  const change = useCallback((role: TaskModelRole, value?: TaskModelOverride) => {
    setSelections((current) => {
      const next = { ...current };
      if (value) next[role] = value;
      else delete next[role];
      return next;
    });
  }, []);
  const overrides: TaskModelOverrides = {};
  const sources: unknown[] = [];
  for (const role of roles) {
    const profile = bootstrap.provider_profiles.find((entry) => entry.id === bootstrap.settings.bindings[role]);
    sources.push([role, profile?.id, profile?.provider_type, profile?.model]);
    const value = selections[role];
    if (value && profile?.provider_type === 'codex_cli' && value.profile_id === profile.id) overrides[role] = value;
  }
  return { overrides: Object.keys(overrides).length ? overrides : undefined, change, sourceKey: JSON.stringify(sources) };
}

export default function TaskModels({ roles, value, onChange, disabled = false }: {
  roles: TaskModelRole[];
  value?: TaskModelOverrides;
  onChange: (role: TaskModelRole, value?: TaskModelOverride) => void;
  disabled?: boolean;
}) {
  const { bootstrap } = useApp();
  const [catalog, setCatalog] = useState<CodexModelCatalog | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const stages = roles.map((role) => ({ role, profile: bootstrap.provider_profiles.find((entry) => entry.id === bootstrap.settings.bindings[role]) }));
  const hasCodex = stages.some(({ profile }) => profile?.provider_type === 'codex_cli');
  const cacheTime = catalog?.fetched_at ? new Date(catalog.fetched_at) : null;
  useEffect(() => {
    if (!hasCodex) return;
    let active = true;
    setLoading(true); setError('');
    void api<CodexModelCatalog>('/codex/models').then((result) => {
      if (!Array.isArray(result.models)) throw new Error('模型目录格式无效');
      if (active) setCatalog(result);
    }).catch((caught) => { if (active) { setCatalog(null); setError(message(caught)); } }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [hasCodex, attempt]);
  useEffect(() => {
    if (!catalog) return;
    for (const { role, profile } of stages) {
      const selection = value?.[role];
      if (!selection?.reasoning_effort || profile?.provider_type !== 'codex_cli') continue;
      const model = catalog.models.find((entry) => entry.id === ((selection.model ?? profile.model) || catalog.default_model));
      if (model && !model.reasoning_efforts.includes(selection.reasoning_effort)) onChange(role, { ...selection, reasoning_effort: null });
    }
  }, [catalog, value, onChange, stages]);
  if (!roles.length) return null;
  return <section className="task-models" aria-label="本次任务模型设置">
    <h3>本次任务的模型</h3>
    <p className="helper">按阶段选择模型与思考等级，仅用于本次任务。未调整时沿用连接设置。</p>
    {stages.map(({ role, profile }) => {
      const label = roleLabels[role];
      const selection = value?.[role];
      const effectiveModel = (selection?.model ?? profile?.model) || catalog?.default_model;
      const model = catalog?.models.find((entry) => entry.id === effectiveModel);
      return <div className="task-model-stage" key={role} aria-label={`${label}模型设置`}>
        <div className="task-model-source"><strong>{label}</strong><span>{profile ? `${profile.provider_type === 'codex_cli' ? 'Codex CLI · ChatGPT 订阅' : 'API'} · ${profile.name}` : '尚未分配连接'}</span></div>
        {profile?.provider_type === 'codex_cli' ? <div className="form-grid">
          <Field label={`${label}模型`}>
            <select disabled={disabled} value={selection?.model === undefined ? 'inherit' : `model:${selection.model}`} onChange={(event) => {
              const selected = event.target.value;
              onChange(role, selected === 'inherit' ? undefined : { provider_type: 'codex_cli', profile_id: profile.id, model: selected.slice(6), reasoning_effort: null });
            }}>
              <option value="inherit">沿用连接（{profileModelLabel(profile)}）</option>
              <option value="model:">Codex 内置默认{catalog?.default_model ? `（${catalog.default_model}）` : ''}</option>
              {catalog?.models.map((entry) => <option key={entry.id} value={`model:${entry.id}`}>{entry.name === entry.id ? entry.id : `${entry.name} · ${entry.id}`}</option>)}
              {selection?.model && !catalog?.models.some((entry) => entry.id === selection.model) && <option value={`model:${selection.model}`}>{selection.model}（目录暂不可用）</option>}
            </select>
          </Field>
          <Field label={`${label}思考等级`} hint={!model ? selection?.reasoning_effort ? '目录暂不可用，保留本次已选等级；重新读取目录后可调整' : '选择目录中的模型后可指定思考等级；默认或目录外模型沿用默认等级' : undefined}>
            <select disabled={disabled || !model?.reasoning_efforts.length} value={selection?.reasoning_effort || ''} onChange={(event) => {
              const effort = event.target.value || null;
              onChange(role, !effort && selection?.model === undefined ? undefined : { provider_type: 'codex_cli', profile_id: profile.id, ...(selection?.model !== undefined ? { model: selection.model } : {}), reasoning_effort: effort });
            }}>
              <option value="">模型默认{model?.default_reasoning_effort ? `（${effortLabel(model.default_reasoning_effort)}）` : ''}</option>
              {model?.reasoning_efforts.map((effort) => <option key={effort} value={effort}>{effortLabel(effort)}</option>)}
              {!model && selection?.reasoning_effort && <option value={selection.reasoning_effort}>{effortLabel(selection.reasoning_effort)} · 已选等级，目录暂不可验证</option>}
            </select>
          </Field>
        </div> : <p className="helper">{profileModelLabel(profile)}{profile ? ' · API 来源暂沿用连接配置' : ''}</p>}
      </div>;
    })}
    {hasCodex && <>
      {loading && <Spinner label="正在读取 Codex 模型目录" />}
      {error && <Alert>无法读取 Codex 模型目录：{error}。仍可沿用连接或选择 Codex 内置默认。<button type="button" className="text-button" disabled={disabled || loading} onClick={() => setAttempt((current) => current + 1)}>重试读取模型</button></Alert>}
      {!loading && !error && catalog?.models.length === 0 && <p className="helper">当前目录没有可选模型，可沿用连接或使用 Codex 内置默认。</p>}
      {!error && <button type="button" className="text-button" disabled={disabled || loading} onClick={() => setAttempt((current) => current + 1)}>重新读取模型</button>}
      {catalog?.message && <p className="helper">{catalog.message}</p>}
      <p className="helper">模型列表来自本机 Codex 缓存{cacheTime && !Number.isNaN(cacheTime.getTime()) ? `（${cacheTime.toLocaleString('zh-CN')}）` : ''}，实际可用性取决于当前账号与 Codex 服务。</p>
    </>}
  </section>;
}
