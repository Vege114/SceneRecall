import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Cable, Save, Terminal } from 'lucide-react';
import { api, message } from '../api';
import { useApp } from '../context';
import { CAPABILITIES, DEFAULT_CONCURRENCY, executionDefaults, profileModelLabel, type ConcurrencySettingsResult, type ExecutionStage, type Settings } from '../types';
import TaskExecution, { ConcurrencyInput } from './TaskExecution';
import { Alert, Field, Spinner } from './ui';

const stages: ExecutionStage[] = ['transcript', 'preparation', 'subtitle'];
const stageNames = { transcript: '台词整理', preparation: '视觉分块', subtitle: '字幕 OCR' };
const validConcurrency = (value: number) => Number.isSafeInteger(value) && value > 0;

export default function ExecutionSettings({ onSaved }: { onSaved: (settings: Settings) => void }) {
  const { bootstrap, refresh, notify } = useApp();
  const [execution, setExecution] = useState(() => executionDefaults(bootstrap.settings.defaults.execution));
  const [limits, setLimits] = useState<Record<string, number>>(() => Object.fromEntries(bootstrap.provider_profiles.map((profile) => [profile.id, profile.max_concurrency ?? DEFAULT_CONCURRENCY])));
  const [bindings, setBindings] = useState(bootstrap.settings.bindings);
  const [unifiedLimit, setUnifiedLimit] = useState(DEFAULT_CONCURRENCY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [refreshError, setRefreshError] = useState('');
  const [saved, setSaved] = useState(false);
  const dirty = useRef(false);
  const sourceKey = JSON.stringify([bootstrap.settings.defaults.execution, bootstrap.settings.bindings, bootstrap.provider_profiles.map((profile) => [profile.id, profile.max_concurrency])]);
  useEffect(() => {
    setBindings(bootstrap.settings.bindings);
    if (dirty.current) return;
    setExecution(executionDefaults(bootstrap.settings.defaults.execution));
    setLimits(Object.fromEntries(bootstrap.provider_profiles.map((profile) => [profile.id, profile.max_concurrency ?? DEFAULT_CONCURRENCY])));
  }, [sourceKey]);
  const changed = () => { dirty.current = true; setSaved(false); setRefreshError(''); };
  const applyUnified = () => {
    if (!validConcurrency(unifiedLimit)) return;
    changed();
    setExecution((current) => ({ ...current, max_concurrency: unifiedLimit }));
    setLimits(Object.fromEntries(bootstrap.provider_profiles.map((profile) => [profile.id, unifiedLimit])));
  };
  const save = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError(''); setRefreshError(''); setSaved(false);
    try {
      const profiles = Object.fromEntries(bootstrap.provider_profiles.map((profile) => [profile.id, limits[profile.id] ?? profile.max_concurrency ?? DEFAULT_CONCURRENCY]));
      if (!validConcurrency(execution.max_concurrency) || Object.values(profiles).some((limit) => !validConcurrency(limit))) throw new Error('并发上限必须是浏览器可精确表示的正整数。');
      const result = await api<ConcurrencySettingsResult>('/concurrency', { method: 'PUT', body: JSON.stringify({ execution, profiles }) });
      setExecution(executionDefaults(result.settings.defaults.execution));
      setLimits(Object.fromEntries(result.provider_profiles.map((profile) => [profile.id, profile.max_concurrency ?? DEFAULT_CONCURRENCY])));
      setBindings(result.settings.bindings);
      dirty.current = false; setSaved(true); onSaved(result.settings);
      try { await refresh(); notify('执行与并发已保存'); }
      catch (error) {
        const text = `执行与并发已保存，但刷新失败：${message(error)}`;
        setRefreshError(text); notify(text, 'error');
      }
    } catch (error) { setError(message(error)); }
    finally { setBusy(false); }
  };
  return <form className="settings-form execution-settings" onSubmit={save}>
    <div className="settings-section-title"><h2>执行与并发</h2><p>集中管理新任务的默认执行方式、任务上限与各模型连接上限。创建任务时仍可临时调整。</p></div>
    <div className="concurrency-unified">
      <Field label="统一并发上限" hint="应用到默认任务与下方所有连接，再保存生效。"><ConcurrencyInput label="统一并发上限" value={unifiedLimit} onChange={setUnifiedLimit} disabled={busy} /></Field>
      <button className="button secondary" type="button" disabled={busy || !validConcurrency(unifiedLimit)} onClick={applyUnified}>全部设为 {validConcurrency(unifiedLimit) ? unifiedLimit : '统一值'}</button>
    </div>
    <TaskExecution defaults stages={stages} value={execution} onChange={(next) => { changed(); setExecution(next); }} disabled={busy} />
    <section aria-label="模型连接并发上限">
      <h3 className="section-label">模型连接并发上限</h3>
      <p className="helper">同一个连接用于多个能力时，共享这里的上限。还可单独微调每个连接。</p>
      <div className="connection-concurrency-list">{bootstrap.provider_profiles.map((profile) => <div className="connection-concurrency-row" key={profile.id}>
        <div className="connection-concurrency-info">
          <div className="connection-concurrency-name">{profile.provider_type === 'codex_cli' ? <Terminal size={17} /> : <Cable size={17} />}<h3>{profile.name}</h3></div>
          <p className="helper">{profile.provider_type === 'codex_cli' ? '本地 Codex CLI' : 'OpenAI-compatible API'} · {profileModelLabel(profile)}</p>
          <div className="profile-tags">{profile.capabilities.map((capability) => <span className="tag" key={capability}>{CAPABILITIES.find((item) => item.key === capability)?.name || capability}</span>)}</div>
        </div>
        <Field label="连接并发上限"><ConcurrencyInput label={`${profile.name} 连接并发上限`} value={limits[profile.id] ?? profile.max_concurrency ?? DEFAULT_CONCURRENCY} disabled={busy} onChange={(limit) => { changed(); setLimits((current) => ({ ...current, [profile.id]: limit })); }} /></Field>
      </div>)}</div>
      {!bootstrap.provider_profiles.length && <p className="helper">尚未添加模型连接；可先保存任务默认值，再到「模型连接」添加。</p>}
    </section>
    <section className="concurrency-preview" aria-label="默认实际并发">
      <h3 className="section-label">默认实际并发</h3>
      {stages.map((stage) => {
        const profile = bootstrap.provider_profiles.find((item) => item.id === bindings[stage === 'transcript' ? 'preparation' : stage]);
        const limit = profile ? limits[profile.id] ?? profile.max_concurrency ?? DEFAULT_CONCURRENCY : null;
        const valid = limit != null && validConcurrency(limit) && validConcurrency(execution.max_concurrency);
        return <div className="concurrency-preview-row" key={stage} aria-label={`${stageNames[stage]}默认并发`}><div><strong>{stageNames[stage]}</strong><span>{profile?.name || '尚未分配模型连接'}</span></div><p>{!profile ? '尚未分配模型连接' : execution[stage] === 'serial' ? '串行 · 最多同时 1 个请求' : valid ? `最多同时 ${Math.min(execution.max_concurrency, limit!)} 个请求（任务 ${execution.max_concurrency} · 连接 ${limit}）` : '请输入有效的并发上限'}</p></div>;
      })}
      <p className="helper">深入理解与连续理解更新始终串行。实际处理速度还取决于剩余批次和模型响应。</p>
    </section>
    {error && <Alert tone="error">{error}</Alert>}
    {refreshError ? <Alert>{refreshError}</Alert> : saved && <Alert tone="success">执行与并发已保存</Alert>}
    <div className="form-actions"><button className="button primary" type="submit" disabled={busy}>{busy ? <Spinner label="保存中" /> : <><Save size={16} />保存执行与并发</>}</button></div>
  </form>;
}
