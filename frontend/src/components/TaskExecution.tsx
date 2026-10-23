import { useEffect, useRef } from 'react';
import { api, message } from '../api';
import { stageLabel, type ExecutionMode, type ExecutionOptions, type ExecutionStage } from '../types';
import { Field } from './ui';

const validConcurrency = (value: number) => Number.isSafeInteger(value) && value > 0;

export async function saveExecutionDefaults(value: ExecutionOptions, stages: ExecutionStage[]) {
  const execution = { max_concurrency: value.max_concurrency, ...Object.fromEntries(stages.map((stage) => [stage, value[stage]])) };
  try { await api('/settings', { method: 'PUT', body: JSON.stringify({ defaults: { execution } }) }); }
  catch (error) { throw new Error(`保存默认执行方式失败：${message(error)}。任务尚未创建，请重试或取消“设为默认值”。`); }
}

export function ConcurrencyInput({ value, onChange, disabled = false, label }: {
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  label: string;
}) {
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    input.current?.setCustomValidity(validConcurrency(value) ? '' : '请输入浏览器可精确表示的正整数。');
  }, [value]);
  return <input ref={input} aria-label={label} type="number" min={1} step={1} required value={value} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))} />;
}

export default function TaskExecution({ value, onChange, stages, disabled = false, connectionLimits = {}, defaults = false, saveDefault = false, onSaveDefaultChange }: {
  value: ExecutionOptions;
  onChange: (value: ExecutionOptions) => void;
  stages: ExecutionStage[];
  disabled?: boolean;
  connectionLimits?: Partial<Record<ExecutionStage, number>>;
  defaults?: boolean;
  saveDefault?: boolean;
  onSaveDefaultChange?: (value: boolean) => void;
}) {
  if (!stages.length) return null;
  const hasParallel = stages.some((stage) => value[stage] === 'parallel');
  return <section className="task-execution" aria-label="任务执行方式">
    <h3>{defaults ? '默认任务执行方式' : '任务执行方式'}</h3>
    <p className="helper">每个阶段可单独选择。并行使用该阶段的模型同时处理多个批次；串行按时间顺序逐批处理。</p>
    <div className="form-grid">{stages.map((stage) => <Field key={stage} label={`${stageLabel(stage)}执行方式`} hint={connectionLimits[stage] != null ? `连接并发上限 ${connectionLimits[stage]}；本次最多同时 ${value[stage] === 'parallel' ? Math.min(value.max_concurrency, connectionLimits[stage]!) : 1} 个请求` : undefined}>
      <select aria-label={`${stageLabel(stage)}执行方式`} value={value[stage]} disabled={disabled} onChange={(event) => onChange({ ...value, [stage]: event.target.value as ExecutionMode })}><option value="parallel">并行 · 同时处理多个批次</option><option value="serial">串行 · 按时间顺序处理</option></select>
    </Field>)}</div>
    <Field label="并发请求上限" hint="自行设置正整数；实际并发取任务与模型连接上限中的较小值，还取决于剩余批次与服务响应。串行阶段最多同时 1 个请求。">
      <ConcurrencyInput label="并发请求上限" value={value.max_concurrency} disabled={disabled || (!defaults && !hasParallel && validConcurrency(value.max_concurrency))} onChange={(max_concurrency) => onChange({ ...value, max_concurrency })} />
    </Field>
    {!defaults && <p className="helper">这里调整本次任务；长期设置在「模型与设置 → 默认任务配置」管理，连接并发上限在「执行与并发」管理。</p>}
    {onSaveDefaultChange && <><label className="check-line"><input type="checkbox" checked={saveDefault} disabled={disabled} onChange={(event) => onSaveDefaultChange(event.target.checked)} /><span>设为默认值</span></label><p className="helper">开始任务时，将本次并发上限与当前阶段的执行方式保存为后续任务的默认值。</p></>}
    {stages.some((stage) => stage === 'transcript' || stage === 'preparation') && <p className="helper">新任务的各集台词先独立并行整理，再按集理解画面；视觉分块参考此前记忆与相邻原始画面，结果按时间顺序合并。深入理解与连续理解更新依赖前序结果，始终串行。</p>}
    {stages.includes('subtitle') && <p className="helper">字幕 OCR 可并行识别图片，完成后按时间顺序合并与去重。</p>}
    {defaults && <p className="helper">保存后用于新任务；创建时会保存执行方式，恢复或重试沿用原任务的选择。</p>}
  </section>;
}
