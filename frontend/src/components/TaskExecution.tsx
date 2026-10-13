import { stageLabel, type ExecutionMode, type ExecutionOptions, type ExecutionStage } from '../types';
import { Field } from './ui';

export default function TaskExecution({ value, onChange, stages, disabled = false, connectionLimits = {}, defaults = false }: {
  value: ExecutionOptions;
  onChange: (value: ExecutionOptions) => void;
  stages: ExecutionStage[];
  disabled?: boolean;
  connectionLimits?: Partial<Record<ExecutionStage, number>>;
  defaults?: boolean;
}) {
  if (!stages.length) return null;
  const hasParallel = stages.some((stage) => value[stage] === 'parallel');
  return <section className="task-execution" aria-label="任务执行方式">
    <h3>{defaults ? '默认任务执行方式' : '任务执行方式'}</h3>
    <p className="helper">每个阶段可单独选择。并行使用该阶段的模型同时处理多个批次；串行按时间顺序逐批处理。</p>
    <div className="form-grid">{stages.map((stage) => <Field key={stage} label={`${stageLabel(stage)}执行方式`} hint={connectionLimits[stage] != null ? `连接并发上限 ${connectionLimits[stage]}；本次最多同时 ${value[stage] === 'parallel' ? Math.min(value.max_concurrency, connectionLimits[stage]!) : 1} 个请求` : undefined}>
      <select aria-label={`${stageLabel(stage)}执行方式`} value={value[stage]} disabled={disabled} onChange={(event) => onChange({ ...value, [stage]: event.target.value as ExecutionMode })}><option value="parallel">并行 · 同时处理多个批次</option><option value="serial">串行 · 按时间顺序处理</option></select>
    </Field>)}</div>
    <Field label="并发请求上限" hint="1–8 个；实际并发还受模型连接的最大并发限制，可在模型连接参数中调整。">
      <input aria-label="并发请求上限" type="number" min={1} max={8} step={1} required value={value.max_concurrency} disabled={disabled || !hasParallel} onChange={(event) => onChange({ ...value, max_concurrency: Number(event.target.value) })} />
    </Field>
    {stages.some((stage) => stage === 'transcript' || stage === 'preparation') && <p className="helper">并行台词整理与视觉分块共享开始时的此前记忆，并参考相邻的原始台词或画面证据，结果按时间顺序合并。深入理解与连续理解更新依赖前序结果，始终串行。</p>}
    {stages.includes('subtitle') && <p className="helper">字幕 OCR 可并行识别图片，完成后按时间顺序合并与去重。</p>}
    {defaults && <p className="helper">保存后用于新任务；创建时会保存执行方式，恢复或重试沿用原任务的选择。</p>}
  </section>;
}
