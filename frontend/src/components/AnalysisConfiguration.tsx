import { useApp } from '../context';
import { DEFAULTS, executionDefaults, profileModelLabel, type Defaults, type ExecutionStage, type Profile, type Settings, type TaskModelRole } from '../types';
import { Alert, Field } from './ui';
import TaskModels from './TaskModels';
import TaskExecution from './TaskExecution';

export function analysisDefaults(value: Partial<Defaults> = {}): Defaults {
  return { ...DEFAULTS, ...value, analysis_strategy: value.analysis_strategy === 'retrieval-first-v1' ? 'retrieval-first-v1' : 'scene-chunks-v1',
    retrieval_mode: value.retrieval_mode || 'core', stages: [...(value.stages || ['vision', 'subtitle'])],
    execution: executionDefaults(value.execution), model_bindings: { ...value.model_bindings }, model_overrides: { ...value.model_overrides },
    subtitle_crop: [...(value.subtitle_crop || DEFAULTS.subtitle_crop)] };
}

export function analysisRoles(value: Defaults, hasEmbedded = true): TaskModelRole[] {
  if (value.analysis_strategy === 'retrieval-first-v1') return value.retrieval_mode === 'complete' ? ['vision'] : [];
  return [...(value.stages?.includes('vision') ? ['preparation' as const, 'vision' as const] : []),
    ...(hasEmbedded && value.stages?.includes('subtitle') ? ['subtitle' as const] : [])];
}

export function analysisProfile(role: TaskModelRole, value: Defaults, settings: Settings, profiles: Profile[]) {
  return profiles.find((profile) => profile.id === (value.model_bindings?.[role] ?? settings.bindings[role]) && profile.capabilities.includes(role));
}

export function resolvedAnalysisDefaults(value: Defaults, settings: Settings, profiles: Profile[]): Defaults {
  return { ...value, model_overrides: Object.fromEntries(Object.entries(value.model_overrides || {}).filter(([role, selection]) => {
    const profile = analysisProfile(role as TaskModelRole, value, settings, profiles);
    return profile?.provider_type === 'codex_cli' && profile.id === selection.profile_id;
  })) };
}

export default function AnalysisConfiguration({ value, onChange, disabled = false, defaults = false, hasEmbedded = true }: {
  value: Defaults; onChange: (value: Defaults) => void; disabled?: boolean; defaults?: boolean; hasEmbedded?: boolean;
}) {
  const { bootstrap } = useApp();
  const encoderChecking = !!bootstrap.startup && ['pending', 'running'].includes(bootstrap.startup.encoder.status);
  const retrieval = value.analysis_strategy === 'retrieval-first-v1';
  const vision = retrieval || !!value.stages?.includes('vision');
  const roles = analysisRoles(value, hasEmbedded);
  const profile = (role: TaskModelRole) => analysisProfile(role, value, bootstrap.settings, bootstrap.provider_profiles);
  const usesCodex = roles.some((role) => profile(role)?.provider_type === 'codex_cli');
  const missing = roles.filter((role) => !profile(role));
  const executionStages: ExecutionStage[] = retrieval ? [] : [...(vision ? ['transcript' as const, 'preparation' as const] : []), ...(hasEmbedded && value.stages?.includes('subtitle') ? ['subtitle' as const] : [])];
  const update = (change: Partial<Defaults>) => onChange({ ...value, ...change });
  const toggle = (stage: 'vision' | 'subtitle', enabled: boolean) => update({ stages: enabled ? [...new Set([...(value.stages || []), stage])] : value.stages?.filter((entry) => entry !== stage) });
  const fixed = (label: string, option: string, hint: string) => <Field label={label} hint={hint}><select aria-label={label} disabled value="current"><option value="current">{option}</option></select></Field>;
  return <fieldset className="analysis-configuration" disabled={disabled} aria-label="任务处理配置">
    <Field label="处理策略" hint="选择这次需要的结果；每个阶段的处理方式和模型列在下面。"><select aria-label="处理策略" value={value.analysis_strategy} onChange={(event) => update({ analysis_strategy: event.target.value as Defaults['analysis_strategy'] })}>
      <option value="scene-chunks-v1">场景理解</option><option value="retrieval-first-v1">画面检索 · 可选 VLM</option>
    </select></Field>
    {retrieval ? <><label className="check-line"><input type="checkbox" checked={value.retrieval_mode === 'complete'} onChange={(event) => update({ retrieval_mode: event.target.checked ? 'complete' : 'core' })} /><span>使用 VLM 生成局部描述与事件</span></label>
      <Alert tone="info">{value.retrieval_mode === 'complete' ? '本地索引与人物匹配完成后，调用所选 VLM 生成局部描述和事件。' : '本地建立画面、已有字幕和人物检索索引，不调用生成式 VLM，也不生成局部描述和事件。人物匹配仍按作品识别设置执行。'}</Alert></> : <div className="choice-grid">
      <label className={`choice selectable ${vision ? 'selected' : ''}`}><input type="checkbox" checked={vision} onChange={(event) => toggle('vision', event.target.checked)} /><strong>画面理解</strong><span>台词整理、视觉分块与复杂片段深入理解</span></label>
      <label className={`choice selectable ${value.stages?.includes('subtitle') && hasEmbedded ? 'selected' : ''}`}><input type="checkbox" checked={!!value.stages?.includes('subtitle') && hasEmbedded} disabled={!hasEmbedded} onChange={(event) => toggle('subtitle', event.target.checked)} /><strong>画面字幕 OCR</strong><span>{hasEmbedded ? '仅对烧录在画面中的字幕执行' : '已有文字字幕在本地使用，无需 OCR'}</span></label>
    </div>}
    <section className="task-models" aria-label="本地处理阶段"><h3>本地处理与人工确认</h3><p className="helper">当前只有一种实现的阶段已锁定，保留为以后扩展的选项。开始任务后统一管理这些阶段的进度。</p>
      <div className="form-grid">
        {vision && fixed('关键帧划分方式', '本地内容变化检测 · PyAV / FFmpeg', '任务内自动划分；复用有效的扫描结果，选帧规则保持不变。')}
        {vision && fixed('角色确认方式', '人工核对候选角色', '在任务队列中确认、否定或明确跳过；未确认的候选不会自动成为实名。')}
        {vision && fixed('人物检测与匹配模型', '本地人物检测 + CCIP · 沿用作品配置', '按作品已启用的识别设置与参考图库执行。未启用时需先在角色档案中配置。')}
        {retrieval && fixed('画面索引模型', bootstrap.visual_encoder?.model || '本地图文编码器 · 当前已配置模型', encoderChecking ? '正在后台检查本机编码器与权重。' : bootstrap.visual_encoder?.available === false ? bootstrap.visual_encoder.reason || bootstrap.visual_encoder.message || '本机编码器尚不可用' : '使用本机编码器和权重，不发送画面给生成式模型。')}
        {fixed('文字字幕处理方式', '本地解析与时间索引', '使用导入时提取的容器字幕或外挂字幕。')}
      </div>
    </section>
    <TaskModels roles={roles} value={value.model_overrides} bindings={value.model_bindings} defaults={defaults} disabled={disabled}
      onBindingChange={(role, id) => { const bindings = { ...value.model_bindings }; const overrides = { ...value.model_overrides }; if (id) bindings[role] = id; else delete bindings[role]; delete overrides[role]; update({ model_bindings: bindings, model_overrides: overrides }); }}
      onChange={(role, selection) => { const overrides = { ...value.model_overrides }; if (selection) overrides[role] = selection; else delete overrides[role]; update({ model_overrides: overrides }); }} />
    {!!roles.length && <p className="helper">{roles.map((role) => `${({ preparation: '台词整理与视觉分块', vision: retrieval ? '局部描述与事件' : '深入理解', subtitle: '字幕 OCR' })[role]}：${profileModelLabel(profile(role))}`).join('；')}。{usesCodex ? '所选阶段会调用 Codex，使用已登录账号的订阅额度。' : '所选阶段会调用模型连接。'}</p>}
    {!!missing.length && <Alert>请为{missing.map((role) => ({ preparation: '台词整理与视觉分块', vision: '画面理解', subtitle: '字幕 OCR' })[role]).join('、')}选择可用连接；可在「模型与设置 → 模型连接」添加连接。</Alert>}
    <TaskExecution value={executionDefaults(value.execution)} stages={executionStages} onChange={(execution) => update({ execution })} defaults={defaults} disabled={disabled}
      connectionLimits={Object.fromEntries(executionStages.map((stage) => [stage, profile(stage === 'transcript' ? 'preparation' : stage)?.max_concurrency ?? 16]))} />
    {!retrieval && hasEmbedded && value.stages?.includes('subtitle') && <section className="task-models"><h3>字幕 OCR 采样与区域</h3>
      <Field label="画面字幕 OCR 采样（帧 / 秒）"><input type="number" min={0.2} max={10} step={0.1} required value={value.subtitle_fps} onChange={(event) => update({ subtitle_fps: Number(event.target.value) })} /></Field>
      <div className="form-grid four">{['左侧 X', '顶部 Y', '宽度', '高度'].map((label, index) => <Field key={label} label={label}><input type="number" min={index > 1 ? 0.01 : 0} max={index < 2 ? 1 - value.subtitle_crop[index + 2] : 1 - value.subtitle_crop[index - 2]} step={0.01} required value={value.subtitle_crop[index]} onChange={(event) => update({ subtitle_crop: value.subtitle_crop.map((current, position) => position === index ? Number(event.target.value) : current) })} /></Field>)}</div>
    </section>}
    <div className="form-grid"><Field label="每个视频最多模型请求数" hint={roles.length ? '达到上限后暂停；可在任务队列调整预算并继续。' : '当前流程不调用生成式模型；此值保留供切换方式后使用。'}><input aria-label="每个视频最多模型请求数" type="number" min={1} max={100000} step={1} required value={value.max_requests} onChange={(event) => update({ max_requests: Number(event.target.value) })} /></Field>
      <Field label="每个视频费用上限（USD）" hint={usesCodex ? 'Codex 订阅无法按 API 单价估算；当前使用请求数限制。' : '留空不限制；需要模型连接配置单价。'}><input aria-label="每个视频费用上限（USD）" type="number" min={0.01} step={0.01} placeholder="不限制" disabled={usesCodex || !roles.length} value={usesCodex ? '' : value.max_cost ?? ''} onChange={(event) => update({ max_cost: event.target.value ? Number(event.target.value) : null })} /></Field>
    </div>
  </fieldset>;
}
