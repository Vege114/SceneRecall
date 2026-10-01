import { useEffect, useRef, useState, type FormEvent } from 'react';
import { ArrowUpRight, Cable, Check, Database, Download, HardDrive, KeyRound, Pencil, PlugZap, Plus, RefreshCw, Save, ShieldCheck, Terminal, Trash2, Upload } from 'lucide-react';
import { api, message, post } from '../api';
import { useApp } from '../context';
import { CAPABILITIES, DEFAULTS, profileModelLabel, type Capability, type CodexStatus, type Profile, type ProviderType, type Settings as SettingsType } from '../types';
import { Alert, Empty, Field, Modal, PageHeader, Spinner } from './ui';

function CodexConnectionStatus() {
  const [status, setStatus] = useState<CodexStatus | null>(null);
  const [checking, setChecking] = useState(true);
  const [error, setError] = useState('');
  const [checkCount, setCheckCount] = useState(0);
  useEffect(() => {
    let active = true;
    setChecking(true); setError(''); setStatus(null);
    api<CodexStatus>('/codex/status')
      .then((result) => { if (active) setStatus(result); })
      .catch((error) => { if (active) setError(message(error)); })
      .finally(() => { if (active) setChecking(false); });
    return () => { active = false; };
  }, [checkCount]);
  const ready = status?.installed && status.authenticated && status.compatible !== false;
  return <section className="codex-connection" aria-label="Codex CLI 状态">
    <div className="section-row"><h3><Terminal size={17} />本地 Codex CLI</h3><button type="button" className="button small secondary" disabled={checking} onClick={() => setCheckCount((count) => count + 1)}>{checking ? <Spinner label="检查中" /> : <><RefreshCw size={14} />重新检查</>}</button></div>
    {status && <Alert tone={ready ? 'success' : 'warning'}><strong>{ready ? '已就绪' : !status.installed ? '未检测到 Codex CLI' : status.compatible === false ? 'Codex CLI 需要升级' : '需要登录 ChatGPT'}</strong>{status.version && <span className="codex-version">{status.version}</span>}<p>{status.message}</p></Alert>}
    {error && <Alert tone="error">{error}</Alert>}
    <p className="helper">在运行 SceneRecall 的电脑上，以同一系统用户安装 Codex CLI，执行 <code>codex login</code> 并选择 ChatGPT 登录，然后重新检查。无需额外启动路由服务。</p>
    <p className="helper">使用已登录账号的 Codex 订阅额度；剩余额度与美元费用无法在此估算。画面和文本会通过 CLI 发送给 Codex。</p>
  </section>;
}

function ProfileModal({ profile, onClose }: { profile?: Profile; onClose: () => void }) {
  const { bootstrap, refresh, notify } = useApp();
  const [providerType, setProviderType] = useState<ProviderType>(profile?.provider_type || 'openai_compatible');
  const isCodex = providerType === 'codex_cli';
  const [name, setName] = useState(profile?.name || '');
  const [url, setUrl] = useState(profile?.base_url || '');
  const [model, setModel] = useState(profile?.model || '');
  const [capabilities, setCapabilities] = useState<Capability[]>(profile?.capabilities || ['vision']);
  const [secretMode, setSecretMode] = useState<Profile['secret_mode']>(profile?.secret_mode || 'session');
  const [apiKey, setApiKey] = useState('');
  const [envVar, setEnvVar] = useState(profile?.env_var || '');
  const [timeout, setTimeout] = useState(profile?.timeout_s || 90);
  const [concurrency, setConcurrency] = useState(profile?.max_concurrency || 2);
  const [inputPrice, setInputPrice] = useState(profile?.input_price_per_million?.toString() || '');
  const [outputPrice, setOutputPrice] = useState(profile?.output_price_per_million?.toString() || '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const removedBindings = profile ? CAPABILITIES.filter((capability) => bootstrap.settings.bindings[capability.key] === profile.id && (!capabilities.includes(capability.key) || (isCodex && capability.key === 'embedding'))) : [];
  const bindingConflict = removedBindings.length ? `请先在能力分配中解除「${removedBindings.map((capability) => capability.name).join('、')}」的绑定并保存，再修改这个连接。` : '';
  const changeProvider = (next: ProviderType) => {
    setProviderType(next); setUrl(''); setModel(''); setApiKey(''); setEnvVar(''); setInputPrice(''); setOutputPrice(''); setError('');
    setSecretMode(next === 'codex_cli' ? 'none' : 'session');
    if (next === 'codex_cli') setCapabilities((current) => current.filter((capability) => capability !== 'embedding'));
  };
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError('');
    try {
      if (bindingConflict) throw new Error(bindingConflict);
      if (!capabilities.length) throw new Error('至少选择一种模型能力。');
      await post('/profiles', {
        ...(profile ? { id: profile.id } : {}), name: name.trim(), provider_type: providerType,
        base_url: isCodex ? '' : url.trim().replace(/\/+$/, ''), model: model.trim(),
        capabilities: isCodex ? capabilities.filter((capability) => capability !== 'embedding') : capabilities,
        secret_mode: isCodex ? 'none' : secretMode, ...(!isCodex && apiKey ? { api_key: apiKey } : {}),
        env_var: isCodex ? null : envVar || null, timeout_s: timeout, max_concurrency: concurrency,
        input_price_per_million: isCodex || inputPrice === '' ? null : Number(inputPrice),
        output_price_per_million: isCodex || outputPrice === '' ? null : Number(outputPrice),
      });
      setApiKey(''); await refresh(); notify('模型连接已保存'); onClose();
    } catch (error) { setError(message(error)); } finally { setBusy(false); }
  };
  return <Modal title={profile ? '编辑模型连接' : '添加模型连接'} subtitle="选择 API 接口或本地 Codex CLI，再分配模型能力。" onClose={onClose} wide>
    <form onSubmit={submit}><div className="modal-body">
      <Field label="接入方式"><select value={providerType} onChange={(e) => changeProvider(e.target.value as ProviderType)}><option value="openai_compatible">OpenAI-compatible API</option><option value="codex_cli">本地 Codex CLI（ChatGPT 订阅）</option></select></Field>
      {isCodex && <CodexConnectionStatus />}
      <div className="form-grid"><Field label="连接名称"><input value={name} onChange={(e) => setName(e.target.value)} placeholder={isCodex ? '例如：我的 Codex 订阅' : '例如：我的视觉模型'} required /></Field><Field label={isCodex ? '模型 ID（可选）' : '模型 ID'} hint={isCodex ? '留空使用 Codex CLI 内置默认模型；自选模型请填写 ID。' : undefined}><input value={model} onChange={(e) => setModel(e.target.value)} placeholder={isCodex ? 'Codex 默认模型' : '供应商提供的模型名'} required={!isCodex} /></Field></div>
      {!isCodex && <><Field label="Base URL" hint="填写兼容接口地址；本地模型同样支持。"><input type="url" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://api.example.com/v1" required /></Field><Field label="凭据保存方式"><select value={secretMode} onChange={(e) => setSecretMode(e.target.value as Profile['secret_mode'])}><option value="session">仅当前服务会话</option><option value="keyring">系统凭据库（推荐持久保存）</option><option value="env">读取环境变量</option><option value="none">无需 Key（本地服务）</option></select></Field>{secretMode === 'env' ? <Field label="环境变量名称" hint="启动服务前设置。这里只填写变量名称，不填写密钥。"><input value={envVar} onChange={(e) => setEnvVar(e.target.value)} placeholder="SCENERECALL_AI_KEY" required /></Field> : secretMode !== 'none' && <Field label="API Key" hint={profile ? '留空保留已有密钥。密钥不会显示在配置、日志或导出文件中。' : secretMode === 'session' ? '仅保存在后端内存；重启服务后需重新填写。' : '保存在操作系统凭据库，资料备份不包含密钥。'}><input type="password" autoComplete="new-password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder={profile ? '••••••••（留空不修改）' : '输入 API Key'} /></Field>}</>}
      <div className="field-label section-label">这个模型支持什么？</div><div className="capability-checks">{CAPABILITIES.map((capability) => {
        const unsupported = isCodex && capability.key === 'embedding';
        return <label key={capability.key} className={unsupported ? 'disabled' : capabilities.includes(capability.key) ? 'selected' : ''}><input type="checkbox" checked={!unsupported && capabilities.includes(capability.key)} disabled={unsupported} onChange={(e) => setCapabilities(e.target.checked ? [...capabilities, capability.key] : capabilities.filter((key) => key !== capability.key))} /><span>{capability.name}</span></label>;
      })}</div>
      {isCodex && <p className="helper">Codex CLI 不提供语义向量能力；可另建 API 连接用于向量检索。</p>}
      {bindingConflict && <Alert>{bindingConflict}</Alert>}
      <details className="advanced"><summary>{isCodex ? '连接参数' : '连接参数与费用单价'}</summary><div className="form-grid"><Field label="超时（秒）"><input type="number" min="5" max="600" value={timeout} onChange={(e) => setTimeout(Number(e.target.value))} /></Field><Field label="最大并发"><input type="number" min="1" max="16" value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))} /></Field>{!isCodex && <><Field label="输入单价（USD / 百万 token）"><input type="number" min="0" step="0.001" value={inputPrice} onChange={(e) => setInputPrice(e.target.value)} placeholder="未知" /></Field><Field label="输出单价（USD / 百万 token）"><input type="number" min="0" step="0.001" value={outputPrice} onChange={(e) => setOutputPrice(e.target.value)} placeholder="未知" /></Field></>}</div></details>
      {!isCodex && <Alert tone="info">只有执行对应能力时，本机后端才会向这个地址发送必要的画面或文本。模型支持情况可在保存后按能力测试。</Alert>}
      {error && <Alert tone="error">{error}</Alert>}
    </div><footer className="modal-footer"><span><ShieldCheck size={14} />{isCodex ? '登录状态由本机 Codex CLI 管理' : '凭据仅交由本机后端使用'}</span><button type="submit" className="button primary" disabled={busy || !!bindingConflict}>{busy ? <Spinner label="保存中" /> : <><Save size={16} />保存连接</>}</button></footer></form>
  </Modal>;
}

function ProfileCard({ profile, onEdit }: { profile: Profile; onEdit: () => void }) {
  const { refresh, notify, bootstrap } = useApp();
  const [testCapability, setTestCapability] = useState<Capability>(profile.capabilities[0] || 'query');
  const [busy, setBusy] = useState('');
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const isCodex = profile.provider_type === 'codex_cli';
  const bound = Object.values(bootstrap.settings.bindings).includes(profile.id);
  const test = async () => {
    setBusy('test'); setResult(null);
    try { setResult(await post<{ ok: boolean; message: string }>(`/profiles/${encodeURIComponent(profile.id)}/test`, { capability: testCapability })); }
    catch (error) { setResult({ ok: false, message: message(error) }); } finally { setBusy(''); }
  };
  const remove = async () => {
    setBusy('delete');
    try { await api(`/profiles/${encodeURIComponent(profile.id)}`, { method: 'DELETE' }); await refresh(); notify('模型连接已删除'); } catch (error) { notify(message(error), 'error'); } finally { setBusy(''); setDeleteConfirm(false); }
  };
  return <div className="profile-card"><div className="profile-header"><div className="profile-icon">{isCodex ? <Terminal size={22} /> : <Cable size={22} />}</div><div><h3>{profile.name}</h3><code>{profileModelLabel(profile)}</code></div><button className="icon-button" onClick={onEdit} aria-label={`编辑 ${profile.name}`}><Pencil size={16} /></button><button className="icon-button" onClick={() => setDeleteConfirm(!deleteConfirm)} aria-label={`删除 ${profile.name}`}><Trash2 size={16} /></button></div><p className="profile-url">{isCodex ? '本地 Codex CLI · ChatGPT 订阅' : profile.base_url}</p><div className="profile-tags">{profile.capabilities.map((capability) => <span className="tag" key={capability}>{CAPABILITIES.find((item) => item.key === capability)?.name || capability}</span>)}</div><div className="profile-secret"><KeyRound size={13} />{isCodex ? '订阅额度 · 费用未知' : ({ session: '会话凭据', keyring: '系统凭据库', env: '环境变量', none: '无需 Key' })[profile.secret_mode]}{bound && <span><Check size={12} />已分配</span>}</div><div className="profile-test"><select value={testCapability} onChange={(e) => setTestCapability(e.target.value as Capability)} aria-label={`测试 ${profile.name} 的能力`}>{profile.capabilities.map((capability) => <option key={capability} value={capability}>{CAPABILITIES.find((item) => item.key === capability)?.name}</option>)}</select><button className="button small secondary" disabled={!!busy} onClick={test}>{busy === 'test' ? <Spinner label="测试中" /> : <><PlugZap size={14} />连接测试</>}</button></div>{result && <Alert tone={result.ok ? 'success' : 'error'}>{result.message || (result.ok ? '连接测试通过' : '连接测试失败')}</Alert>}{deleteConfirm && <div className="delete-confirm"><p>{bound ? '此连接正在使用。请先在能力分配中解除绑定，再删除。' : '删除此连接配置及保存的凭据？已保存的影片资料不会受影响。'}</p><button className="button small danger" disabled={bound || !!busy} onClick={remove}>确认删除</button><button className="button small ghost" onClick={() => setDeleteConfirm(false)}>取消</button></div>}</div>;
}

export default function Settings() {
  const { bootstrap, refresh, notify } = useApp();
  const [tab, setTab] = useState<'models' | 'defaults' | 'data'>('models');
  const [profileModal, setProfileModal] = useState<Profile | 'new' | null>(null);
  const [settings, setSettings] = useState<SettingsType>({ bindings: { ...bootstrap.settings.bindings }, defaults: { ...DEFAULTS, ...bootstrap.settings.defaults } });
  useEffect(() => {
    setSettings((current) => {
      const bindings = { ...current.bindings };
      let changed = false;
      for (const capability of CAPABILITIES) {
        const id = bindings[capability.key];
        if (!id) continue;
        const profile = bootstrap.provider_profiles.find((item) => item.id === id);
        if (!profile?.capabilities.includes(capability.key) || (profile.provider_type === 'codex_cli' && capability.key === 'embedding')) {
          bindings[capability.key] = null; changed = true;
        }
      }
      return changed ? { ...current, bindings } : current;
    });
  }, [bootstrap.provider_profiles]);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [includeMedia, setIncludeMedia] = useState(false);
  const [backup, setBackup] = useState<string | null>(null);
  const [restoreFile, setRestoreFile] = useState<File | null>(null);
  const [restoreReport, setRestoreReport] = useState<Record<string, unknown> | null>(null);
  const [indexStatus, setIndexStatus] = useState<Record<string, unknown> | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const save = async (event: FormEvent) => {
    event.preventDefault(); setBusy('save'); setError('');
    try { const result = await api<SettingsType>('/settings', { method: 'PUT', body: JSON.stringify(settings) }); setSettings(result); await refresh(); notify('设置已保存'); } catch (error) { setError(message(error)); } finally { setBusy(''); }
  };
  const backupLibrary = async () => {
    setBusy('backup'); setError(''); setBackup(null);
    try { const result = await post<{ id: string; download_url: string }>('/backups', { include_media: includeMedia }); setBackup(result.download_url); notify('备份已生成，可以下载保存'); } catch (error) { setError(message(error)); } finally { setBusy(''); }
  };
  const restoreLibrary = async () => {
    if (!restoreFile) return;
    setBusy('restore'); setError(''); setRestoreReport(null);
    try { const body = new FormData(); body.append('file', restoreFile); setRestoreReport(await api<Record<string, unknown>>('/restore', { method: 'POST', body })); await refresh(); notify('备份恢复已完成，请查看恢复报告'); setRestoreFile(null); if (fileRef.current) fileRef.current.value = ''; } catch (error) { setError(message(error)); } finally { setBusy(''); }
  };
  const rebuild = async (embeddings: boolean) => {
    setBusy(embeddings ? 'embedding' : 'index'); setError('');
    try { await post('/index/rebuild', { embeddings }); await refresh(); notify('索引重建已加入任务队列'); } catch (error) { setError(message(error)); } finally { setBusy(''); }
  };
  return <><PageHeader eyebrow="MADE TO BE YOURS" title="模型与设置">使用你选择的 AI。识别、检索与决策，分别配置、独立运行。</PageHeader><div className="section-row"><div className="tabs">{[['models', '模型连接'], ['defaults', '分析默认值'], ['data', '本地数据']].map(([value, label]) => <button key={value} className={tab === value ? 'active' : ''} onClick={() => { setTab(value as typeof tab); setError(''); }}>{label}</button>)}</div></div>{error && <Alert tone="error">{error}</Alert>}{tab === 'models' && <><div className="section-row settings-section-title"><div><h2>我的模型连接</h2><p>连接供应商 API、本机模型，或使用 Codex CLI 订阅。</p></div><button className="button primary" onClick={() => setProfileModal('new')}><Plus size={17} />添加连接</button></div>{bootstrap.provider_profiles.length === 0 ? <div className="settings-empty"><Empty icon={<Cable size={29} />} title="连接你的第一个 AI" action={<button className="button secondary" onClick={() => setProfileModal('new')}><Plus size={16} />添加模型连接</button>}>支持 OpenAI-compatible 接口和本地 Codex CLI。画面分析与烧录字幕 OCR 需要图像模型，容器文本字幕无需模型。</Empty></div> : <div className="profile-grid">{bootstrap.provider_profiles.map((profile) => <ProfileCard key={`${profile.id}-${profile.provider_type}-${profile.model}-${profile.capabilities.join(',')}`} profile={profile} onEdit={() => setProfileModal(profile)} />)}</div>}<form onSubmit={save}><div className="section-row settings-section-title"><div><h2>能力分配</h2><p>同一个连接可以用于多个能力；修改检索模型不需要重新识别作品。</p></div><button className="button secondary" type="submit" disabled={!!busy}>{busy === 'save' ? <Spinner label="保存中" /> : <><Save size={16} />保存分配</>}</button></div><div className="bindings-list">{CAPABILITIES.map((capability, index) => <div className="binding-row" key={capability.key}><span className="binding-index">0{index + 1}</span><div><h3>{capability.name}{capability.key === 'decision' && <span className="tag">Jev READY</span>}</h3><p>{capability.description}</p><small>{capability.data}</small></div><select aria-label={`分配${capability.name}模型`} value={settings.bindings[capability.key] || ''} onChange={(e) => setSettings({ ...settings, bindings: { ...settings.bindings, [capability.key]: e.target.value || null } })}><option value="">{capability.key === 'answer' ? '使用模板解释' : capability.key === 'embedding' ? '仅字面检索' : '暂不启用'}</option>{bootstrap.provider_profiles.filter((profile) => profile.capabilities.includes(capability.key) && !(profile.provider_type === 'codex_cli' && capability.key === 'embedding')).map((profile) => <option key={profile.id} value={profile.id}>{profile.name} · {profileModelLabel(profile)}</option>)}</select></div>)}</div><p className="helper binding-footnote">更换语义向量模型后，到「本地数据」重建向量索引。决策接口初期使用传统 LLM；Jev 实际接入留待后续版本。</p></form></>}{tab === 'defaults' && <form className="settings-form" onSubmit={save}><div className="settings-section-title"><h2>下一次分析的默认参数</h2><p>单次任务可以覆盖这些设置；已创建任务不受影响。</p></div><div className="form-grid"><Field label="最长分析窗口（秒）" hint="按镜头切分，单个窗口不会跨越镜头"><input type="number" min="1" max="30" step="0.5" value={settings.defaults.window_ms / 1000} onChange={(e) => setSettings({ ...settings, defaults: { ...settings.defaults, window_ms: Math.round(Number(e.target.value) * 1000) } })} required /></Field><Field label="每窗口画面数"><input type="number" min="2" max="12" value={settings.defaults.frames_per_window} onChange={(e) => setSettings({ ...settings, defaults: { ...settings.defaults, frames_per_window: Number(e.target.value) } })} required /></Field><Field label="画面字幕 OCR 采样（帧 / 秒）"><input type="number" min="0.2" max="10" step="0.1" value={settings.defaults.subtitle_fps} onChange={(e) => setSettings({ ...settings, defaults: { ...settings.defaults, subtitle_fps: Number(e.target.value) } })} required /></Field><Field label="最大模型请求数"><input type="number" min="1" max="100000" value={settings.defaults.max_requests} onChange={(e) => setSettings({ ...settings, defaults: { ...settings.defaults, max_requests: Number(e.target.value) } })} required /></Field><Field label="费用上限（USD，可选）" hint="需要配置 API 模型单价；Codex CLI 使用请求次数控制用量"><input type="number" min="0.01" step="0.01" placeholder="未设置" value={settings.defaults.max_cost ?? ''} onChange={(e) => setSettings({ ...settings, defaults: { ...settings.defaults, max_cost: e.target.value ? Number(e.target.value) : null } })} /></Field></div><h3 className="section-label">默认字幕裁剪区域</h3><p className="helper">使用 0–1 归一化坐标；左侧 + 宽度、顶部 + 高度均不能超过 1。</p><div className="form-grid four">{['左侧 X', '顶部 Y', '宽度', '高度'].map((label, index) => <Field key={label} label={label}><input type="number" min={index > 1 ? 0.01 : 0} max="1" step="0.01" value={settings.defaults.subtitle_crop[index]} onChange={(e) => setSettings({ ...settings, defaults: { ...settings.defaults, subtitle_crop: settings.defaults.subtitle_crop.map((value, i) => i === index ? Number(e.target.value) : value) } })} required /></Field>)}</div><div className="form-actions"><button className="button primary" type="submit" disabled={!!busy}>{busy === 'save' ? <Spinner label="保存中" /> : <><Save size={16} />保存默认值</>}</button></div></form>}{tab === 'data' && <div className="data-settings"><div className="storage-banner"><HardDrive size={29} strokeWidth={1.5} /><div><h3>你的资料，保存在本机</h3><code>{bootstrap.library_path}</code><p>JSON / JSONL 是原始资料；SQLite 与向量索引随时可以重建。</p></div><ShieldCheck size={28} /></div><div className="data-grid"><section className="data-card"><Database size={23} /><h3>重建检索索引</h3><p>从已有资料重建索引。不会重新调用画面或字幕识别。</p><div className="button-row"><button className="button secondary" disabled={!!busy} onClick={() => rebuild(false)}>{busy === 'index' ? <Spinner label="提交中" /> : <><RefreshCw size={15} />字面索引</>}</button><button className="button secondary" disabled={!!busy || !bootstrap.settings.bindings.embedding} onClick={() => rebuild(true)}>{busy === 'embedding' ? <Spinner label="提交中" /> : '语义向量索引'}</button></div><small>向量重建会向绑定的 embedding 接口发送文本并产生用量。</small><button className="text-button" onClick={async () => { try { setIndexStatus(await api('/index/status')); } catch (error) { setError(message(error)); } }}>查看索引状态 <ArrowUpRight size={13} /></button>{indexStatus && <pre className="json-report">{JSON.stringify(indexStatus, null, 2)}</pre>}</section><section className="data-card"><Download size={23} /><h3>备份资料库</h3><p>包含作品信息、识别资料、人工纠错与收藏笔记，不包含 API Key。</p><label className="check-line"><input type="checkbox" checked={includeMedia} onChange={(e) => setIncludeMedia(e.target.checked)} />同时包含原视频（备份体积较大）</label><button className="button secondary" disabled={!!busy} onClick={backupLibrary}>{busy === 'backup' ? <Spinner label="正在生成备份" /> : <><Download size={15} />生成{includeMedia ? '完整' : '资料'}备份</>}</button>{backup && <a className="backup-link" href={backup} download><Check size={15} />备份已就绪，点击下载</a>}</section><section className="data-card"><Upload size={23} /><h3>恢复备份</h3><p>安全合并资料，遇到冲突不会覆盖现有文件。恢复后按需重建索引与重新定位视频。</p><input ref={fileRef} type="file" accept=".zip,application/zip" aria-label="选择资料库备份文件" onChange={(e) => setRestoreFile(e.target.files?.[0] || null)} /><button className="button secondary" disabled={!!busy || !restoreFile} onClick={restoreLibrary}>{busy === 'restore' ? <Spinner label="正在恢复" /> : <><Upload size={15} />恢复所选备份</>}</button>{restoreReport && <pre className="json-report">{JSON.stringify(restoreReport, null, 2)}</pre>}</section><section className="data-card"><ShieldCheck size={23} /><h3>本地服务状态</h3><p>单用户本机服务。浏览器通过本地会话访问资料与视频。</p><div className="tool-status"><span>FFmpeg</span><span className={bootstrap.tools.ffmpeg ? 'success-text' : 'warning-text'}>{bootstrap.tools.ffmpeg ? '可用' : '未检测到'}</span></div><div className="tool-status"><span>ffprobe</span><span className={bootstrap.tools.ffprobe ? 'success-text' : 'warning-text'}>{bootstrap.tools.ffprobe ? '可用' : '未检测到'}</span></div><div className="tool-status"><span>SceneRecall</span><span>v{bootstrap.version}</span></div></section></div></div>}{profileModal && <ProfileModal profile={profileModal === 'new' ? undefined : profileModal} onClose={() => setProfileModal(null)} />}</>;
}
