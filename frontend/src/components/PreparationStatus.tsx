import { useState } from 'react';
import { message } from '../api';
import { useApp, type LibraryResource } from '../context';
import { Alert, Spinner } from './ui';

const resourceLabels = { assets: '作品列表', projects: '项目列表', folders: '文件夹', jobs: '任务列表' };
export function ResourceStatus({ resource }: { resource: LibraryResource | 'jobs' }) {
  const { resources, retryResource } = useApp();
  const state = resources?.[resource];
  if (!state) return null;
  if (!state.loaded && state.loading) return <div className="resource-status"><Spinner label={`正在加载${resourceLabels[resource]}`} /></div>;
  if (!state.error) return null;
  return <Alert tone="error">{resourceLabels[resource]}{state.loaded ? '刷新' : '加载'}失败：{state.error} {retryResource && <button className="text-button" onClick={() => void retryResource(resource).catch(() => undefined)}>重试{resourceLabels[resource]}</button>}</Alert>;
}

export function PreparationStatus({ task }: { task: 'index' | 'statistics' | 'encoder' }) {
  const { bootstrap, retryStartup } = useApp();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const state = bootstrap.startup?.[task];
  if (!state || state.status === 'ready') return null;
  const label = { index: '检索索引', statistics: '资料统计', encoder: '本机视觉编码器' }[task];
  const failed = state.status === 'error';
  const retry = async () => {
    setBusy(true); setError('');
    try { await retryStartup?.(); } catch (caught) { setError(message(caught)); } finally { setBusy(false); }
  };
  return <Alert tone={failed ? 'error' : 'info'}>
    {failed ? `${label}准备失败` : task === 'encoder' ? '正在检查本机视觉编码器' : `正在准备${label}`}
    {state.message && `：${state.message}`}
    {task === 'index' && !failed && '。可以先浏览资料库；准备完成后即可检索。'}
    {task === 'statistics' && !failed && '。现有资料可以正常浏览。'}
    {failed && retryStartup && <button type="button" className="text-button" disabled={busy} onClick={() => void retry()}>{busy ? <Spinner label="正在重试" /> : `重试${label}`}</button>}
    {error && <span>：{error}</span>}
  </Alert>;
}
