import type { ImportItem, Job } from './types';

function phaseMessage(phase?: string): string {
  if (phase === 'import_profile') return '正在同步作品档案';
  if (phase === 'import_index') return '正在建立搜索索引';
  return '正在本机校验与提取字幕';
}

export function importProgressMessage(job: Job): string {
  if (job.message?.trim()) return job.message;
  if (['queued', 'pending'].includes(job.status)) return '等待开始导入';
  if (['completed', 'complete'].includes(job.status)) return '导入完成';
  if (job.status === 'partial') return '部分视频导入失败';
  if (job.status === 'failed') return '导入失败';
  if (job.status === 'paused') return '导入已暂停';
  if (job.status === 'interrupted') return '导入已中断';
  if (['cancelled', 'canceled'].includes(job.status)) return '导入已取消';
  return phaseMessage(job.stage);
}

export function importItemProgress(job: Job, item: ImportItem): string | undefined {
  if (item.status !== 'running' || !['running', 'pausing', 'cancelling'].includes(job.status)) return undefined;
  return phaseMessage(item.phase || job.stage);
}
