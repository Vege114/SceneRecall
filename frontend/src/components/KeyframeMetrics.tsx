import type { Job } from '../types';

export default function KeyframeMetrics({ job }: { job: Job }) {
  const metrics = job.keyframe_metrics;
  if (!metrics) return null;
  return <div className="job-stage-metrics" aria-label="本次关键帧处理耗时">
    <div><strong>本次关键帧处理</strong>
      <span>选帧 {(metrics.selection_ms / 1000).toFixed(1)} 秒{metrics.selection_cache_hit ? '（复用计划）' : ''}</span>
      <span>提取预览 {(metrics.extraction_ms / 1000).toFixed(1)} 秒</span>
      <span>登记 {(metrics.registration_ms / 1000).toFixed(1)} 秒</span>
      {metrics.validation_ms !== undefined && <span>验证发布 {(metrics.validation_ms / 1000).toFixed(1)} 秒</span>}
      <span>总耗时 {(metrics.elapsed_ms / 1000).toFixed(1)} 秒</span>
      <span>已保存 {metrics.extracted_frames + metrics.reused_frames} / {metrics.selected_frames} 张 · 复用 {metrics.reused_frames} 张</span>
    </div>
  </div>;
}
