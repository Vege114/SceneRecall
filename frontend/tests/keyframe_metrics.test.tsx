import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it } from 'vitest';
import KeyframeMetrics from '../src/components/KeyframeMetrics';
import type { Job } from '../src/types';

afterEach(cleanup);

it('shows current-attempt timings and reused previews without model request counts', () => {
  const job = { keyframe_metrics: { selection_ms: 0, extraction_ms: 1200, registration_ms: 30,
    validation_ms: 100, elapsed_ms: 1500, selected_frames: 200, reused_frames: 128, extracted_frames: 72,
    selection_cache_hit: true } } as Job;
  render(<KeyframeMetrics job={job} />);
  expect(screen.getByText('选帧 0.0 秒（复用计划）')).toBeTruthy();
  expect(screen.getByText('总耗时 1.5 秒')).toBeTruthy();
  expect(screen.getByText('验证发布 0.1 秒')).toBeTruthy();
  expect(screen.getByText('已保存 200 / 200 张 · 复用 128 张')).toBeTruthy();
  expect(screen.queryByText(/请求/)).toBeNull();
});

it('does not invent timings for historical jobs', () => {
  const { container } = render(<KeyframeMetrics job={{} as Job} />);
  expect(container.textContent).toBe('');
});
