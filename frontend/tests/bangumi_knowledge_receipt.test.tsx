import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { BangumiKnowledgeReceipt } from '../src/components/Jobs';

afterEach(cleanup);

describe('Bangumi knowledge import receipts', () => {
  it('distinguishes all saved cards from newly created candidates', () => {
    render(<BangumiKnowledgeReceipt receipt={{ status: 'completed', source_count: 5, card_count: 8, created_cards: 2, warnings: [] }} />);
    expect(screen.getByText('Bangumi 知识资料：5 份来源 · 8 张资料卡，新增候选 2 张')).toBeTruthy();
    expect(screen.queryByText(/8 张待核对/)).toBeNull();
  });

  it('preserves partial-failure warnings without inventing successful counts', () => {
    render(<BangumiKnowledgeReceipt receipt={{ status: 'failed', warnings: ['角色简介暂未保存，可在作品设定库重试'] }} />);
    expect(screen.getByRole('status').textContent).toBe('角色简介暂未保存，可在作品设定库重试');
    expect(screen.queryByText(/份来源/)).toBeNull();
  });
});
