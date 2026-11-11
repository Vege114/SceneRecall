import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RecognitionSelection from '../src/components/RecognitionSelection';
import type { CharacterConfirmationRole } from '../src/identityTypes';

const roles = (): CharacterConfirmationRole[] => [
  {
    character_id: 'summer', recognition_enabled: true, status: 'pending', trusted_reference_count: 2,
    matching_reference_count: 3, available_count: 1, candidates: [],
    profile: { id: 'summer', name: '小夏', aliases: ['夏同学'], description: '', notes: '', external_source: {
      source: 'bangumi', subject_id: 1, character_id: 2, url: 'https://bgm.tv/character/2', image_url: '/synthetic/summer.jpg',
    } },
  },
  {
    character_id: 'spring', recognition_enabled: false, status: 'pending', trusted_reference_count: 0,
    matching_reference_count: 0, available_count: 0, candidates: [],
    profile: { id: 'spring', name: '小春', aliases: ['春同学'], description: '', notes: '' },
  },
  {
    character_id: 'autumn', recognition_enabled: false, status: 'confirmed', trusted_reference_count: 1,
    matching_reference_count: 1, available_count: 0, candidates: [],
    profile: { id: 'autumn', name: '小秋', aliases: ['秋同学'], description: '', notes: '' },
  },
];

function setup(options: { busy?: boolean; hasDraftCrops?: boolean } = {}) {
  const onSave = vi.fn();
  const onCancel = vi.fn();
  const onDraftChange = vi.fn();
  render(<RecognitionSelection roles={roles()} busy={options.busy ?? false} hasDraftCrops={options.hasDraftCrops ?? false}
    onSave={onSave} onCancel={onCancel} onDraftChange={onDraftChange} />);
  return { user: userEvent.setup(), onSave, onCancel, onDraftChange };
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('Recognition targets in the illustrated role list', () => {
  it('shows a role image, local matching reference coverage and candidate state alongside its target checkbox', () => {
    setup();
    expect(screen.getByAltText('小夏的角色头像').getAttribute('src')).toBe('/synthetic/summer.jpg');
    expect(screen.getByText(/本地参考 3 张/)).toBeTruthy();
    expect(screen.getByText(/候选 1 张/)).toBeTruthy();
    expect(screen.getByText('已确认')).toBeTruthy();
    expect(screen.getByLabelText('小春暂无头像')).toBeTruthy();
    expect((screen.getByRole('checkbox', { name: '识别小夏' }) as HTMLInputElement).checked).toBe(true);
    expect((screen.getByRole('checkbox', { name: '识别小春' }) as HTMLInputElement).checked).toBe(false);
    expect(screen.getByText(/同一作品/)).toBeTruthy();
  });

  it('prefers an available local reference preview over the external profile image', () => {
    const localRoles = roles();
    localRoles[0].reference_preview_url = '/synthetic/local-reference.jpg';
    render(<RecognitionSelection roles={localRoles} busy={false} hasDraftCrops={false} onSave={vi.fn()} onCancel={vi.fn()} />);
    expect(screen.getByAltText('小夏的角色头像').getAttribute('src')).toBe('/synthetic/local-reference.jpg');
  });

  it('limits bulk changes to the filtered roles and preserves hidden selections', async () => {
    const { user, onSave } = setup();
    await user.type(screen.getByRole('searchbox'), '春同学');
    expect(screen.queryByRole('checkbox', { name: '识别小夏' })).toBeNull();
    expect(screen.queryByRole('checkbox', { name: '识别小秋' })).toBeNull();
    await user.click(screen.getByRole('button', { name: '全选当前显示' }));
    expect((screen.getByRole('checkbox', { name: '识别小春' }) as HTMLInputElement).checked).toBe(true);
    expect(onSave).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: '保存名单并生成候选' }));
    expect(onSave).toHaveBeenLastCalledWith(['summer', 'spring']);
    await user.click(screen.getByRole('button', { name: '取消当前显示' }));
    expect((screen.getByRole('checkbox', { name: '识别小春' }) as HTMLInputElement).checked).toBe(false);
    await user.click(screen.getByRole('button', { name: '保存名单并生成候选' }));
    expect(onSave).toHaveBeenLastCalledWith(['summer']);
    await user.clear(screen.getByRole('searchbox'));
    expect((screen.getByRole('checkbox', { name: '识别小夏' }) as HTMLInputElement).checked).toBe(true);
  });

  it('reports target drafts and resets an individual change on cancel without saving', async () => {
    const { user, onSave, onCancel, onDraftChange } = setup({ hasDraftCrops: true });
    await user.click(screen.getByRole('checkbox', { name: '识别小春' }));
    expect(onDraftChange).toHaveBeenLastCalledWith(true);
    expect(onSave).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: '取消修改' }));
    expect(onSave).not.toHaveBeenCalled();
    expect(onCancel).toHaveBeenCalledOnce();
    expect((screen.getByRole('checkbox', { name: '识别小春' }) as HTMLInputElement).checked).toBe(false);
    expect((screen.getByRole('checkbox', { name: '识别小夏' }) as HTMLInputElement).checked).toBe(true);
    expect(onDraftChange).toHaveBeenLastCalledWith(false);
  });

  it('locks role and bulk mutations while the saved selection is being generated', async () => {
    const { user, onSave, onCancel } = setup({ busy: true });
    expect(screen.getAllByRole('checkbox').every((element) => (element as HTMLInputElement).disabled)).toBe(true);
    for (const name of ['全选当前显示', '取消当前显示', '取消修改', '保存名单并生成候选']) {
      const button = screen.getByRole('button', { name });
      expect((button as HTMLButtonElement).disabled).toBe(true);
      await user.click(button);
    }
    expect(onSave).not.toHaveBeenCalled();
    expect(onCancel).not.toHaveBeenCalled();
  });
});
