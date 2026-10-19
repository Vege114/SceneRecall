export const FRAME_MS = 40;
export const clone = (value) => JSON.parse(JSON.stringify(value));
export const timecode = (ms) => {
  const value = Math.max(0, Math.round(ms));
  const seconds = Math.floor(value / 1000);
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map((part) => String(part).padStart(2, '0')).join(':') + '.' + String(value % 1000).padStart(3, '0');
};

export function validatePreparation(start, end, duration) {
  if (![start, end, duration].every(Number.isFinite) || !Number.isInteger(start) || !Number.isInteger(end)
      || start < 0 || end > duration || start % FRAME_MS || end % FRAME_MS
      || end - start < 640 || end - start > 8000) {
    throw new Error('请选择原片范围内的 0.64–8 秒片段，起止时间须以 0.04 秒为步长。');
  }
}

export function validateBox(box) {
  if (box === null) return;
  if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)
      || box[0] < 0 || box[1] < 0 || box[2] <= 0 || box[3] <= 0
      || box[0] + box[2] > 1.000000001 || box[1] + box[3] > 1.000000001) {
    throw new Error('人物框须位于画面内，并且有实际宽度和高度。');
  }
}

export function validateRange(start, end, count) {
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || start >= end || end > count) {
    throw new Error('帧范围须满足 0 ≤ 起始帧 < 结束帧 ≤ 总帧数，包含起点、不包含终点。');
  }
}

export function newTrack(count, id) {
  return { id, character_id: null, boxes: Array(count).fill(null), labels: Array(count).fill(-1),
    reviewed: false, excluded: false, note: '' };
}

export function updateTrack(clip, id, changes, invalidateReview = true) {
  if (!clip.tracks.some((track) => track.id === id)) throw new Error('请先选择人物轨迹。');
  const result = clone(clip);
  result.tracks = result.tracks.map((track) => track.id === id
    ? { ...track, ...clone(changes), ...(invalidateReview ? { reviewed: false } : {}) } : track);
  return result;
}

export function labelRange(clip, id, start, end, label) {
  validateRange(start, end, clip.frame_count);
  if (![-1, 0, 1].includes(label)) throw new Error('无效的说话标签。');
  const track = clip.tracks.find((row) => row.id === id);
  if (!track) throw new Error('请先选择人物轨迹。');
  if (label !== -1 && track.boxes.slice(start, end).some((box) => box === null)) {
    throw new Error('所选范围有缺少人物框的帧。请先补框，或保留“不确定”。');
  }
  const labels = [...track.labels];
  labels.fill(label, start, end);
  return updateTrack(clip, id, { labels });
}

function batchTargets(clip, ids) {
  const selected = new Set(ids);
  if (!selected.size) throw new Error('请先勾选要批量处理的人物轨迹。');
  if ([...selected].some((id) => !clip.tracks.some((track) => track.id === id))) {
    throw new Error('选中的人物轨迹已变化，请重新选择。');
  }
  return selected;
}

export function missingBoxRanges(track, start = 0, end = track.boxes.length) {
  validateRange(start, end, track.boxes.length);
  const ranges = [];
  for (let frame = start; frame < end; frame += 1) {
    if (track.boxes[frame] !== null) continue;
    const from = frame;
    while (frame + 1 < end && track.boxes[frame + 1] === null) frame += 1;
    ranges.push([from, frame + 1]);
  }
  return ranges;
}

// A batch is one immutable edit, so undo restores every affected track together.
export function batchLabelRange(clip, ids, start, end, { label, speaker = null, onlyUnknown = false } = {}) {
  validateRange(start, end, clip.frame_count);
  const selected = batchTargets(clip, ids);
  if (speaker !== null && !selected.has(speaker)) throw new Error('单人说话模式要求当前人物也在勾选范围内。');
  if (speaker === null && ![-1, 0, 1].includes(label)) throw new Error('无效的说话标签。');
  const result = clone(clip);
  const summary = { changedFrames: 0, changedTracks: 0, missingFrames: 0, keptFrames: 0, missingByTrack: [] };
  for (const track of result.tracks) {
    if (!selected.has(track.id)) continue;
    const target = speaker === null ? label : Number(track.id === speaker);
    const ranges = target === -1 ? [] : missingBoxRanges(track, start, end);
    if (ranges.length) summary.missingByTrack.push({ id: track.id, ranges });
    let changed = false;
    for (let frame = start; frame < end; frame += 1) {
      if (onlyUnknown && track.labels[frame] !== -1) { summary.keptFrames += 1; continue; }
      if (target !== -1 && track.boxes[frame] === null) { summary.missingFrames += 1; continue; }
      if (track.labels[frame] === target) continue;
      track.labels[frame] = target; changed = true; summary.changedFrames += 1;
    }
    if (changed) { track.reviewed = false; summary.changedTracks += 1; }
  }
  return { clip: result, ...summary };
}

export function batchUpdateTracks(clip, ids, changes) {
  const selected = batchTargets(clip, ids);
  const allowed = ['character_id', 'reviewed', 'excluded'];
  if (!Object.keys(changes).length || Object.keys(changes).some((key) => !allowed.includes(key))) {
    throw new Error('无效的批量人物操作。');
  }
  if (Object.hasOwn(changes, 'character_id') && changes.character_id !== null && typeof changes.character_id !== 'string') {
    throw new Error('无效的角色身份。');
  }
  for (const key of ['reviewed', 'excluded']) {
    if (Object.hasOwn(changes, key) && typeof changes[key] !== 'boolean') throw new Error('无效的复核或排除状态。');
  }
  const result = clone(clip);
  for (const track of result.tracks) {
    if (!selected.has(track.id)) continue;
    const changed = Object.keys(changes).some((key) => track[key] !== changes[key]);
    if (!changed) continue;
    Object.assign(track, changes);
    if (Object.hasOwn(changes, 'character_id') || Object.hasOwn(changes, 'excluded')) track.reviewed = false;
  }
  return result;
}

export function matchingTracks(clip, filter) {
  return clip.tracks.filter((track) => filter === 'all' || (filter === 'active' && !track.excluded)
    || (filter === 'unknown' && !track.excluded && track.character_id === null)
    || (filter.startsWith('character:') && !track.excluded && track.character_id === filter.slice(10))).map((track) => track.id);
}

export function batchAction(clip, ids, action, { start = 0, end = clip.frame_count, onlyUnknown = false, character = null } = {}) {
  const labels = { speaking: 1, silent: 0, uncertain: -1 };
  if (Object.hasOwn(labels, action)) return batchLabelRange(clip, ids, start, end, { label: labels[action], onlyUnknown });
  const changes = { reviewed: { reviewed: true }, unreviewed: { reviewed: false }, exclude: { excluded: true },
    restore: { excluded: false }, identity: { character_id: character } }[action];
  if (!changes) throw new Error('无效的批量操作。');
  const result = batchUpdateTracks(clip, ids, changes);
  return { clip: result, changedFrames: 0, missingFrames: 0, keptFrames: 0,
    changedTracks: result.tracks.filter((track, index) => JSON.stringify(track) !== JSON.stringify(clip.tracks[index])).length };
}

export function setFrameBox(clip, id, frame, box) {
  validateRange(frame, frame + 1, clip.frame_count);
  validateBox(box);
  const track = clip.tracks.find((row) => row.id === id);
  if (!track) throw new Error('请先选择人物轨迹。');
  const boxes = clone(track.boxes);
  const labels = [...track.labels];
  boxes[frame] = clone(box);
  if (box === null) labels[frame] = -1;
  return updateTrack(clip, id, { boxes, labels });
}

export function applyCurrentBox(clip, id, frame, start, end) {
  validateRange(start, end, clip.frame_count);
  const track = clip.tracks.find((row) => row.id === id);
  const box = track?.boxes[frame];
  if (!box) throw new Error('当前帧尚无人物框，请先在暂停画面上绘制。');
  validateBox(box);
  const boxes = clone(track.boxes);
  for (let index = start; index < end; index += 1) boxes[index] = [...box];
  return updateTrack(clip, id, { boxes });
}

export function editableSnapshot(clip) {
  return clone({ note: clip.note || '', tracks: clip.tracks });
}

export function restoreSnapshot(clip, snapshot) {
  return { ...clip, ...clone(snapshot) };
}

export function savePayload(clip) {
  const seen = new Set();
  if (!clip || clip.status !== 'ready' || clip.fps !== 25 || !Number.isInteger(clip.frame_count) || clip.frame_count < 1) {
    throw new Error('请等待片段准备完成后再保存。');
  }
  const tracks = clip.tracks.map((track) => {
    if (typeof track.id !== 'string' || !track.id || seen.has(track.id)) throw new Error('人物轨迹编号缺失或重复。');
    seen.add(track.id);
    if (!Array.isArray(track.boxes) || !Array.isArray(track.labels)
        || track.boxes.length !== clip.frame_count || track.labels.length !== clip.frame_count) {
      throw new Error('人物框与标签数量必须与片段帧数一致。');
    }
    track.boxes.forEach(validateBox);
    track.labels.forEach((label, index) => {
      if (![-1, 0, 1].includes(label) || (label !== -1 && track.boxes[index] === null)) {
        throw new Error('缺框帧只能保留“不确定”；请先修正再保存。');
      }
    });
    if (track.character_id !== null && typeof track.character_id !== 'string') throw new Error('无效的角色身份。');
    if (typeof track.reviewed !== 'boolean' || typeof track.excluded !== 'boolean') throw new Error('请明确人物复核与排除状态。');
    return { id: track.id, character_id: track.character_id, boxes: clone(track.boxes), labels: [...track.labels],
      reviewed: track.reviewed, excluded: track.excluded, note: String(track.note || '') };
  });
  return { revision: clip.revision, note: String(clip.note || ''), tracks };
}
