import { FRAME_MS, clone, timecode, validatePreparation, newTrack, updateTrack,
  setFrameBox, applyCurrentBox, editableSnapshot, restoreSnapshot, savePayload, validateRange,
  batchLabelRange, batchUpdateTracks, batchAction, matchingTracks, missingBoxRanges } from './editor.js';

const $ = (id) => document.getElementById(id);
const base = '/api/asd-annotations';
const state = { index: null, profiles: [], clip: null, original: '', history: [], selected: null,
  frame: 0, rangeStart: 0, rangeEnd: 1, busy: false, dirty: false, drawing: false, drag: null,
  serial: 0, poll: null, animation: null, conflict: false,
  checkedTracks: new Set(), checkedClips: new Set(), batchUndo: [] };
const video = $('clip-video');
const svg = $('boxes');
const labelNames = { '-1': '不确定', 0: '未说话', 1: '说话' };
const labelClasses = { '-1': 'state-uncertain', 0: 'state-silent', 1: 'state-speaking' };

function element(tag, attrs = {}, ...children) {
  const result = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'className') result.className = value;
    else if (key.startsWith('on')) result.addEventListener(key.slice(2).toLowerCase(), value);
    else if (value !== undefined && value !== null) result.setAttribute(key, String(value));
  }
  for (const child of children.flat()) if (child !== undefined && child !== null) {
    result.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return result;
}

function showMessage(id, value = '') { $(id).textContent = value; $(id).hidden = !value; }
function fail(error) { showMessage('error', error instanceof Error ? error.message : String(error)); }
function notice(text) { showMessage('notice', text); }
function clearMessages() { showMessage('error'); showMessage('notice'); }
function assignValue(id, value) {
  // A playback frame must not overwrite an unfinished number or steal its caret.
  if (document.activeElement !== $(id) && $(id).value !== String(value)) $(id).value = String(value);
}

class RequestError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { ...options, credentials: 'same-origin',
      headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) } });
  } catch { throw new RequestError('无法连接本机标注服务，请确认小程序仍在运行。', 0); }
  const value = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) $('session').hidden = false;
    const detail = typeof value.detail === 'string' ? value.detail : value.message || '请求未完成，请重试。';
    throw new RequestError(detail, response.status);
  }
  return value;
}

const post = (path, body) => request(path, { method: 'POST', body: JSON.stringify(body) });
const currentTrack = () => state.clip?.tracks.find((track) => track.id === state.selected) || null;
const nameOf = (id) => state.profiles.find((profile) => profile.id === id)?.name || id || '未知人物';
const trackName = (id) => '人物' + (state.clip.tracks.findIndex((track) => track.id === id) + 1);
const operationTargets = () => $('label-target').value === 'checked' ? [...state.checkedTracks]
  : state.selected ? [state.selected] : [];
const taskItems = () => state.index?.annotation_plan?.items || [];
const taskForClip = (clip) => taskItems().find((item) => item.clip_id === clip?.id
  || (item.start_ms === clip?.start_ms && item.end_ms === clip?.end_ms));
const splitName = (split) => ({ train: '训练', val: '验证', test: '测试' }[split] || '待分配');

function canDiscard() {
  return !state.dirty || window.confirm('本段有未保存的修改。切换会放弃这些修改，是否继续？');
}

function renderDirty() {
  state.dirty = !!state.clip && JSON.stringify(editableSnapshot(state.clip)) !== state.original;
  $('save-state').textContent = state.dirty ? '有未保存修改' : '与服务器版本一致';
  $('save-state').classList.toggle('dirty', state.dirty);
  $('conflict-actions').hidden = !state.conflict;
}

function renderDisabled() {
  const ready = state.clip?.status === 'ready';
  $('save').textContent = state.busy ? '正在处理，请稍候…' : '保存本段';
  $('save-next').textContent = state.busy ? '正在处理，请稍候…' : '保存并下一任务';
  $('prepare').disabled = state.busy || state.clip?.status === 'preparing';
  for (const control of document.querySelectorAll('#editor button, #editor input, #editor select, #editor textarea')) {
    control.disabled = state.busy || !ready;
  }
  for (const control of document.querySelectorAll('#track-editor button, #track-editor input, #track-editor select, #track-editor textarea')) {
    control.disabled = state.busy || !ready || !currentTrack();
  }
  $('undo').disabled = state.busy || !state.history.length;
  let validRange = true;
  try { validateRange(state.rangeStart, state.rangeEnd, state.clip?.frame_count || 0); } catch { validRange = false; }
  document.querySelectorAll('[data-label]').forEach((control) => {
    control.disabled = state.busy || !ready || !operationTargets().length || !validRange;
  });
  $('copy-box').disabled = state.busy || !currentTrack()?.boxes[state.frame];
  $('clear-box').disabled = state.busy || !currentTrack()?.boxes[state.frame];
  document.querySelectorAll('#progress-panel button, #progress-panel input, #progress-panel select').forEach((control) => { control.disabled = state.busy; });
  document.querySelectorAll('[data-clip-ready="false"]').forEach((control) => { control.disabled = true; });
  document.querySelectorAll('[data-batch-label], [data-batch-track], #batch-identity, #batch-speaker')
    .forEach((control) => { control.disabled = state.busy || !ready || !state.checkedTracks.size; });
  document.querySelectorAll('[data-batch-label], #batch-speaker').forEach((control) => {
    control.disabled = control.disabled || !validRange;
  });
  $('saved-batch-apply').disabled = state.busy || !state.checkedClips.size;
  $('saved-batch-undo').disabled = state.busy || !state.batchUndo.length;
  $('saved-unknown').disabled = state.busy || !['silent', 'speaking', 'uncertain'].includes($('saved-action').value);
  const taskIndex = taskItems().findIndex((item) => item === taskForClip(state.clip));
  $('task-select').disabled = state.busy;
  $('previous-task').disabled = state.busy || taskIndex <= 0;
  $('next-task').disabled = state.busy || !taskItems().length || taskIndex === taskItems().length - 1;
}

function characterOptions() {
  return [element('option', { value: '' }, '未知 / 暂不关联'),
    ...state.profiles.map((profile) => element('option', { value: profile.id }, profile.name))];
}

function renderTracks() {
  state.checkedTracks = new Set([...state.checkedTracks].filter((id) => state.clip?.tracks.some((track) => track.id === id)));
  $('track-list').replaceChildren(...(state.clip?.tracks || []).map((track, index) => {
    const count = track.labels.filter((label) => label !== -1).length;
    const select = element('input', { type: 'checkbox', 'aria-label': '批量选择人物' + (index + 1),
      onchange: (event) => { if (event.target.checked) state.checkedTracks.add(track.id); else state.checkedTracks.delete(track.id);
        $('label-target').value = state.checkedTracks.size ? 'checked' : 'current';
        renderBatch(); renderDisabled(); } });
    select.checked = state.checkedTracks.has(track.id);
    const identity = element('select', { 'aria-label': '人物' + (index + 1) + '整轨角色',
      onchange: (event) => performEdit(() => batchUpdateTracks(state.clip, [track.id], { character_id: event.target.value || null })) }, characterOptions());
    identity.value = track.character_id || '';
    return element('div', { className: 'track-row' + (track.id === state.selected ? ' selected' : '') }, select,
      element('button', { type: 'button', className: 'button secondary' + (track.id === state.selected ? ' selected' : ''),
      'aria-pressed': track.id === state.selected, onclick: () => selectTrack(track.id) },
    '人物' + (index + 1), element('small', {}, (track.excluded ? '已排除' : track.reviewed ? '已复核' : '待复核')
      + ' · 已标 ' + count + '/' + state.clip.frame_count + ' · 有框 '
      + track.boxes.filter((box) => box !== null).length + ' 帧')), identity);
  }));
  if (!state.clip?.tracks.length) $('track-list').append(element('p', { className: 'muted' }, '没有初始人物框。可新增人物，再在暂停画面中画框。'));
}

function renderTrackEditor() {
  const track = currentTrack();
  $('track-editor').hidden = !track;
  if (!track) return;
  if (track.character_id && !Array.from($('character').options).some((option) => option.value === track.character_id)) {
    $('character').append(element('option', { value: track.character_id }, '原存角色：' + nameOf(track.character_id)));
  }
  assignValue('character', track.character_id || '');
  assignValue('track-note', track.note || '');
  $('reviewed').checked = track.reviewed;
  $('excluded').checked = track.excluded;
  $('draw-box').textContent = state.drawing ? '结束画框，恢复播放器操作' : '暂停后拖拽画框 / 重画';
  $('draw-box').classList.toggle('draw-active', state.drawing);
}

function renderTimeline() {
  const track = currentTrack();
  const labels = track?.labels || Array(state.clip?.frame_count || 0).fill(-1);
  $('timeline').style.gridTemplateColumns = 'repeat(' + Math.max(1, labels.length) + ', minmax(0,1fr))';
  $('timeline').replaceChildren(...labels.map((label, index) => element('button', {
    type: 'button', className: track?.boxes[index] === null ? 'state-missing' : labelClasses[label],
    title: '帧 ' + index + ' · ' + (track?.boxes[index] === null ? '缺人物框，保持不确定' : labelNames[label]),
    'aria-label': '帧 ' + index + '：' + (track?.boxes[index] === null ? '缺人物框' : labelNames[label]), onclick: () => seekFrame(index),
  })));
  renderFrame();
}

function renderFrame() {
  if (!state.clip) return;
  const count = state.clip.frame_count;
  assignValue('frame-slider', state.frame);
  $('frame-slider').max = Math.max(0, count - 1);
  $('frame-position').textContent = '原片 ' + timecode(state.clip.start_ms + state.frame * FRAME_MS)
    + ' · 片段帧 ' + state.frame + ' / ' + Math.max(0, count - 1) + '（从 0 计数）';
  assignValue('range-start', state.rangeStart);
  assignValue('range-end', state.rangeEnd);
  $('range-start').max = Math.max(0, count - 1);
  $('range-end').max = count;
  let valid = true;
  try { validateRange(state.rangeStart, state.rangeEnd, count); } catch { valid = false; }
  $('range-description').textContent = valid ? '当前范围 [' + state.rangeStart + ', ' + state.rangeEnd + ')，共 '
    + (state.rangeEnd - state.rangeStart) + ' 帧。'
    : '请输入有效帧范围：起始帧小于结束帧，结束帧最多为 ' + count + '。';
  $('range-description').classList.toggle('warning', !valid);
  Array.from($('timeline').children).forEach((bar, index) => {
    bar.classList.toggle('current', index === state.frame);
    bar.classList.toggle('selected', index >= state.rangeStart && index < state.rangeEnd);
  });
  const box = currentTrack()?.boxes[state.frame];
  $('box-description').textContent = box ? '当前帧已有框，可暂停后重画或清除。' : '当前帧缺框，标签只能保留“不确定”。';
  renderBoxes();
  renderBatch();
  renderDisabled();
}

function batchBounds() { return [state.rangeStart, state.rangeEnd]; }

function renderBatch() {
  if (!state.clip) return;
  const [start, end] = batchBounds();
  const track = currentTrack();
  const index = state.clip.tracks.findIndex((row) => row.id === state.selected);
  $('batch-summary').textContent = '已勾选 ' + state.checkedTracks.size + ' / ' + state.clip.tracks.length
    + ' 人物 · 帧范围 [' + start + ', ' + end + ') · 当前查看：' + (track ? '人物' + (index + 1) + '（' + nameOf(track.character_id) + '）' : '无');
  $('batch-speaker').textContent = '仅当前人物' + (track ? '（人物' + (index + 1) + '）' : '') + '说话，其余勾选人物未说话';
  const ids = operationTargets();
  $('operation-summary').textContent = '本次处理：' + (ids.length ? ids.map(trackName).join('、') : '尚未选择人物')
    + ' · 帧 [' + start + ', ' + end + ')'
    + ($('batch-unknown').checked ? ' · 只填不确定帧' : ' · 覆盖范围内已有标签');
  let missing = [];
  try { missing = state.clip.tracks.filter((row) => ids.includes(row.id)).map((row) => ({ id: row.id,
    ranges: missingBoxRanges(row, start, end) })).filter((row) => row.ranges.length); } catch { /* Invalid typed range has its own hint. */ }
  $('missing-guidance').hidden = !missing.length;
  $('missing-description').textContent = missing.length ? '所选范围有缺框：' + missing.map((row) => trackName(row.id) + ' '
    + row.ranges.map(([a, b]) => '[' + a + ', ' + b + ')').join('、')).join('；')
    + '。这些斜纹帧不能写说话 / 未说话标签；其余有框帧仍可标记。' : '';
}

function svgElement(tag, attrs) {
  const result = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) result.setAttribute(key, String(value));
  return result;
}

function renderBoxes() {
  svg.replaceChildren();
  svg.classList.toggle('drawing', state.drawing && video.paused);
  svg.classList.toggle('playing', !video.paused);
  if (!state.clip) return;
  state.clip.tracks.forEach((track, index) => {
    const box = track.boxes[state.frame];
    if (!box) return;
    const selected = track.id === state.selected;
    const color = selected ? '#86efac' : '#ffe190';
    const rect = svgElement('rect', { x: box[0] * 1000, y: box[1] * 1000, width: box[2] * 1000, height: box[3] * 1000,
      fill: selected ? '#55d78a18' : 'transparent', stroke: color, 'stroke-width': selected ? 3 : 2,
      'aria-label': '选择人物' + (index + 1), role: 'button', tabindex: 0 });
    rect.addEventListener('pointerdown', (event) => { if (!state.drawing) { event.stopPropagation(); selectTrack(track.id); } });
    rect.addEventListener('keydown', (event) => { if (event.key === 'Enter') selectTrack(track.id); });
    svg.append(rect);
    const title = svgElement('text', { x: box[0] * 1000 + 5, y: Math.max(28, box[1] * 1000 + 28), fill: color,
      'font-size': 25, 'font-weight': 700, 'pointer-events': 'none' });
    title.textContent = '人物' + (index + 1);
    svg.append(title);
  });
  if (state.drag) {
    const box = dragBox(state.drag);
    svg.append(svgElement('rect', { x: box[0] * 1000, y: box[1] * 1000, width: box[2] * 1000, height: box[3] * 1000,
      fill: '#86efac22', stroke: '#fff', 'stroke-width': 2, 'stroke-dasharray': '8 5' }));
  }
}

function renderProgress() {
  const index = state.index;
  if (!index) return;
  $('progress-panel').hidden = false;
  const stats = index.stats || {};
  $('stats').textContent = '已保存片段 ' + (stats.clips ?? index.clips.length) + ' · 已复核轨迹 '
    + (stats.reviewed_tracks || 0) + ' · 说话帧 ' + (stats.positive_frames || 0) + ' · 未说话帧 ' + (stats.negative_frames || 0);
  $('clip-list').replaceChildren(...index.clips.map((clip) => {
    const tracks = clip.tracks || [];
    const reviewed = tracks.filter((track) => track.reviewed && !track.excluded).length;
    const text = clip.status === 'preparing' ? '正在准备' : clip.status === 'failed' ? '准备失败'
      : '已复核 ' + reviewed + ' / ' + tracks.filter((track) => !track.excluded).length + ' 条人物轨迹';
    const checkbox = element('input', { type: 'checkbox', 'data-clip-ready': clip.status === 'ready', 'aria-label': '批量选择片段 ' + timecode(clip.start_ms),
      onchange: (event) => { if (event.target.checked) state.checkedClips.add(clip.id); else state.checkedClips.delete(clip.id);
        renderSavedBatch(); renderDisabled(); } });
    checkbox.checked = state.checkedClips.has(clip.id);
    checkbox.disabled = clip.status !== 'ready';
    return element('div', { className: 'clip-row' }, checkbox,
      element('button', { type: 'button', className: 'button secondary' + (state.clip?.id === clip.id ? ' selected' : ''),
      onclick: () => void openClip(clip.id) }, timecode(clip.start_ms) + ' — ' + timecode(clip.end_ms), element('small', {}, text)));
  }));
  if (!index.clips.length) $('clip-list').append(element('p', { className: 'muted' }, '还没有准备好的标注片段。'));
  renderSavedBatch(); renderTasks();
}

function renderTasks() {
  const items = taskItems();
  $('task-panel').hidden = !items.length;
  const current = taskForClip(state.clip);
  const completed = items.filter((item) => {
    const clip = state.index.clips.find((row) => row.id === item.clip_id
      || (row.start_ms === item.start_ms && row.end_ms === item.end_ms));
    const tracks = clip?.tracks.filter((track) => !track.excluded) || [];
    return tracks.length && tracks.every((track) => track.reviewed);
  }).length;
  $('task-progress').textContent = '共 ' + items.length + ' 个推荐片段 · 已复核 ' + completed + ' 个';
  $('task-select').replaceChildren(element('option', { value: '' }, '请选择推荐任务'), ...items.map((item, index) => {
    const clip = state.index.clips.find((row) => row.id === item.clip_id);
    return element('option', { value: item.id }, String(index + 1).padStart(2, '0') + ' · '
      + (item.title || timecode(item.start_ms)) + ' · ' + splitName(item.split)
      + (clip?.status === 'preparing' ? ' · 正在准备' : clip?.status === 'failed' ? ' · 准备失败' : ''));
  }));
  $('task-select').value = current?.id || '';
  $('task-description').textContent = current ? (current.description || '') : '选择一个任务开始；原有练习片段仍在下方“所有片段”中。';
  $('task-context').textContent = current ? '推荐任务 ' + (items.indexOf(current) + 1) + '/' + items.length + ' · '
    + (current.title || '') + ' · 用于' + splitName(current.split) : '原有 / 自选片段 · 不会自动加入推荐任务划分';
}

function renderSavedBatch() {
  const filter = $('saved-track-filter').value;
  const clips = (state.index?.clips || []).filter((clip) => state.checkedClips.has(clip.id) && clip.status === 'ready');
  const tracks = clips.reduce((total, clip) => total + matchingTracks(clip, filter).length, 0);
  const action = $('saved-action');
  $('saved-character-label').hidden = action.value !== 'identity';
  $('saved-unknown').disabled = state.busy || !['silent', 'speaking', 'uncertain'].includes(action.value);
  $('saved-batch-summary').textContent = '已选 ' + clips.length + ' 个片段，匹配 ' + tracks + ' 条人物轨迹。操作：'
    + action.selectedOptions[0].textContent + (action.value === 'identity' ? ' → ' + nameOf($('saved-character').value) + '（所有命中轨迹将是同一角色）' : '')
    + '。跨片段处理会直接保存，每个片段保留修订记录。';
}

function batchResultText(result) {
  return '修改 ' + result.changedTracks + ' 条轨迹、' + result.changedFrames + ' 个人物帧标签；跳过 '
    + result.missingFrames + ' 个缺框帧，保留 ' + result.keptFrames + ' 个已有标签。'
    + (result.missingByTrack?.length ? '缺框位置：' + result.missingByTrack.map((row) => trackName(row.id) + ' '
      + row.ranges.map(([a, b]) => '[' + a + ', ' + b + ')').join('、')).join('；') + '。' : '');
}

function reportEdit(result) {
  commitEdit(result.clip); showMessage('error');
  showMessage('batch-result', '上次操作帧 [' + state.rangeStart + ', ' + state.rangeEnd + ')：' + batchResultText(result)
    + (result.changedTracks ? ' 当前修改尚未保存。' : '没有产生新修改。'));
  $('batch-result').classList.toggle('warning', !!result.missingFrames || !!result.keptFrames);
}

function editBatch(action, extra = {}) {
  if (state.busy) return;
  try {
    const [start, end] = batchBounds();
    const result = action === 'speaker' ? batchLabelRange(state.clip, [...state.checkedTracks], start, end,
      { speaker: state.selected, onlyUnknown: $('batch-unknown').checked })
      : batchAction(state.clip, [...state.checkedTracks], action, { start, end,
        onlyUnknown: $('batch-unknown').checked, character: $('batch-character').value || null, ...extra });
    reportEdit(result);
  } catch (error) { fail(error); }
}

async function savedBatch(undo = false) {
  if (state.busy) return;
  if (state.dirty) { fail(new Error('当前片段有未保存修改，请先保存，再执行跨片段操作。')); return; }
  const selected = (state.index?.clips || []).filter((clip) => clip.status === 'ready' && state.checkedClips.has(clip.id));
  const records = undo ? [...state.batchUndo] : selected.map((clip) => ({ id: clip.id, revision: clip.revision }));
  if (!records.length) { fail(new Error('请先勾选已就绪的片段。')); return; }
  const action = $('saved-action').value;
  const filter = $('saved-track-filter').value;
  const options = { onlyUnknown: $('saved-unknown').checked, character: $('saved-character').value || null };
  state.busy = true; video.pause(); clearMessages(); renderDisabled();
  let completed = 0;
  const totals = { changedTracks: 0, changedFrames: 0, missingFrames: 0, keptFrames: 0 };
  try {
    // Check every selected revision before the first write; subsequent writes still use optimistic locking.
    const plans = [];
    for (const record of records) {
      const fresh = await request(base + '/clips/' + encodeURIComponent(record.id));
      if (fresh.revision !== record.revision || fresh.status !== 'ready') {
        throw new Error('片段 ' + timecode(fresh.start_ms) + ' 的版本已变化。尚未覆盖该片段，请刷新并重新选择。');
      }
      const ids = matchingTracks(fresh, filter);
      const result = undo ? { clip: restoreSnapshot(fresh, record.before), changedTracks: 0, changedFrames: 0, missingFrames: 0, keptFrames: 0 }
        : ids.length ? batchAction(fresh, ids, action, options) : null;
      if (!result) continue;
      for (const key of Object.keys(totals)) totals[key] += result[key];
      if (JSON.stringify(editableSnapshot(fresh)) === JSON.stringify(editableSnapshot(result.clip))) continue;
      plans.push({ record, fresh, result, payload: savePayload(result.clip) });
    }
    if (!undo && plans.length) state.batchUndo = [];
    for (const plan of plans) {
      $('saved-batch-result').textContent = (undo ? '正在撤销 ' : '正在保存 ') + (completed + 1) + ' / ' + plans.length + '…';
      const saved = await request(base + '/clips/' + encodeURIComponent(plan.fresh.id), { method: 'PUT', body: JSON.stringify(plan.payload) });
      if (undo) state.batchUndo = state.batchUndo.filter((record) => record.id !== saved.id);
      else state.batchUndo.push({ id: saved.id, revision: saved.revision, before: editableSnapshot(plan.fresh) });
      if (state.clip?.id === saved.id) adoptClip(saved, true);
      completed += 1;
    }
    if (undo) state.batchUndo = [];
    $('saved-batch-result').textContent = (undo ? '已撤销并保存 ' : '已处理并保存 ') + completed + ' 个片段。'
      + (undo ? '' : batchResultText(totals)) + (completed ? '' : '内容未变化，无需写入。');
  } catch (error) {
    $('saved-batch-result').textContent = '已保存 ' + completed + ' 个片段后停止；其余片段未覆盖。已完成部分可用“撤销上次跨片段批量操作”恢复。';
    fail(error.status === 409 ? new Error('保存时发生版本冲突，已停止后续操作。请刷新并核对片段。') : error);
  } finally {
    try { await refreshIndex(); } catch (error) { fail(error); }
    state.busy = false; renderDisabled(); renderSavedBatch();
  }
}

function renderReferences() {
  $('reference-grid').replaceChildren(...state.profiles.map((profile) => {
    const references = (profile.references || []).filter((row) => row.status !== 'revoked');
    const card = element('article', { className: 'reference-card' }, element('strong', {}, profile.name));
    if (!references.length) card.append(element('p', { className: 'muted' }, '尚无本机参考图'));
    for (const reference of references.slice(0, 3)) card.append(element('img', {
      src: reference.image_url, loading: 'lazy', alt: profile.name + '的本机参考图',
    }), element('p', { className: 'muted' }, reference.status === 'confirmed' ? '已确认参考图' : '候选参考图 · 待核对'));
    card.append(element('button', { type: 'button', className: 'button secondary',
      onclick: () => { if (!currentTrack()) { fail(new Error('请先选择一条人物轨迹。')); return; }
        performEdit(() => updateTrack(state.clip, state.selected, { character_id: profile.id })); } }, '选为本轨角色'));
    return card;
  }));
  $('character').replaceChildren(element('option', { value: '' }, '未知 / 暂不关联'),
    ...state.profiles.map((profile) => element('option', { value: profile.id }, profile.name)));
  for (const id of ['batch-character', 'saved-character']) $(id).replaceChildren(...characterOptions());
  $('saved-track-filter').append(...state.profiles.map((profile) => element('option', { value: 'character:' + profile.id }, profile.name)));
}

function renderEditor() {
  const clip = state.clip;
  $('editor').hidden = !clip || clip.status !== 'ready';
  $('preparing').hidden = !clip || clip.status === 'ready';
  if (!clip) return;
  if (clip.status !== 'ready') {
    $('preparing').textContent = clip.status === 'preparing' ? '正在准备片段：解码音视频并生成待核对人物框，不预测说话标签。'
      : '片段准备失败：' + (clip.reason || '请检查服务日志后重试。');
    renderDisabled(); return;
  }
  $('clip-heading').textContent = timecode(clip.start_ms) + ' — ' + timecode(clip.end_ms);
  const sourceChanged = video.getAttribute('src') !== clip.clip_url;
  if (sourceChanged) { video.pause(); video.src = clip.clip_url; }
  assignValue('clip-note', clip.note || '');
  renderTracks(); renderTrackEditor(); renderTimeline(); renderDirty(); renderDisabled();
}

function adoptClip(clip, preserve = false) {
  video.pause();
  const previous = { frame: state.frame, selected: state.selected, start: state.rangeStart, end: state.rangeEnd };
  state.clip = clone(clip);
  state.original = JSON.stringify(editableSnapshot(clip));
  state.history = []; state.dirty = false; state.conflict = false; state.drag = null; state.drawing = false;
  state.frame = preserve ? Math.min(previous.frame, clip.frame_count - 1) : 0;
  state.selected = preserve && clip.tracks.some((track) => track.id === previous.selected) ? previous.selected : clip.tracks[0]?.id || null;
  state.checkedTracks = preserve ? new Set([...state.checkedTracks].filter((id) => clip.tracks.some((track) => track.id === id))) : new Set();
  showMessage('batch-result');
  if (!preserve) $('label-target').value = 'current';
  state.rangeStart = preserve ? previous.start : 0;
  state.rangeEnd = preserve ? previous.end : clip.frame_count;
  assignValue('start-seconds', clip.start_ms / 1000); assignValue('end-seconds', clip.end_ms / 1000);
  renderEditor(); renderProgress();
}

async function refreshIndex() {
  const index = await request(base);
  state.index = index;
  state.checkedClips = new Set([...state.checkedClips].filter((id) => index.clips.some((clip) => clip.id === id && clip.status === 'ready')));
  $('asset-title').textContent = index.asset_title || '本机动画素材';
  $('prepare-panel').hidden = false;
  $('start-seconds').max = index.duration_ms / 1000;
  $('end-seconds').max = index.duration_ms / 1000;
  renderProgress();
  return index;
}

function stopPolling() { state.serial += 1; clearTimeout(state.poll); }

function pollClip(id, serial) {
  state.poll = setTimeout(async () => {
    try {
      const clip = await request(base + '/clips/' + encodeURIComponent(id));
      if (serial !== state.serial) return;
      adoptClip(clip);
      if (clip.status === 'preparing') pollClip(id, serial);
      else await refreshIndex();
    } catch (error) { if (serial === state.serial) fail(error); }
  }, 1000);
}

async function openClip(id, discard = true) {
  if (state.busy || (discard && !canDiscard())) return;
  stopPolling(); const serial = state.serial;
  state.busy = true; clearMessages(); renderDisabled();
  try {
    const clip = await request(base + '/clips/' + encodeURIComponent(id));
    if (serial !== state.serial) return;
    adoptClip(clip);
    if (clip.status === 'preparing') pollClip(id, serial);
  } catch (error) { fail(error); } finally { state.busy = false; renderDisabled(); }
}

async function prepareClip(start, end, discard = true) {
  if (state.busy || (discard && !canDiscard())) return;
  try { validatePreparation(start, end, state.index.duration_ms); } catch (error) { fail(error); return; }
  const existing = state.index.clips.find((clip) => clip.start_ms === start && clip.end_ms === end && clip.status !== 'failed');
  if (existing) { await openClip(existing.id, false); return; }
  stopPolling(); const serial = state.serial;
  state.busy = true; clearMessages(); renderDisabled();
  try {
    const clip = await post(base + '/clips', { start_ms: start, end_ms: end });
    if (serial !== state.serial) return;
    adoptClip(clip); await refreshIndex();
    if (clip.status === 'preparing') pollClip(clip.id, serial);
  } catch (error) { fail(error); } finally { state.busy = false; renderDisabled(); }
}

function commitEdit(next) {
  if (state.busy || !state.clip || state.clip.status !== 'ready') return;
  if (JSON.stringify(editableSnapshot(next)) === JSON.stringify(editableSnapshot(state.clip))) return;
  state.history.push(editableSnapshot(state.clip));
  if (state.history.length > 60) state.history.shift();
  state.clip = next;
  renderEditor();
}

function performEdit(action) {
  if (state.busy) return;
  try { const next = action(); commitEdit(next); showMessage('error'); } catch (error) { fail(error); }
}

function selectTrack(id) {
  if (state.busy || state.drag) return;
  state.selected = id; state.drawing = false;
  renderTracks(); renderTrackEditor(); renderTimeline(); renderDisabled();
}

function setRange(start, end) {
  state.rangeStart = start; state.rangeEnd = end; renderFrame();
}

function seekFrame(frame, select = false) {
  if (!state.clip || state.clip.status !== 'ready' || state.drag) return;
  video.pause();
  state.frame = Math.max(0, Math.min(state.clip.frame_count - 1, Math.round(frame)));
  video.currentTime = state.frame * FRAME_MS / 1000;
  if (select) { state.rangeStart = state.frame; state.rangeEnd = state.frame + 1; }
  renderFrame();
}

function syncVideoFrame() {
  if (!state.clip || state.clip.status !== 'ready') return;
  const frame = Math.max(0, Math.min(state.clip.frame_count - 1, Math.floor((video.currentTime * 1000 + 0.01) / FRAME_MS)));
  if (frame !== state.frame) { state.frame = frame; renderFrame(); }
}

function animateVideo() {
  syncVideoFrame();
  if (!video.paused) state.animation = requestAnimationFrame(animateVideo);
}

async function togglePlay() {
  if (!state.clip || state.clip.status !== 'ready') return;
  if (!video.paused) video.pause();
  else {
    state.drawing = false; state.drag = null; renderTrackEditor(); renderBoxes();
    try { await video.play(); } catch { fail(new Error('浏览器未能播放片段，请重新准备或检查本机视频编码。')); }
  }
}

function applyLabel(label) {
  if (state.busy) return;
  try {
    const result = batchLabelRange(state.clip, operationTargets(), state.rangeStart, state.rangeEnd,
      { label, onlyUnknown: $('batch-unknown').checked });
    reportEdit(result);
  } catch (error) { fail(error); }
}

async function saveClip() {
  if (state.busy || !state.clip) return null;
  let payload;
  try { payload = savePayload(state.clip); } catch (error) { fail(error); return null; }
  state.busy = true; video.pause(); clearMessages(); renderDisabled();
  try {
    const saved = await request(base + '/clips/' + encodeURIComponent(state.clip.id), { method: 'PUT', body: JSON.stringify(payload) });
    adoptClip(saved, true); await refreshIndex(); notice('本段人工标注已保存。未确定的帧仍保留“不确定”。');
    return saved;
  } catch (error) {
    if (error.status === 409) { state.conflict = true; renderDirty();
      fail(new Error('服务器已有更新，本地修改已保留。请先下载本地草稿，再重新读取服务器版本进行核对。')); }
    else fail(error);
    return null;
  } finally { state.busy = false; renderDisabled(); }
}

async function saveNext() {
  const saved = await saveClip();
  if (!saved) return;
  if (taskItems().length) {
    const index = taskItems().indexOf(taskForClip(saved));
    const next = taskItems()[index + 1];
    if (next) await openTask(next, false);
    else notice('本段已保存，已到推荐任务末尾。可在任务列表中检查尚未完成的场景。');
    return;
  }
  const end = Math.min(saved.end_ms + 4000, Math.floor(state.index.duration_ms / FRAME_MS) * FRAME_MS);
  if (end - saved.end_ms < 640) { notice('本段已保存，原片剩余时长不足一个标注片段。'); return; }
  await prepareClip(saved.end_ms, end, false);
}

async function openTask(item, discard = true) {
  if (!item || state.busy) return;
  if (discard && !canDiscard()) { renderTasks(); return; }
  const known = state.index.clips.find((clip) => clip.id === item.clip_id
    || (clip.start_ms === item.start_ms && clip.end_ms === item.end_ms && clip.status !== 'failed'));
  if (known) await openClip(known.id, false);
  else await prepareClip(item.start_ms, item.end_ms, false);
  renderTasks(); renderDisabled();
}

function jumpMissing(draw = false) {
  const ids = operationTargets();
  const track = state.clip?.tracks.find((row) => ids.includes(row.id)
    && row.boxes.slice(state.rangeStart, state.rangeEnd).some((box) => box === null));
  if (!track) return;
  const frame = track.boxes.findIndex((box, index) => index >= state.rangeStart && index < state.rangeEnd && box === null);
  selectTrack(track.id); seekFrame(frame);
  if (draw) { state.drawing = true; $('box-tools').open = true; renderTrackEditor(); renderBoxes(); }
}

function pointerPosition(event) {
  const bounds = svg.getBoundingClientRect();
  return [Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
    Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height))];
}

function dragBox(drag) {
  return [Math.min(drag.from[0], drag.to[0]), Math.min(drag.from[1], drag.to[1]),
    Math.abs(drag.to[0] - drag.from[0]), Math.abs(drag.to[1] - drag.from[1])];
}

svg.addEventListener('pointerdown', (event) => {
  if (!state.drawing || !video.paused || !currentTrack() || state.busy || event.button !== 0) return;
  event.preventDefault();
  const point = pointerPosition(event);
  state.drag = { from: point, to: point, track: state.selected, frame: state.frame, pointer: event.pointerId };
  try { svg.setPointerCapture(event.pointerId); } catch { /* Capture may be unavailable on older browsers. */ }
  renderBoxes();
});
svg.addEventListener('pointermove', (event) => {
  if (!state.drag || event.pointerId !== state.drag.pointer) return;
  state.drag.to = pointerPosition(event); renderBoxes();
});
svg.addEventListener('pointerup', (event) => {
  if (!state.drag || event.pointerId !== state.drag.pointer) return;
  const drag = state.drag; drag.to = pointerPosition(event); state.drag = null;
  const box = dragBox(drag);
  if (box[2] < .003 || box[3] < .003) { renderBoxes(); return; }
  performEdit(() => setFrameBox(state.clip, drag.track, drag.frame, box));
});
svg.addEventListener('pointercancel', () => { state.drag = null; renderBoxes(); });

$('prepare-form').addEventListener('submit', (event) => {
  event.preventDefault();
  void prepareClip(Math.round(Number($('start-seconds').value) * 1000), Math.round(Number($('end-seconds').value) * 1000));
});
$('previous-frame').addEventListener('click', () => seekFrame(state.frame - 1));
$('next-frame').addEventListener('click', () => seekFrame(state.frame + 1));
$('toggle-play').addEventListener('click', () => void togglePlay());
$('frame-slider').addEventListener('input', (event) => seekFrame(Number(event.target.value)));
$('range-start').addEventListener('input', (event) => { video.pause(); state.rangeStart = Number(event.target.value); renderFrame(); });
$('range-end').addEventListener('input', (event) => { video.pause(); state.rangeEnd = Number(event.target.value); renderFrame(); });
$('current-frame-range').addEventListener('click', () => setRange(state.frame, state.frame + 1));
$('whole-track-range').addEventListener('click', () => setRange(0, state.clip.frame_count));
document.querySelectorAll('[data-select-tracks]').forEach((button) => button.addEventListener('click', () => {
  const action = button.dataset.selectTracks;
  const tracks = state.clip.tracks;
  const current = currentTrack();
  if (action === 'same' && !current?.character_id) { fail(new Error('当前人物尚未指定角色；请先指定身份后再选同角色轨迹。')); return; }
  state.checkedTracks = new Set(tracks.filter((track) => action === 'all'
    || (action === 'invert' && !state.checkedTracks.has(track.id))
    || (action === 'visible' && track.boxes[state.frame] !== null && !track.excluded)
    || (action === 'unreviewed' && !track.reviewed && !track.excluded)
    || (action === 'same' && track.character_id === current.character_id && !track.excluded)).map((track) => track.id));
  $('label-target').value = state.checkedTracks.size ? 'checked' : 'current';
  renderTracks(); renderBatch(); renderDisabled();
}));
$('label-target').addEventListener('change', () => { renderBatch(); renderDisabled(); });
$('batch-unknown').addEventListener('change', renderBatch);
$('jump-missing').addEventListener('click', () => jumpMissing());
$('fix-missing').addEventListener('click', () => jumpMissing(true));
$('task-select').addEventListener('change', (event) => void openTask(taskItems().find((item) => item.id === event.target.value)));
$('previous-task').addEventListener('click', () => void openTask(taskItems()[taskItems().indexOf(taskForClip(state.clip)) - 1]));
$('next-task').addEventListener('click', () => void openTask(taskItems()[taskItems().indexOf(taskForClip(state.clip)) + 1]));
document.querySelectorAll('[data-batch-label]').forEach((button) => button.addEventListener('click', () =>
  editBatch({ 1: 'speaking', 0: 'silent', '-1': 'uncertain' }[button.dataset.batchLabel])));
document.querySelectorAll('[data-batch-track]').forEach((button) => button.addEventListener('click', () => editBatch(button.dataset.batchTrack)));
$('batch-speaker').addEventListener('click', () => editBatch('speaker'));
$('batch-identity').addEventListener('click', () => editBatch('identity'));
document.querySelectorAll('[data-select-clips]').forEach((button) => button.addEventListener('click', () => {
  const action = button.dataset.selectClips;
  state.checkedClips = new Set(state.index.clips.filter((clip) => clip.status === 'ready' && (action === 'all'
    || (action === 'invert' && !state.checkedClips.has(clip.id))
    || (action === 'unreviewed' && clip.tracks.some((track) => !track.reviewed && !track.excluded)))).map((clip) => clip.id));
  renderProgress(); renderDisabled();
}));
for (const id of ['saved-track-filter', 'saved-action', 'saved-character']) $(id).addEventListener('change', renderSavedBatch);
$('saved-batch-apply').addEventListener('click', () => void savedBatch());
$('saved-batch-undo').addEventListener('click', () => void savedBatch(true));
document.querySelectorAll('[data-label]').forEach((button) => button.addEventListener('click', () => applyLabel(Number(button.dataset.label))));
$('character').addEventListener('change', (event) => performEdit(() => updateTrack(state.clip, state.selected, { character_id: event.target.value || null })));
$('reviewed').addEventListener('change', (event) => performEdit(() => updateTrack(state.clip, state.selected, { reviewed: event.target.checked }, false)));
$('excluded').addEventListener('change', (event) => performEdit(() => updateTrack(state.clip, state.selected, { excluded: event.target.checked })));
$('track-note').addEventListener('input', (event) => performEdit(() => updateTrack(state.clip, state.selected, { note: event.target.value }, false)));
$('clip-note').addEventListener('input', (event) => performEdit(() => ({ ...clone(state.clip), note: event.target.value })));
$('draw-box').addEventListener('click', () => { video.pause(); state.drawing = !state.drawing; renderTrackEditor(); renderBoxes(); });
$('clear-box').addEventListener('click', () => performEdit(() => setFrameBox(state.clip, state.selected, state.frame, null)));
$('copy-box').addEventListener('click', () => performEdit(() => applyCurrentBox(state.clip, state.selected, state.frame, state.rangeStart, state.rangeEnd)));
$('add-track').addEventListener('click', () => {
  const id = 'track_' + crypto.randomUUID().replace(/-/g, '');
  performEdit(() => ({ ...clone(state.clip), tracks: [...clone(state.clip.tracks), newTrack(state.clip.frame_count, id)] }));
  state.selected = id; video.pause(); state.drawing = true;
  renderTracks(); renderTrackEditor(); renderTimeline(); renderDisabled();
});
$('undo').addEventListener('click', () => {
  const previous = state.history.pop();
  if (!previous) return;
  state.clip = restoreSnapshot(state.clip, previous);
  if (!state.clip.tracks.some((track) => track.id === state.selected)) state.selected = state.clip.tracks[0]?.id || null;
  renderEditor();
  showMessage('batch-result', '已撤销上一步修改，整批操作会一起恢复。');
});
$('save').addEventListener('click', () => void saveClip());
$('save-next').addEventListener('click', () => void saveNext());
  $('refresh-progress').addEventListener('click', () => void refreshIndex().then(renderDisabled).catch(fail));
$('reload-clip').addEventListener('click', () => void openClip(state.clip.id));
$('download-draft').addEventListener('click', () => {
  const body = { draft_only: true, asset_id: state.index.asset_id, clip_id: state.clip.id,
    start_ms: state.clip.start_ms, end_ms: state.clip.end_ms, ...savePayload(state.clip) };
  const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], { type: 'application/json' }));
  const link = element('a', { href: url, download: 'asd-local-draft-' + state.clip.id + '.json' });
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('loop').addEventListener('change', (event) => { video.loop = event.target.checked; });
$('speed').addEventListener('change', (event) => { video.playbackRate = Number(event.target.value); });
video.loop = $('loop').checked;
video.addEventListener('loadedmetadata', () => {
  if (video.videoWidth && video.videoHeight) $('stage').style.aspectRatio = video.videoWidth + '/' + video.videoHeight;
  video.playbackRate = Number($('speed').value); video.loop = $('loop').checked;
});
video.addEventListener('play', () => { state.drawing = false; renderTrackEditor(); renderBoxes(); cancelAnimationFrame(state.animation); animateVideo(); });
video.addEventListener('pause', () => { cancelAnimationFrame(state.animation); syncVideoFrame(); renderBoxes(); });
video.addEventListener('timeupdate', syncVideoFrame);
video.addEventListener('seeked', syncVideoFrame);
video.addEventListener('error', () => { if (state.clip?.status === 'ready') fail(new Error('片段暂时无法播放。可重新准备该片段，或检查本机服务。')); });
window.addEventListener('keydown', (event) => {
  if (!state.clip || state.clip.status !== 'ready' || state.busy || state.drag
      || event.target.closest('input,textarea,select,[contenteditable="true"]') || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.code === 'Space') { event.preventDefault(); void togglePlay(); }
  else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); seekFrame(state.frame + (event.key === 'ArrowLeft' ? -1 : 1)); }
  else if (['s', 'n', 'u'].includes(event.key.toLowerCase())) { event.preventDefault(); applyLabel({ s: 1, n: 0, u: -1 }[event.key.toLowerCase()]); }
});
window.addEventListener('beforeunload', (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ''; } });

async function bootstrap() {
  const index = await refreshIndex();
  const characters = index.profiles?.every((profile) => Array.isArray(profile.references))
    ? { profiles: index.profiles } : await request('/api/characters');
  state.profiles = characters.profiles || []; renderReferences();
  $('session').hidden = true;
  if (!state.clip) {
    assignValue('start-seconds', index.default_start_ms / 1000);
    assignValue('end-seconds', index.default_end_ms / 1000);
    const task = taskItems().find((item) => {
      const clip = index.clips.find((row) => row.id === item.clip_id);
      return !clip || clip.status !== 'ready' || !clip.tracks.length || clip.tracks.some((track) => !track.reviewed && !track.excluded);
    }) || taskItems()[0];
    if (task) { await openTask(task, false); return; }
    const candidate = index.clips.find((clip) => clip.status === 'ready' && clip.tracks?.some((track) => !track.reviewed && !track.excluded))
      || index.clips.at(-1);
    if (candidate) await openClip(candidate.id, false);
  } else renderTrackEditor();
}

$('session-form').addEventListener('submit', async (event) => {
  event.preventDefault(); clearMessages();
  try { await post('/api/session', { token: $('token').value.trim() }); $('token').value = ''; await bootstrap(); }
  catch (error) { fail(error); }
});

const address = new URL(window.location.href);
const token = address.searchParams.get('token');
if (token) { address.searchParams.delete('token'); history.replaceState({}, '', address.pathname + address.search + address.hash); }
try {
  if (token) await post('/api/session', { token });
  await bootstrap();
} catch (error) { if (error.status === 401) $('session').hidden = false; fail(error); }
