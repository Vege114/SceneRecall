// Optional DOM tests use the repository's existing dev dependency; the tool itself has no npm dependency.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as editor from './editor.js';

let JSDOM;
try { ({ JSDOM } = await import('../../frontend/node_modules/jsdom/lib/api.js')); } catch { /* optional dev runtime */ }
const uiTest = JSDOM ? test : test.skip;
const copy = editor.clone;
const html = await readFile(new URL('./index.html', import.meta.url), 'utf8');
const source = (await readFile(new URL('./app.js', import.meta.url), 'utf8'))
  .replace(/^import[\s\S]*?from '\.\/editor.js';/, 'const {' + Object.keys(editor).join(',') + '} = window.annotationEditor;');

async function fixture({ mutateClips = () => {}, plan = false } = {}) {
  const clips = [0, 1].map((index) => ({ id: 'clip_' + index, revision: 1, status: 'ready', fps: 25,
    start_ms: 1204000 + index * 4000, end_ms: 1208000 + index * 4000, frame_count: 100, note: '',
    clip_url: '/clip-' + index + '.mp4', tracks: [0, 1].map((person) => {
      const track = editor.newTrack(100, 'track_' + index + '_' + person);
      track.boxes = Array.from({ length: 100 }, () => [.1, .1, .2, .4]); return track;
    }) }));
  mutateClips(clips);
  const annotation_plan = plan ? { schema_version: '1.0', items: clips.map((clip, index) => ({
    id: 'task_' + index, clip_id: clip.id, start_ms: clip.start_ms, end_ms: clip.end_ms,
    title: '推荐场景 ' + index, split: index ? 'val' : 'train', description: '请人工核对画面及声音',
    scene_id: 'scene_' + index,
  })) } : null;
  const dom = new JSDOM(html, { url: 'http://127.0.0.1:8766/', runScripts: 'outside-only', pretendToBeVisual: true });
  const w = dom.window;
  w.annotationEditor = editor;
  w.HTMLMediaElement.prototype.pause = function () {};
  w.confirm = () => true;
  const writes = [];
  const control = { failPut: null };
  w.fetch = async (path, options = {}) => {
    let value;
    const id = path.split('/').at(-1);
    const current = clips.find((clip) => clip.id === id);
    if (options.method === 'PUT') {
      const payload = JSON.parse(options.body);
      if (control.failPut === id || payload.revision !== current.revision) {
        return { ok: false, status: 409, json: async () => ({ detail: 'revision conflict' }) };
      }
      writes.push({ id, payload: copy(payload) });
      Object.assign(current, payload, { revision: current.revision + 1 }); value = copy(current);
    } else if (current) value = copy(current);
    else if (path === '/api/asd-annotations') value = { asset_title: 'Test movie', duration_ms: 9999999,
      default_start_ms: 1204000, default_end_ms: 1208000, clips: copy(clips), stats: {},
      annotation_plan, profiles: [{ id: 'a', name: '角色 A', references: [] }, { id: 'b', name: '角色 B', references: [] }] };
    else throw new Error('Unexpected mock request: ' + path);
    return { ok: true, status: 200, json: async () => value };
  };
  await w.eval('(async()=>{' + source + '\n})()');
  const $ = (id) => w.document.getElementById(id);
  const click = (selector) => w.document.querySelector(selector).click();
  const change = (id, value) => { $(id).value = value; $(id).dispatchEvent(new w.Event('change', { bubbles: true })); };
  const input = (id, value) => { $(id).value = value; $(id).dispatchEvent(new w.Event('input', { bubbles: true })); };
  const settled = async (predicate) => {
    for (let count = 0; count < 200; count += 1) {
      if (predicate()) return;
      await new Promise(setImmediate);
    }
    assert.fail('Mock UI did not settle: ' + $('error').textContent);
  };
  return { dom, w, $, click, change, input, settled, clips, writes, control };
}

uiTest('DOM: select two people, label whole clip with one speaker, and undo the entire batch', async (t) => {
  const f = await fixture(); t.after(() => f.dom.window.close());
  f.click('[data-select-tracks="all"]');
  assert.equal(f.w.document.querySelectorAll('#track-list input:checked').length, 2);
  f.$('batch-speaker').click();
  assert.match(f.$('batch-result').textContent, /修改 2 条轨迹、200/);
  assert.match(f.$('track-list').textContent, /已标 100\/100/);
  assert.equal(f.writes.length, 0);
  f.$('undo').click();
  assert.doesNotMatch(f.$('track-list').textContent, /已标 100/);
  assert.equal(f.$('save-state').textContent, '与服务器版本一致');
  f.click('[data-select-tracks="invert"]');
  assert.equal(f.w.document.querySelectorAll('#track-list input:checked').length, 0);
});

uiTest('DOM: row identity is one whole-track edit and batch review is explicitly saved once', async (t) => {
  const f = await fixture(); t.after(() => f.dom.window.close());
  const select = f.w.document.querySelector('select[aria-label="人物1整轨角色"]');
  select.value = 'a'; select.dispatchEvent(new f.w.Event('change', { bubbles: true }));
  f.click('[data-select-tracks="all"]'); f.click('[data-batch-track="reviewed"]'); f.$('save').click();
  await f.settled(() => f.writes.length === 1 && !f.$('save').disabled);
  assert.equal(f.writes[0].payload.tracks[0].character_id, 'a');
  assert.equal(f.writes[0].payload.tracks[1].character_id, null);
  assert.ok(f.writes[0].payload.tracks.every((track) => track.reviewed && track.labels.every((label) => label === -1)));
});

uiTest('DOM: cross-clip batch saves each selected revision and undo creates restoring revisions', async (t) => {
  const f = await fixture(); t.after(() => f.dom.window.close());
  f.click('[data-select-clips="all"]'); f.$('saved-batch-apply').click();
  await f.settled(() => f.$('saved-batch-result').textContent.startsWith('已处理并保存 2'));
  assert.equal(f.writes.length, 2);
  assert.ok(f.clips.every((clip) => clip.tracks.every((track) => track.reviewed)));
  f.$('saved-batch-undo').click();
  await f.settled(() => f.$('saved-batch-result').textContent.startsWith('已撤销并保存 2'));
  assert.equal(f.writes.length, 4);
  assert.ok(f.clips.every((clip) => clip.revision === 3 && clip.tracks.every((track) => !track.reviewed)));
});

uiTest('DOM: preflight revision mismatch prevents every write and retains saved data', async (t) => {
  const f = await fixture(); t.after(() => f.dom.window.close());
  f.click('[data-select-clips="all"]'); f.clips[1].revision += 1; f.$('saved-batch-apply').click();
  await f.settled(() => f.$('error').textContent.includes('版本已变化'));
  assert.equal(f.writes.length, 0);
});

uiTest('DOM: mid-batch conflict stops later writes and completed part remains undoable', async (t) => {
  const f = await fixture(); t.after(() => f.dom.window.close());
  f.click('[data-select-clips="all"]'); f.control.failPut = 'clip_1'; f.$('saved-batch-apply').click();
  await f.settled(() => f.$('error').textContent.includes('保存时发生版本冲突') && !f.$('saved-batch-undo').disabled);
  assert.equal(f.writes.length, 1);
  assert.match(f.$('saved-batch-result').textContent, /已保存 1 个片段后停止/);
  f.control.failPut = null; f.$('saved-batch-undo').click();
  await f.settled(() => f.$('saved-batch-result').textContent.startsWith('已撤销并保存 1'));
  assert.ok(f.clips.every((clip) => clip.tracks.every((track) => !track.reviewed)));
});

uiTest('DOM: cross-clip processing cannot discard an unsaved current draft', async (t) => {
  const f = await fixture(); t.after(() => f.dom.window.close());
  f.click('[data-select-tracks="all"]'); f.$('batch-speaker').click();
  f.click('[data-select-clips="all"]'); f.$('saved-batch-apply').click();
  assert.match(f.$('error').textContent, /未保存修改/);
  assert.equal(f.writes.length, 0);
  assert.match(f.$('track-list').textContent, /已标 100\/100/);
});

uiTest('DOM: screenshot reproduction labels only boxed frames but explains every gap next to the main action', async (t) => {
  const f = await fixture({ mutateClips: (clips) => {
    clips[0].tracks[1].boxes = Array.from({ length: 100 }, (_, index) => index >= 13 && index < 32 ? [.1, .1, .2, .4] : null);
  } });
  t.after(() => f.dom.window.close());
  f.w.document.querySelectorAll('#track-list button')[1].click();
  f.input('range-start', 0); f.input('range-end', 37);
  f.w.document.querySelectorAll('#timeline button')[36].click();
  assert.equal(f.$('range-start').value, '0'); assert.equal(f.$('range-end').value, '37');
  assert.equal(f.w.document.querySelectorAll('#timeline .state-missing').length, 81);
  f.click('[data-label="1"]');
  assert.match(f.$('batch-result').textContent, /19 个人物帧标签；跳过 18 个缺框帧/);
  assert.match(f.$('batch-result').textContent, /\[0, 13\).*\[32, 37\)/);
  assert.equal(f.$('batch-result').hidden, false); assert.equal(f.$('missing-guidance').hidden, false);
  f.$('jump-missing').click();
  assert.match(f.$('frame-position').textContent, /片段帧 0 \/ 99/);
  assert.equal(f.$('range-end').value, '37');
  f.$('save').click(); await f.settled(() => f.writes.length === 1 && !f.$('save').disabled);
  const track = f.writes[0].payload.tracks[1];
  assert.ok(track.labels.slice(0, 13).every((label) => label === -1));
  assert.ok(track.labels.slice(13, 32).every((label) => label === 1));
  assert.ok(track.labels.slice(32).every((label) => label === -1));
  assert.equal(track.reviewed, false); assert.equal(track.boxes[0], null);
});

uiTest('DOM: a single shared range and unknown-only option work identically for current and checked people', async (t) => {
  const f = await fixture({ mutateClips: (clips) => { clips[0].tracks.forEach((track) => { track.labels[4] = 1; }); } });
  t.after(() => f.dom.window.close());
  f.input('range-start', 4); f.input('range-end', 8); f.$('batch-unknown').click();
  f.click('[data-label="0"]');
  assert.match(f.$('batch-result').textContent, /3 个人物帧标签；跳过 0 个缺框帧，保留 1 个已有标签/);
  f.click('[data-select-tracks="all"]'); assert.equal(f.$('label-target').value, 'checked');
  f.click('[data-label="0"]');
  assert.match(f.$('batch-result').textContent, /3 个人物帧标签；跳过 0 个缺框帧，保留 5 个已有标签/);
  f.$('batch-unknown').click(); f.click('[data-label="-1"]');
  assert.match(f.$('batch-result').textContent, /8 个人物帧标签/);
  assert.equal(f.$('range-start').value, '4'); assert.equal(f.$('range-end').value, '8');
  f.$('save').click(); await f.settled(() => f.writes.length === 1 && !f.$('save').disabled);
  assert.ok(f.writes[0].payload.tracks.every((track) => track.labels.every((label) => label === -1)));
});

uiTest('DOM: scrubbing, previous/next frame and playback keep the chosen range and an unfinished focused input', async (t) => {
  const f = await fixture(); t.after(() => f.dom.window.close());
  f.input('range-start', 2); f.input('range-end', 37);
  f.input('frame-slider', 36); f.$('next-frame').click(); f.$('previous-frame').click();
  assert.equal(f.$('range-start').value, '2'); assert.equal(f.$('range-end').value, '37');
  f.$('range-end').focus(); f.input('range-end', '');
  f.$('clip-video').currentTime = .8;
  f.$('clip-video').dispatchEvent(new f.w.Event('timeupdate'));
  assert.equal(f.w.document.activeElement, f.$('range-end')); assert.equal(f.$('range-end').value, '');
});

uiTest('DOM: recommended queue keeps scene split visible, saves before next task and leaves original clips accessible', async (t) => {
  const f = await fixture({ plan: true }); t.after(() => f.dom.window.close());
  assert.equal(f.$('task-panel').hidden, false); assert.equal(f.$('task-select').value, 'task_0');
  assert.match(f.$('task-context').textContent, /用于训练/);
  f.click('[data-label="0"]'); f.$('save-next').click();
  await f.settled(() => f.$('task-select').value === 'task_1' && !f.$('save').disabled);
  assert.equal(f.writes.length, 1); assert.match(f.$('task-context').textContent, /用于验证/);
  assert.equal(f.w.document.querySelectorAll('#clip-list .clip-row').length, 2);
  assert.equal(f.$('next-task').disabled, true); assert.equal(f.$('previous-task').disabled, false);
});

uiTest('DOM: changing tasks protects unsaved labels and restores the selector when discard is cancelled', async (t) => {
  const f = await fixture({ plan: true }); t.after(() => f.dom.window.close());
  f.click('[data-label="1"]'); f.w.confirm = () => false;
  f.change('task-select', 'task_1'); await new Promise(setImmediate);
  assert.equal(f.$('task-select').value, 'task_0'); assert.equal(f.writes.length, 0);
  assert.equal(f.$('save-state').textContent, '有未保存修改');
});
