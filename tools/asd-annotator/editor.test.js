import test from 'node:test';
import assert from 'node:assert/strict';
import { clone, timecode, validatePreparation, validateRange, newTrack, updateTrack,
  labelRange, setFrameBox, applyCurrentBox, editableSnapshot, restoreSnapshot, savePayload,
  batchLabelRange, batchUpdateTracks, batchAction, matchingTracks } from './editor.js';

const box = [.1, .2, .3, .4];
function clip() {
  return { id: 'clip_1', revision: 7, status: 'ready', start_ms: 1204000, end_ms: 1208000,
    fps: 25, frame_count: 100, clip_url: '/api/asd-annotations/clips/clip_1/media', note: '',
    source_fingerprint: 'server-owned-fingerprint', tracks: [newTrack(100, 'track_1')] };
}

test('new human tracks contain no predicted identity, speech or review state', () => {
  const track = newTrack(100, 'track_new');
  assert.equal(track.character_id, null);
  assert.equal(track.reviewed, false);
  assert.equal(track.excluded, false);
  assert.equal(track.labels.length, 100);
  assert.ok(track.labels.every((label) => label === -1));
  assert.ok(track.boxes.every((value) => value === null));
});

test('preparation uses actual millisecond boundaries on the 25fps grid', () => {
  validatePreparation(1204000, 1208000, 7200000);
  assert.throws(() => validatePreparation(1204001, 1208000, 7200000));
  assert.throws(() => validatePreparation(0, 600, 7200000));
  assert.throws(() => validatePreparation(0, 8040, 7200000));
  assert.throws(() => validatePreparation(1204000, 1208000, 1206000));
  assert.equal(timecode(1204040), '00:20:04.040');
});

test('a missing box prevents positive and negative labeling without partially changing the draft', () => {
  const original = clip();
  original.tracks[0].boxes[0] = [...box];
  const before = clone(original);
  for (const label of [0, 1]) assert.throws(() => labelRange(original, 'track_1', 0, 2, label), /缺少人物框/);
  assert.deepEqual(original, before);
  assert.deepEqual(labelRange(original, 'track_1', 0, 2, -1).tracks[0].labels, before.tracks[0].labels);
});

test('labeling is half-open, invalidates review and leaves neighboring frames unchanged', () => {
  const original = clip();
  original.tracks[0].boxes = Array.from({ length: 100 }, () => [...box]);
  original.tracks[0].reviewed = true;
  const changed = labelRange(original, 'track_1', 2, 5, 1);
  assert.deepEqual(changed.tracks[0].labels.slice(0, 7), [-1, -1, 1, 1, 1, -1, -1]);
  assert.equal(changed.tracks[0].reviewed, false);
  assert.equal(original.tracks[0].reviewed, true);
  assert.ok(original.tracks[0].labels.every((label) => label === -1));
  assert.throws(() => validateRange(3, 3, 100));
  assert.throws(() => validateRange(0, 101, 100));
});

test('clearing a box masks only that frame and revokes its track review', () => {
  const original = clip();
  original.tracks[0].boxes[4] = [...box];
  original.tracks[0].labels[4] = 1;
  original.tracks[0].reviewed = true;
  const changed = setFrameBox(original, 'track_1', 4, null);
  assert.equal(changed.tracks[0].boxes[4], null);
  assert.equal(changed.tracks[0].labels[4], -1);
  assert.equal(changed.tracks[0].reviewed, false);
  assert.equal(original.tracks[0].labels[4], 1);
});

test('range box application copies the one explicitly chosen box without motion interpolation or speech labels', () => {
  let original = setFrameBox(clip(), 'track_1', 9, box);
  original.tracks[0].reviewed = true;
  const changed = applyCurrentBox(original, 'track_1', 9, 10, 14);
  assert.equal(changed.tracks[0].boxes[8], null);
  assert.deepEqual(changed.tracks[0].boxes[10], box);
  assert.deepEqual(changed.tracks[0].boxes[13], box);
  assert.equal(changed.tracks[0].boxes[14], null);
  assert.notEqual(changed.tracks[0].boxes[10], changed.tracks[0].boxes[11]);
  assert.ok(changed.tracks[0].labels.every((label) => label === -1));
  assert.equal(changed.tracks[0].reviewed, false);
  assert.throws(() => applyCurrentBox(clip(), 'track_1', 0, 0, 100), /尚无人物框/);
});

test('identity and box edits invalidate review while only an explicit review action restores it', () => {
  const original = clip();
  original.tracks[0].reviewed = true;
  const identified = updateTrack(original, 'track_1', { character_id: 'local-roster-character' });
  assert.equal(identified.tracks[0].reviewed, false);
  const reviewed = updateTrack(identified, 'track_1', { reviewed: true }, false);
  assert.equal(reviewed.tracks[0].reviewed, true);
  assert.equal(setFrameBox(reviewed, 'track_1', 0, box).tracks[0].reviewed, false);
});

test('undo restores editable content while retaining server revision and provenance', () => {
  const original = clip();
  const snapshot = editableSnapshot(original);
  const changed = setFrameBox(original, 'track_1', 0, box);
  changed.note = 'local draft';
  const restored = restoreSnapshot(changed, snapshot);
  assert.deepEqual(restored, original);
  assert.equal(restored.revision, 7);
  assert.equal(restored.source_fingerprint, original.source_fingerprint);
});

test('save payload has only contract edit fields and rejects missing-box labels and invalid boxes', () => {
  const original = clip();
  const payload = savePayload(original);
  assert.deepEqual(Object.keys(payload).sort(), ['note', 'revision', 'tracks']);
  assert.deepEqual(Object.keys(payload.tracks[0]).sort(), ['boxes', 'character_id', 'excluded', 'id', 'labels', 'note', 'reviewed']);
  assert.equal(payload.revision, 7);
  assert.equal(payload.tracks[0].character_id, null);
  original.tracks[0].labels[0] = 0;
  assert.throws(() => savePayload(original), /缺框帧/);
  original.tracks[0].labels[0] = -1;
  assert.throws(() => setFrameBox(original, 'track_1', 0, [.8, .1, .4, .3]), /人物框/);
  original.tracks.push(clone(original.tracks[0]));
  assert.throws(() => savePayload(original), /重复/);
});

function twoPeople() {
  const value = clip();
  value.tracks.push(newTrack(100, 'track_2'), newTrack(100, 'track_3'));
  value.tracks.forEach((track) => { track.boxes = Array.from({ length: 100 }, () => [...box]); track.reviewed = true; });
  return value;
}

test('one batch sets a speaker and the other selected person silent in a half-open range', () => {
  const original = twoPeople();
  const result = batchLabelRange(original, ['track_1', 'track_2'], 20, 50, { speaker: 'track_2' });
  assert.equal(result.changedFrames, 60);
  assert.equal(result.changedTracks, 2);
  assert.deepEqual(result.clip.tracks[0].labels.slice(19, 22), [-1, 0, 0]);
  assert.deepEqual(result.clip.tracks[1].labels.slice(49, 52), [1, -1, -1]);
  assert.equal(result.clip.tracks[0].reviewed, false);
  assert.ok(result.clip.tracks[2].labels.every((label) => label === -1));
  assert.equal(result.clip.tracks[2].reviewed, true);
  assert.deepEqual(restoreSnapshot(result.clip, editableSnapshot(original)), original);
});

test('batch preserves known labels and missing-box masks, counts skips and does not revoke unchanged review', () => {
  const original = twoPeople();
  original.tracks[0].labels[1] = 1;
  original.tracks[1].boxes[0] = null;
  const result = batchLabelRange(original, ['track_1', 'track_2'], 0, 2, { label: 0, onlyUnknown: true });
  assert.equal(result.changedFrames, 2);
  assert.equal(result.keptFrames, 1);
  assert.equal(result.missingFrames, 1);
  assert.deepEqual(result.clip.tracks[0].labels.slice(0, 3), [0, 1, -1]);
  assert.deepEqual(result.clip.tracks[1].labels.slice(0, 3), [-1, 0, -1]);
  const unchanged = batchLabelRange(original, ['track_1'], 1, 2, { label: 1 });
  assert.equal(unchanged.changedTracks, 0);
  assert.equal(unchanged.clip.tracks[0].reviewed, true);
  savePayload(result.clip);
});

test('explicit overwrite clears known labels including missing-box unknown frames without inventing boxes', () => {
  const original = twoPeople();
  original.tracks[0].labels.fill(1);
  const cleared = batchAction(original, ['track_1'], 'uncertain');
  assert.equal(cleared.changedFrames, 100);
  assert.ok(cleared.clip.tracks[0].labels.every((label) => label === -1));
  assert.deepEqual(cleared.clip.tracks[0].boxes, original.tracks[0].boxes);
});

test('batch identity and review apply to full tracks and never auto-merge identities or change labels', () => {
  const original = twoPeople();
  const identified = batchUpdateTracks(original, ['track_1', 'track_2'], { character_id: 'character_a' });
  assert.equal(identified.tracks[0].character_id, 'character_a');
  assert.equal(identified.tracks[1].character_id, 'character_a');
  assert.equal(identified.tracks[2].character_id, null);
  assert.equal(identified.tracks[0].reviewed, false);
  assert.deepEqual(identified.tracks[0].labels, original.tracks[0].labels);
  const reviewed = batchAction(identified, ['track_1', 'track_2'], 'reviewed').clip;
  assert.equal(reviewed.tracks[1].reviewed, true);
  assert.equal(batchAction(reviewed, ['track_1'], 'exclude').clip.tracks[0].reviewed, false);
});

test('cross-clip filters use human character identity instead of track number and respect exclusions', () => {
  const original = twoPeople();
  original.tracks[0].character_id = 'a'; original.tracks[1].character_id = 'b'; original.tracks[2].excluded = true;
  assert.deepEqual(matchingTracks(original, 'character:b'), ['track_2']);
  assert.deepEqual(matchingTracks(original, 'active'), ['track_1', 'track_2']);
  assert.deepEqual(matchingTracks(original, 'unknown'), []);
  assert.equal(matchingTracks(original, 'all').length, 3);
});

test('invalid batch selections and ranges reject the whole edit without partial writes', () => {
  const original = twoPeople(); const before = clone(original);
  assert.throws(() => batchLabelRange(original, [], 0, 10, { label: 1 }), /勾选/);
  assert.throws(() => batchLabelRange(original, ['track_1', 'missing'], 0, 10, { label: 1 }), /已变化/);
  assert.throws(() => batchLabelRange(original, ['track_1'], 0, 10, { speaker: 'track_2' }), /当前人物/);
  assert.throws(() => batchAction(original, ['track_1'], 'speaking', { start: 40, end: 20 }), /帧范围/);
  assert.throws(() => batchUpdateTracks(original, ['track_1'], { boxes: [] }), /无效/);
  assert.deepEqual(original, before);
});
