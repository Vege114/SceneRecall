"""Prepare a frozen local scene queue with unknown labels using the standalone API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--supersede-unlabeled-plan', action='store_true', help='Archive an untouched proposal plan before refining its ranges')
    args = parser.parse_args()
    data = args.data_dir.resolve()
    selection = json.loads(args.selection.read_text(encoding='utf-8-sig'))
    state = json.loads((data / 'server.private.json').read_text(encoding='utf-8'))
    if state.get('service') != 'scenerecall-asd-annotator' or state.get('host') != '127.0.0.1':
        raise ValueError('Unexpected standalone service identity')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None):
        current = json.loads((data / 'server.private.json').read_text(encoding='utf-8'))
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(f'http://127.0.0.1:{state["port"]}/api/asd-annotations' + path, data=body,
                                     headers={'Authorization': 'Bearer ' + current['token'], 'Content-Type': 'application/json'})
        with opener.open(req, timeout=30) as response:
            return json.load(response)

    index = request('')
    if selection['source_video_sha256'] != index['source_fingerprint']:
        raise ValueError('Selection source differs from the configured movie')
    if not index['clips']:
        raise ValueError('Prepare a source-bound initial clip before installing a scene plan')
    source_stat = index['clips'][0]['source_stat']
    config_stream = index['clips'][0]['audio_stream_index']
    items = []
    for position, scene in enumerate(selection['scenes'], 1):
        identity = ['scenerecall-human-asd-v1', index['source_fingerprint'], source_stat,
                    scene['start_ms'], scene['end_ms'], config_stream]
        key = 'asd_' + hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
        items.append({'id': scene['candidate_id'], 'clip_id': key, 'scene_id': scene['scene_group'],
                      'start_ms': scene['start_ms'], 'end_ms': scene['end_ms'], 'split': scene['split'],
                      'title': scene['title'], 'category': ', '.join(scene['visual_categories']),
                      'description': scene['visual_description'] + '。请听日语原声判断说话区间；画面类型不是说话标签。',
                      'priority': position, 'human_labels_created': False})
    plan = {'schema_version': 'scenerecall-asd-scene-plan-v1', 'asset_id': index['asset_id'],
            'source_fingerprint': index['source_fingerprint'], 'title': '辉夜姬第一轮人工标注',
            'minimum_cross_split_gap_ms': 90000, 'items': items,
            'selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest(),
            'selection_method': 'dialogue_timing_plus_reviewed_center_frames_and_detector_quality',
            'validation_scope': 'within_title_scene_holdout', 'cross_title_validation': False,
            'baseline_prior_film_exposure': 'previous_weak_adaptation_not_pristine',
            'prior_training_audit': selection.get('prior_training_ranges', []),
            'scene_groups': selection.get('scene_groups', []), 'labels_generated': False}
    target = data / 'annotation-plan.json'
    if target.exists():
        existing = json.loads(target.read_text(encoding='utf-8'))
        if existing.get('selection_sha256') != plan['selection_sha256']:
            if not args.supersede_unlabeled_plan:
                raise ValueError('A different scene plan exists; preserve it and start a new round explicitly')
            old_groups = {item['id']: (item['scene_id'], item['split']) for item in existing['items']}
            new_groups = {item['id']: (item['scene_id'], item['split']) for item in plan['items']}
            if old_groups != new_groups or existing['source_fingerprint'] != plan['source_fingerprint']:
                raise ValueError('Range refinement cannot change scenes, source, or frozen split assignments')
            for item in existing['items']:
                saved = data / 'clips' / item['clip_id'] / 'clip.json'
                if saved.is_file() and json.loads(saved.read_text(encoding='utf-8'))['revision'] != 0:
                    raise ValueError('A planned proposal already has human edits; preserve its current task plan')
            archive = data / 'backups' / ('proposal-plan-' + hashlib.sha256(target.read_bytes()).hexdigest()[:16] + '.json')
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                archive.write_bytes(target.read_bytes())
            superseded = sorted({item['clip_id'] for item in existing['items']} - {item['clip_id'] for item in plan['items']})
            atomic(data / 'hidden-proposals.json', {'schema_version': '1.0', 'asset_id': index['asset_id'],
                                                   'source_fingerprint': index['source_fingerprint'], 'clip_ids': superseded})
            atomic(target, plan)
        else:
            plan = existing
    else:
        atomic(target, plan)
    log = data / 'scene-preparation.json'
    progress = {'items': [], 'starts_training': False, 'human_labels_created': False}
    for position, item in enumerate(items, 1):
        stamp = time.perf_counter()
        clip = request('/clips', {'start_ms': item['start_ms'], 'end_ms': item['end_ms']})
        if clip['id'] != item['clip_id']:
            raise ValueError('Planned clip identity no longer matches the configured audio/source')
        while clip['status'] == 'preparing':
            time.sleep(.5)
            clip = request('/clips/' + item['clip_id'])
        if clip['status'] != 'ready':
            raise ValueError('Scene preparation failed: ' + item['id'] + ': ' + clip.get('reason', ''))
        tracks = clip['tracks']
        if clip['revision'] == 0 and any(track['reviewed'] or track['character_id'] is not None
                                         or any(label != -1 for label in track['labels']) for track in tracks):
            raise ValueError('New scene unexpectedly contains inferred human labels')
        summary = {'id': item['id'], 'clip_id': clip['id'], 'status': 'ready', 'seconds': round(time.perf_counter()-stamp, 3),
                   'tracks': len(tracks), 'track_box_frames': [sum(box is not None for box in track['boxes']) for track in tracks],
                   'frame_count': clip['frame_count'], 'revision': clip['revision']}
        progress['items'].append(summary)
        atomic(log, progress)
        print(json.dumps({'prepared': position, 'total': len(items), **summary}, ensure_ascii=False), flush=True)
    print(json.dumps({'status': 'all_ready', 'count': len(items), 'plan': str(target), 'starts_training': False}), flush=True)


if __name__ == '__main__':
    main()
