import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Subtitles } from 'lucide-react';
import { api, message, post } from '../api';
import { useApp } from '../context';
import type { Asset, SubtitleTracks } from '../types';
import { Alert, Field, Modal, Spinner } from './ui';
import SubtitleTrackSelect from './SubtitleTrackSelect';

export default function ExtractSubtitlesModal({ asset, onClose, onSaved }: { asset: Asset; onClose: () => void; onSaved: () => Promise<void> }) {
  const { notify } = useApp();
  const [tracks, setTracks] = useState<SubtitleTracks | null>(null);
  const [streamIndex, setStreamIndex] = useState<number | null>(null);
  const [offset, setOffset] = useState(String((asset.subtitle_offset_ms || 0) / 1000));
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const currentStream = useRef(asset.subtitle_stream_index);
  currentStream.current = asset.subtitle_stream_index;
  useEffect(() => {
    let active = true;
    setLoading(true); setError(''); setTracks(null); setStreamIndex(null);
    void api<SubtitleTracks>(`/assets/${encodeURIComponent(asset.id)}/subtitle-tracks`).then((result) => {
      if (!active) return;
      setTracks(result);
      const current = result.tracks.find((track) => track.supported && track.index === currentStream.current);
      setStreamIndex(current?.index ?? result.recommended_stream_index ?? result.tracks.find((track) => track.supported)?.index ?? null);
    }).catch((error) => { if (active) setError(message(error)); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [asset.id, attempt]);
  const selected = tracks?.tracks.some((track) => track.index === streamIndex && track.supported);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!selected || busy) return;
    setBusy(true); setError('');
    try {
      await post(`/assets/${encodeURIComponent(asset.id)}/subtitles/extract`, { subtitle_stream_index: streamIndex, subtitle_offset_ms: Math.round(Number(offset) * 1000) });
      await onSaved(); notify('字幕轨已在本地提取，可直接检索台词'); onClose();
    } catch (error) { setError(message(error)); } finally { setBusy(false); }
  };
  return <Modal title="提取 / 切换字幕轨" subtitle={asset.title} onClose={onClose} wide><form onSubmit={submit}>
    <div className="modal-body">
      <p className="character-modal-copy">从原视频提取文本字幕，无需模型。提取后更新当前台词记录与检索索引。</p>
      {loading ? <Spinner label="正在探测字幕轨" /> : tracks ? <SubtitleTrackSelect result={tracks} value={streamIndex} onChange={setStreamIndex} disabled={busy} /> : <button type="button" className="button secondary" onClick={() => setAttempt((value) => value + 1)}>重新探测字幕轨</button>}
      {selected && <Field label="时间偏移（秒）" hint="相对原始字幕时间轴，正值表示延后"><input type="number" step="0.001" required value={offset} disabled={busy} onChange={(event) => setOffset(event.target.value)} /></Field>}
      {error && <Alert tone="error">{error}</Alert>}
    </div>
    <footer className="modal-footer"><span>本地提取，不发送字幕图片</span><button type="submit" className="button primary" disabled={loading || busy || !selected}>{busy ? <Spinner label="正在提取字幕" /> : <><Subtitles size={17} />提取并更新台词</>}</button></footer>
  </form></Modal>;
}
