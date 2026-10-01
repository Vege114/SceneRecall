import type { SubtitleTrack, SubtitleTracks } from '../types';
import { Alert, Field } from './ui';

export function subtitleTrackLabel(track: SubtitleTrack): string {
  return [`轨道 ${track.index}`, track.language || '语言未知', track.title, track.codec, track.default && '默认', track.forced && '强制', !track.supported && (track.reason || '暂不支持提取')].filter(Boolean).join(' · ');
}

export default function SubtitleTrackSelect({ result, value, onChange, allowAuto = false, disabled = false }: { result: SubtitleTracks; value: number | null; onChange: (value: number | null) => void; allowAuto?: boolean; disabled?: boolean }) {
  if (!result.tracks.length) return <Alert>{allowAuto ? '未发现容器字幕轨。可先导入作品；若字幕烧录在画面中，之后配置模型进行 OCR，或改用外挂字幕。' : '原视频未发现容器字幕轨，无法提取。若字幕烧录在画面中，可配置模型进行 OCR。'}</Alert>;
  const supported = result.tracks.some((track) => track.supported);
  const recommended = result.tracks.find((track) => track.index === result.recommended_stream_index && track.supported);
  return <>
    <Field label="视频内的字幕轨" hint="文本轨在本地提取，保留台词与时间轴，无需视觉模型。">
      <select value={value ?? ''} onChange={(event) => onChange(event.target.value === '' ? null : Number(event.target.value))} disabled={disabled || !supported} required={!allowAuto}>
        {allowAuto ? <option value="">{recommended ? `自动选择：${subtitleTrackLabel(recommended)}` : '自动选择文本字幕轨'}</option> : <option value="" disabled>选择文本字幕轨</option>}
        {result.tracks.map((track) => <option key={track.index} value={track.index} disabled={!track.supported}>{subtitleTrackLabel(track)}</option>)}
      </select>
    </Field>
    {!supported && <Alert>当前字幕轨不支持文本提取。图像字幕轨不能直接作为画面 OCR 的来源，请使用外挂字幕；仅当字幕确实烧录在视频画面中时选择画面字幕 OCR。</Alert>}
  </>;
}
