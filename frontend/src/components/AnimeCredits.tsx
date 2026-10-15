import { useEffect, useState } from 'react';
import { api, friendlyDate, message, post } from '../api';
import { Alert, Spinner } from './ui';

interface Credit {
  person_id: number; name: string; name_cn: string; position: string;
  episodes: string; scope: 'episode' | 'subject' | 'unparsed'; url: string;
}
interface Credits {
  status: 'missing' | 'unavailable' | 'completed'; episode?: number | null;
  fetched_at?: string; url?: string; rows: Credit[]; warnings: string[];
}

export default function AnimeCredits({ assetId }: { assetId: string }) {
  const [value, setValue] = useState<Credits>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    let current = true;
    setValue(undefined); setError('');
    void api<Credits>(`/assets/${encodeURIComponent(assetId)}/credits`).then((result) => { if (current) setValue(result); }).catch((reason) => { if (current) setError(message(reason)); });
    return () => { current = false; };
  }, [assetId]);
  const fetchCredits = async () => {
    setBusy(true); setError('');
    try { setValue(await post<Credits>(`/assets/${encodeURIComponent(assetId)}/credits`)); }
    catch (reason) { setError(message(reason)); } finally { setBusy(false); }
  };
  const rows = (items: Credit[]) => <div className="credit-table-wrap"><table className="credits-table"><thead><tr><th>职位</th><th>人员</th><th>来源参与集数</th></tr></thead><tbody>{items.map((item) => <tr key={`${item.person_id}:${item.position}:${item.episodes}`}><td>{item.position}</td><td><a href={item.url} target="_blank" rel="noreferrer">{item.name_cn || item.name}</a></td><td>{item.episodes || '未注明'}</td></tr>)}</tbody></table></div>;
  const episode = value?.rows?.filter((row) => row.scope === 'episode') || [];
  const subject = value?.rows?.filter((row) => row.scope === 'subject') || [];
  const unparsed = value?.rows?.filter((row) => row.scope === 'unparsed') || [];
  return <section className="asset-anime-credits" aria-label="演职员资料"><div className="section-row"><div><h3>演职员资料</h3><p className="helper">从 Bangumi 读取并按参与集数对应，不调用画面理解模型。</p></div><button type="button" className="button secondary small" disabled={busy || value?.status === 'unavailable'} onClick={() => void fetchCredits()}>{busy ? <Spinner label="正在读取演职员资料" /> : value?.status === 'completed' ? '更新演职员资料' : '读取演职员资料'}</button></div>
    {error && <Alert tone="error">{error}</Alert>}
    {value?.status === 'completed' && <><p className="helper">{value.fetched_at && `读取于 ${friendlyDate(value.fetched_at)} · `}<a href={value.url} target="_blank" rel="noreferrer">Bangumi 来源</a></p>
      <details open><summary>本集演职员 · {episode.length} 条</summary>{episode.length ? rows(episode) : <p className="helper">来源尚无可明确对应本集的演职员记录。</p>}</details>
      {subject.length > 0 && <details><summary>作品级人员，来源未注明参与集数 · {subject.length} 条</summary>{rows(subject)}</details>}
      {unparsed.length > 0 && <details><summary>参与范围待核对 · {unparsed.length} 条</summary>{rows(unparsed)}</details>}
      {value.warnings?.map((warning) => <Alert key={warning}>{warning}</Alert>)}
    </>}
  </section>;
}
