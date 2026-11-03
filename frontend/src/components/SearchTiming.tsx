import { searchPhaseLabel, type SearchClientTiming, type SearchPresentationTiming, type SearchTiming } from '../searchTiming';

const ms = (value?: number | null) => value == null ? '未测得' : `${value.toFixed(1)} ms`;

export default function SearchTimingDetails({ server, client, presentation }: {
  server?: SearchTiming; client?: SearchClientTiming; presentation?: SearchPresentationTiming;
}) {
  if (!server && !client) return null;
  const record = { schema: 'search-client-timing-v1', server, client, presentation,
    boundaries: { click: 'pointer click event timestamp (or click handler entry fallback); excludes pointer-down to release; keyboard/programmatic submit has no click measurement',
      submit: 'form submit handler entry', results: 'React result DOM committed',
      paint: 'second requestAnimationFrame opportunity; excludes image loading and is not proof of pixels displayed',
      remainder: 'request-to-JSON duration minus application middleware duration; includes transport, unmeasured ingress, body transfer, parsing and scheduling',
      parallel: 'parent spans contain children; parallel child durations overlap and must not be added' } };
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url;
    link.download = `search-timing-${server?.request_id || 'client'}.json`; link.click(); URL.revokeObjectURL(url);
  };
  return <details data-search-request-id={server?.request_id}>
    <summary>搜索总耗时与阶段记录{presentation && ` · ${presentation.trigger === 'pointer_click' ? '点击起' : '提交起'} ${ms(presentation.click_to_result_commit_ms ?? presentation.submit_to_result_commit_ms)}`}</summary>
    <p className="helper">{presentation?.trigger === 'pointer_click' ? `本次从搜索按钮的${presentation.click_time_source === 'event_timestamp' ? '点击事件时间戳' : '点击处理函数入口'}计时。` : '本次从表单提交计时，未记录指针点击。'}结果文字可用不含图片加载；当前没有输入防抖。父阶段包含子阶段，并行召回会重叠，请勿把下列时长直接相加。</p>
    <table><thead><tr><th>测量边界</th><th>时长</th></tr></thead><tbody>
      {presentation?.trigger === 'pointer_click' && <>
        <tr><td>点击到结果 DOM 可用</td><td>{ms(presentation.click_to_result_commit_ms)}</td></tr>
        <tr><td>点击到表单提交</td><td>{ms(presentation.click_to_submit_ms)}</td></tr>
        <tr><td>点击到绘制机会</td><td>{ms(presentation.click_to_paint_opportunity_ms)}</td></tr>
      </>}
      <tr><td>提交到结果 DOM 可用</td><td>{ms(presentation?.submit_to_result_commit_ms)}</td></tr>
      <tr><td>提交到绘制机会</td><td>{ms(presentation?.submit_to_paint_opportunity_ms)}</td></tr>
      <tr><td>前端准备请求</td><td>{ms(presentation?.submit_to_fetch_ms)}</td></tr>
      <tr><td>请求发起到响应头</td><td>{ms(client?.fetch_to_headers_ms)}</td></tr>
      <tr><td>响应头到 JSON 已解析</td><td>{ms(client?.headers_to_json_ms)}</td></tr>
      <tr><td>后端应用处理（含序列化）</td><td>{ms(server?.total_ms)}</td></tr>
      <tr><td>请求到 JSON 的剩余项（传输、未测入站等待、解析及调度）</td><td>{ms(client?.client_server_remainder_ms)}</td></tr>
      <tr><td>JSON 已解析到结果 DOM 可用</td><td>{ms(presentation?.json_to_result_commit_ms)}</td></tr>
    </tbody></table>
    {client?.resource && <p className="helper">浏览器资源记录：请求前准备 {ms(client.resource.browser_before_request_ms)}；发出请求到首字节 {ms(client.resource.request_to_first_byte_ms)}（包含服务处理）；响应体传输 {ms(client.resource.body_transfer_ms)}。</p>}
    {server && <><p className="helper">请求 ID：{server.request_id} · 后端从应用中间件入口测到响应头就绪；操作系统／服务器入站排队未单独测量。</p>
      <table><thead><tr><th>阶段（ID / 父 ID）</th><th>开始 → 结束</th><th>时长</th><th>实际状态</th></tr></thead><tbody>
        {server.spans.map((span) => <tr key={span.id}><td>{searchPhaseLabel[span.name] || span.name} ({span.id} / {span.parent_id ?? '—'})</td><td>{ms(span.start_ms)} → {ms(span.end_ms)}</td><td>{ms(span.elapsed_ms)}</td><td>{JSON.stringify(span.attributes)}</td></tr>)}
      </tbody></table></>}
    <button type="button" className="text-button" onClick={download}>下载本次耗时记录 JSON</button>
    <pre className="json-report" aria-label="搜索耗时机器记录">{JSON.stringify(record, null, 2)}</pre>
  </details>;
}
