export interface SearchSpan {
  id: number;
  parent_id: number | null;
  name: string;
  start_ms: number;
  end_ms: number;
  elapsed_ms: number;
  status: 'ok' | 'error';
  attributes: Record<string, string | number | boolean | null>;
}

export interface SearchTiming {
  schema: 'search-timing-v1';
  request_id: string;
  total_ms: number;
  boundary: string;
  attributes: Record<string, string | number | boolean | null>;
  spans: SearchSpan[];
}

export interface SearchClientTiming {
  fetch_started_at: number;
  json_parsed_at: number;
  fetch_to_headers_ms: number;
  headers_to_json_ms: number;
  request_to_json_ms: number;
  client_server_remainder_ms: number | null;
  resource: null | {
    browser_before_request_ms: number;
    request_to_first_byte_ms: number;
    body_transfer_ms: number;
  };
}

export interface SearchPresentationTiming {
  trigger: 'pointer_click' | 'form_submit';
  click_time_source?: 'event_timestamp' | 'click_handler';
  click_to_submit_ms?: number;
  click_to_result_commit_ms?: number;
  click_to_paint_opportunity_ms?: number;
  submit_to_result_commit_ms: number;
  submit_to_paint_opportunity_ms?: number;
  submit_to_fetch_ms: number;
  json_to_result_commit_ms: number;
  debounce_ms: 0;
}

export function searchClickStart(timeStamp: number, handlerStartedAt = performance.now(), timeOrigin = performance.timeOrigin) {
  // Modern browsers use the performance time origin. Older event timestamps
  // may be epoch-based; never mix either clock with performance.now().
  const relative = timeStamp > handlerStartedAt ? timeStamp - timeOrigin : timeStamp;
  const valid = Number.isFinite(relative) && relative > 0 && relative <= handlerStartedAt;
  return { clickedAt: valid ? relative : handlerStartedAt,
    source: valid ? 'event_timestamp' as const : 'click_handler' as const };
}

export function searchTransportTiming(response: Response, started: number, headers: number, parsed: number) {
  let server: SearchTiming | undefined;
  try {
    const candidate = JSON.parse(response.headers.get('X-Search-Timing') || 'null');
    if (candidate?.schema === 'search-timing-v1' && typeof candidate.request_id === 'string' &&
        Number.isFinite(candidate.total_ms) && Array.isArray(candidate.spans)) server = candidate;
  } catch { /* Older servers and malformed optional diagnostics cannot fail search. */ }
  const entries = typeof performance.getEntriesByName === 'function'
    ? performance.getEntriesByName(new URL('/api/search', window.location.href).href, 'resource') as PerformanceResourceTiming[] : [];
  const resource = entries.filter((entry) => entry.startTime >= started - 1 && entry.responseEnd > 0).at(-1);
  const client: SearchClientTiming = {
    fetch_started_at: started, json_parsed_at: parsed, fetch_to_headers_ms: headers - started,
    headers_to_json_ms: parsed - headers, request_to_json_ms: parsed - started,
    client_server_remainder_ms: server ? parsed - started - server.total_ms : null,
    resource: resource && resource.requestStart > 0 && resource.responseStart > 0 ? {
      browser_before_request_ms: resource.requestStart - started,
      request_to_first_byte_ms: resource.responseStart - resource.requestStart,
      body_transfer_ms: resource.responseEnd - resource.responseStart,
    } : null,
  };
  return { search_timing: server, client_timing: client };
}

export const searchPhaseLabel: Record<string, string> = {
  'http.application': '后端应用总时长', 'http.handler': '搜索接口处理',
  'search.engine': '搜索引擎', 'query.interpret': '可选查询理解',
  'query.character_names': '角色简称与别名解析',
  'query.character_names.revision': '核对角色名录版本',
  'snapshot.acquire': '取得索引与资料版本', 'snapshot.revision.work': '读取资料版本（线程执行）',
  'snapshot.validate': '检查资料版本', 'recall.parallel': '并行召回墙钟时长',
  'recall.lexical': '字面召回', 'recall.semantic': '文字语义召回',
  'recall.visual': '直接画面召回', 'recall.concept': '作品资料召回',
  'lexical.sql_tokenize.work': '字面 SQL 与分词（线程执行）',
  'semantic.snapshot.work': '语义索引元数据读取', 'semantic.query_encode': '文字查询编码',
  'semantic.index_read_rank.work': '语义索引读取与排序',
  'visual.model_identity': '核对图文模型身份', 'encoder.identity.work': '模型文件身份与依赖核对',
  'visual.query_cache_lookup': '查询向量缓存读取', 'visual.query_encode': '图文查询编码',
  'encoder.request.work': '编码器请求（线程执行）', 'encoder.prepare_and_wait': '编码器准备与资源等待（未拆分）',
  'encoder.worker_start': '启动或复用编码进程',
  'encoder.worker_exchange': '编码进程往返（含本次模型加载与推理）',
  'visual.rank': '画面索引召回（含线程调度）', 'visual.rank.work': '画面索引召回（线程执行）',
  'visual.matrix_cache': '读取或构建矩阵缓存', 'visual.matrix_build.work': '从索引构建矩阵',
  'visual.matrix_score': '矩阵相似度计算', 'visual.top_results': '画面候选排序与读取',
  'visual.streaming_rank.work': '内存受限时流式索引召回',
  'results.merge_rank': '合并各路召回与排序', 'results.provider_rerank': '可选模型重排',
  'results.deduplicate_select': '去重与选择结果', 'results.compose': '可选结果解释',
  'canonical.source_records': '读取最新来源证据（含线程调度）',
  'canonical.source_records.work': '读取最新来源证据（线程执行）',
  'canonical.enrich': '核对、补齐展示结果', 'visual_verification': '可选视觉复核', 'response.json': '序列化 JSON 响应',
};
