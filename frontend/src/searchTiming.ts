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
  operations?: Record<string, SearchOperationTiming>;
}

export interface SearchOperationTiming {
  count: number;
  elapsed_ms: number;
  identity_calls?: number;
  unique_count?: number;
  repeated_count?: number;
  identity_limit_reached?: boolean;
  [key: string]: number | boolean | undefined;
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
  snapshot: '资料版本', knowledge: '作品资料', character: '角色名录',
  canonical: '来源核对', records: '来源记录重建', identity: '角色身份核对', retrieval: '检索窗口证据',
  'http.application': '后端应用总时长', 'http.handler': '搜索接口处理',
  'search.engine': '搜索引擎', 'query.interpret': '可选查询理解',
  'query.character_names': '角色简称与别名解析',
  'query.character_names.revision': '核对角色名录版本',
  'snapshot.acquire': '取得索引与资料版本', 'snapshot.revision.work': '读取资料版本（线程执行）',
  'snapshot.revision.sqlite': '读取 SQLite 索引版本',
  'snapshot.revision.knowledge': '核对作品资料版本', 'snapshot.revision.encoder': '核对编码器文件版本',
  'snapshot.validate': '检查资料版本', 'recall.parallel': '并行召回墙钟时长',
  'knowledge.scopes': '确定作品资料范围', 'knowledge.projection': '重建已审核作品资料投影',
  'knowledge.revision.runtime': '读取知识运行时版本', 'knowledge.revision.identity': '核对角色身份资料版本',
  'knowledge.revision.baseline': '核对基础作品资料版本', 'knowledge.baseline.snapshot': '取得基础作品资料快照',
  'character.snapshot': '取得角色名录快照',
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
  'canonical.owner_lookup': '查找候选来源素材', 'canonical.asset_records': '重建单个素材的来源记录',
  'records.asset_metadata': '读取素材元信息', 'records.annotations': '读取人工注释',
  'records.observations': '读取画面观察', 'records.scenes': '读取场景记录',
  'records.observation_filter': '筛选有效画面观察', 'records.audio': '读取音频记录',
  'records.retrieval': '读取检索窗口记录', 'records.subtitles': '读取字幕记录',
  'records.assemble': '组装来源记录', 'records.work_projection': '补充作品资料字段',
  'records.subtitle_compact': '合并字幕记录', 'retrieval.load_validate': '读取并核验检索窗口证据',
  'identity.decorate': '补充角色身份字段', 'identity.resolver_init': '初始化身份解析资料',
  'identity.resolve_batch': '批量解析角色身份',
  'canonical.enrich': '核对、补齐展示结果', 'visual_verification': '可选视觉复核', 'response.json': '序列化 JSON 响应',
};

// New nested phases remain identifiable before a dedicated display label is added.
export function searchPhaseDisplayName(name: string) {
  if (searchPhaseLabel[name]) return searchPhaseLabel[name];
  const prefix = Object.keys(searchPhaseLabel)
    .filter((key) => name.startsWith(`${key}.`))
    .sort((left, right) => right.length - left.length)[0];
  return prefix ? `${searchPhaseLabel[prefix]} · ${name.slice(prefix.length + 1)}` : name;
}

export const searchOperationLabel: Record<string, string> = {
  fs_exists: '检查文件是否存在', fs_stat: '读取文件元信息', file_stat: '读取文件元信息', file_glob: '匹配目录文件',
  file_read_text: '读取文本（含解码）', file_read_bytes: '读取文件字节',
  json_parse: '解析 JSON', digest_serialize: '哈希前序列化（含 UTF-8 编码）', digest_sha256: '计算 SHA-256',
  source_read: '读取知识来源字节', source_sha256: '计算知识来源哈希', source_decode: '解码知识来源文本',
  projection_sha256: '计算作品资料投影哈希', reference_sha256: '计算角色参考图哈希',
  scope_view: '读取作品范围', knowledge_cache_load: '载入作品资料缓存',
  knowledge_cache_transform: '转换作品资料缓存', knowledge_cache_size: '估算作品资料缓存大小',
  character_profile_load: '载入角色档案', character_names_build: '构建角色姓名表',
  record_copy: '复制记录', identity_merge: '合并身份字段',
  lock_wait: '等待锁', library_lock_wait: '等待素材库锁',
  knowledge_cache_lock_wait: '等待作品资料缓存锁', character_cache_lock_wait: '等待角色缓存锁',
};
