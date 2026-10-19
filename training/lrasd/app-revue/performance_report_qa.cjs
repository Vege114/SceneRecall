/* Offline DOM/interaction QA only. Does not launch a browser or server. */
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const { JSDOM, ResourceLoader, VirtualConsole } = require('../../../frontend/node_modules/jsdom');
const here = __dirname;
const html = fs.readFileSync(path.join(here, 'performance_report.html'), 'utf8');
const errors = [], network = [], checks = [];
class NoNetwork extends ResourceLoader { fetch(url) { network.push(url); return null; } }
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', error => errors.push(error.message));
const dom = new JSDOM(html, { runScripts: 'dangerously', resources: new NoNetwork(), virtualConsole,
  beforeParse(window) {
    window.fetch = () => { throw new Error('Network forbidden in offline QA'); };
    Object.defineProperty(window, 'innerWidth', { value: 320, configurable: true });
  } });
const document = dom.window.document;
const data = JSON.parse(document.getElementById('report-data').textContent);
function check(name, action) { action(); checks.push(name); }
check('inline JavaScript executes without errors', () => assert.deepEqual(errors, []));
check('only episode1 completes the target; excluded history is cancelled', () => {
  assert.equal(data.all_complete, true); assert.equal(data.complete_count, 1); assert.equal(data.target_count, 1);
  assert.equal(data.requests, 583); assert.equal(data.excluded_requests, 15);
  assert.deepEqual(data.episodes.map(ep => ep.status), ['completed', 'cancelled', 'cancelled']);
  assert.match(document.getElementById('report-status').textContent, /目标已达成/);
  assert.equal(document.getElementById('report-status').classList.contains('done'), true);
});
const sectionCounts = {};
check('all sections and upstream timing limits render', () => {
  for (const id of ['overview', 'upstream', 'timeline', 'steps', 'stalls', 'optimization', 'outcomes', 'sources']) {
    const section = document.getElementById(id);
    assert.ok(section && section.textContent.trim().length > 40);
    sectionCounts[id] = { textCharacters: section.textContent.length, childElements: section.querySelectorAll('*').length };
  }
  assert.ok(document.getElementById('upstream').textContent.includes('字幕提取'));
});
check('default target view renders nine steps and one episode', () => {
  assert.equal(document.querySelectorAll('.episode-table').length, 1);
  assert.equal(document.querySelectorAll('.steps-table tbody tr').length, 9);
  assert.equal(document.querySelectorAll('.episode-heading').length, 1);
  assert.ok(document.querySelectorAll('.timeline-row').length >= 6);
});
check('three timeout repairs and final completion are visible', () => {
  const text = document.getElementById('events').textContent;
  for (const value of ['112/223', '125/223', '205/223', '第一集本次处理目标全部完成']) assert.ok(text.includes(value));
  assert.equal((text.match(/该单元已恢复/g) || []).length, 3);
  assert.ok(text.includes('失败调用链计数 2 次'));
});
check('excluded and all-episode controls preserve target counts', () => {
  document.querySelector('[data-scope="all"]').click();
  assert.equal(document.querySelectorAll('.episode-table').length, 3);
  document.querySelector('[data-scope="2"]').click();
  assert.equal(document.querySelectorAll('.episode-table').length, 1);
  assert.match(document.querySelector('.episode-table summary').textContent, /第 2 集.*已取消/);
  document.querySelector('[data-scope="target"]').click();
  assert.equal(document.querySelectorAll('.steps-table tbody tr').length, 9);
});
check('source search, empty state and pagination work', () => {
  const input = document.getElementById('source-search');
  input.value = 'jobs.sqlite'; input.dispatchEvent(new dom.window.Event('input'));
  assert.ok(document.querySelectorAll('.source-entry').length > 0);
  assert.ok([...document.querySelectorAll('.source-entry')].every(e => e.textContent.includes('jobs.sqlite')));
  input.value = 'no-such-report-source-xyz'; input.dispatchEvent(new dom.window.Event('input'));
  assert.equal(document.querySelectorAll('.source-entry').length, 0);
  input.value = ''; input.dispatchEvent(new dom.window.Event('input'));
  assert.equal(document.querySelectorAll('.source-entry').length, 20);
  document.getElementById('more-sources').click();
  assert.equal(document.querySelectorAll('.source-entry').length, 40);
});
check('evidence dialog opens by click and closes', () => {
  document.querySelector('#events [data-detail]').click();
  assert.equal(document.getElementById('evidence-dialog').hasAttribute('open'), true);
  assert.ok(document.querySelector('#dialog-body pre').textContent.includes('source'));
  document.getElementById('close-dialog').click();
  assert.equal(document.getElementById('evidence-dialog').hasAttribute('open'), false);
});
check('ASD, dossier, voice and visual QA limits remain explicit', () => {
  const text = document.getElementById('outcomes').textContent;
  assert.ok(text.includes('2 个得分窗口') && text.includes('1 个无轨迹'));
  assert.ok(text.includes('部分完成') && text.includes('不可用'));
  assert.ok(text.includes('未做真实浏览器排版') && text.includes('缺其中 1 份完整档案'));
  assert.ok(document.getElementById('optimization').textContent.includes('身份页分区加载与超时提示'));
});
check('offline resources and 320px CSS are statically bounded', () => {
  assert.deepEqual(network, []);
  assert.equal(document.querySelectorAll('script[src],link[href],img[src],iframe[src]').length, 0);
  assert.match(html, /connect-src 'none'/);
  assert.match(html, /name="viewport" content="width=device-width,initial-scale=1"/);
  assert.match(html, /@media\(max-width:700px\)/);
  assert.match(html, /\.steps-table tr\{display:grid;grid-template-columns:1fr 1fr/);
  assert.match(html, /\.timeline-viewport\{overflow:auto/);
  assert.match(html, /overflow-wrap:anywhere/);
});
const result = { kind: 'offline-jsdom-and-static-css-qa', passed: checks.length, checks, sectionCounts,
  sourceCount: data.sources.length, dataFinal: data.all_complete, externalRequests: network.length,
  browserUsed: false, serverStarted: false, visualLayoutInspected: false,
  limitations: ['DOM interaction and static CSS assertions do not prove real browser layout or 320px pixel rendering.'] };
fs.writeFileSync(path.join(here, 'performance_report_qa.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
dom.window.close();
