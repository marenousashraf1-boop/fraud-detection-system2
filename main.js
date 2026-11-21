// Frontend logic: upload a CSV, call the Flask API (/predict), render the results.
const API = location.protocol === 'file:' ? 'http://localhost:5000' : '';   // works when index.html is opened directly
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = n => Number(n).toLocaleString('en-US', {maximumFractionDigits: 0});
let file = null;

function setFile(f) {
  file = f;
  $('fname').textContent = f ? `${f.name} (${(f.size / 1048576).toFixed(2)} MB)` : 'No file selected';
  $('go').disabled = !f;
}
function showMsg(text, info) {
  const m = $('msg'); m.textContent = text || ''; m.hidden = !text; m.classList.toggle('info', !!info);
}

const drop = $('drop');
drop.addEventListener('click', () => $('file').click());
drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); } });
$('file').addEventListener('change', e => { setFile(e.target.files[0] || null); showMsg(''); });
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { setFile(e.dataTransfer.files[0] || null); showMsg(''); });
$('thr').addEventListener('input', e => { $('thrv').textContent = Number(e.target.value).toFixed(2); });
$('go').addEventListener('click', analyze);
$('sample').addEventListener('click', async () => {
  try {
    const r = await fetch(API + '/samples/sample_3_messy_names.csv');
    if (!r.ok) throw new Error('Sample file not available.');
    setFile(new File([await r.blob()], 'sample_3_messy_names.csv', {type: 'text/csv'}));
    analyze();
  } catch (e) { showMsg(e.message || 'Could not load the sample. Is the server running?'); }
});

async function analyze() {
  if (!file) return;
  const btn = $('go'); btn.disabled = true; btn.textContent = 'Analyzing…';
  showMsg('Analyzing your file…', true); $('results').hidden = true;
  const fd = new FormData(); fd.append('file', file); fd.append('threshold', $('thr').value);
  try {
    const r = await fetch(API + '/predict', {method: 'POST', body: fd});
    let data = null;
    try { data = await r.json(); } catch (_) { /* non-JSON error page */ }
    if (!r.ok) throw new Error((data && data.error) || `Server error (${r.status}).`);
    showMsg(''); render(data);
  } catch (e) {
    showMsg(e instanceof TypeError ? 'Cannot reach the server. Make sure the backend is running.' : e.message);
  } finally { btn.disabled = false; btn.textContent = 'Analyze'; }
}

const stat = (n, l, cls) => `<div class="stat"><div class="n ${cls || ''}">${n}</div><div class="l">${l}</div></div>`;
const table = (head, rows) => `<table><tr>${head.map(h => `<th>${esc(h)}</th>`).join('')}</tr>` +
  rows.map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join('')}</tr>`).join('') + '</table>';

function render(d) {
  const s = d.summary;
  $('rfile').textContent = '— ' + d.filename;
  $('dl').href = `${API}/download/${d.download_id}`;
  $('stats').innerHTML = stat(num(s.total_transactions), 'Total transactions') +
    stat(num(s.suspicious_count), 'Suspicious', 'bad') + stat(s.suspicious_pct + '%', 'Suspicious rate') +
    stat(num(s.suspicious_amount), `Suspicious amount (of ${num(s.total_amount)})`);
  $('modelline').textContent = `Model: ${s.model_name} · threshold ${s.threshold}`;

  $('warnbox').hidden = !d.warnings.length;
  $('warns').innerHTML = d.warnings.map(w => `<li>${esc(w)}</li>`).join('');

  const ev = s.evaluation; $('evalbox').hidden = !ev;
  if (ev) $('evalstats').innerHTML = stat(ev.precision, 'Precision') + stat(ev.recall, 'Recall') + stat(ev.f1, 'F1-score') +
    stat(`TP ${ev.tp} · FP ${ev.fp}<br>FN ${ev.fn} · TN ${ev.tn}`, 'Confusion matrix');

  const tot = d.hist.reduce((a, b) => a + b, 0) || 1, max = Math.log10(Math.max(...d.hist) + 1) || 1;
  const thrBucket = Math.min(9, Math.floor(s.threshold * 10));
  $('hist').innerHTML = d.hist.map((c, i) => `<div class="${i >= thrBucket ? 'hot' : ''}" style="height:${Math.max(2, 100 * Math.log10(c + 1) / max)}%"><span>${(100 * c / tot).toFixed(1)}%</span></div>`).join('');
  $('hlab').innerHTML = d.hist.map((_, i) => `<span>${(i / 10).toFixed(1)}</span>`).join('');
  $('bytype').innerHTML = s.by_type.map(t => `<div class="bt"><div class="h"><span>${esc(t.type)}</span><span>${num(t.suspicious)} / ${num(t.total)} (${t.pct}%)</span></div><div class="bar"><i style="width:${Math.min(100, t.pct)}%"></i></div></div>`).join('');

  $('top').innerHTML = d.top.rows.length ? table(d.top.columns, d.top.rows) : '<p class="ok">No suspicious transactions at this threshold.</p>';
  $('map').innerHTML = table(['Standard field', 'Your column', 'Confidence', 'Method'], d.mapping.map(m => [m.field, m.column, m.score + '%', m.method]));
  $('unmapped').textContent = d.unmapped.length ? 'Ignored columns: ' + d.unmapped.join(', ') : '';
  const r = d.report;
  $('repsum').textContent = `Data understanding report (${num(r.rows)} rows × ${r.columns} columns, ${num(r.total_missing_cells)} missing cells, ${num(r.duplicate_rows)} duplicate rows)`;
  $('rep').innerHTML = table(['Column', 'Kind', 'Missing', 'Unique'], r.column_info.map(c => [c.column, c.kind, `${c.missing} (${c.missing_pct}%)`, c.unique]));
  $('results').hidden = false; $('results').scrollIntoView({behavior: 'smooth', block: 'start'});
}
