import * as duckdb from '@duckdb/duckdb-wasm';
import duckdbWasmMvp from '@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url';
import mvpWorker from '@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url';
import duckdbWasmEh from '@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url';
import ehWorker from '@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url';
import './style.css';

const DATA_ROOT = new URL('data/mentions/', window.location.href).toString();
const PAGE_SIZE = 25;

interface Manifest {
  n_mentions: number;
  n_papers: number;
  years: number[];
  by_field: { value: string | null; n: number }[];
  by_manufacturer: { value: string | null; n: number }[];
  by_usage_context: { value: string | null; n: number }[];
}

interface State {
  yearMin: number;
  yearMax: number;
  field: string;
  manufacturer: string;
  usage: string;
  text: string;
  page: number;
}

let db: duckdb.AsyncDuckDB;
let manifest: Manifest;
const state: State = { yearMin: 0, yearMax: 0, field: '', manufacturer: '', usage: '', text: '', page: 0 };

async function initDuckDB(): Promise<duckdb.AsyncDuckDB> {
  const bundles: duckdb.DuckDBBundles = {
    mvp: { mainModule: duckdbWasmMvp, mainWorker: mvpWorker },
    eh: { mainModule: duckdbWasmEh, mainWorker: ehWorker },
  };
  const bundle = await duckdb.selectBundle(bundles);
  const worker = new Worker(bundle.mainWorker!);
  const logger = new duckdb.ConsoleLogger(duckdb.LogLevel.WARNING);
  const instance = new duckdb.AsyncDuckDB(logger, worker);
  await instance.instantiate(bundle.mainModule, bundle.pthreadWorker);
  return instance;
}

function esc(s: string): string {
  return s.replace(/'/g, "''");
}

function setLoading(loading: boolean): void {
  (document.getElementById('spinner') as HTMLElement).hidden = !loading;
}

// Only the year files the current filter actually needs are ever fetched --
// selecting a narrow year range keeps this to a handful of megabytes instead
// of the whole ~35MB bundle.
function filesForYears(min: number, max: number): string[] {
  return manifest.years.filter((y) => y >= min && y <= max)
    .map((y) => `${DATA_ROOT}year=${y}/part-0.parquet`);
}

function whereClause(): string {
  const parts: string[] = [];
  if (state.field) parts.push(`field = '${esc(state.field)}'`);
  if (state.manufacturer) parts.push(`manufacturer = '${esc(state.manufacturer)}'`);
  if (state.usage) parts.push(`usage_context = '${esc(state.usage)}'`);
  if (state.text.trim()) {
    const t = esc(state.text.trim().toLowerCase());
    parts.push(`(lower(title) LIKE '%${t}%' OR lower(evidence_quote) LIKE '%${t}%')`);
  }
  return parts.length ? `WHERE ${parts.join(' AND ')}` : '';
}

async function runQuery(): Promise<void> {
  const statusEl = document.getElementById('status')!;
  const bodyEl = document.getElementById('results-body')!;
  setLoading(true);
  statusEl.textContent = '';

  const files = filesForYears(state.yearMin, state.yearMax);
  if (!files.length) {
    bodyEl.innerHTML = '';
    setLoading(false);
    statusEl.textContent = 'No years selected.';
    return;
  }
  const fileList = files.map((f) => `'${f}'`).join(', ');
  const from = `read_parquet([${fileList}])`;
  const where = whereClause();

  const conn = await db.connect();
  try {
    const countRes = await conn.query(`SELECT count(*) AS n FROM ${from} ${where}`);
    const total = Number(countRes.toArray()[0].n);

    const rowsRes = await conn.query(`
      SELECT year, title, arxiv_id, display, model_raw, manufacturer, usage_context, evidence_quote
      FROM ${from} ${where}
      ORDER BY year DESC
      LIMIT ${PAGE_SIZE} OFFSET ${state.page * PAGE_SIZE}
    `);
    const rows = rowsRes.toArray();

    bodyEl.innerHTML = rows.map((r) => {
      const modelName = (r.display ?? r.model_raw ?? '?') as string;
      const title = (r.title ?? '(no title)') as string;
      const arxivId = r.arxiv_id as string;
      return `<tr>
        <td>${r.year}</td>
        <td><a href="https://arxiv.org/abs/${arxivId}" target="_blank" rel="noopener">${escapeHtml(title)}</a></td>
        <td>${escapeHtml(modelName)}</td>
        <td>${escapeHtml((r.manufacturer ?? '') as string)}</td>
        <td>${escapeHtml((r.usage_context ?? '') as string)}</td>
        <td class="quote">&ldquo;${escapeHtml((r.evidence_quote ?? '') as string)}&rdquo;</td>
      </tr>`;
    }).join('');

    const from1 = total === 0 ? 0 : state.page * PAGE_SIZE + 1;
    const to = Math.min(total, (state.page + 1) * PAGE_SIZE);
    setLoading(false);
    statusEl.textContent = `${total.toLocaleString()} matching mentions — showing ${from1}-${to}`;
    (document.getElementById('page-label') as HTMLElement).textContent =
      `page ${state.page + 1} of ${Math.max(1, Math.ceil(total / PAGE_SIZE))}`;
    (document.getElementById('prev-page') as HTMLButtonElement).disabled = state.page === 0;
    (document.getElementById('next-page') as HTMLButtonElement).disabled = to >= total;
  } catch (err) {
    console.error(err);
    setLoading(false);
    statusEl.textContent = 'Query failed. Try a narrower filter or reload the page.';
  } finally {
    await conn.close();
  }
}

function escapeHtml(s: string): string {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function populateSelect(id: string, values: { value: string | null; n: number }[]): void {
  const el = document.getElementById(id) as HTMLSelectElement;
  for (const { value, n } of values) {
    if (!value) continue;
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = `${value} (${n.toLocaleString()})`;
    el.appendChild(opt);
  }
}

function populateYearSelects(): void {
  const min = Math.min(...manifest.years);
  const max = Math.max(...manifest.years);
  const minEl = document.getElementById('f-year-min') as HTMLSelectElement;
  const maxEl = document.getElementById('f-year-max') as HTMLSelectElement;
  for (let y = min; y <= max; y++) {
    minEl.appendChild(new Option(String(y), String(y)));
    maxEl.appendChild(new Option(String(y), String(y)));
  }
  // Default to the last 10 years -- the full range is one click away, but most
  // browsing is about recent activity.
  state.yearMin = Math.max(min, max - 9);
  state.yearMax = max;
  minEl.value = String(state.yearMin);
  maxEl.value = String(state.yearMax);
}

function wireFilters(): void {
  const debounce = (fn: () => void, ms: number) => {
    let t: number | undefined;
    return () => { window.clearTimeout(t); t = window.setTimeout(fn, ms); };
  };
  const rerun = debounce(() => { state.page = 0; void runQuery(); }, 250);

  document.getElementById('f-text')!.addEventListener('input', (e) => {
    state.text = (e.target as HTMLInputElement).value; rerun();
  });
  document.getElementById('f-year-min')!.addEventListener('change', (e) => {
    state.yearMin = Number((e.target as HTMLSelectElement).value); state.page = 0; void runQuery();
  });
  document.getElementById('f-year-max')!.addEventListener('change', (e) => {
    state.yearMax = Number((e.target as HTMLSelectElement).value); state.page = 0; void runQuery();
  });
  document.getElementById('f-field')!.addEventListener('change', (e) => {
    state.field = (e.target as HTMLSelectElement).value; state.page = 0; void runQuery();
  });
  document.getElementById('f-manufacturer')!.addEventListener('change', (e) => {
    state.manufacturer = (e.target as HTMLSelectElement).value; state.page = 0; void runQuery();
  });
  document.getElementById('f-usage')!.addEventListener('change', (e) => {
    state.usage = (e.target as HTMLSelectElement).value; state.page = 0; void runQuery();
  });
  document.getElementById('prev-page')!.addEventListener('click', () => {
    if (state.page > 0) { state.page -= 1; void runQuery(); }
  });
  document.getElementById('next-page')!.addEventListener('click', () => {
    state.page += 1; void runQuery();
  });
}

async function main(): Promise<void> {
  // The spinner is visible by default (papers.html has no `hidden` attribute
  // on it) so it's already showing from first paint through this setup step.
  const [manifestRes, instance] = await Promise.all([
    fetch(`${DATA_ROOT}manifest.json`), initDuckDB(),
  ]);
  manifest = await manifestRes.json();
  db = instance;

  populateYearSelects();
  populateSelect('f-field', manifest.by_field);
  populateSelect('f-manufacturer', manifest.by_manufacturer);
  populateSelect('f-usage', manifest.by_usage_context);
  wireFilters();

  await runQuery();
}

main().catch((err) => {
  console.error(err);
  setLoading(false);
  document.getElementById('status')!.textContent =
    'Failed to load the paper index. Please try reloading the page.';
});
