import Plotly from 'plotly.js-dist-min';
import './style.css';

// Okabe-Ito palette, same ordering the notebooks use, so field/series colors
// read consistently between the PDFs and this page.
const PAL = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9',
             '#DDCC77', '#882255', '#999999'];

const DATA_ROOT = 'data/charts';

async function loadJSON<T>(name: string): Promise<T> {
  const res = await fetch(`${DATA_ROOT}/${name}.json`);
  if (!res.ok) throw new Error(`failed to load ${name}.json: ${res.status}`);
  return res.json() as Promise<T>;
}

function plot(divId: string, data: unknown[], layout: Record<string, unknown>): void {
  const merged = {
    margin: { t: 28, r: 16, b: 42, l: 56 },
    font: { size: 11 },
    legend: { font: { size: 10 } },
    ...layout,
  };
  Plotly.newPlot(divId, data, merged, { responsive: true, displaylogo: false });
}

// --- shared row shapes (mirrors accelscan.webexport's to_dicts() output) ----

interface YearSeries { year: number; n: number; med: number; p90: number; total: number; }
interface FrontierRow { year: number; frontier: number; }
interface FieldSeries { seg: string; year: number; n: number; med: number; }
interface SpreadRow { seg: string; year: number; n: number; med: number; p90: number;
                      ratio: number; ratio_ma: number; }
interface CompositionRow { year: number; seg: string; n: number; }
interface PrevalenceOverallRow { year: number; n: number; any_share: number;
                                 used_share: number; modelspec_share: number; cand_share: number; }
interface PrevalenceFieldRow { seg: string; year: number; n: number; used: number; }
interface SpecificityRow { year: number; n: number; specific: number; }
interface DeviceLifeRow { canonical_model: string; year: number; display: string;
                          release_year: number; since_release: number; papers: number; }

// --- compute/memory vs frontier ---------------------------------------------

interface AxisPayload { per_device: YearSeries[]; per_paper: YearSeries[]; frontier: FrontierRow[]; }
interface ComputeMemoryChart { compute: AxisPayload; memory: AxisPayload; }

function renderComputeMemory(divId: string, axis: AxisPayload, scale: number, unit: string): void {
  const series = (rows: YearSeries[], key: 'med' | 'p90', name: string, color: string, dash?: string) => ({
    x: rows.map((r) => r.year), y: rows.map((r) => r[key] / scale),
    type: 'scatter', mode: 'lines+markers', name,
    line: { color, dash: dash ?? 'solid' }, marker: { size: 5 },
  });
  const traces = [
    series(axis.per_device, 'med', 'per device, median', '#0072B2'),
    series(axis.per_device, 'p90', 'per device, 90%-ile', '#0072B2', 'dash'),
    series(axis.per_paper, 'med', 'per paper, median', '#009E73'),
    series(axis.per_paper, 'p90', 'per paper, 90%-ile', '#009E73', 'dash'),
    {
      x: axis.frontier.map((r) => r.year), y: axis.frontier.map((r) => r.frontier / scale),
      type: 'scatter', mode: 'lines', name: 'frontier GPU (single device)',
      line: { color: '#D55E00', width: 3 },
    },
  ];
  // dtick: 1 restricts a log axis to major ticks at each power of ten -- Plotly's
  // default also draws minor ticks at 2 and 5 within each decade.
  plot(divId, traces, {
    yaxis: { type: 'log', dtick: 1, title: unit }, xaxis: { title: 'Year' },
  });
}

// --- by-field median ---------------------------------------------------------

function renderByField(divId: string, rows: FieldSeries[], fields: string[], scale: number,
                       unit: string): void {
  const traces = fields.map((f, i) => {
    const s = rows.filter((r) => r.seg === f);
    return {
      x: s.map((r) => r.year), y: s.map((r) => r.med / scale),
      type: 'scatter', mode: 'lines+markers', name: f,
      line: { color: PAL[i % PAL.length], dash: f.startsWith('Other') ? 'dash' : 'solid' },
      marker: { size: 4 },
    };
  });
  plot(divId, traces, { xaxis: { title: 'Year' }, yaxis: { title: unit } });
}

// --- spread (p90/median ratio, smoothed) -------------------------------------

function renderSpread(divId: string, rows: SpreadRow[], fields: string[], subtitle: string): void {
  const traces = fields.map((f, i) => {
    const s = rows.filter((r) => r.seg === f).sort((a, b) => a.year - b.year);
    return {
      x: s.map((r) => r.year), y: s.map((r) => r.ratio_ma),
      type: 'scatter', mode: 'lines', name: f,
      line: {
        color: f === 'All fields' ? '#111111' : PAL[i % PAL.length],
        width: f === 'All fields' ? 3 : 1.5,
        dash: f.startsWith('Other') ? 'dash' : 'solid',
      },
    };
  });
  plot(divId, traces, {
    title: { text: subtitle, font: { size: 12 } },
    xaxis: { title: 'Year' },
    yaxis: { title: '90%-ile / median (3-yr moving average)' },
    shapes: [{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 1, y1: 1,
              line: { color: '#999999', dash: 'dot', width: 1 } }],
  });
}

// --- corpus composition (stacked bar) ---------------------------------------

function renderCorpusComposition(divId: string, rows: CompositionRow[], segments: string[]): void {
  const years = Array.from(new Set(rows.map((r) => r.year))).sort((a, b) => a - b);
  const traces = segments.map((seg, i) => ({
    x: years,
    y: years.map((y) => rows.find((r) => r.year === y && r.seg === seg)?.n ?? 0),
    type: 'bar', name: seg,
    marker: { color: seg.startsWith('Other') ? '#BBBBBB' : seg.startsWith('No field')
      ? '#EEEEEE' : PAL[i % PAL.length] },
  }));
  plot(divId, traces, {
    barmode: 'stack', xaxis: { title: 'Year' }, yaxis: { title: 'Full-text papers' },
  });
}

// --- prevalence ---------------------------------------------------------------

function renderPrevalenceOverall(divId: string, rows: PrevalenceOverallRow[]): void {
  const line = (key: keyof PrevalenceOverallRow, name: string, color: string) => ({
    x: rows.map((r) => r.year), y: rows.map((r) => 100 * (r[key] as number)),
    type: 'scatter', mode: 'lines+markers', name, line: { color }, marker: { size: 5 },
  });
  plot(divId, [
    line('any_share', 'any accelerator mention', '#0072B2'),
    line('used_share', 'reports using an accelerator', '#D55E00'),
    line('modelspec_share', 'names and uses a specific model', '#009E73'),
  ], { xaxis: { title: 'Year' }, yaxis: { title: 'Percentage of full-text papers' } });
}

function renderPrevalenceByField(divId: string, rows: PrevalenceFieldRow[], fields: string[]): void {
  const traces = fields.map((f, i) => {
    const s = rows.filter((r) => r.seg === f);
    return {
      x: s.map((r) => r.year), y: s.map((r) => 100 * r.used),
      type: 'scatter', mode: 'lines', name: f,
      line: { color: f.startsWith('Other') ? '#BBBBBB' : PAL[i % PAL.length],
              dash: f.startsWith('Other') ? 'dash' : 'solid' },
    };
  });
  plot(divId, traces, {
    xaxis: { title: 'Year' }, yaxis: { title: 'Percentage of field papers reporting use' },
  });
}

// --- funnel + specificity ------------------------------------------------------

function renderFunnel(divId: string, funnel: Record<string, number>): void {
  const labels = Object.keys(funnel);
  const values = labels.map((l) => funnel[l]);
  const total = values[0];
  plot(divId, [{
    type: 'bar', orientation: 'h', x: values, y: labels,
    text: values.map((v) => `${v.toLocaleString()} (${(100 * v / total).toFixed(2)}%)`),
    textposition: 'outside',
    marker: { color: ['#999999', '#56B4E9', '#0072B2', '#D55E00', '#009E73'] },
  }], {
    xaxis: { type: 'log', dtick: 1, title: 'Papers (log)' },
    yaxis: { autorange: 'reversed' },
    margin: { t: 28, r: 90, b: 42, l: 190 },
  });
}

function renderSpecificity(divId: string, rows: SpecificityRow[]): void {
  plot(divId, [{
    x: rows.map((r) => r.year), y: rows.map((r) => 100 * r.specific),
    type: 'scatter', mode: 'lines+markers', line: { color: '#0072B2' }, marker: { size: 5 },
  }], {
    xaxis: { title: 'Year' },
    yaxis: { title: 'Percentage naming a specific model', range: [0, 100] },
  });
}

// --- device life ---------------------------------------------------------------

function renderDeviceLife(divId: string, rows: DeviceLifeRow[]): void {
  const models = Array.from(new Set(rows.map((r) => r.canonical_model)))
    .map((id) => rows.find((r) => r.canonical_model === id)!)
    .sort((a, b) => a.release_year - b.release_year);
  const traces = models.map((m, i) => {
    const s = rows.filter((r) => r.canonical_model === m.canonical_model)
      .sort((a, b) => a.since_release - b.since_release);
    return {
      x: s.map((r) => r.since_release), y: s.map((r) => r.papers),
      type: 'scatter', mode: 'lines+markers', name: `${m.display} (${m.release_year})`,
      line: { color: PAL[i % PAL.length] }, marker: { size: 4 },
    };
  });
  plot(divId, traces, {
    xaxis: { title: 'Years since release' },
    yaxis: { type: 'log', dtick: 1, title: 'Papers supported (log)' },
  });
}

// --- wire-up ---------------------------------------------------------------------

async function main(): Promise<void> {
  const [computeMemory, byField, spread, composition, prevalence, funnel, deviceLife] =
    await Promise.all([
      loadJSON<ComputeMemoryChart>('compute_memory'),
      loadJSON<{ fields: string[]; compute: FieldSeries[]; memory: FieldSeries[] }>(
        'compute_fp32_memory_vram_by_field'),
      loadJSON<{ fields: string[]; compute: SpreadRow[]; memory: SpreadRow[] }>(
        'compute_fp32_memory_vram_spread'),
      loadJSON<{ fields: string[]; other: string; no_label: string; series: CompositionRow[] }>(
        'corpus_composition'),
      loadJSON<{ fields: string[]; overall: PrevalenceOverallRow[]; by_field: PrevalenceFieldRow[] }>(
        'prevalence'),
      loadJSON<{ funnel: Record<string, number>; specificity: SpecificityRow[] }>('funnel'),
      loadJSON<{ series: DeviceLifeRow[] }>('device_life'),
    ]);

  renderComputeMemory('chart-compute', computeMemory.compute, 1000,     // GFLOPS -> TFLOPS
    'Estimated FP32 Capacity (TFLOPS)');
  renderComputeMemory('chart-memory', computeMemory.memory, 1,          // already GB
    'Estimated VRAM (GB)');

  renderByField('chart-by-field-compute', byField.compute, byField.fields, 1000,
    'Median FP32 Capacity (TFLOPS)');
  renderByField('chart-by-field-memory', byField.memory, byField.fields, 1,
    'Median VRAM (GB)');

  renderSpread('chart-spread-compute', spread.compute, spread.fields, 'Compute (FP32 TFLOPS)');
  renderSpread('chart-spread-memory', spread.memory, spread.fields, 'Memory (VRAM GB)');

  renderCorpusComposition('chart-corpus-composition', composition.series,
    [...composition.fields, composition.other, composition.no_label]);

  renderPrevalenceOverall('chart-prevalence-overall', prevalence.overall);
  renderPrevalenceByField('chart-prevalence-field', prevalence.by_field, prevalence.fields);

  renderFunnel('chart-funnel', funnel.funnel);
  renderSpecificity('chart-specificity', funnel.specificity);

  renderDeviceLife('chart-device-life', deviceLife.series);
}

main().catch((err) => {
  console.error(err);
  document.querySelectorAll('.plot').forEach((el) => {
    el.textContent = 'Failed to load chart data.';
  });
});
