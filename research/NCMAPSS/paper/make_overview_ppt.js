/* 5-slide research overview deck — photo-heavy, minimal text.
   Palette: Okabe-Ito blue 0072B2 + vermillion D55E00 on white (matches the
   paper figures inside).  Fonts: Arial.  Usage: node make_overview_ppt.js */
const pptxgen = require('pptxgenjs');
const P = new pptxgen();
P.layout = 'LAYOUT_WIDE';                       // 13.33 x 7.5

const BLUE = '0072B2', VERM = 'D55E00', INK = '1A1A1A', GRY = '666666';
const DIR = require('path').join(__dirname, '');

function title(s, n, t) {
  s.addText(t, { x: 0.5, y: 0.28, w: 11.0, h: 0.6, fontFace: 'Arial',
                 fontSize: 28, bold: true, color: INK, margin: 0 });
  s.addText(String(n), { x: 12.5, y: 0.28, w: 0.4, h: 0.6, fontFace: 'Arial',
                 fontSize: 16, bold: true, color: BLUE, align: 'right',
                 margin: 0 });
}
function cap(s, x, y, w, t, align) {
  s.addText(t, { x, y, w, h: 0.3, fontFace: 'Arial', fontSize: 10.5,
                 italic: true, color: GRY, align: align || 'center',
                 margin: 0 });
}

/* ---------------- slide 1: data + GPR ---------------- */
let s = P.addSlide();
title(s, 1, 'Data selection and the sparse-window GP model');
const AR = 919 / 480;                            // raw thumb aspect
const tw = 1.75, th = tw / AR;                   // 1.75 x 0.914
const inX = 0.55, inY = 1.45;
const INPUTS = ['alt', 'Mach', 'TRA', 'T2'];
s.addText('Inputs — operating conditions', { x: inX, y: 1.05, w: 3.7, h: 0.3,
  fontFace: 'Arial', fontSize: 13, bold: true, color: INK, margin: 0 });
INPUTS.forEach((n, i) => {
  const x = inX + (i % 2) * (tw + 0.15), y = inY + Math.floor(i / 2) * (th + 0.42);
  s.addImage({ path: DIR + `raw_u5/input_u5_${n}_clean.png`, x, y, w: tw, h: th });
  cap(s, x, y + th + 0.02, tw, n);
});
const OUTS = ['T30', 'T48', 'T50', 'Nc', 'Wf'];
const outX = 8.75, outY = 1.45;
s.addText('Outputs — 5 sensors, screened from 14', { x: outX, y: 1.05, w: 4.2,
  h: 0.3, fontFace: 'Arial', fontSize: 13, bold: true, color: INK, margin: 0 });
OUTS.forEach((n, i) => {
  const x = outX + (i % 2) * (tw + 0.15), y = outY + Math.floor(i / 2) * (th + 0.42);
  s.addImage({ path: DIR + `raw_u5/raw_u5_${n}_clean.png`, x, y, w: tw, h: th });
  cap(s, x, y + th + 0.02, tw, n);
});
s.addShape(P.ShapeType.roundRect, { x: 4.85, y: 2.55, w: 3.4, h: 2.1,
  rectRadius: 0.08, fill: { color: 'EAF2F9' }, line: { color: BLUE, width: 1.5 } });
s.addText([
  { text: 'Vecchia MOGP', options: { fontSize: 17, bold: true, color: INK, breakLine: true } },
  { text: 'RBF kernel, rank 1', options: { fontSize: 13, color: INK, breakLine: true } },
  { text: 'trained on flight cycles 1–3 only', options: { fontSize: 13, color: BLUE, bold: true, breakLine: true } },
  { text: '27,000 stratified rows per seed', options: { fontSize: 13, color: INK } },
], { x: 4.95, y: 2.7, w: 3.2, h: 1.8, fontFace: 'Arial', align: 'center',
     valign: 'middle', margin: 0, lineSpacingMultiple: 1.35 });
s.addShape(P.ShapeType.rightArrow, { x: 4.15, y: 3.35, w: 0.62, h: 0.42,
  fill: { color: BLUE } });
s.addShape(P.ShapeType.rightArrow, { x: 8.32, y: 3.35, w: 0.62, h: 0.42,
  fill: { color: BLUE } });
s.addShape(P.ShapeType.rect, { x: 4.0, y: 6.55, w: 0.22, h: 0.22,
  fill: { color: '1F77B4' } });
s.addText('flight cycles 1–3 (training window)', { x: 4.3, y: 6.5, w: 3.1,
  h: 0.3, fontFace: 'Arial', fontSize: 12, color: INK, margin: 0 });
s.addShape(P.ShapeType.rect, { x: 7.45, y: 6.55, w: 0.22, h: 0.22,
  fill: { color: 'D62728' } });
s.addText('remaining life (never trained)', { x: 7.75, y: 6.5, w: 3.0, h: 0.3,
  fontFace: 'Arial', fontSize: 12, color: INK, margin: 0 });

/* ---------------- slide 2: residuals + detcov ---------------- */
s = P.addSlide();
title(s, 2, 'Residuals and the confidence gate on detcov');
const rAR = 1280 / 640;
const rw = 5.9, rh = rw / rAR;                   // 5.9 x 2.95
s.addImage({ path: DIR + 'fig_slide_resid.png', x: 0.55, y: 1.05, w: rw, h: rh });
cap(s, 0.55, 1.05 + rh + 0.03, rw,
    'residual = measured − GP prediction  (unit u5, T48)');
s.addImage({ path: DIR + 'fig_slide_gate.png', x: 6.85, y: 1.05, w: rw, h: rh });
cap(s, 6.85, 1.05 + rh + 0.03, rw,
    'gate keeps detcov ≥ V · V = 20.2, or 19.0 when the baseline holds under 10 cycles');
const bw = 3.7;
const gAR = 960 / 600, sAR = 840 / 555;
s.addImage({ path: DIR + 'fig_vsweep_class.png', x: 0.55, y: 4.6, w: bw, h: bw / gAR });
cap(s, 0.55, 4.6 + bw / gAR + 0.03, bw, 'class-stratified V sweep → V dense = 20.2, V sparse = 19.0');
s.addImage({ path: DIR + 'fig_v4_sens_cycle.png', x: 4.8, y: 4.6, w: bw, h: bw / sAR });
cap(s, 4.8, 4.6 + bw / sAR + 0.03, bw, 'cycle-count baselines — rejected');
s.addImage({ path: DIR + 'fig_v4_sens_flighth.png', x: 9.05, y: 4.6, w: bw, h: bw / sAR });
cap(s, 9.05, 4.6 + bw / sAR + 0.03, bw, 'flight-hour baselines → 30 h chosen, 5.65');

/* ---------------- slide 3: HI ---------------- */
s = P.addSlide();
title(s, 3, 'Health-index modelling');
const hAR = 960 / 600;
const hw = 5.9, hh = hw / hAR;                   // 5.9 x 3.69
s.addImage({ path: DIR + 'fig_v4_hi_curves.png', x: 0.55, y: 1.3, w: hw, h: hh });
cap(s, 0.55, 1.3 + hh + 0.05, hw, 'development units — 9 engines, LOO reference 7.30 ± 0.16');
s.addImage({ path: DIR + 'fig_v4_hi_curves_test.png', x: 6.85, y: 1.3, w: hw, h: hh });
cap(s, 6.85, 1.3 + hh + 0.05, hw, 'sealed test units — dev-trained network, input only');
s.addText([
  { text: 'gated trim25 residuals  →  pooled z-norm  →  MLP  →  median-3 filter',
    options: { fontSize: 15, bold: true, color: INK, breakLine: true } },
  { text: 'loss = level anchors + 8·monotonicity + 0.25·convexity + tail-flatness',
    options: { fontSize: 13, color: GRY } },
], { x: 0.55, y: 6.1, w: 12.2, h: 0.9, fontFace: 'Arial', align: 'center',
     margin: 0, lineSpacingMultiple: 1.4 });

/* ---------------- slide 4: onset benchmark ---------------- */
s = P.addSlide();
title(s, 4, 'Benchmark — onset detection');
const oAR = 1240 / 620;
const ow = 5.9;
s.addImage({ path: DIR + 'fig_onset_bars.png', x: 0.5, y: 1.15, w: ow, h: ow / oAR });
cap(s, 0.5, 1.15 + ow / oAR + 0.02, ow,
    'pooled onset RMSE per model');
s.addText([
  { text: 'dev, 27 cases: every competitor significantly worse than the gated chain (p ≤ 0.021, all seeds).', options: { breakLine: true } },
  { text: 'test cells hold 6–9 cases — descriptive only; no unit-level separation (n = 6).', options: { breakLine: true } },
  { text: 'LLKE flips on short+medium between splits (8.17 ↔ 4.18) — instability, not skill; its long-haul consistency is noted as a mirror profile.', options: {} },
], { x: 7.3, y: 1.3, w: 5.5, h: 2.6, fontFace: 'Arial', fontSize: 12.5,
     color: INK, margin: 0, lineSpacingMultiple: 1.3, paraSpaceAfter: 8 });
const T2_HDR = { fill: { color: BLUE }, color: 'FFFFFF', bold: true, fontSize: 10.5 };
s.addText('Onset RMSE — pooled and by flight class (p: paired Wilcoxon vs the gated chain, dev 27 cases)',
  { x: 0.5, y: 4.45, w: 12.3, h: 0.35, fontFace: 'Arial', fontSize: 14,
    bold: true, color: INK, margin: 0 });
s.addTable([
  [{ text: 'Model', options: T2_HDR }, { text: 'dev overall', options: T2_HDR },
   { text: 'dev p', options: T2_HDR }, { text: 'dev short+med', options: T2_HDR },
   { text: 'dev long', options: T2_HDR }, { text: 'test overall', options: T2_HDR },
   { text: 'test short+med', options: T2_HDR }, { text: 'test long', options: T2_HDR }],
  [{ text: 'MOGP gate (coverage-conditional)', options: { bold: true, color: 'B03A00' } },
   { text: '5.21', options: { bold: true } }, '—',
   { text: '4.75', options: { bold: true } }, { text: '6.60', options: { bold: true } },
   { text: '8.54', options: { bold: true } }, { text: '5.54', options: { bold: true } },
   { text: '10.73', options: { bold: true } }],
  ['MOGP ungated', '6.78', '0.021', '6.51', '7.65', '9.01', '5.73', '11.38'],
  ['LLKE', '7.67', '0.008', '8.17', '5.55', '6.87', '4.18', '8.77'],
  ['B-spline', '8.69', '0.001', '8.63', '8.93', '10.03', '6.87', '12.41'],
  ['LR', '10.02', '<0.001', '9.52', '11.61', '8.88', '5.37', '11.34'],
  ['CaBN', '10.54', '<0.001', '10.03', '12.17', '10.66', '5.37', '14.08'],
], { x: 0.5, y: 4.9, w: 12.3, colW: [3.3, 1.3, 1.1, 1.6, 1.3, 1.4, 1.6, 1.3],
     fontFace: 'Arial', fontSize: 10, color: INK, align: 'center',
     valign: 'middle', border: { type: 'solid', color: 'CCCCCC', pt: 0.5 },
     rowH: 0.3 });

/* ---------------- slide 5: RUL benchmark ---------------- */
s = P.addSlide();
title(s, 5, 'Benchmark — test RUL and computational cost');
const bAR = 960 / 570;
const bwi = 5.6;
s.addImage({ path: DIR + 'fig_bench_rul_box.png', x: 0.55, y: 1.5, w: bwi, h: bwi / bAR });
cap(s, 0.55, 1.5 + bwi / bAR + 0.05, bwi,
    'test RUL per unit — all models tie, pairwise p ≥ 0.26');
const T1_HDR = { fill: { color: BLUE }, color: 'FFFFFF', bold: true, fontSize: 11 };
s.addTable([
  [{ text: 'Model', options: T1_HDR }, { text: 'Train s', options: T1_HDR },
   { text: 'Infer s', options: T1_HDR }, { text: 'Test RUL RMSE', options: T1_HDR }],
  [{ text: 'MOGP', options: { bold: true } }, '921.3', '9.1',
   { text: '7.57 ± 0.10', options: { bold: true } }],
  ['LLKE', '2.9', '15.3', '7.74 ± 0.06'],
  ['B-spline', '0.1', '0.2', '7.39 ± 0.11'],
  ['LR = CaBN', '< 0.1', '< 0.1', '7.38 ± 0.11'],
], { x: 6.8, y: 1.5, w: 6.0, colW: [1.8, 1.2, 1.2, 1.8], fontFace: 'Arial',
     fontSize: 11, color: INK, align: 'center', valign: 'middle',
     border: { type: 'solid', color: 'CCCCCC', pt: 0.5 }, rowH: 0.34 });
s.addText([
  { text: 'The truncation-RUL end metric does not discriminate normal models — the degradation drift lives in T48/T50 and every model captures it.', options: { breakLine: true } },
  { text: 'The gate is not an RUL component: removing it leaves test RUL unchanged (7.47 vs 7.57, paired p = 0.99).', options: { breakLine: true } },
  { text: 'MOGP pays a one-off training cost; its inference is faster than LLKE.', options: {} },
], { x: 6.8, y: 3.6, w: 6.0, h: 2.6, fontFace: 'Arial', fontSize: 12.5,
     color: INK, margin: 0, lineSpacingMultiple: 1.3, paraSpaceAfter: 8 });

/* ---------------- slide 5: ablation ---------------- */
s = P.addSlide();
title(s, 6, 'Ablation — every frozen choice carries its weight');
const cards = [
  ['+28%', 'RUL error without\nresidualization', VERM],
  ['40.5', 'RUL collapse without\nthe HI end anchor', VERM],
  ['p = 0.99', 'RUL unchanged\nwithout the gate', BLUE],
];
cards.forEach(([big, small, col], i) => {
  const x = 0.6 + i * 4.25;
  s.addShape(P.ShapeType.roundRect, { x, y: 1.1, w: 3.8, h: 1.7,
    rectRadius: 0.08, fill: { color: 'F5F5F5' }, line: { color: 'DDDDDD', width: 1 } });
  s.addText(big, { x, y: 1.22, w: 3.8, h: 0.75, fontFace: 'Arial', fontSize: 34,
    bold: true, color: col, align: 'center', margin: 0 });
  s.addText(small, { x, y: 2.0, w: 3.8, h: 0.7, fontFace: 'Arial', fontSize: 12,
    color: GRY, align: 'center', margin: 0 });
});
const T3_HDR = { fill: { color: BLUE }, color: 'FFFFFF', bold: true, fontSize: 10.5 };
s.addTable([
  [{ text: 'Component removed', options: T3_HDR }, { text: 'Chain', options: T3_HDR },
   { text: 'Ablated', options: T3_HDR }, { text: 'Verdict', options: T3_HDR }],
  ['detcov gate', 'onset 5.65', 'onset 6.78 (p = 0.021)',
   'detection component; RUL robust to it'],
  ['residualization (raw sensor means)', 'RUL 7.30', 'RUL 9.33, shape 0/3',
   'residuals are indispensable'],
  ['cycle≤3 window (healthy-range retrain)', 'RUL 7.30', 'RUL 7.57',
   'earliest cycles are the only clean data'],
], { x: 0.6, y: 3.2, w: 12.1, colW: [3.6, 1.7, 3.0, 3.8], fontFace: 'Arial',
     fontSize: 10.5, color: INK, align: 'left', valign: 'middle',
     border: { type: 'solid', color: 'CCCCCC', pt: 0.5 }, rowH: 0.34 });
s.addTable([
  [{ text: 'HI loss term removed', options: T3_HDR }, { text: 'RUL RMSE', options: T3_HDR },
   { text: 'Shape', options: T3_HDR }],
  [{ text: 'none — full recipe', options: { bold: true } },
   { text: '7.30 ± 0.16', options: { bold: true } }, '3/3'],
  ['initial-level anchor', '12.63', '0/3'],
  ['end-level anchor', '40.53', '0/3'],
  ['tail-flatness', '7.54', '3/3'],
], { x: 0.6, y: 5.15, w: 7.2, colW: [3.4, 2.2, 1.6], fontFace: 'Arial',
     fontSize: 10.5, color: INK, align: 'center', valign: 'middle',
     border: { type: 'solid', color: 'CCCCCC', pt: 0.5 }, rowH: 0.3 });
s.addText([
  { text: 'Armor-removal check:', options: { bold: true, color: INK, breakLine: true } },
  { text: 'replacing trim25 with plain means and removing the median filter collapses no competitor — the RUL tie is structural, not a downstream artefact.',
    options: { color: GRY } },
], { x: 8.2, y: 5.3, w: 4.5, h: 1.6, fontFace: 'Arial', fontSize: 11.5,
     margin: 0, lineSpacingMultiple: 1.25 });

P.writeFile({ fileName: DIR + 'research_overview.pptx' })
  .then(() => console.log('written research_overview.pptx'));
