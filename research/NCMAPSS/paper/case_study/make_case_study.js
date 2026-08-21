/* make_case_study.js — fills the "IV. Case study" template with the DS03
   record first, then the transfer sub-datasets (DS01, DS02, DS08a, DS07).
   Numbers: case_study_data.json (make_cs_data.py).  Usage: node make_case_study.js */
const fs = require('fs');
const path = require('path');
const D = require('docx');
const HERE = __dirname;
const FIG = path.join(HERE, '..') + '/';
const J = JSON.parse(fs.readFileSync(path.join(HERE, 'case_study_data.json'), 'utf8'));
const FONT = 'Times New Roman', TW = 9000;

const run = (x, o = {}) => new D.TextRun({ text: x, font: FONT, size: 20, ...o });
const P = (text, o = {}) => new D.Paragraph({ spacing: { before: 60, after: 120 }, alignment: D.AlignmentType.JUSTIFIED, children: [run(text, o)] });
const H1 = x => new D.Paragraph({ spacing: { before: 360, after: 120 }, children: [run(x, { bold: true, size: 26 })] });
const H2 = x => new D.Paragraph({ spacing: { before: 280, after: 100 }, children: [run(x, { bold: true, italics: true, size: 22 })] });
const cap = (label, text) => new D.Paragraph({ spacing: { before: 200, after: 80 }, children: [run(label + ' ', { bold: true }), run(text)] });
const note = text => new D.Paragraph({ spacing: { before: 40, after: 160 }, children: [run('Note. ' + text, { italics: true, size: 18, color: '444444' })] });
function cell(content, { header = false, bold = false, w = 1000, align = 'center', size = 17, span = 1, rowSpan = 1 } = {}) {
  return new D.TableCell({ width: { size: w, type: D.WidthType.DXA }, columnSpan: span > 1 ? span : undefined, rowSpan: rowSpan > 1 ? rowSpan : undefined,
    shading: header ? { type: D.ShadingType.CLEAR, fill: 'E8E8E8' } : undefined, verticalAlign: D.VerticalAlign.CENTER,
    margins: { top: 30, bottom: 30, left: 50, right: 50 },
    children: [new D.Paragraph({ alignment: align === 'left' ? D.AlignmentType.LEFT : D.AlignmentType.CENTER, children: [run(String(content), { bold: header || bold, size })] })] });
}
function table(headers, rows, { widths, boldRows = [], leftCol = true } = {}) {
  const n = headers.length; const w = widths || Array(n).fill(Math.floor(TW / n));
  return new D.Table({ width: { size: TW, type: D.WidthType.DXA }, columnWidths: w,
    rows: [new D.TableRow({ tableHeader: true, children: headers.map((h, j) => cell(h, { header: true, w: w[j] })) }),
      ...rows.map((r, i) => new D.TableRow({ children: r.map((c, j) => cell(c, { w: w[j], bold: boldRows.includes(i), align: leftCol && j === 0 ? 'left' : 'center' })) }))] });
}
function figure(file, wPx, hPx, label, text) {
  const p = file.startsWith('/') ? file : FIG + file;
  return [new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { before: 200, after: 40 },
      children: [new D.ImageRun({ type: 'png', data: fs.readFileSync(p), transformation: { width: wPx, height: hPx } })] }),
    new D.Paragraph({ alignment: D.AlignmentType.CENTER, spacing: { after: 200 }, children: [run(label + ' ', { bold: true, size: 18 }), run(text, { size: 18 })] })];
}
const f2 = x => (x === undefined || x === null || Number.isNaN(x)) ? 'n/r' : x.toFixed(2);
const f1 = x => (x === undefined || x === null || Number.isNaN(x)) ? 'n/r' : x.toFixed(1);
const ft = x => x < 0.1 ? '< 0.1' : x.toFixed(1);
const LV = ['0.2', '0.4', '0.6', '0.8', 'all'];

/* level table with RMSE/Score sub-columns (Table 4/5/6 layout) */
function levelTable(rowsObj, { times = false, boldFirst = true } = {}) {
  const names = Object.keys(rowsObj);
  const w0 = times ? 1600 : 2100, wl = times ? 580 : 690, wt = 800;
  const widths = [w0, ...Array(10).fill(wl), ...(times ? [wt, wt] : [])];
  const hdr1 = new D.TableRow({ tableHeader: true, children: [
    cell('Benchmarks', { header: true, w: w0, rowSpan: 2 }),
    ...['20%', '40%', '60%', '80%', 'Average'].map(l => cell(l, { header: true, w: wl * 2, span: 2 })),
    ...(times ? [cell('Training time (s)', { header: true, w: wt, rowSpan: 2 }), cell('Inference time (s)', { header: true, w: wt, rowSpan: 2 })] : []) ] });
  const hdr2 = new D.TableRow({ tableHeader: true, children: [
    ...LV.flatMap(() => [cell('RMSE', { header: true, w: wl }), cell('Score', { header: true, w: wl })]) ] });
  const rows = names.map((nm, i) => {
    const r = rowsObj[nm]; const b = boldFirst && i === 0;
    return new D.TableRow({ children: [cell(nm, { w: w0, align: 'left', bold: b }),
      ...LV.flatMap(l => [cell(r[l] ? f2(r[l].rmse) : 'n/r', { w: wl, bold: b }), cell(r[l] ? f1(r[l].score) : 'n/r', { w: wl, bold: b })]),
      ...(times ? [cell(ft(r.train), { w: wt, bold: b }), cell(ft(r.infer), { w: wt, bold: b })] : [])] });
  });
  return new D.Table({ width: { size: TW, type: D.WidthType.DXA }, columnWidths: widths, rows: [hdr1, hdr2, ...rows] });
}

const kids = [];
kids.push(new D.Paragraph({ spacing: { after: 200 }, children: [run('IV. Case study', { bold: true, size: 28 })] }));

/* ---------------- A ---------------- */
kids.push(H2('A. Dataset Description'));
kids.push(P('The case study uses the N-CMAPSS turbofan run-to-failure data set (Arias Chao et al., 2021). Every flight is recorded at 1 Hz; each record holds four operating parameters (the scenario descriptors W) and fourteen monitoring measurements (the sensor readings Xs) listed in Table 1. The primary sub-dataset is DS03-012, in which the high-pressure turbine and the low-pressure turbine degrade simultaneously (HPT efficiency, LPT efficiency and LPT flow). Its fifteen units are split as delivered by the data set: nine historical units (units 1–9, development) that are the only data used for model selection, sweeps and calibration, and six in-service units (units 10–15, test) that enter the evaluation as prediction input only. The fleet mixes the three flight classes (short, medium and long haul: 3/4/2 historical and 2/1/3 in-service units), with 60–93 cycles of life and a labelled degradation onset (hs transition) between cycles 18 and 38. Section IV-G repeats the full procedure on two further sub-datasets to test whether the method transfers: DS08a-009, in which all ten health parameters degrade on a fleet with the same 9/6 design and class mix as DS03, and DS07, in which the low-pressure turbine degrades (efficiency and flow) on a 6/4 fleet of mixed flight classes.'));
kids.push(cap('Table 1', 'Detailed information on 14 monitoring measurements and 4 operating parameters.'));
const T1 = [['Operating parameters', 'alt', 'Altitude', 'ft'], ['', 'Mach', 'Flight Mach number', '–'], ['', 'TRA', 'Throttle-resolver angle', '%'], ['', 'T2', 'Total temperature at fan inlet', '°R'],
  ['Monitoring measurements', 'T24', 'Total temperature at LPC outlet', '°R'], ['', 'T30', 'Total temperature at HPC outlet', '°R'], ['', 'T48', 'Total temperature at HPT outlet', '°R'], ['', 'T50', 'Total temperature at LPT outlet', '°R'],
  ['', 'P15', 'Total pressure in bypass duct', 'psia'], ['', 'P2', 'Total pressure at fan inlet', 'psia'], ['', 'P21', 'Total pressure at fan outlet', 'psia'], ['', 'P24', 'Total pressure at LPC outlet', 'psia'],
  ['', 'Ps30', 'Static pressure at HPC outlet', 'psia'], ['', 'P40', 'Total pressure at burner outlet', 'psia'], ['', 'P50', 'Total pressure at LPT outlet', 'psia'], ['', 'Nf', 'Physical fan speed', 'rpm'], ['', 'Nc', 'Physical core speed', 'rpm'], ['', 'Wf', 'Fuel flow', 'pps']];
kids.push(table(['', 'Symbol', 'Description', 'Units'], T1, { widths: [2300, 1200, 4100, 1400] }));

/* ---------------- B ---------------- */
kids.push(H2('B. Evaluation metrics'));
kids.push(P('Two tasks are scored. State division is scored by the onset RMSE, the root-mean-square difference in cycles between the detected degradation onset and the labelled hs transition over all unit–seed cases (27 historical cases on DS03, three seeds per unit). RUL prediction is scored by truncation: the health-index trajectory of an in-service unit is cut at 20 %, 40 %, 60 % and 80 % of its life, the remaining life is predicted from the truncated trajectory, and the RUL RMSE is reported per truncation level and averaged over the four levels. The NASA asymmetric score s = Σ[exp(−d/13) − 1] for early predictions (d = predicted − true < 0) and Σ[exp(d/10) − 1] for late ones penalises late predictions more heavily; it is summed over the predictions of one seed and averaged over the three seeds. Every figure in this section is the mean over three training seeds unless stated otherwise; ± denotes the seed standard deviation.'));

/* ---------------- C ---------------- */
kids.push(H2('C. Model training settings'));
kids.push(P('All choices are made on the historical units only. Sensor screening is model-free: a kNN (k = 10) healthy reference is built over the standardised operating parameters from the first three flight cycles of every unit, the residual of each sensor on the degraded stretch of life is expressed in units of its own healthy residual standard deviation (zRMS), and the sensors are ranked by this score (Fig. 1); an insensitive sensor scores about 1. The top-3/5/7 candidate sets were carried into the normal-model grid, and the five-sensor set [T30, T48, T50, Nc, Wf] was selected by the downstream onset detector (best both ungated, 6.78 vs 7.15/6.95, and gated, 5.65 vs 6.41/5.76), together with the numerical instability of the three-sensor set.'));
kids.push(...figure('fig_v4_sensitivity.png', 520, 269, 'Fig. 1.', 'Sensor selection: degradation sensitivity (zRMS in healthy-noise units) of the fourteen measurements under the cycle≤3 reference; the blue bars form the selected set.'));
kids.push(P('The normal model is a multi-output Gaussian process with a Vecchia approximation (m = 18 nearest neighbours) trained on 27,000 rows stratified as 1,000 rows per unit and cycle over the first three cycles. Kernel and rank are chosen on a held-out zRMSE — the RMSE of the standardised predictions on 8,192 healthy rows of the same cycle≤3 pool that were held out of training — over {RBF, Matérn 3/2, Matérn 1/2} × rank {1, 2, 3} and three seeds (Fig. 2). Within every sensor set the tie-band rule (minimum mean, one-seed-s.d. band, simplest member inside) selects RBF with rank 1; Matérn 5/2 is excluded because its float32 Cholesky factorisation fails chronically in the three- and five-sensor sets.'));
kids.push(...figure('fig_v4_grid_heatmap.png', 600, 176, 'Fig. 2.', 'Kernel selection: held-out zRMSE over kernel × rank for the three candidate sensor sets (one shared colour scale); the frame marks the selection.'));
kids.push(P('The onset detector reads the per-cycle 75th percentile of the conditional log-likelihood curve, fixes a baseline from the first 30 flight hours, clips the curve at μ0 − 10σ0 and places the onset at the full-range mean-drop change point. Before the cycle summary a confidence gate drops the sampled points whose confidence index log|Σ(w)|⁻¹ᐟ² (detcov) falls below a threshold V; the threshold is swept on an absolute axis past its minimum. The historical units are divided by the observable baseline cycle count: a unit with fewer than ten cycles inside the 30-hour baseline (the long-haul units) is sparse, otherwise dense. Figure 3 shows why one threshold cannot serve both: the dense optimum is V = 20.2 while the sparse units reach their optimum on a plateau at 18.8–19.2, whose centre 19.0 is adopted; the resulting coverage-conditional rule (V = 19.0 if nbase < 10, else 20.2) was frozen on the historical units and verified once on the in-service units (onset RMSE 5.21 historical / 8.54 in-service against 6.78 / 9.01 ungated, paired p = 0.003 on the 27 historical cases).'));
kids.push(...figure('fig_vsweep_class.png', 500, 313, 'Fig. 3.', 'The effect of the threshold on the onset RMSE of the historical units, stratified by baseline density; dotted lines are the ungated references.'));
kids.push(P('The health index is a small MLP trained on the per-flight trimmed-mean residual tables with a shape-constrained loss. Five loss terms are used (Table 3): the initial-level, end-level and tail-flatness terms are fixed constants of the recipe, and only the two shape weights λ1 (monotonicity) and λ2 (convexity) are tuned. They are selected on a 6 × 5 grid by the same two-step rule every time: a combination is eligible only if the trained index passes the shape criteria on all three seeds, and eligible combinations are ranked by a composite property score, the equal-weight mean of the min–max-normalised monotonicity (Mono), curvature (Curv) and range (Rng) of the historical trajectories; the truncation RUL is reported but never used for selection, because RUL-only selection had been shown to favour trajectories that flatten near the end of life. Table 2 is the grid on the final healthy-range tables: (12, 0.25) had been selected on the stage-1 tables as the outright composite winner (0.718 vs 0.691 for the runner-up, with the λ1 = 16 column failing the shape gate there), and on the final tables the composite top (16, 0.25) leads it by only 0.010 (0.737 vs 0.727) with a seed-unstable RUL (7.80 ± 0.20 vs 7.53 ± 0.05), so the recipe is retained by the tie-band rule.'));
kids.push(cap('Table 2', 'Selecting the coefficients λ1 and λ2 (historical units, three seeds, final healthy-range tables). Mono = 100 × Σ negative increments; Curv = 100 × Σ second differences; Rng = mean(end − start); RUL = leave-one-unit-out truncation RMSE in cycles (med3-free grid protocol). The bold cell is the selected recipe.'));
{
  const L1S = ['2.0', '4.0', '6.0', '8.0', '12.0', '16.0'], L2S = ['0.25', '0.5', '1.0', '2.0', '4.0'];
  const g = J.ds03.lambda_grid; const lw = 760, pw = Math.floor((TW - lw) / 20);
  const hdr1 = new D.TableRow({ tableHeader: true, children: [cell('', { header: true, w: lw, size: 14 }), ...L2S.map(v => cell('λ2 = ' + v, { header: true, w: pw * 4, span: 4, size: 14 }))] });
  const hdr2 = new D.TableRow({ tableHeader: true, children: [cell('', { header: true, w: lw, size: 14 }), ...L2S.flatMap(() => ['Mono', 'Curv', 'Rng', 'RUL'].map(p => cell(p, { header: true, w: pw, size: 13 })))] });
  const rows = L1S.map(l1 => new D.TableRow({ children: [cell('λ1 = ' + parseFloat(l1), { w: lw, size: 14, align: 'left', bold: l1 === '12.0' }),
    ...L2S.flatMap(l2 => { const c = g[`${l1}_${l2}`]; const b = l1 === '12.0' && l2 === '0.25';
      return [cell(c.mon.toFixed(1), { w: pw, size: 13, bold: b }), cell(c.curv.toFixed(2), { w: pw, size: 13, bold: b }), cell(c.rng.toFixed(3), { w: pw, size: 13, bold: b }), cell(c.rul.toFixed(2), { w: pw, size: 13, bold: b })]; })] }));
  kids.push(new D.Table({ width: { size: TW, type: D.WidthType.DXA }, columnWidths: [lw, ...Array(20).fill(pw)], rows: [hdr1, hdr2, ...rows] }));
}
kids.push(cap('Table 3', 'Hyperparameter settings of the proposed method on DS03 (all selected on the historical units).'));
const hp = J.ds03.hparams;
kids.push(table(['Component', 'Hyperparameter', 'Selected value', 'Selection basis'], [
  ['Sensor screening', 'sensor set', hp.sensors, 'kNN zRMS ranking, top-5 by downstream detector sweep'],
  ['Normal model', 'kernel, rank, Vecchia m', 'RBF, rank 1, m = 18', 'held-out zRMSE grid, tie band'],
  ['Normal model', 'training budget', hp.budget, 'matched between stages (coverage is the only variable)'],
  ['Onset detector', 'baseline window, clip', hp.window + ', ' + hp.clip, 'window × clip sweep on onset RMSE'],
  ['Confidence gate', 'threshold rule', hp.V, 'absolute-axis sweep, class-stratified'],
  ['Health index', 'λ1, λ2', hp.lam, 'shape gate 3/3 + composite property score'],
  ['Health index', 'fixed loss constants', hp.recipe, 'recipe constants (structural)'],
  ['RUL model', 'β (exponential first passage)', hp.beta + ' per seed', 'leave-one-unit-out NASA score on historical units'],
], { widths: [1700, 2000, 2700, 2600] }));

/* ---------------- D ---------------- */
kids.push(H2('D. Results and comparison'));
const o3 = J.ds03.onset;
kids.push(P(`Figure 4 summarises the onset RMSE of the in-service units, one box per sub-dataset (points are per-unit RMSE over the three seeds). On DS03 the conditional gate reaches ${f2(o3.test_cond)} cycles against ${f2(o3.test_ung)} ungated (historical units ${f2(o3.dev_cond)} vs ${f2(o3.dev_ung)}, paired p = ${o3.dev_p}). The other boxes are discussed in Section IV-G.`));
kids.push(...figure(path.join(HERE, 'cs_fig4_onset_subsets.png'), 470, 307, 'Fig. 4.', 'Onset RMSE of the in-service units over the sub-datasets (conditional gate; boxes span the full per-unit range).'));
kids.push(P('Figure 5 shows the health-index trajectories of the nine historical units produced by the final chain (stage-2 healthy-range residuals, seed 0): the index stays flat over the healthy stretch, rises monotonically and reaches the failure level at the end of life on every unit.'));
kids.push(...figure('fig_hi2_curves.png', 500, 313, 'Fig. 5.', 'Health index of the historical units (DS03, seed 0).'));
kids.push(P('Figure 6 gives the RUL RMSE of the in-service units per truncation level (one point per unit, RMSE over three seeds): the error shrinks monotonically as the truncation moves towards the end of life, from 10.92 cycles at 20 % to 2.48 cycles at 80 %.'));
kids.push(...figure(path.join(HERE, 'cs_fig6_rul_levels_ds03.png'), 470, 307, 'Fig. 6.', 'RUL RMSE of the in-service units of DS03 over the truncation levels (proposed method).'));
kids.push(P('Figure 7 compares the per-cycle residuals of the historical units produced by the benchmark normal models, each trained on the identical 27,000 cycle≤3 rows with its own cross-validated hyperparameters (LLKE h = 0.1, B-spline 30 knots, LR, CaBN λ1 = 0). The proposed healthy-range MOGP shows the flattest healthy stretch at a preserved end-of-life swing; LLKE shows coverage spikes, the B-spline isolated large spikes and LR broadband noise. CaBN shares the covariate-adjusted mean of LR, so its residuals — and therefore its RUL — coincide with LR by construction.'));
kids.push(...figure('fig_bench_dev_resid_prop.png', 560, 496, 'Fig. 7.', 'Per-cycle mean residuals of the historical units per normal model and sensor (raw units, seed 0); top row: the proposed stage-2 MOGP.'));
kids.push(cap('Table 4', 'RUL RMSE (cycles) and NASA score of the in-service units of DS03 per truncation level, and running time under one protocol (seed 0, 12 threads; inference on the 87,600-point in-service grid). The downstream is identical for every row; the proposed row differs from the MOGP row only by its training window.'));
kids.push(levelTable(J.ds03.bench, { times: true }));
kids.push(note('All pairwise RUL differences on this cycle≤3 benchmark are within seed noise. When the competitors are retrained on the same healthy-range window as the proposed model (seventeenth opening), they reach at best 7.44 ± 0.11 (LLKE) against 7.18 ± 0.05, and the LLKE difference becomes significant (p = 0.003); the healthy-range retraining improves LLKE, leaves the B-spline unchanged and degrades LR significantly (7.31 → 7.44, p < 0.001), so the benefit of the window is specific to the MOGP.'));

/* ---------------- E ---------------- */
kids.push(H2('E. Sensitive Analysis'));
kids.push(P('Figure 8 sweeps the baseline window of the onset detector at the frozen clip (10σ) on the conditional-gate curves of the historical units. Flight-hour windows of 20–30 h beat every cycle-count window (15 h still trails 12 cycles, 9.42 vs 8.66, and 50 h deteriorates again to 11.95); 30 h is the grid minimum with the tightest per-unit spread and reproduces the final chain value of 5.21.'));
kids.push(...figure('fig_v5_sens_flighth.png', 440, 291, 'Fig. 8(a).', 'Effect of the flight-hour baseline window on the onset RMSE (historical units, per-unit boxes over three seeds).'));
kids.push(...figure('fig_v5_sens_cycle.png', 440, 291, 'Fig. 8(b).', 'Effect of the cycle-count baseline window on the onset RMSE.'));
kids.push(P('Figure 9 maps the leave-one-unit-out RUL RMSE of the historical units over the λ1 × λ2 grid of Table 2 (selection is confined to the historical units, so the in-service units are not re-opened per grid cell). The surface is flat across λ2 ≤ 1 and λ1 ≤ 12 (7.39–7.54) and deteriorates with seed-unstable values once λ1 = 16 or λ2 ≥ 2; the selected (12, 0.25) sits inside the flat region.'));
kids.push(...figure(path.join(HERE, 'cs_fig9_lambda_heatmap.png'), 470, 325, 'Fig. 9.', 'Effect of λ1 and λ2 on the average RUL RMSE (leave-one-unit-out, historical units, final healthy-range tables); the red frame marks the selected recipe.'));

/* ---------------- F ---------------- */
kids.push(H2('F. Ablation Study'));
kids.push(P('Table 5 removes the two main modules. Without the state division the normal model cannot be retrained on a detected healthy range and falls back to the sparse cycle≤3 window (the stage-1 chain); without the residual extraction the raw per-cycle sensor means replace the residuals at the same five-channel interface (no normal model at all). Table 6 deletes the health-index loss terms one at a time; each variant is retrained on the historical units with β by the frozen rule and scored once on the in-service units.'));
kids.push(cap('Table 5', 'Ablation study on the main modules of the proposed method (in-service units of DS03).'));
kids.push(levelTable(J.ds03.abl_main));
kids.push(note('Without the state division = the stage-1 chain (sparse cycle≤3 window, ungated, same recipe); without the residual extraction = raw per-cycle sensor means through the same five-channel interface (recorded at the eleventh opening as 8.31 ± 0.35, shape 0/3; the per-level scores come from a re-summary with identical predictions).'));
kids.push(cap('Table 6', 'Ablation study on the components of the health-index construction (in-service units of DS03).'));
kids.push(levelTable(J.ds03.abl_hi));
kids.push(note('The two level anchors are existence conditions (shape 0/3 on every seed; outright collapse without the end anchor); the tail-flatness term costs 1.1 cycles on the in-service units (p = 0.02).'));

/* ---------------- G: transfer ---------------- */
const FINDINGS = {
  ds08a: 'Findings. The detector itself transfers — the ungated onset RMSE is 7.34 on the historical and 10.79 on the in-service units, and the re-selection returns the 30-flight-hour baseline of DS03 — but the conditional gate adds nothing on this fleet (7.62 / 10.68, paired p = 0.37 / 0.73): with all ten health parameters degrading at once the likelihood drop is sharp enough without coverage filtering. The healthy-range retraining reproduces the residual mechanism (healthy floor × 0.58, SNR 18.7 → 34.2, better in 25 of 27 cases) but not the RUL gain: the stage-1 MOGP chain is the best model on the in-service units (7.28 ± 0.45 against B-spline 8.49, LLKE 9.11, LR 9.34), whereas the stage-2 chain falls to 9.43 ± 0.34 (p = 0.53 against stage 1), losing at the 80 % level (15.39 vs 10.21) where the in-service trajectories reach the failure level early; the same 80 % weakness appears in every model on this short-lived fleet.',
  ds07: 'Findings. The conditional gate is significant on the historical units (7.49 vs 9.11, p = 0.008) but does not carry to the four in-service units (10.90 vs 10.35, p = 0.53), where the sparse branch fires late on units 9 and 10. Residual quality again transfers (healthy floor × 0.29, SNR 22.2 → 77.5, better in 18 of 18 cases) while RUL does not: the stage-2 chain (10.08 ± 0.61, recipe re-frozen to (16, 0.25) on the stage-2 tables) trails the stage-1 chain (9.40 ± 0.22, p = 0.075), and LLKE (8.82) and LR (8.87) beat both MOGP chains on the in-service units; the B-spline collapses (47.07, shape 0/3) because its knots, placed on the cycle≤3 operating range, are extrapolated on the in-service flights. The historical-to-in-service gap is large for every model (about 6.2 → 8.8–10.1), so the four in-service units are harder than the six historical ones.',
};
const SYNTHESIS = 'Across the two transfers the components behave as on DS03 — the MOGP normal model transfers (it is the best in-service model on DS08a), the detector re-selection yields sensible constants, and the healthy-range retraining lowers the healthy residual floor on every sub-dataset and every seed — but the two end-metric gains reported on DS03 do not generalise: the conditional gate improves the onset RMSE only where the historical fleet supports its calibration and even then not on the in-service units, and the stage-2 RUL gain, already not significant on DS03 (p = 0.49), reverses on DS08a and DS07. The residual-quality mechanism is therefore robust, whereas its conversion into RUL accuracy is dataset-specific; both directions are reported as measured.';
const TR = [['ds08a', 'DS08a-009', 'all ten health parameters; 9 historical / 6 in-service units with the DS03 fleet design'],
            ['ds07', 'DS07', 'LPT efficiency and flow; 6 / 4 units; mixed flight classes']];
const have = TR.filter(t => J[t[0]]);
if (have.length) {
  kids.push(H2('G. Transfer to other sub-datasets'));
  kids.push(P('The whole procedure of Section IV-C was repeated on each sub-dataset with the model classes, the pipeline structure, the loss-term types and every selection rule carried over unchanged, and every numeric constant — baseline window, clip, gate thresholds, λ1/λ2, β, normalisation — re-selected on that sub-dataset\'s historical units. The in-service units of each sub-dataset were opened once, after pre-registration, for the onset check, the RUL comparison and the benchmark below. Score sums are not comparable across sub-datasets because the number of in-service units differs.'));
  for (const [k, name, desc] of have) {
    const d = J[k]; const r = d.rule;
    kids.push(P(`${name} — ${desc}.`, { bold: true }));
    kids.push(table(['Hyperparameter', 'Re-selected value'], [
      ['baseline window, clip', `${r.wmode === 'fh' ? r.wval + ' flight-hours' : r.wval + ' cycles'}, ${r.clip} σ`],
      ['gate rule', `V_sparse = ${f2(r.V_sparse)} (nbase < ${r.NB}), V_dense = ${f2(r.V_dense)}` + (r.V_sparse === r.V_dense ? ' (single historical class: one branch uncalibratable, shares the value)' : '')],
      ['λ1, λ2 (stage 1 / stage 2)', `(${d.recipe1[0]}, ${d.recipe1[1]}) / (${d.recipe2[0]}, ${d.recipe2[1]})` + (d.recipe1.join() !== d.recipe2.join() ? ' — re-frozen on the stage-2 tables (stage-1 recipe outside the tie band)' : '')],
      ['historical LOO RUL: stage 1 / stage 2', `${f2(d.dev_hi1[0])} ± ${f2(d.dev_hi1[1])} / ${f2(d.dev_hi2[0])} ± ${f2(d.dev_hi2[1])}`],
      ['onset RMSE, historical: conditional / ungated', `${f2(d.onset.dev_cond)} / ${f2(d.onset.dev_ung)}`],
      ['onset RMSE, in-service: conditional / ungated', `${f2(d.onset.test_cond)} / ${f2(d.onset.test_ung)}`],
      ['stage-2 residual quality (historical)', `healthy floor × ${f2(d.snr.floor_ratio)} (lower in ${d.snr.floor_won}), SNR ${f1(d.snr.snr_c3)} → ${f1(d.snr.snr_hr)} (better in ${d.snr.snr_won})`],
    ], { widths: [3400, 5600] }));
    if (FINDINGS[k]) kids.push(P(FINDINGS[k]));
    kids.push(cap(`Table 4-${name}`, `RUL RMSE and NASA score of the in-service units of ${name} per truncation level (${d.nunits_test} units, three seeds; frozen downstream, cycle≤3 benchmark rows).`));
    kids.push(levelTable(d.bench));
  }
  kids.push(P(SYNTHESIS));
}
const doc = new D.Document({ styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: kids }] });
const out = path.join(HERE, 'Case_study_filled.docx');
D.Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log('written', out); });
