/* FINAL REPORT — complete experimental record of the v5 chain
   (coverage-conditional gate everywhere) + every paper-figure candidate.
   Usage: node make_final_report.js en|ko -> final_report_{en,ko}.docx */
const fs = require('fs');
const D = require('docx');

const LANG = (process.argv[2] || 'en').toLowerCase();
const KO = LANG === 'ko';
const FONT = KO ? 'Malgun Gothic' : 'Times New Roman';
const TW = 9000;
const t = (en, ko) => (KO ? ko : en);

const run = (x, o = {}) => new D.TextRun({ text: x, font: FONT, size: 20, ...o });
const caption = (label, text) => new D.Paragraph({
  spacing: { before: 240, after: 80 },
  children: [run(label + '. ', { bold: true }), run(text)],
});
const note = text => new D.Paragraph({
  spacing: { before: 40, after: 160 },
  children: [run(t('Note. ', '주. ') + text, { italics: true, size: 18, color: '444444' })],
});
const body = (text, o = {}) => new D.Paragraph({
  spacing: { before: 60, after: 120 }, children: [run(text, o)],
});
function cell(content, { header = false, bold = false, w = 1000, align = 'center',
                         size = 18, span = 1 } = {}) {
  return new D.TableCell({
    width: { size: w, type: D.WidthType.DXA },
    columnSpan: span > 1 ? span : undefined,
    shading: header ? { type: D.ShadingType.CLEAR, fill: 'E8E8E8' } : undefined,
    verticalAlign: D.VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    children: [new D.Paragraph({
      alignment: align === 'left' ? D.AlignmentType.LEFT : D.AlignmentType.CENTER,
      children: [run(String(content), { bold: header || bold, size })],
    })],
  });
}
function table(headers, rows, { widths, boldCells = [], leftCol = true } = {}) {
  const n = headers.length;
  const w = widths || Array(n).fill(Math.floor(TW / n));
  return new D.Table({
    width: { size: TW, type: D.WidthType.DXA }, columnWidths: w,
    rows: [new D.TableRow({
      tableHeader: true,
      children: headers.map((h, j) => cell(h, { header: true, w: w[j] })),
    }), ...rows.map((r, i) => new D.TableRow({
      children: r.map((c, j) => cell(c, {
        w: w[j], bold: boldCells.some(([bi, bj]) => bi === i && bj === j),
        align: leftCol && j === 0 ? 'left' : 'center',
      })),
    }))],
  });
}
function figure(file, wPx, hPx, label, text) {
  return [new D.Paragraph({
    alignment: D.AlignmentType.CENTER, spacing: { before: 200, after: 40 },
    children: [new D.ImageRun({
      type: 'png', data: fs.readFileSync(require('path').join(__dirname, file)),
      transformation: { width: wPx, height: hPx },
    })],
  }), new D.Paragraph({
    alignment: D.AlignmentType.CENTER, spacing: { after: 200 },
    children: [run(label + '. ', { bold: true, size: 18 }), run(text, { size: 18 })],
  })];
}
const H = x => new D.Paragraph({
  spacing: { before: 300, after: 100 },
  children: [new D.TextRun({ text: x, font: FONT, bold: true, size: 24 })],
});

const kids = [];
kids.push(new D.Paragraph({
  alignment: D.AlignmentType.CENTER, spacing: { after: 200 },
  children: [run(t(
    'N-CMAPSS DS03 Sparse-Training RUL Chain — Final Report',
    'N-CMAPSS DS03 희소 학습 RUL 체인 — 최종 보고서'), { bold: true, size: 28 })],
}));
kids.push(new D.Paragraph({
  alignment: D.AlignmentType.CENTER, spacing: { after: 200 },
  children: [run(t(
    'Complete experimental record and paper-figure candidates. Final configuration: coverage-conditional confidence gate applied throughout.',
    '전체 실험 기록과 논문 그림 후보. 최종 구성: 커버리지 조건부 신뢰도 게이트 전면 적용.'), { italics: true, size: 20 })],
}));

/* ---------- 0. Protocol ---------- */
kids.push(H(t('0. Protocol', '0. 프로토콜')));
kids.push(body(t(
  'Data: N-CMAPSS DS03-012. Development: 9 run-to-failure units — the only data used for every model choice, sweep and calibration. Test: 6 sealed units, prediction input only. Disclosure — the DS03 test split has been opened twelve times in total: one earlier cycle<5 chain, two by the earlier cycle<3 chain, three by the v4 chain (first rendering, HI recipe re-fix, filter edge refinement — all re-fixed on development data alone), one benchmark batch, one pre-registered verification of the coverage-conditional gate revision, one final-chain re-summary unifying the gate on the RUL path, one pre-registered batch re-verification after the chain re-freeze on the conditional-table lambda grid, one pre-registered batch completing the ablation tables on test, and one scoring-robustness batch on test. Final official numbers in this report are the re-frozen coverage-conditional chain (lambda1 = 12); fixed-threshold and lambda8 history is reported where each selection was originally made.',
  '데이터: N-CMAPSS DS03-012. 개발(dev): run-to-failure 9유닛 — 모든 모델 선택·스윕·캘리브레이션에 사용된 유일한 데이터. 검정(test): 봉인 6유닛, 예측 입력 전용. 공개 — DS03 test 분할은 총 12회 개봉됨: 이전 cycle<5 체인 1회, 이전 cycle<3 체인 2회, v4 체인 3회(1차 렌더링, HI 레시피 재확정, 필터 가장자리 개선 — 전부 dev만으로 재확정), 벤치마크 배치 1회, 커버리지 조건부 게이트 개정의 사전 등록 검증 1회, RUL 경로 게이트 통일을 위한 최종 체인 재요약 1회, 조건부 테이블 λ그리드 재동결 후 사전 등록 배치 재검증 1회, 절제표의 test 완성 배치 1회, 채점 강건성 배치 1회. 본 보고서의 최종 공식 수치는 재동결 커버리지 조건부 체인(λ1 = 12)이며, 각 선정이 이뤄진 지점에서는 고정 임계값·λ8 이력을 함께 보고함.')));
kids.push(body(t(
  'Metrics: (i) held-out zRMSE for model comparison; (ii) onset RMSE = RMS of detected minus true hs-transition, dev 27 cases; (iii) LOO truncation RUL at 20/40/60/80 % of life; (iv) NASA asymmetric score. Methodology rules: sweeps run on absolute axes past their minima until collapse; tie bands instead of nominal minima; selection eligibility requires all three seeds; no blank table cells; results reported regardless of direction.',
  '지표: (i) 모델 비교용 held-out zRMSE; (ii) 온셋 RMSE = (검출−실제 hs 전환)의 RMS, dev 27케이스; (iii) 수명 20/40/60/80 % 절단 LOO RUL; (iv) NASA 비대칭 점수. 방법론 규칙: 스윕은 절대축에서 최소점을 지나 붕괴까지; 명목 최소가 아닌 동률밴드; 선정 자격은 3시드 전부 완주; 표에 빈칸 금지; 결과는 유불리 무관 보고.')));

/* ---------- 1. Framework ---------- */
kids.push(H(t('1. Overall framework', '1. 전체 프레임워크')));
kids.push(...figure('fig_blockdiagram.png', 660, 273, 'Figure 1',
  t('Overall framework of the proposed chain: normal-behaviour modelling, residuals and confidence, degradation diagnosis, health assessment and prognosis. Shaded blocks are learned or calibrated components.',
    '제안 체인의 전체 프레임워크: 정상거동 모델링, 잔차·신뢰도, 열화 진단, 건강 평가·예지. 음영 블록은 학습·캘리브레이션 구성요소.')));

/* ---------- 2. Sensor screening ---------- */
kids.push(H(t('2. Sensor sensitivity screening (model-free)', '2. 센서 민감도 스크리닝 (모델 무관)')));
kids.push(caption('Table 1', t(
  'Degradation-sensitivity screening of all 14 sensors under the cycle<=3 window (kNN k=10 healthy reference over standardized operating conditions; zRMS of degraded residuals in healthy-noise units — an insensitive sensor scores ~1). Candidate sets are the top-3/5/7.',
  '전체 14개 센서의 열화 민감도 스크리닝, cycle≤3 창 기준 (표준화 운전조건 위 kNN k=10 정상 기준모델; 열화 잔차의 zRMS, 정상 잡음 단위 — 무감 센서는 ~1점). 후보 세트는 top-3/5/7.')));
kids.push(table(
  [t('Rank', '순위'), '1', '2', '3', '4', '5', '6', '7', '8', '9–14'],
  [[t('Sensor', '센서'), 'T50', 'T48', 'Wf', 'Nc', 'T30', 'P40', 'Ps30', 'Nf', t('others', '기타')],
   ['zRMS', '2.99', '2.17', '1.19', '1.17', '1.06', '1.06', '1.05', '1.03', '≤1.02']],
  { widths: Array(10).fill(900), boldCells: [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]] }));
kids.push(...figure('fig_v4_sensitivity.png', 600, 310, 'Figure 2',
  t('Sensor sensitivity ranking. Blue bars are the finally selected set.',
    '센서 민감도 순위. 파랑 막대가 최종 선택 세트.')));

/* ---------- 3. Normal-model grid ---------- */
kids.push(H(t('3. Normal model: kernel x rank x sensor-set grid', '3. 정상모델: 커널 × rank × 센서세트 그리드')));
kids.push(...figure('fig_v4_grid_heatmap.png', 650, 191, 'Figure 3',
  t('Full grid, held-out zRMSE: {RBF, Matern 3/2, Matern 1/2} x rank {1,2,3} x candidate set {3,5,7}, 3 seeds, one shared Blues scale capped at 0.020. The black frame marks the selection. Matern 5/2 is excluded for chronic float32 Cholesky failures.',
    '전면 그리드, held-out zRMSE: {RBF, Matérn 3/2, Matérn 1/2} × rank {1,2,3} × 후보 세트 {3,5,7}, 3시드, 0.020 상한의 공용 Blues 스케일. 검정 테두리가 선정. Matérn 5/2는 만성 float32 Cholesky 실패로 제외.')));
kids.push(note(t(
  'Top-5 [T30, T48, T50, Nc, Wf] is best under both selection rules — the grid tie band (minimum mean, one-seed-s.d. band, simplest inside; RBF rank 1 in every set) and the downstream detector sweep; pairwise differences are not significant, so the decision rests on the consistent minimum, simplicity, and the numerical instability of top-3. Final model: Vecchia MOGP, RBF rank 1, m = 18 neighbours, trained on flight cycles 1..3 only, 27,000 stratified rows per seed.',
  'Top-5 [T30, T48, T50, Nc, Wf]가 두 선정 규칙 — 그리드 동률 밴드(평균 최소 → 시드 sd 1배 밴드 → 밴드 내 최단순; 전 세트에서 RBF rank 1)와 하류 검출기 스윕 — 모두에서 최선; 쌍대 차이는 비유의하므로 결정 근거는 일관된 최소 + 단순성 + top-3의 수치 불안정. 최종 모델: Vecchia MOGP, RBF rank 1, 이웃 m = 18, 비행 사이클 1..3만으로 학습, 시드당 층화 27,000행.')));

/* ---------- 4. Residuals ---------- */
kids.push(H(t('4. Residual and confidence generation', '4. 잔차·신뢰도 생성')));
kids.push(...figure('fig_slide_resid.png', 600, 300, 'Figure 4',
  t('Residual construction on unit u5: measured T48 cycle means against the GP prediction; the shaded gap is the residual r = x − x̂. The GP tracks operating conditions but not degradation, so the gap grows with damage.',
    'u5의 잔차 구성: 실측 T48 사이클 평균 vs GP 예측 — 음영 간격이 잔차 r = x − x̂. GP는 운전조건은 추적하지만 열화는 모르므로 간격이 손상과 함께 커짐.')));
kids.push(...figure('fig_resid_curves.png', 600, 300, 'Figure 5',
  t('Per-cycle mean residuals of unit u5 over the five selected channels, standardized by the mean and s.d. of the first ten cycles. The healthy stretch stays near zero on every channel; after damage begins, the hot-section channels T48 and T50 drift upward, Nc downward, Wf upward — all five channels stay alive over the extrapolated life.',
    'u5의 다섯 선택 채널 사이클 평균 잔차 — 첫 10사이클의 평균·표준편차로 표준화. 정상 구간은 전 채널이 0 부근; 손상 시작 후 고온부 T48·T50은 상승, Nc는 하강, Wf는 상승 — 외삽 수명 전체에서 다섯 채널이 모두 살아 있음.')));
kids.push(...figure('fig_slide_gate.png', 600, 300, 'Figure 6',
  t('Confidence index detcov per sampled point, unit u5. detcov = −(1/2) log det Sigma(w) is computed from operating conditions only. Points below the applicable threshold V are dropped before the cycle summary; the shaded band is the 30 flight-hour baseline window.',
    '표본점별 신뢰도 지수 detcov, u5. detcov = −½ log det Σ(w)는 운전조건만으로 계산됨. 해당 임계값 V 미만 점은 사이클 요약 전에 제거; 음영 구간은 30 비행시간 기준선 창.')));

/* ---------- 5. Gate ---------- */
kids.push(H(t('5. Coverage-conditional confidence gate (final)', '5. 커버리지 조건부 신뢰도 게이트 (최종)')));
kids.push(body(t(
  'Final rule: count the flight cycles inside the 30 flight-hour baseline window; if fewer than 10, apply V = 19.0, otherwise V* = 20.2. Both constants are absolute development-calibrated values, transferred without re-derivation. Selection history, disclosed in order: (1) the absolute-axis sweep selected V* = 20.2 (sweep minimum, sole tie-band member, all seeds); (2) the test verification exposed a long-haul reversal, diagnosed to sigma0 inflation on sparse baselines (the gate keeps confident, not accurate, points within an isolated cycle); (3) the class-stratified re-scoring showed the long-haul optimum is a plateau at 18.8–19.2, and the conditional rule was frozen on development data (seed and leave-one-unit-out stability) and verified once on test.',
  '최종 규칙: 30 비행시간 기준선 창의 사이클 수를 세어 10 미만이면 V = 19.0, 아니면 V* = 20.2. 두 상수 모두 dev 캘리브레이션 절대값이며 재산출 없이 이식. 선정 이력(순서대로 공개): (1) 절대축 스윕이 V* = 20.2 선정(스윕 최소·동률밴드 단독·전 시드); (2) test 검정에서 장거리 반전이 드러났고, 희박 기준선의 σ0 팽창(게이트는 고립 사이클 안에서 정확한 점이 아니라 확신하는 점을 선택)으로 진단됨; (3) 클래스 층화 재채점이 장거리 최적이 18.8~19.2 평탄대임을 보였고, 조건부 규칙을 dev에서 동결(시드·유닛제외 안정성)한 뒤 test에서 1회 검증.')));
kids.push(...figure('fig_vsweep_class.png', 560, 350, 'Figure 7',
  t('Class-stratified gate sweep. One absolute V trades the classes off: the short+medium optimum is 20.2 while the long-haul optimum is a plateau at 18.8–19.2 whose centre 19.0 becomes V sparse. Dotted lines are the ungated references.',
    '클래스 층화 게이트 스윕. 절대 V 하나는 두 클래스를 맞바꿈: 단·중거리 최적은 20.2, 장거리 최적은 18.8~19.2 평탄대이며 그 중심 19.0이 V sparse. 점선은 무게이트 기준선.')));
kids.push(caption('Table 2', t(
  'One-time test verification of the conditional gate (eighth opening, pre-registered): onset RMSE by flight class. The sparse rule fires on test units u10/u11/u13 (7–9 baseline cycles).',
  '조건부 게이트의 1회 test 검증 (8번째 개봉, 사전 등록): 클래스별 온셋 RMSE. 희박 규칙은 test u10/u11/u13(기준선 7~9사이클)에서 발동.')));
kids.push(table(
  [t('Detector', '검출기'), t('Dev overall', 'dev 전체'), t('Dev short+med', 'dev 단·중'),
   t('Dev long', 'dev 장거리'), t('Test overall', 'test 전체'),
   t('Test short+med', 'test 단·중'), t('Test long', 'test 장거리')],
  [[t('Conditional gate (final)', '조건부 게이트 (최종)'), '5.21', '4.75', '6.60', '8.54', '5.54', '10.73'],
   [t('Ungated', '무게이트'), '6.78', '6.51', '7.65', '9.01', '5.73', '11.38']],
  { widths: [2400, 1150, 1150, 1050, 1150, 1150, 950],
    boldCells: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]] }));
kids.push(note(t(
  'The conditional gate beats the ungated reference on every subset of both splits. Mechanism verified: the sigma0 inflation that a single absolute threshold suffers on the sparse-baseline units u10/u11 (2.54x/2.77x) collapses to 0.91x/0.79x under the conditional rule. Dev advantage over ungated is significant (p = 0.003, 27 cases); test subsets hold 9 cases each and are descriptive. The rule is label-free and deployable: the trigger is the observable baseline cycle count.',
  '조건부 게이트가 양 분할의 모든 부분집합에서 무게이트 기준을 이김. 메커니즘 검증: 단일 절대 임계값이 희박 기준선 유닛 u10/u11에서 겪던 σ0 팽창(2.54×/2.77×)이 조건부 규칙에서 0.91×/0.79×로 소멸. dev 우위는 유의(p = 0.003, 27케이스); test 부분집합은 각 9케이스로 기술 통계. 규칙은 라벨프리·배포 가능 — 트리거가 관측 가능한 기준선 사이클 수임.')));

/* ---------- 6. Onset detector + sensitivity ---------- */
kids.push(H(t('6. Onset detector and its sensitivity analysis', '6. 온셋 검출기와 민감도 분석')));
kids.push(body(t(
  'Detector, frozen: per-cycle q75 of the gated log-likelihood curve; baseline = first 30 flight-hours; clip at mu0 − 10 sigma0; full-range mean-drop change point. Final dev onset RMSE 5.21 cycles.',
  '검출기(동결): 게이트 로그가능도 곡선의 사이클 q75; 기준선 = 첫 30 비행시간; 클립 μ0 − 10σ0; 전구간 mean-drop 변화점. 최종 dev 온셋 RMSE 5.21 사이클.')));
kids.push(caption('Table 3', t(
  'Baseline-window sensitivity at the frozen clip 10 sigma, re-scored on the conditional-gate curves (gate rule frozen as deployed). 30 h — originally frozen on the fixed-gate grid — remains the grid minimum and reproduces the final chain value.',
  '동결 클립 10σ에서의 기준선 창 민감도 — 조건부 게이트 곡선으로 재채점(게이트 규칙은 배포형 그대로 동결). 고정 게이트 그리드에서 동결됐던 30h가 여기서도 그리드 최소이며 최종 체인 수치를 재현.')));
kids.push(table(
  [t('Window', '창'), '3cy', '4cy', '5cy', '6cy', '8cy', '12cy', '20cy', '10h', '15h', '20h', '25h', '30h', '50h'],
  [[t('RMSE', 'RMSE'), '20.01', '17.91', '14.66', '10.31', '10.47', '8.66', '10.21',
    '12.40', '9.42', '8.24', '6.02', '5.21', '11.95']],
  { widths: [1000, ...Array(13).fill(615)] }));
kids.push(...figure('fig_v5_sens_cycle.png', 520, 344, 'Figure 8',
  t('Onset RMSE per cycle-count baseline window at clip 10 sigma. Each box holds the per-unit RMSE over 3 seeds, 9 units per box; whiskers span the full range.',
    '클립 10σ에서 사이클 수 기준선 창별 온셋 RMSE. 상자 안 값은 유닛별 RMSE(3시드 집계), 상자당 9유닛; 수염은 전 범위.')));
kids.push(...figure('fig_v5_sens_flighth.png', 520, 344, 'Figure 9',
  t('Onset RMSE per flight-hour baseline window. Hour-based windows beat every cycle-count window from 15 h upward; 30 h is the grid minimum with the tightest per-unit spread.',
    '비행시간 기준선 창별 온셋 RMSE. 15h 이상의 시간 창이 모든 사이클 수 창을 이김; 30h가 그리드 최소이자 유닛별 산포 최소.')));

/* ---------- 7. HI ---------- */
kids.push(H(t('7. Health-index network', '7. 건강지수(HI) 네트워크')));
kids.push(body(t(
  'Inputs: per-flight trim25 of gate-passing residuals, z-normalized with the pooled development mean and s.d.; MLP with shape losses; width-3 median post-filter with a non-decreasing endpoint (dev-justified on the conditional tables: LOO 7.37 -> 7.22). Recipe provenance, disclosed in order: (8, 0.25) was selected on the fixed-gate tables and initially inherited; the re-sweep on the conditional-gate tables (grid below) promotes (12, 0.25) — the lambda1 = 12 column passes the shape gate 3/3 only on the conditional tables — and the chain was re-frozen on the composite winner before a single pre-registered test verification. The lambda8 conditional chain is retained as disclosed history: dev LOO 7.10 +/- 0.07, paired difference to the re-frozen chain not significant (p = 0.39). Final chain: lambda1 = 12, lambda2 = 0.25, dev LOO 7.22 +/- 0.08, beta per seed 30/30/35.',
  '입력: 게이트 통과 잔차의 비행별 trim25를 dev 풀링 평균·표준편차로 z정규화; 형태 손실의 MLP; 폭 3 중앙값 후처리(끝점 비감소; 조건부 테이블 dev 정당화 — LOO 7.37 → 7.22). 레시피 출처(순서대로 공개): (8, 0.25)는 고정 게이트 테이블에서 선정되어 처음에는 계승됐으나, 조건부 게이트 테이블 재스윕(아래 그리드)에서 (12, 0.25)가 승격 — λ1 = 12 열은 조건부 테이블에서만 형태 게이트 3/3 통과 — 하여 합성점수 승자로 체인을 재동결한 뒤 단일 사전등록 test 검증을 수행. λ8 조건부 체인은 이력으로 공개: dev LOO 7.10 ± 0.07, 재동결 체인과의 쌍대 차이 비유의(p = 0.39). 최종 체인: λ1 = 12, λ2 = 0.25, dev LOO 7.22 ± 0.08, β 시드별 30/30/35.')));
kids.push(caption('Table 4', t(
  'Frozen loss terms and the two tuned shape weights.',
  '동결 손실 항과 튜닝 대상인 두 형태 가중치.')));
kids.push(table(
  [t('Loss term', '손실 항'), t('Parameter', '파라미터'), t('Value', '값'), t('Status', '상태')],
  [[t('Initial level', '초기 레벨'), 'lambda0, thr', '1, 0.20', t('frozen', '동결')],
   [t('End level', '종점 레벨'), 'end_target', '1.03', t('frozen', '동결')],
   [t('Tail flatness', '말기 평탄화 방지'), 'w, m', '800, 0.004', t('frozen', '동결')],
   [t('Monotonicity', '단조성'), 'lambda1', '{2, 4, 6, 8, 12, 16}', t('tuned', '튜닝')],
   [t('Convexity', '볼록성'), 'lambda2', '{0.25, 0.5, 1, 2, 4}', t('tuned', '튜닝')]],
  { widths: [2300, 1900, 2100, 2700] }));
kids.push(caption('Table 5', t(
  'lambda1 x lambda2 property grid, re-swept on the conditional-gate tables (9 dev units x 3 seeds). Mono = 100 x sum of negative increments; Curv = 100 x sum of second differences; Rng = mean(end − start); Pass = seeds passing the shape gate — below 3/3 is excluded regardless of any other number. Winner (12, 0.25) by the property composite among 3/3 passers; the lambda1 = 12 column passes 3/3 only on the conditional tables, which is what moves the winner from the fixed-gate choice (8, 0.25).',
  'λ1 × λ2 성질 그리드 — 조건부 게이트 테이블로 재스윕 (dev 9유닛 × 3시드). Mono = 음의 증분 합 × 100; Curv = 2차 차분 합 × 100; Rng = 평균(끝 − 시작); Pass = 형태 기준 통과 시드 수 — 3/3 미만은 다른 수치와 무관하게 제외. 3/3 통과 조합 중 성질 합성점수 1위 (12, 0.25) 선정; λ1 = 12 열은 조건부 테이블에서만 3/3을 통과하며, 이것이 고정 게이트 선정 (8, 0.25)에서 승자가 이동한 이유.')));
{
  const L2S = ['0.25', '0.5', '1.0', '2.0', '4.0'];
  const P = {
    '2':  { '0.25': ['-26.2','3.76','0.757','3/3'], '0.5': ['-25.6','3.80','0.753','3/3'],
            '1.0': ['-24.9','3.84','0.745','3/3'], '2.0': ['-24.0','3.86','0.731','3/3'],
            '4.0': ['-22.2','3.82','0.703','3/3'] },
    '4':  { '0.25': ['-24.8','3.90','0.749','3/3'], '0.5': ['-24.5','3.91','0.745','3/3'],
            '1.0': ['-24.1','3.91','0.738','3/3'], '2.0': ['-23.2','3.91','0.723','3/3'],
            '4.0': ['-21.3','3.85','0.694','1/3'] },
    '6':  { '0.25': ['-24.0','3.96','0.740','3/3'], '0.5': ['-23.8','3.96','0.737','3/3'],
            '1.0': ['-23.4','3.96','0.730','3/3'], '2.0': ['-22.5','3.92','0.715','3/3'],
            '4.0': ['-20.5','3.86','0.686','0/3'] },
    '8':  { '0.25': ['-23.3','4.00','0.733','3/3'], '0.5': ['-23.1','4.00','0.729','3/3'],
            '1.0': ['-22.6','3.98','0.722','3/3'], '2.0': ['-21.0','3.97','0.707','3/3'],
            '4.0': ['-19.9','3.86','0.679','0/3'] },
    '12': { '0.25': ['-21.4','4.05','0.716','3/3'], '0.5': ['-21.2','4.04','0.713','3/3'],
            '1.0': ['-20.9','4.01','0.706','3/3'], '2.0': ['-20.1','3.95','0.692','0/3'],
            '4.0': ['-18.4','3.84','0.663','0/3'] },
    '16': { '0.25': ['-20.1','4.04','0.702','2/3'], '0.5': ['-19.8','4.02','0.698','1/3'],
            '1.0': ['-19.4','3.99','0.691','0/3'], '2.0': ['-18.5','3.93','0.676','0/3'],
            '4.0': ['-17.2','3.77','0.649','0/3'] },
  };
  const TW2 = 9360;
  const lw = 700, pw = Math.floor((TW2 - lw) / 20);
  const props = ['Mono', 'Curv', 'Rng', 'Pass'];
  const hdr1 = new D.TableRow({ tableHeader: true, children: [
    cell(t('Tuning', '파라미터'), { header: true, w: lw, size: 13 }),
    ...L2S.map(v => cell(`λ2 = ${v}`, { header: true, w: pw * 4, span: 4, size: 13 })),
  ]});
  const hdr2 = new D.TableRow({ tableHeader: true, children: [
    cell(t('Properties', '성질'), { header: true, w: lw, size: 13 }),
    ...L2S.flatMap(() => props.map(p => cell(p, { header: true, w: pw, size: 12 }))),
  ]});
  const rows = Object.keys(P).map(l1 => new D.TableRow({ children: [
    cell(`λ1 = ${l1}`, { w: lw, size: 13, align: 'left', bold: l1 === '12' }),
    ...L2S.flatMap(l2 => P[l1][l2].map(v =>
      cell(v, { w: pw, size: 12, bold: l1 === '12' && l2 === '0.25' }))),
  ]}));
  kids.push(new D.Table({
    width: { size: TW2, type: D.WidthType.DXA },
    columnWidths: [lw, ...Array(20).fill(pw)],
    rows: [hdr1, hdr2, ...rows],
  }));
}
kids.push(...figure('fig_v5_hi_curves.png', 560, 350, 'Figure 10',
  t('HI trajectories of the 9 development units, seed 0, conditional-gate tables.',
    'dev 9유닛의 HI 궤적, 시드 0, 조건부 게이트 테이블.')));
kids.push(...figure('fig_v5_hi_curves_test.png', 560, 350, 'Figure 11',
  t('HI trajectories of the 6 test units — dev-trained network with the same normalization; test data enters only as input.',
    'TEST 6유닛의 HI 궤적 — dev 학습 네트워크와 동일 정규화; test 데이터는 입력으로만 사용.')));

/* ---------- 8. RUL ---------- */
kids.push(H(t('8. RUL prediction and final test verification', '8. RUL 예측과 최종 test 검정')));
kids.push(body(t(
  'Stage 5: Bayesian exponential regression on the cycle axis, prior from the development HI curves, beta per seed by the frozen dev-LOO NASA rule (all seeds select 30); RUL = first passage of HI = 1 minus the truncation point.',
  'Stage 5: 사이클 축 베이지안 지수 회귀, 사전분포는 dev HI 곡선, β는 동결 dev-LOO NASA 규칙으로 시드별 선정(전 시드 30); RUL = HI = 1의 first-passage − 절단 시점.')));
kids.push(caption('Table 6', t(
  'Final-chain truncation RUL (coverage-conditional gate throughout). Dev = LOO over 9 units; test = 6 sealed units, everything frozen beforehand. 3 seeds.',
  '최종 체인 절단 RUL (커버리지 조건부 게이트 전면 적용). dev = 9유닛 LOO; test = 봉인 6유닛, 전부 사전 동결. 3시드.')));
kids.push(table(
  ['', '20 %', '40 %', '60 %', '80 %', t('Overall RMSE', '전체 RMSE'), 'NASA'],
  [[t('dev LOO', 'dev LOO'), '10.60', '7.50', '5.02', '3.82', '7.22 ± 0.08', '27.3 ± 0.8'],
   ['test', '11.05', '8.75', '4.67', '2.47', '7.52 ± 0.09', '19.2 ± 0.5']],
  { boldCells: [[1, 5]] }));
kids.push(note(t(
  'The test RMSE sits within 0.30 cycles of the development reference with matching truncation profiles and the per-seed beta carried over unchanged — the chain transfers with every constant frozen. The tenth opening also confirms both insensitivities: to the gate (ungated test 7.47 vs conditional 7.52, p = 0.40) and to the lambda re-freeze (lambda8 chain 7.44 vs 7.52, p = 0.72) — the gate is a detection component, and the recipe choice moves the end metric within noise.',
  'test RMSE는 dev 참고치와 0.30 사이클 이내, 절단 프로파일 일치, 시드별 β 그대로 이식 — 모든 상수를 동결한 채 체인이 이식됨. 10번째 개봉은 두 둔감성도 확인함: 게이트 둔감 (무게이트 test 7.47 vs 조건부 7.52, p = 0.40), λ 재동결 둔감 (λ8 체인 7.44 vs 7.52, p = 0.72) — 게이트는 검출 구성요소이고, 레시피 선택은 끝단 지표를 잡음 범위 안에서만 움직임.')));
kids.push(...figure('fig_v5_rul_box.png', 520, 347, 'Figure 12',
  t('Test RUL RMSE by truncation. Each box holds the per-unit RMSE over 3 seeds, 6 test units per box; whiskers span the full range. Rendered from the re-frozen chain test verification (tenth opening).',
    '절단 시점별 test RUL RMSE. 상자 안 값은 유닛별 RMSE(3시드 집계), 상자당 6유닛; 수염은 전 범위. 재동결 체인 test 검증(10번째 개봉)으로 렌더링.')));

/* ---------- 9. Benchmark ---------- */
kids.push(H(t('9. Benchmark against alternative normal models', '9. 대안 정상모델 벤치마크')));
kids.push(body(t(
  'Competitors: LLKE (h = 0.1), B-spline (30 knots), LR, CaBN (lambda_1 = 0 selected — bit-identical residuals to LR). Component-swap convention: only the normal model changes; the downstream is byte-identical. Competitors run ungated — their leverage-based detcov is degenerate, so the gate cannot be constructed on them.',
  '경쟁: LLKE(h = 0.1), B-spline(매듭 30), LR, CaBN(λ1 = 0 선정 — 잔차가 LR과 비트 동일). component-swap 관행: 정상모델만 교체, 하류는 바이트 동일. 경쟁 모델은 무게이트 — 레버리지 기반 detcov가 퇴화해 게이트 구성이 불가.')));
kids.push(caption('Table 7', t(
  'Experiment 1 — residual quality over the full development life (each model in its own healthy-noise units).',
  '실험 1 — dev 전 수명 잔차 품질 (각 모델 자신의 정상 잡음 단위).')));
kids.push(table(
  [t('Model', '모델'), t('Separation median', '분리도 중앙값'), t('Worst unit', '최악 유닛'),
   t('Usable channels', '가용 채널'), t('Artefacts', '인공물')],
  [['MOGP', 'x7.1', 'x4.5', '5 / 5', t('early low-coverage wobble, handled by the gate', '초반 저커버리지 출렁임 — 게이트가 처리')],
   ['LLKE', 'x4.7', 'x0.8', '~2 / 5', t('x100-300 spikes across the whole life', '수명 전체에 x100~300 스파이크')],
   ['B-spline', 'x10.4', 'x1.7', '3-4 / 5', t('isolated x100 spikes; Wf channel lost', '고립 x100 스파이크; Wf 채널 상실')],
   ['LR/CaBN', 'x5.4', 'x3.5', '2 / 5', t('T30, Nc, Wf buried in noise', 'T30·Nc·Wf가 잡음에 매몰')]],
  { widths: [1300, 1700, 1500, 1500, 3000], boldCells: [[0, 0], [0, 2], [0, 3]] }));
kids.push(...figure('fig_bench_dev_resid_raw.png', 600, 531, 'Figure 13',
  t('Per-cycle mean residual per model and sensor, raw units, 9 development units, seed 0.',
    '모델·센서별 사이클 평균 잔차, 원단위, dev 9유닛, 시드 0.')));
kids.push(...figure('fig_bench_test_resid_raw.png', 600, 531, 'Figure 14',
  t('Per-cycle mean residual per model and sensor on the test units, raw units, seed 0.',
    'test 유닛의 모델·센서별 사이클 평균 잔차, 원단위, 시드 0.')));
kids.push(caption('Table 8', t(
  'Experiment 2 — test truncation RUL with the frozen downstream, and running time per seed. The MOGP row is the final conditional-gate chain.',
  '실험 2 — 동결 하류의 test 절단 RUL과 시드당 실행 시간. MOGP 행은 최종 조건부 게이트 체인.')));
kids.push(table(
  [t('Model', '모델'), '20 %', '40 %', '60 %', '80 %', t('Overall RMSE', '전체 RMSE'),
   'NASA', t('Train s', '학습 s'), t('Infer s', '추론 s')],
  [[t('MOGP, conditional gate', 'MOGP, 조건부 게이트'), '11.05', '8.75', '4.67', '2.47', '7.52 ± 0.09', '19.2 ± 0.5', '921.3', '9.1'],
   [t('MOGP ungated', 'MOGP 무게이트'), '11.09', '8.47', '4.74', '2.44', '7.47 ± 0.06', '19.5 ± 0.8', '921.3', '9.1'],
   ['LLKE', '11.14', '8.65', '5.21', '3.07', '7.67 ± 0.07', '19.4 ± 0.3', '2.9', '15.3'],
   ['B-spline', '11.12', '7.95', '4.62', '2.44', '7.32 ± 0.11', '18.6 ± 0.2', '0.1', '0.2'],
   ['LR = CaBN', '10.85', '8.24', '4.66', '2.57', '7.31 ± 0.10', '18.3 ± 0.5', '< 0.1', '< 0.1']],
  { widths: [1900, 750, 750, 750, 750, 1450, 1150, 800, 700], boldCells: [[0, 0], [0, 5]] }));
kids.push(...figure('fig_bench_rul_box.png', 560, 333, 'Figure 15',
  t('Test RUL RMSE per unit and model. All models tie: no pairwise difference is significant (p >= 0.08 over 72 paired predictions), and stripping the robust downstream (trim25, median filter) collapses no competitor — the insensitivity of the end metric is structural.',
    '유닛·모델별 test RUL RMSE. 전 모델 동률: 어떤 쌍대 차이도 비유의(72예측 p ≥ 0.08)하고, 로버스트 하류(trim25·중앙값 필터)를 제거해도 붕괴하는 경쟁 모델이 없음 — 끝단 지표의 둔감성은 구조적.')));
/* ---------- 10. Ablations ---------- */
kids.push(H(t('10. Ablation studies (dev-trained variants, scored on the sealed test units)', '10. 절제 연구 (dev 학습 변형, 봉인 test 채점)')));
kids.push(caption('Table 9', t(
  'Component ablations against the final conditional-gate chain. Every variant is trained on development data only; test enters as prediction input (eleventh opening).',
  '최종 조건부 게이트 체인 대비 구성요소 절제. 모든 변형은 dev로만 학습; test는 예측 입력으로만 사용 (11번째 개봉).')));
kids.push(table(
  [t('Ablation', '절제'), t('Chain, test', '체인 (test)'), t('Ablated, test', '절제 후 (test)')],
  [[t('Remove the gate', '게이트 제거'),
    t('onset 8.54 / RUL 7.52', '온셋 8.54 / RUL 7.52'),
    t('onset 9.01 / RUL 7.47 (p = 0.40)', '온셋 9.01 / RUL 7.47 (p = 0.40)')],
   [t('Raw sensor means replace residuals', '잔차를 원시 센서 평균으로 대체'),
    'RUL 7.52 ± 0.09', t('RUL 8.31 ± 0.35 (+11 %), NASA 22.6', 'RUL 8.31 ± 0.35 (+11 %), NASA 22.6')],
   [t('Healthy-range window replaces cycle<=3', '학습창을 검출 정상범위로 교체'),
    'RUL 7.52 ± 0.09', t('RUL 7.26 ± 0.08 (p = 0.82)', 'RUL 7.26 ± 0.08 (p = 0.82)')]],
  { widths: [3600, 2500, 2900] }));
kids.push(caption('Table 10', t(
  'HI loss-term ablation (delete one term at a time; dev-trained on the conditional tables, test truncation RUL).',
  'HI 손실 항 절제 (한 항씩 삭제; 조건부 테이블 dev 학습, test 절단 RUL).')));
kids.push(table(
  [t('Variant', '변형'), t('RUL RMSE', 'RUL RMSE'), 'NASA', '20 %', '40 %', '60 %', '80 %'],
  [[t('full recipe', '전체 레시피'), '7.52 ± 0.09', '19.2', '11.05', '8.75', '4.67', '2.47'],
   [t('without the initial-level term', '초기 레벨 항 제거'), '18.89 ± 0.98', '90.5', '23.26', '19.77', '16.74', '14.81'],
   [t('without the end-level term', '종점 레벨 항 제거'), '40.71 ± 0.50', '1740', '38.38', '41.13', '39.28', '43.85'],
   [t('without the tail-flatness term', '말기 평탄화 항 제거'), '7.81 ± 0.16', '19.7', '10.77', '9.57', '5.00', '3.35']],
  { widths: [2700, 1750, 1150, 850, 850, 850, 850], boldCells: [[0, 0], [0, 1]] }));
kids.push(caption('Table 11', t(
  'Downstream armor-removal grid on the sealed test units (dev-trained; every row on the re-frozen lambda12 downstream; the gated MOGP row uses the final conditional gate, all other rows ungated). Summary {trim25, mean} x HI filter {median-3, none}.',
  '봉인 test 유닛의 하류 무장해제 그리드 (dev 학습; 전 행이 재동결 λ12 하류; MOGP 게이트 행은 최종 조건부 게이트, 나머지 행은 무게이트). 요약 {trim25, mean} × HI 필터 {med3, 없음}.')));
kids.push(table(
  [t('Model', '모델'), 'trim25+med3', t('trim25 only', 'trim25만'), 'mean+med3', t('mean only', 'mean만')],
  [[t('MOGP, conditional gate', 'MOGP, 조건부 게이트'), '7.52', '7.43', '7.61', '7.60'],
   [t('MOGP ungated', 'MOGP 무게이트'), '7.47', '7.54', '7.46', '7.44'],
   ['LLKE', '7.67', '7.52', '7.61', '7.47'],
   ['B-spline', '7.32', '7.27', '7.63', '7.87'],
   ['LR = CaBN', '7.31', '7.35', '7.54', '7.51']],
  { widths: [2200, 1700, 1700, 1700, 1700] }));
kids.push(note(t(
  'All 24 test cells sit in the narrow 7.27-7.87 band and no model collapses: the RUL tie does not depend on the robust downstream — the insensitivity of the end metric is deep, absorbed by the pooled z-normalization scale, the HI network and the exponential first passage. Together, Tables 9–11 close the loop on test: the two HI level anchors are existence conditions (18.9 / 40.7 without them), residualization contributes +11 %, tail-flatness +0.29, and the gate remains a detection component (onset 8.54 vs 9.01; RUL p = 0.40). Reported as-is: the median filter (dev-justified +0.15) and the sparse training window (dev-justified +0.39 over healthy-range retraining) both reverse within noise on test (7.43 without the filter; 7.26 with the healthy-range window, p = 0.82) — further instances of the structural end-metric insensitivity; their justification is development-side by protocol.',
  'test 24셀 전부가 7.27~7.87 좁은 밴드에 있고 붕괴 모델 없음: RUL 동률은 로버스트 하류 덕이 아니며 — 끝단 지표의 둔감성은 풀링 z정규화 스케일·HI 네트워크·지수 first-passage가 흡수하는 깊은 구조임. Table 9~11이 test에서 폐루프를 완성함: HI 레벨 앵커 2개는 존립 조건(제거 시 18.9 / 40.7), 잔차화 +11 %, 말기 평탄화 +0.29, 게이트는 검출 구성요소로 유지(온셋 8.54 vs 9.01; RUL p = 0.40). 유불리 무관 보고: dev에서 정당화된 중앙값 필터(+0.15)와 희소 학습창(정상범위 재학습 대비 +0.39) 모두 test에서 잡음 범위 내 반전(필터 미적용 7.43; 정상범위창 7.26, p = 0.82) — 끝단 지표의 구조적 둔감성의 추가 사례이며, 두 선택의 정당화는 프로토콜상 dev 측에 있음.')));

/* ---------- 11. Conclusions ---------- */
kids.push(H(t('11. Conclusions', '11. 결론')));
kids.push(body(t(
  '(1) GP-based degradation modelling under varying operating conditions: the sparse-window Vecchia MOGP is the only benchmarked model with all five degradation channels alive over the full extrapolated life, the best worst-unit separation, a stable predictive distribution (fold s.d. 0.27 vs LLKE 7.46) and a usable confidence index; the full chain transfers to sealed test data with every constant frozen (dev 7.22 -> test 7.52). (2) State-change detection on the conditional likelihood: the GPR chain detects the labelled degradation onset at 5.21 cycles on development and 8.54 on the sealed test units. (3) Uncertainty-based filtering: the coverage-conditional gate — strength conditioned on the baseline sample support — improves onset on both splits and every flight class, with the failure mode of the fixed threshold diagnosed (sigma0 inflation, confidence is not accuracy) and repaired by a label-free deployable rule. (4) Honest negative finding: the truncation-RUL end metric does not discriminate normal models, before or after downstream armor removal — RUL evaluations alone cannot justify a normal-model choice; the discriminating evidence lives in residual quality and in the confidence machinery that only the GP provides.',
  '(1) 운전조건 가변 하의 GP 열화 모델링: 희소 창 Vecchia MOGP는 벤치마크에서 전 외삽 수명 동안 다섯 열화 채널이 모두 살아 있는 유일한 모델이며, 최악 유닛 분리도·안정 예측분포(폴드 sd 0.27 vs LLKE 7.46)·사용 가능한 신뢰도 지수를 갖추고, 모든 상수를 동결한 채 봉인 test로 이식됨 (dev 7.22 → test 7.52). (2) 조건부 가능도 기반 상태변화 검출: GPR 체인이 라벨된 열화 온셋을 dev 5.21 / 봉인 test 8.54 사이클 오차로 검출. (3) 불확실성 기반 필터링: 커버리지 조건부 게이트 — 강도를 기준선 표본 지지에 조건화 — 는 양 분할·전 클래스에서 온셋을 개선하며, 고정 임계값의 고장 모드를 진단(σ0 팽창, 확신도 ≠ 정확도)하고 라벨프리 배포 가능 규칙으로 해결함. (4) 정직한 부정적 발견: 절단 RUL 끝단 지표는 하류 무장해제 전후 모두 정상모델을 변별하지 못함 — RUL 평가만으로는 정상모델 선택을 정당화할 수 없고, 변별 증거는 잔차 품질과 GP만이 제공하는 신뢰도 기구에 있음.')));

const doc = new D.Document({
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: kids,
  }],
});
const out = require('path').join(__dirname, `final_report_${KO ? 'ko' : 'en'}.docx`);
D.Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(out, b);
  console.log('written', out);
});
