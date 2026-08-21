# N-CMAPSS RUL 연구 기록 (RESEARCH_LOG)

> 이 파일은 세션·기기 간 핸드오버의 단일 기준입니다. 새 세션은 이 파일의 "현재 상태"부터 읽으세요.
> 마지막 갱신: 2026-08-21 (맥북 → 맥미니 이관 시점)

## 0. 현재 상태 (2026-08-21)

**현행 공식 = DS03-012 2-pass 자기지도 체인** (2026-08-15 확정)
- 1단 검출: cycle≤3 층화 27k Vecchia MOGP(RBF rank1, m=18, 5센서 T30/T48/T50/Nc/Wf) + 커버리지 조건부 게이트(30h 기준선에 10사이클 미만이면 V=19.0, 아니면 20.2) → q75 우도 곡선 → 클립 10σ → mean-drop 온셋. dev 5.21 / test 8.54 사이클.
- 2단 RUL: 1단 온셋 이전 정상범위에서 유닛당 3,000행(27k) MOGP 재학습(무게이트) → trim25 → 풀링 z → HI#2(λ1=12, λ2=0.25, med3) → 지수 first-passage, β dev-LOO NASA 규칙(30/30/30).
- **dev 7.56±0.05 / test 7.18±0.05, NASA 17.7±0.2** (기록상 최저 test; 1단 HI#1 대비 쌍대 비유의 p=0.49). 1단 HI#1 무게이트: dev 7.15±0.10 / test 7.47±0.06.
- 스크립트: `experiments/exp_v5c_*.py` (dev/test/hi_ablate/test_ablate/cross_hi/resid_quality/bench_quality/lambda_grid/timing/report_figs/bench_hr), 캐시 `v5c_*.npz`, GP `v5c_gp_s*.pt`(git 제외).

**DS03 test 개봉 원장**: 16회(final_report §0 전수 기재) + 17차 = 정상범위 벤치마크(`exp_v5c_bench_hr.py`, 2026-08-18) + raw 절제 재요약(`exp_v5c_raw_resummary.py`, 11차와 동일 예측, 레벨별 score 추가). **final_report §0는 아직 16회로 기재 — 17차·재요약 반영 미완.**

**문서**: `paper/final_report_{en,ko}.docx/.pdf` (생성기 `paper/make_final_report.js en|ko`, 2026-08-18 전수 감사 완료: 443개 수치 대조, 5건 교정) / `paper/Case_study.docx/.pdf` (논문 IV장 템플릿, 생성기 `paper/case_study/make_cs_data.py` → `make_case_study.js`; DS03 A~F + IV-G 전이 DS08a·DS07).

**전이 실험 v6 (2026-08-19~21)** — 원칙: 구조·손실항 종류·선정 절차만 이식, 수치(창·클립·V·λ1·λ2·β·z)는 데이터셋 dev에서 재선정(Table 3 비튜닝 상수는 유지). 조건부 게이트 구조 필수(시간 창 동률밴드 내 선택, nbase 갭으로 NB 재유도, 단일 클래스면 NB=10 구조값). 2단 λ 재스윕 동률밴드 0.02(밖이면 재동결).
- 스크립트: `exp_dsx_lib.py`(공용), `exp_dsx_c3.py`(1단 GP), `exp_dsx_detsel.py`(검출기 재선정), `exp_dsx_hi1.py`(λ 그리드+HI#1), `exp_dsx_stage2.py`(2단), `exp_dsx_test.py`(사전등록 test 1회 개봉), `exp_dsx_gate_diag.py`(넓은 V축 게이트 진단), `build_cache_dsx.py`(캐시). 오케스트레이터 `run_dsx_v3.sh`, `run_dsx_screen.sh`, `run_dsx_test_screen.sh`.
- 결론: **SNR 메커니즘(정상 바닥 0.3~0.6×)은 9개 데이터셋 전부 재현, 그러나 끝단 이득은 이식 안 됨.** 게이트가 DS03처럼 넓게 작동하는 데이터셋 없음(넓은 V축에서 무게이트를 이기는 V 비율: DS03 46/120, DS02 20/120, DS07 10/120, DS08a·08c 0/120). 2단 RUL test: 개선 방향은 DS03(−4%)·DS05(−4%, 시드0)·DS08c(−2%, 시드0)뿐이고 전부 비유의, 나머지 악화(DS02 유의 악화). 게이트의 전제 = 강한 우도 하락(drop-SNR 4~10) + 저신뢰 점 오염(25~46%) — DS03만 충족.
- 데이터셋별 test 결과 표는 `experiments/*v6_test_results*.txt`, 스크린 진단 `dsx_screen_gate_results.txt`, `dsx_gate_diag_results.txt`.
- 단일 시드 개봉(DS04/05/06/08c)은 방향 확인용 — 논문 수치로 쓰려면 3시드 2차 개봉 필요(원장 기록).

**하드웨어 정책**: Vecchia GP 학습은 프로세스당 ~1.7코어 → 시드×데이터셋 동시 프로세스(`SEEDS_ONLY`, `STATS_ONLY` env) + 프로세스 내부 청크 병렬(`stats_parallel` LLKE 3.6×, `point_stats_parallel` GP 추론 3.9×, 결과 비트 동일). `NJOBS`로 워커 수 지정(기본 코어/2).

**미결**: ⓐ final_report §0 원장 17차·재요약 반영 ⓑ PPT(research_overview.pptx) 동기화 ⓒ 논문 본문 집필 ⓓ DS05·DS08c 3시드 완주 여부 결정 ⓔ Case study IV-G에 9개 스크린 표 추가 여부.

## 1. 방법론 규칙 (불변)
절대축 스윕·붕괴까지 / 동률밴드 / 선택은 dev만·개봉 정직 공개 / 표 빈칸 금지 / 그림 라벨 영어·괄호 금지 / 유불리 무관 정직 보고 / 사후수리 금지 / test 신규 지표 = 개봉 카운트(사전등록 필수, 스크립트 헤더가 사전등록문).

## 2. 상세 연대기 (세션 메모리 원문)
