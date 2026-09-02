#!/usr/bin/env python3
"""run.py — 플레이북 실행기.

    python3 run.py check          # config 검증 + 데이터 점검
    python3 run.py plan           # 11단계 계획 출력
    python3 run.py list           # 전체 기법 카탈로그 (상태별)
    python3 run.py list --stage 6 # 특정 단계의 기법만
    python3 run.py show <id>      # 기법 하나의 상세 (근거·주의·시그니처)
    python3 run.py stage <n>      # 단계 지시서 출력 + 진행 기록
    python3 run.py status         # 진행 상황
    python3 run.py next           # 다음에 할 일 하나만
"""
from __future__ import annotations
import argparse, inspect, json, os, sys, textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG                                    # noqa: E402
from methods._base import load_all, STATUS_ORDER, STATUS_MARK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, 'state.json')

STAGES = [
    (0, '규정·데이터 파악', [
        '대회 규정 원문에서 **금지 항목과 허용 항목**을 표로 정리한다',
        '  → 무엇이 허용인지도 확인 (늦게 알면 손해: 이 대회는 과거요약 피처가 허용이었다)',
        '`config.py` 를 채우고 `python3 run.py check` 통과',
        '데이터 크기·결측·카디널리티·시간 범위 확인',
        '제출 형식·슬롯 수·순위 산정 방식 확인 (최고점 기준이면 탐색 하방이 0)']),
    (1, '가드레일 설치 ★ 1일차에 반드시', [
        '`toolkit/check_row_independence.py` 를 베이스라인에 연결',
        '`toolkit/check_submission.py` 를 제출 스크립트에 묶기',
        'SSOT 로그 파일 하나를 정하고 모든 판정을 append 하기 시작',
        '⚠️ 나중에 붙이면 이미 오염된 것을 못 본다']),
    (2, 'GBDT 기준선', [
        'LightGBM / CatBoost / XGBoost 기준선 각각 (gbdt.library_trio)',
        '  → 목적함수를 분류/회귀 두 벌로 (gbdt.objective_split) — 가장 싼 다양성',
        '시드 5개로 예측 배깅 (ens.seed_bagging)',
        'GBDT 가 선 뒤에 MLP 한 종 추가 (nn.mlp_member) — 이 대회 +36.24',
        '  → 목적함수를 나눠 멤버를 늘린다 (nn.multi_objective) — 구현 비용 거의 0',
        '  → ⚠️ 종류는 늘리지 말 것. ResNet/Transformer/TabNet/TabM 전부 음수였다',
        '⚠️ "권장 설정"을 믿지 말 것 — 이 대회는 cat_features 지정이 −15.49 였다']),
    (3, '노이즈 바닥 측정 ★', [
        '동일 설정을 시드만 바꿔 5회 이상 (val.noise_floor)',
        '이 값 미만의 차이는 앞으로 개선으로 간주하지 않는다',
        '`config.gate_min_gain` 을 여기서 확정',
        '⚠️ 이건 시드 노이즈다. 결정적 사후처리엔 0 이므로 그대로 쓰면 안 된다']),
    (4, '폴드 설계 + 레짐 점검', [
        'val.time_folds — 전방 폴드. **예측 시계를 배포와 맞출 것**',
        'val.regime_check — 세그먼트 붕괴 구간 확인 → 제외하거나 분리 채점',
        'val.nested_cv — 튜닝 폴드와 평가 폴드 분리',
        'LB 실측점이 쌓이면 어느 폴드가 가장 근사한지 대조하라']),
    (5, '피처 엔지니어링', [
        'feat.asof_decompose ★ — 이 대회 최대 이득 (+146.8)',
        'feat.era_relative — 분포 밀림이 크면',
        'feat.domain_physics — ⚠️ 만들기 전에 check_determinism 필수',
        '새 피처 블록은 stage 7 의 ρ 스크린으로 상한부터 재라']),
    (6, '엔티티 룩업 ★★ 이 대회 두 번째 이득', [
        'lookup.entity_residual — 5요소를 전부 지킬 것',
        'lookup.axis_sweep — 제3축 후보를 **무작위 대조군과 함께** 훑기',
        'lookup.contrast_target — 보조 라벨이 있으면',
        '⚠️ lookup.target_encoding_eb 는 수축 없이 쓰면 재앙 (LB 103 사례)']),
    (7, '정보원 발굴 + 상한 스크린', [
        'feat.decode_hidden_labels ★ — 누적 컬럼이 있으면 무조건',
        'feat.state_reconstruction — 창 대수로 숨은 상태 복원',
        'val.rho_screen ★★ — **새 arm 을 학습하기 전에** 상한을 재라',
        '  ρ < 필요치면 그 방향은 원리상 불가능하다. GPU 를 쓰지 마라']),
    (8, '앙상블', [
        'ens.gram — 종이 계산으로 실험 대체',
        'ens.rho_gate — 새 arm 채택 판단',
        'ens.gated_blend — arm 의 약점 구간 우회',
        'ens.library_ceiling — 캠페인을 계속할지 접을지 판단',
        '⚠️ 가중치는 반드시 다른 폴드에서 적합']),
    (9, '캘리브레이션', [
        'cal.affine — 파라미터 2개. LB 로 직접 최적화가 가장 정확',
        'cal.logit_cell_offset — 전역 스케일 U 로 자유도 통제',
        'cal.lb_quadratic_fit — 제출 3회로 최적 계수 역산',
        'cal.shrinkage_decomposition ★ — 이득이 정보인지 수축인지 항상 확인']),
    (10, '제출', [
        '`check_submission.py` 8항목 통과',
        '`check_row_independence.py` 통과',
        '**한 번에 하나만** 바꾼다 — 그래야 LB 델타가 그 축의 순수 신호',
        '⚠️ 의존성 목록을 함부로 바꾸지 마라 (설치 오류로 전체 실패 사례)',
        'val.transfer_ratio 에 (로컬Δ, 실전Δ) 쌍을 기록']),
]


def load_state():
    return json.load(open(STATE, encoding='utf-8')) if os.path.exists(STATE) else {'done': [], 'log': []}


def save_state(s):
    json.dump(s, open(STATE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def cmd_check(a):
    print('[config 검증]')
    bad = CFG.validate()
    for b in bad:
        print(f'  🔴 {b}')
    if not bad:
        print('  ✅ 통과')
    print('\n[스키마]')
    for k in ('id_col', 'target_col', 'time_col', 'entity_cols', 'context_cols',
              'cumulative_prefixes', 'metric', 'row_independent_required'):
        print(f'  {k:24} = {getattr(CFG, k)}')
    print('\n[적용 가능한 기법]')
    R = load_all()
    ok = [m for m in R.values() if m.runnable(CFG)[0]]
    ng = [(m, m.runnable(CFG)[1]) for m in R.values() if not m.runnable(CFG)[0]]
    print(f'  사용 가능 {len(ok)} / 전체 {len(R)}')
    for m, why in ng[:8]:
        print(f'  ⏸ {m.id:32} — {why}')
    return 0 if not bad else 1


def cmd_plan(a):
    st = load_state()
    print('플레이북 11단계\n' + '─' * 66)
    for n, title, items in STAGES:
        mark = '✅' if n in st['done'] else '  '
        print(f'{mark} [{n:2}] {title}')
        for it in items:
            print(f'        {it}')
        print()
    return 0


def cmd_list(a):
    R = load_all()
    ms = sorted(R.values(), key=lambda m: (m.stage, STATUS_ORDER.get(m.status, 9), m.id))
    if a.stage is not None:
        ms = [m for m in ms if m.stage == a.stage]
    if a.status:
        ms = [m for m in ms if m.status == a.status.upper()]
    cur = None
    for m in ms:
        if m.stage != cur:
            cur = m.stage
            t = next((s[1] for s in STAGES if s[0] == cur), '')
            print(f'\n── 단계 {cur} · {t} ' + '─' * max(0, 44 - len(t)))
        print(f'  {STATUS_MARK.get(m.status,"?")} {m.id:34} {m.title}')
        print(f'      이득: {m.gain}')
    n = len(ms)
    cnt = {s: sum(1 for m in R.values() if m.status == s) for s in STATUS_ORDER}
    print(f'\n총 {n}개 표시 / 전체 {len(R)}개  '
          f'(✅{cnt["ADOPTED"]} ❌{cnt["REJECTED"]} ⏸{cnt["SHELVED"]} ·{cnt["UNTESTED"]})')
    return 0


def cmd_show(a):
    R = load_all()
    m = R.get(a.id)
    if not m:
        cands = [k for k in R if a.id in k]
        print(f'없음: {a.id}' + (f'\n  비슷한 것: {cands}' if cands else ''))
        return 1
    print(f'{STATUS_MARK.get(m.status,"?")} {m.id}  [{m.status}]  단계 {m.stage}  비용 {m.cost}')
    print(f'\n  {m.title}')
    print(f'\n  이득 : {m.gain}')
    print(f'  근거 : {m.evidence}')
    if m.note:
        print('\n' + textwrap.indent(textwrap.fill(m.note, 78), '  '))
    if m.requires:
        okk, why = m.runnable(CFG)
        print(f'\n  필요 : {m.requires}  → {"✅ 충족" if okk else "⏸ " + why}')
    if m.fn:
        try:
            print(f'\n  시그니처: {m.fn.__name__}{inspect.signature(m.fn)}')
            print(f'  구현    : methods/{m.fn.__module__.split(".")[-1]}.py')
        except (TypeError, ValueError):
            pass
    return 0


def cmd_stage(a):
    s = next((x for x in STAGES if x[0] == a.n), None)
    if not s:
        print(f'단계 {a.n} 없음 (0~{STAGES[-1][0]})'); return 1
    n, title, items = s
    print(f'━━ 단계 {n} · {title} ━━\n')
    for it in items:
        print(f'  □ {it}')
    R = load_all()
    ms = [m for m in R.values() if m.stage == n]
    if ms:
        print('\n  이 단계의 기법:')
        for m in sorted(ms, key=lambda x: STATUS_ORDER.get(x.status, 9)):
            okk, why = m.runnable(CFG)
            tail = '' if okk else f'  ⏸ {why}'
            print(f'    {STATUS_MARK.get(m.status,"?")} {m.id:32} {m.gain}{tail}')
        print(f'\n  상세: python3 run.py show <id>')
    if a.done:
        st = load_state()
        if n not in st['done']:
            st['done'].append(n); st['done'].sort()
        save_state(st); print(f'\n  ✅ 단계 {n} 완료로 기록')
    return 0


def cmd_status(a):
    st = load_state()
    done = set(st['done'])
    print('진행 상황\n' + '─' * 50)
    for n, title, _ in STAGES:
        print(f'  {"✅" if n in done else "  "} [{n:2}] {title}')
    nxt = next((n for n, _, _ in STAGES if n not in done), None)
    print('─' * 50)
    print(f'  {len(done)}/{len(STAGES)} 완료' +
          (f'  → 다음: 단계 {nxt}' if nxt is not None else '  → 전부 완료'))
    return 0


def cmd_next(a):
    st = load_state(); done = set(st['done'])
    nxt = next((s for s in STAGES if s[0] not in done), None)
    if not nxt:
        print('모든 단계 완료.'); return 0
    return cmd_stage(argparse.Namespace(n=nxt[0], done=False))


def main():
    ap = argparse.ArgumentParser(description='플레이북 실행기')
    sub = ap.add_subparsers(dest='cmd')
    sub.add_parser('check')
    sub.add_parser('plan')
    p = sub.add_parser('list'); p.add_argument('--stage', type=int); p.add_argument('--status')
    p = sub.add_parser('show'); p.add_argument('id')
    p = sub.add_parser('stage'); p.add_argument('n', type=int); p.add_argument('--done', action='store_true')
    sub.add_parser('status')
    sub.add_parser('next')
    a = ap.parse_args()
    if not a.cmd:
        ap.print_help(); print('\n' + __doc__); return 0
    return globals()[f'cmd_{a.cmd}'](a)


if __name__ == '__main__':
    sys.exit(main())
