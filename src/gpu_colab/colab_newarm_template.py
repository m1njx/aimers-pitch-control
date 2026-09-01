# =============================================================================
#  새 arm 을 만들 때의 표준 절차 — 판정용 + 배포용을 한 번에 뽑는다
#
#  v33 의 실패 원인: 배포 모델이 `train<2024` 라 2025 까지 **격차 2년**이었는데,
#  가치는 **1년 격차**(train<2024 → 2024)에서 재서 과대평가했다.
#  → LB 1078.03 (v29 1082 대비 −3.97), 폴드내 오라클조차 w_C=0.
#
#  그래서 새 arm 은 **반드시 두 벌**을 만든다:
#    (a) 판정용 : train<2024 로 학습 → 2024 예측  (= 1년 격차, 배포와 동일 조건)
#    (b) 배포용 : 전 시즌 학습                    (= 2025 예측 시 1년 격차)
#  (a) 로 판정하고, 통과하면 (b) 를 싣는다.
#
#  ⚠️ 착수 전 확정 판정 기준 (결과 보기 전에 고정):
#     G1  d_AC ≥ 0.020 그리고 d_BC ≥ 0.020        (다양성)
#     G2  최적가중 3-arm − 2-arm > +12 (2024)      (기여)
#     G3  2024 를 반으로 갈라 한쪽에서 가중치·재보정 적합 → 다른쪽에서도 양수 (자유도 방어)
#     셋 다 통과해야 배포. 하나라도 실패하면 REJECT.
# =============================================================================
import numpy as np, pandas as pd, json, time, os, shutil

CSV = 'armc_slim.csv.gz'      # 82피처를 쓰려면 그 컬럼이 포함된 CSV 로 교체할 것
df = pd.read_csv(CSV)

# ---------------------------------------------------------------------------
# 여기에 본인의 피처 빌더를 넣는다.  지켜야 할 것 두 가지:
#   1) 라벨을 쓰는 인코딩(EB/타깃인코딩)은 **fit 구간에서만** 적합할 것.
#      판정용은 fit=train<2024, 배포용은 fit=전 시즌.  섞으면 검증이 무효가 된다.
#   2) 학습에 쓴 fit 구간과 추론에 쓸 fit 구간을 **일치**시킬 것.
# ---------------------------------------------------------------------------
def build(fit_df, trans_df):
    raise NotImplementedError('본인 피처 빌더로 교체')


def train_bag(X, y, make_model, seeds, tag):
    models = []
    t0 = time.time()
    for i, s in enumerate(seeds, 1):
        m = make_model(s); m.fit(X, y); models.append(m)
        print(f'  [{tag}] {i}/{len(seeds)} seed {s} ({time.time()-t0:.0f}s)', flush=True)
    return models


def run(make_model, seeds):
    # ---- (a) 판정용: train<2024 → 2024 --------------------------------------
    past = df[df.season < 2024]
    f24  = df[df.season == 2024].reset_index(drop=True)
    Xtr, Xva = build(past, past), build(past, f24)
    ms = train_bag(Xtr, past.control_success.values, make_model, seeds, 'judge')
    p24 = np.mean([m.predict_proba(Xva)[:, 1] for m in ms], 0)
    np.save('newarm_judge_pred2024.npy', p24)
    r = f24.control_success.values.mean()
    sk = 1e5*(1-((p24-f24.control_success.values)**2).mean()/(r*(1-r)))
    print(f'\n[판정용] 2024 skill={sk:.2f}  mean(p)={p24.mean():.5f}  (r={r:.5f})')

    # ---- (b) 배포용: 전 시즌 -------------------------------------------------
    Xall = build(df, df)
    md = train_bag(Xall, df.control_success.values, make_model, seeds, 'deploy')
    os.makedirs('out_model', exist_ok=True)
    for s, m in zip(seeds, md):
        m.save_model(f'out_model/model_seed{s}.cbm'
                     if hasattr(m, 'save_model') else f'out_model/model_seed{s}.json')
    shutil.make_archive('newarm_deploy', 'zip', 'out_model')
    print('\n배포용 zip 생성 완료')

# from google.colab import files
# files.download('newarm_judge_pred2024.npy')   # ← 이걸 먼저 보내면 내가 판정한다
# files.download('newarm_deploy.zip')           # ← 통과하면 이걸 싣는다
