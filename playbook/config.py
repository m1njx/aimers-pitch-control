"""config.py — 새 대회에 맞춰 이 파일 하나만 고치면 된다.

플레이북의 모든 단계·기법이 여기를 읽는다. 값을 채우고 `python3 run.py check` 로 확인하라.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── 경로 ────────────────────────────────────────────────────────────
    root: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = './data'                 # train/test/sample 이 있는 폴더
    work_dir: str = './pb_work'              # 산출물 (캐시·예측·리포트)
    train_file: str = 'train.csv'
    test_file: str = 'test.csv'
    sample_file: str = 'sample_submission.csv'

    # ── 스키마 ──────────────────────────────────────────────────────────
    id_col: str = 'row_id'
    target_col: str = 'target'
    pred_col: str = 'prediction'             # sample_submission 의 예측 컬럼명

    # 시간 축 (없으면 None → 랜덤 K-fold 로 대체)
    time_col: str | None = 'season'          # 폴드를 가르는 컬럼 (연/월/주)
    fold_edges: list = field(default_factory=list)
    # 예: [2021, 2022, 2023, 2024] → (≤2021→2022), (≤2022→2023), (≤2023→2024)

    # 반복 등장하는 엔티티 (엔티티 룩업의 재료). 없으면 빈 리스트
    entity_cols: list = field(default_factory=list)     # 예: ['pitcher_id', 'batter_id']
    # 엔티티의 관측량 컬럼 (게이트용). 없으면 None → 코드가 직접 센다
    entity_count_col: str | None = None                 # 예: 'asof_pitcher_n'

    # 문맥 축 후보 (룩업의 '셀'과 '제3축'이 될 저카디널리티 컬럼)
    context_cols: list = field(default_factory=list)    # 예: ['balls','strikes','outs']

    # 누적 통계 컬럼 접두사 (라벨 디코딩의 재료). 없으면 빈 리스트
    cumulative_prefixes: list = field(default_factory=list)   # 예: ['asof_']

    # ── 지표 ────────────────────────────────────────────────────────────
    metric: str = 'brier_skill'   # 'brier_skill' | 'mse' | 'auc' | 'logloss'
    metric_scale: float = 1e5     # brier_skill 의 C
    greater_is_better: bool = True
    task: str = 'binary'          # 'binary' | 'regression'

    # ── 규정 ────────────────────────────────────────────────────────────
    row_independent_required: bool = True   # 행 독립 규정이 있는가
    daily_submission_limit: int = 5
    best_score_ranking: bool = True         # 최고점 기준이면 탐색 하방이 0
    inference_time_limit_s: int = 600

    # ── 판정 기준 (착수 전 확정) ────────────────────────────────────────
    n_seeds: int = 5              # 최소 5. 2~3 은 노이즈를 4~5배 과소평가한다
    n_random_controls: int = 20   # 대량 탐색 시 무작위 대조군 개수
    gate_min_gain: float | None = None   # None 이면 노이즈 바닥 측정 후 자동 설정

    def paths(self):
        d = lambda f: os.path.join(self.data_dir, f)
        return dict(train=d(self.train_file), test=d(self.test_file), sample=d(self.sample_file))

    def validate(self) -> list[str]:
        p, bad = self.paths(), []
        for k, v in p.items():
            if not os.path.isfile(v):
                bad.append(f'{k} 파일 없음: {v}')
        if self.time_col and len(self.fold_edges) < 2:
            bad.append('time_col 이 있는데 fold_edges 가 비었다 (최소 2개)')
        if self.n_seeds < 5:
            bad.append(f'n_seeds={self.n_seeds} — 최소 5 권장 (2~3 은 노이즈 4~5배 과소평가)')
        return bad


CFG = Config()
