"""_base.py — 기법 레지스트리.

이 대회에서 **실제로 시도한 모든 기법**을 코드로 등록한다.
채택된 것뿐 아니라 **실패·보류한 것도 등록**한다 — 다음 대회에서
"이미 해봤고 이런 결과였다" 를 코드와 함께 즉시 확인하기 위해서다.

상태 값
  ADOPTED   채택됨. 실전 점수를 냈다
  REJECTED  시도했고 실패. 근거가 명확해 재시도 금지
  SHELVED   보류. 원리상 가능하나 이 대회에선 여건이 안 됐다 (다른 대회에선 볼 것)
  UNTESTED  이 대회에서 시도하지 못함
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

REGISTRY: dict[str, "Method"] = {}


@dataclass
class Method:
    id: str
    stage: int                      # 플레이북 단계 번호
    status: str                     # ADOPTED | REJECTED | SHELVED | UNTESTED
    title: str
    gain: str                       # 이 대회 실측 이득
    evidence: str                   # 근거 한 줄
    cost: str = 'low'               # low | med | high (시간·자원)
    requires: list = field(default_factory=list)   # cfg 에 있어야 하는 항목
    fn: Callable | None = None
    note: str = ''

    def runnable(self, cfg) -> tuple[bool, str]:
        for r in self.requires:
            v = getattr(cfg, r, None)
            if v in (None, [], ''):
                return False, f'cfg.{r} 가 비어 있음'
        return True, ''


def method(**kw):
    def deco(fn):
        m = Method(fn=fn, title=kw.pop('title', fn.__name__), **kw)
        if m.id in REGISTRY:
            raise ValueError(f'중복 id: {m.id}')
        REGISTRY[m.id] = m
        return fn
    return deco


def load_all():
    """모든 기법 모듈을 import 해 레지스트리를 채운다."""
    from . import features, lookups, calibration, ensemble, validation, rejected  # noqa
    return REGISTRY


STATUS_ORDER = {'ADOPTED': 0, 'UNTESTED': 1, 'SHELVED': 2, 'REJECTED': 3}
STATUS_MARK = {'ADOPTED': '✅', 'REJECTED': '❌', 'SHELVED': '⏸', 'UNTESTED': '·'}
