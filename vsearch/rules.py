"""Merchandising rules (engine-side and application-side) and the page metrics B5 scores them with."""
from collections.abc import Iterable
from typing import Literal

import numpy as np
from pymilvus import Function, FunctionChain, FunctionChainStage, FunctionScore, FunctionType
from pymilvus.function_chain import col, fn


def boost(filter: str, weight: float, name: str = "boost") -> Function:
    """Multiply the score of candidates matching `filter` (weight < 1 buries). Only the `limit` candidates a
    segment returns are rescored, so a rule can't reach deeper than the request's limit."""
    return Function(name=name, function_type=FunctionType.RERANK, input_field_names=[],
                    params={"reranker": "boost", "filter": filter, "weight": weight})


def combined(boosts: list[Function], mode: Literal["Multiply", "Sum"]) -> FunctionScore:
    return FunctionScore(functions=boosts, params={"boost_mode": mode, "function_mode": mode})


def margin_chain(weight: float) -> FunctionChain:
    return FunctionChain(FunctionChainStage.L0_RERANK, name="margin").map(
        "$score", fn.num_combine(col("$score"), col("margin"), mode="weighted", weights=[1.0, weight]))


def apply_app(hits: list[tuple[str, float]], mult: dict[str, float]) -> list[str]:
    """Application-side rules: multiply each hit's score, keep the engine's order on ties."""
    return [i for i, _ in sorted(hits, key=lambda h: -h[1] * mult.get(h[0], 1.0))]


def page_margin(ids: Iterable[str], margin: dict[str, float]) -> float | None:
    vals = [margin[i] for i in ids]
    return float(np.mean(vals)) if vals else None


def _holds(page: list[str], wanted: set[str]) -> bool:
    return any(p in wanted for p in page)


def reach(pages: list[list[str]], campaign: set[str], eligible: list[bool]) -> float | None:
    """Share of eligible queries whose page holds a campaign product."""
    got = [_holds(p, campaign) for p, e in zip(pages, eligible, strict=True) if e]
    return float(np.mean(got)) if got else None


def leak(pages: list[list[str]], hidden: set[str]) -> float:
    """Share of pages holding any product that should be gone."""
    return float(np.mean([_holds(p, hidden) for p in pages]))


def slot_share(pages: list[list[str]], wanted: set[str]) -> float | None:
    """Share of all returned page slots held by `wanted`: rises with over-boosting after reach has saturated."""
    slots = [p in wanted for page in pages for p in page]
    return float(np.mean(slots)) if slots else None
