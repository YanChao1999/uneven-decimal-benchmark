"""Minimal RASP-style operators (Weiss et al.) for executable sketches.

These are sequence→sequence primitives used to express Transformer-friendly
algorithms. They are not a full Tracr compiler; they document which
``select`` / ``aggregate`` patterns the decimal algorithm needs.
"""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def indices(n: int) -> list[int]:
    return list(range(n))


def map_seq(fn: Callable[[T], U], xs: Sequence[T]) -> list[U]:
    return [fn(x) for x in xs]


def zipmap(fn: Callable[[T, U], V], xs: Sequence[T], ys: Sequence[U]) -> list[V]:
    return [fn(x, y) for x, y in zip(xs, ys)]


def select(
    keys: Sequence[T],
    queries: Sequence[U],
    pred: Callable[[T, U], bool],
) -> list[list[bool]]:
    """Return attention mask ``pred(key_j, query_i)`` with shape [len(queries), len(keys)]."""
    return [[pred(k, q) for k in keys] for q in queries]


def aggregate(
    mask: Sequence[Sequence[bool]],
    values: Sequence[T],
    *,
    default: T,
    combine: Callable[[T, T], T] | None = None,
) -> list[T]:
    """Aggregate values selected by a boolean mask (mean for numbers if combine is None)."""
    out: list[T] = []
    for row in mask:
        chosen = [v for bit, v in zip(row, values) if bit]
        if not chosen:
            out.append(default)
        elif combine is None:
            # Default: last selected (stable for unique selects) / sum for ints via caller.
            out.append(chosen[-1])
        else:
            acc = chosen[0]
            for v in chosen[1:]:
                acc = combine(acc, v)
            out.append(acc)
    return out


def selector_width(mask: Sequence[Sequence[bool]]) -> list[int]:
    return [sum(1 for bit in row if bit) for row in mask]
