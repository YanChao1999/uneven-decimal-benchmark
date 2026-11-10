/-!
# Reference alignment + carry transform (documentation module)

This module records the intended **explicit algorithm** that the paper hypothesizes
LLMs may not stably learn:

1. **Align** addends by the decimal point (equivalently: rewrite each addend to a
   common fractional width by right-padding zeros).
2. **Add column-wise** from the least significant fractional place toward the
   integer places.
3. **Propagate carries** whenever a column sum is at least 10 (or at most -10 for
   signed variants).

The executable semantics used for ground-truth checking live in
`UnevenDecimal.Decimal` (`addDec` / `chainSum`), which are mathematically
equivalent to alignment+carry over base-10 place values, but expressed via
scaled integer arithmetic.

Future work may attach difficulty measures (carry density, unevenness) to these
definitions; phase 1 only documents the transform and keeps checks decidable.

The Python RASP sketch in `experiments/src/rasp/` mirrors the same three stages
and exposes hardness proxies (`max_carry_run`, `frac_unevenness`) used to test
whether scaling trends of LLMs are predictable from algorithmic bottlenecks.
-/

namespace UnevenDecimal.AlignmentCarry

/-- Maximum fractional scale among a list of decimals (alignment width). -/
def maxScale : List UnevenDecimal.Dec → Nat
  | [] => 0
  | x :: xs => Nat.max x.scale (maxScale xs)

/--
Conceptual column index: `0` is the least significant place after aligning to
`maxScale`. This is a documentation helper for later difficulty metrics.
-/
def alignedWidth (xs : List UnevenDecimal.Dec) : Nat :=
  maxScale xs

end UnevenDecimal.AlignmentCarry
