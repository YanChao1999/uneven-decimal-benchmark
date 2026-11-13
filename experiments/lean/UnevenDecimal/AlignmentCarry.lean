import UnevenDecimal.Decimal

/-!
# Reference alignment + carry transform (documentation module)

Records the place-value algorithm whose RASP bottlenecks predict sample difficulty:

1. **Align** addends by the decimal point (common fractional width via right-pad).
2. **Add column-wise** from the least significant fractional place upward.
3. **Propagate carries** when a column sum reaches 10 (or ≤ -10 for signed variants).

Executable ground-truth semantics live in `UnevenDecimal.Decimal`
(`addDec` / `chainSum`). The Python RASP sketch mirrors the same stages and
exposes hardness proxies used to test relative difficulty trends across models.
-/

namespace UnevenDecimal.AlignmentCarry
open UnevenDecimal

/-- Maximum fractional scale among a list of decimals (alignment width). -/
def maxScale : List Dec → Nat
  | [] => 0
  | x :: xs => Nat.max x.scale (maxScale xs)

/--
Conceptual helper: alignment width equals `maxScale` after place-value padding.
-/
def alignedWidth (xs : List Dec) : Nat :=
  maxScale xs

end UnevenDecimal.AlignmentCarry
