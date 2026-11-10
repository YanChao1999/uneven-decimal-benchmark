/-!
# Exact decimals as significand + scale

A value `mkDec n k` denotes the rational `n / 10^k`.
All operations are computable so that `native_decide` can batch-check
Python-generated ground-truth chain sums.
-/

namespace UnevenDecimal

/-- Decimal encoded as integer significand and non-negative scale. -/
structure Dec where
  numer : Int
  scale : Nat
  deriving DecidableEq, Repr

/-- Construct a decimal `numer / 10^scale`. -/
def mkDec (numer : Int) (scale : Nat) : Dec :=
  ⟨numer, scale⟩

/-- `10^n` as a natural number. -/
def pow10 : Nat → Nat
  | 0 => 1
  | n + 1 => 10 * pow10 n

/-- Equality of decimals as rationals with denominator a power of ten. -/
def decEq (a b : Dec) : Bool :=
  a.numer * Int.ofNat (pow10 b.scale) == b.numer * Int.ofNat (pow10 a.scale)

/-- Align both decimals to a common scale and add significands. -/
def addDec (a b : Dec) : Dec :=
  let s := Nat.max a.scale b.scale
  let a' := a.numer * Int.ofNat (pow10 (s - a.scale))
  let b' := b.numer * Int.ofNat (pow10 (s - b.scale))
  ⟨a' + b', s⟩

/-- Chain sum of a list of decimals (left fold). -/
def chainSum : List Dec → Dec
  | [] => mkDec 0 0
  | x :: xs => List.foldl addDec x xs

/-- Reference partial sums `s_k` via a left scan. -/
def partialSums : List Dec → List Dec
  | [] => []
  | x :: xs =>
    (List.foldl
      (fun (acc : Dec × List Dec) y =>
        let acc' := addDec acc.1 y
        (acc', acc.2 ++ [acc']))
      (x, [x])
      xs).2

end UnevenDecimal
