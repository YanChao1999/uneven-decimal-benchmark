import UnevenDecimal.Decimal

namespace UnevenDecimal.Generated
open UnevenDecimal

/-! Auto-generated ground-truth checks. Do not edit by hand. -/

/-- Auto-generated check for aligned_00000 (aligned) -/
example : decEq (chainSum [(mkDec 868 3), (mkDec 26387 3), (mkDec 9206 3)]) (mkDec 36461 3) = true := by native_decide

/-- Auto-generated check for aligned_00001 (aligned) -/
example : decEq (chainSum [(mkDec 7371 3), (mkDec 40146 3), (mkDec 4218 3)]) (mkDec 51735 3) = true := by native_decide

/-- Auto-generated check for uneven_low_carry_00000 (uneven_low_carry) -/
example : decEq (chainSum [(mkDec 21201 4), (mkDec 124 2), (mkDec 5292 3)]) (mkDec 86521 4) = true := by native_decide

/-- Auto-generated check for uneven_low_carry_00001 (uneven_low_carry) -/
example : decEq (chainSum [(mkDec 512 1), (mkDec 242 1), (mkDec 43020 4)]) (mkDec 79702 3) = true := by native_decide

/-- Auto-generated check for uneven_high_carry_00000 (uneven_high_carry) -/
example : decEq (chainSum [(mkDec 16 1), (mkDec 5569 2), (mkDec 7167 3)]) (mkDec 64457 3) = true := by native_decide

/-- Auto-generated check for uneven_high_carry_00001 (uneven_high_carry) -/
example : decEq (chainSum [(mkDec 808 1), (mkDec 238 1), (mkDec 796877 4)]) (mkDec 1842877 4) = true := by native_decide

end UnevenDecimal.Generated
