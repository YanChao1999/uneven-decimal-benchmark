import Lake
open Lake DSL

package «uneven-decimal» where
  -- Ground-truth verification package for the uneven decimal benchmark.

@[default_target]
lean_lib UnevenDecimal where
  -- Sources under ./UnevenDecimal and root UnevenDecimal.lean
