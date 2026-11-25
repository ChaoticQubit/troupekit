import Base
import ./Index.bend as I

law top_k_returns_min_k_n:
  for k: Nat
  for xs: I.Scores
  {I.len(I.top_k(k, xs)) == Nat.min(k, I.len(xs)) : Nat}

law top_k_is_sorted:
  for k: Nat
  for xs: I.Scores
  {I.sorted(I.top_k(k, xs)) == True{} : Bool}

law rrf_ignores_list_order:
  for id: Nat
  for a: I.Ranks
  for b: I.Ranks
  for rest: I.Lists
  {I.rrf(id, I.Lists.Cons{a, I.Lists.Cons{b, rest}}) == I.rrf(id, I.Lists.Cons{b, I.Lists.Cons{a, rest}}) : Nat}
