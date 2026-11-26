import Base
import ./Graph.bend as G

law reducer_result_ignores_finish_order:
  for s: G.Slots
  for i: Nat
  for j: Nat
  for a: Nat
  for b: Nat
  for ne: {Nat.is_eq(i, j) == False{} : Bool}
  {G.merge(G.fill(G.fill(s, i, a), j, b)) == G.merge(G.fill(G.fill(s, j, b), i, a)) : G.Slot}

law last_finisher_wins_breaks_order:
  {G.finish_faulty(G.finish_faulty(G.Slot.Empty{}, 1n), 2n) != G.finish_faulty(G.finish_faulty(G.Slot.Empty{}, 2n), 1n) : G.Slot}

law kept_write_runs_branch_zero_more_times:
  for s: G.Slots
  for i: Nat
  for v: Nat
  {G.run_at(G.rerun(G.fill(s, i, v)), i) == 0n : Nat}

law ignored_write_reruns_branch:
  {G.run_at(G.rerun_faulty(G.fill(G.Slots.Con{G.Slot.Empty{}, G.Slots.Nil{}}, 0n, 5n)), 0n) != 0n : Nat}

law fork_keeps_parent_checkpoints:
  for s: G.Store
  for k: Nat
  for p: Nat
  for n: Nat
  for lt: {Nat.is_lt(k, G.size(s)) == True{} : Bool}
  {G.get(G.fork(s, p, n), k) == G.get(s, k) : G.Steps}

law in_place_fork_changes_parent:
  {G.get(G.fork_faulty(G.Store.Con{G.Steps.Con{1n, G.Steps.Con{2n, G.Steps.Nil{}}}, G.Store.Nil{}}, 0n, 1n), 0n) != G.Steps.Con{1n, G.Steps.Con{2n, G.Steps.Nil{}}} : G.Steps}
