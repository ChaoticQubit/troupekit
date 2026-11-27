import Base
import ./Graph.bend as G
import ./LAWS.bend as Laws

def TrueDisc(b: Bool) -> Type:
  match b:
    case True{}:
      Unit
    case False{}:
      Empty

def FalseDisc(b: Bool) -> Type:
  match b:
    case True{}:
      Empty
    case False{}:
      Unit

def true_ne_false(e: {True{} == False{} : Bool}) -> Empty:
  %e : TrueDisc(_)
  Unit{}

def false_ne_true(e: {False{} == True{} : Bool}) -> Empty:
  %e : FalseDisc(_)
  Unit{}

law fill_swap:
  for s: G.Slots
  for i: Nat
  for j: Nat
  for +a: Nat
  for +b: Nat
  for ne: {Nat.is_eq(i, j) == False{} : Bool}
  {G.fill(G.fill(s, i, a), j, b) == G.fill(G.fill(s, j, b), i, a) : G.Slots}

def fill_swap(s, i, j, a, b, ne):
  match s i j:
    case G.Slots.Nil{} _ _:
      {==}
    case G.Slots.Con{+h, +t} 0n 0n:
      Empty.absurd({G.fill(G.fill(G.Slots.Con{h, t}, 0n, a), 0n, b) == G.fill(G.fill(G.Slots.Con{h, t}, 0n, b), 0n, a) : G.Slots}, true_ne_false(ne))
    case G.Slots.Con{_, _} 0n 1n+q:
      {==}
    case G.Slots.Con{_, _} 1n+p 0n:
      {==}
    case G.Slots.Con{+h, t} 1n+p 1n+q:
      %fill_swap(t, p, q, a, b, ne) : {G.Slots.Con{h, G.fill(G.fill(t, p, a), q, b)} == G.Slots.Con{h, _} : G.Slots}
      {==}

def Laws.reducer_result_ignores_finish_order(s, i, j, a, b, ne):
  %fill_swap(s, i, j, a, b, ne) : {G.merge(G.fill(G.fill(s, i, a), j, b)) == G.merge(_) : G.Slot}
  {==}

def NatIsTwo(n: Nat) -> Type:
  match n:
    case 0n:
      Empty
    case 1n+p:
      match p:
        case 0n:
          Empty
        case 1n+q:
          match q:
            case 0n:
              Unit
            case 1n+r:
              Empty

def SlotIsTwo(s: G.Slot) -> Type:
  match s:
    case G.Slot.Empty{}:
      Empty
    case G.Slot.Full{v}:
      NatIsTwo(v)

def two_ne_one(e: {G.Slot.Full{2n} == G.Slot.Full{1n} : G.Slot}) -> Empty:
  %e : SlotIsTwo(_)
  Unit{}

def Laws.last_finisher_wins_breaks_order():
  e => two_ne_one(e)

def Laws.kept_write_runs_branch_zero_more_times(s, i, v):
  match s i:
    case G.Slots.Nil{} _:
      {==}
    case G.Slots.Con{_, _} 0n:
      {==}
    case G.Slots.Con{_, t} 1n+p:
      Laws.kept_write_runs_branch_zero_more_times(t, p, v)

def NatIsPos(n: Nat) -> Type:
  match n:
    case 0n:
      Empty
    case 1n+p:
      Unit

def one_ne_zero(e: {1n == 0n : Nat}) -> Empty:
  %e : NatIsPos(_)
  Unit{}

def Laws.ignored_write_reruns_branch():
  e => one_ne_zero(e)

law get_snoc:
  for s: G.Store
  for k: Nat
  for -x: G.Steps
  for lt: {Nat.is_lt(k, G.size(s)) == True{} : Bool}
  {G.get(G.snoc(s, x), k) == G.get(s, k) : G.Steps}

def get_snoc(s, k, x, lt):
  match s k:
    case G.Store.Nil{} 0n:
      Empty.absurd({G.get(G.snoc(G.Store.Nil{}, x), 0n) == G.get(G.Store.Nil{}, 0n) : G.Steps}, false_ne_true(lt))
    case G.Store.Nil{} 1n+p:
      Empty.absurd({G.get(G.snoc(G.Store.Nil{}, x), 1n+p) == G.get(G.Store.Nil{}, 1n+p) : G.Steps}, false_ne_true(lt))
    case G.Store.Con{_, _} 0n:
      {==}
    case G.Store.Con{_, t} 1n+p:
      get_snoc(t, p, x, lt)

def Laws.fork_keeps_parent_checkpoints(s, k, p, n, lt):
  get_snoc(s, k, G.prefix(G.get(s, p), n), lt)

def StepsIsShort(s: G.Steps) -> Type:
  match s:
    case G.Steps.Nil{}:
      Empty
    case G.Steps.Con{_, t}:
      match t:
        case G.Steps.Nil{}:
          Unit
        case G.Steps.Con{_, _}:
          Empty

def short_ne_long(e: {G.Steps.Con{1n, G.Steps.Nil{}} == G.Steps.Con{1n, G.Steps.Con{2n, G.Steps.Nil{}}} : G.Steps}) -> Empty:
  %e : StepsIsShort(_)
  Unit{}

def Laws.in_place_fork_changes_parent():
  e => short_ne_long(e)
