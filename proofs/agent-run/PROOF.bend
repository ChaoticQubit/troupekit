import Base
import ./AgentRun.bend as A
import ./LAWS.bend as Laws

def BoolDisc(b: Bool) -> Type:
  match b:
    case True{}:
      Unit
    case False{}:
      Empty

def bool_clash(e: {True{} == False{} : Bool}) -> Empty:
  %e : BoolDisc(_)
  Unit{}

law succ_le:
  for c: Nat
  for m: Nat
  for e: {Nat.is_ge(c, m) == False{} : Bool}
  {Nat.is_le(1n+c, m) == True{} : Bool}

def succ_le(c, m, e):
  match c m:
    case 0n 0n:
      Empty.absurd({Nat.is_le(1n, 0n) == True{} : Bool}, bool_clash(e))
    case 0n 1n+mp:
      match mp:
        case 0n:
          {==}
        case 1n+q:
          {==}
    case 1n+cp 0n:
      Empty.absurd({Nat.is_le(2n+cp, 0n) == True{} : Bool}, bool_clash(e))
    case 1n+cp 1n+mp:
      succ_le(cp, mp, e)

law model_reply_ok:
  for a: A.Answer
  for +calls: Nat
  for +max: Nat
  for rep: Bool
  for ok: {Nat.is_le(calls, max) == True{} : Bool}
  for s: {Nat.is_le(1n+calls, max) == True{} : Bool}
  {Nat.is_le(A.Run.model_calls(A.model_reply(a, calls, max, rep)), A.Run.max_turns(A.model_reply(a, calls, max, rep))) == True{} : Bool}

def model_reply_ok(a, calls, max, rep, ok, s):
  match a:
    case A.Answer.Text{}:
      s
    case A.Answer.ToolCalls{}:
      s
    case A.Answer.PauseTurn{}:
      s
    case A.Answer.Fatal{}:
      ok
    case A.Answer.Retry{}:
      ok
    case A.Answer.Budget{}:
      ok
    case A.Answer.Pass{}:
      ok
    case A.Answer.CheckFail{}:
      ok

law call_model_ok:
  for over: Bool
  for a: A.Answer
  for +calls: Nat
  for +max: Nat
  for rep: Bool
  for e: {Nat.is_ge(calls, max) == over : Bool}
  for ok: {Nat.is_le(calls, max) == True{} : Bool}
  {Nat.is_le(A.Run.model_calls(A.call_model(over, a, calls, max, rep)), A.Run.max_turns(A.call_model(over, a, calls, max, rep))) == True{} : Bool}

def call_model_ok(over, a, calls, max, rep, e, ok):
  match over:
    case True{}:
      ok
    case False{}:
      model_reply_ok(a, calls, max, rep, ok, succ_le(calls, max, e))

law check_failed_ok:
  for c: Bool
  for +calls: Nat
  for +max: Nat
  for rep: Bool
  for ok: {Nat.is_le(calls, max) == True{} : Bool}
  {Nat.is_le(A.Run.model_calls(A.check_failed(c, calls, max, rep)), A.Run.max_turns(A.check_failed(c, calls, max, rep))) == True{} : Bool}

def check_failed_ok(c, calls, max, rep, ok):
  match c:
    case True{}:
      ok
    case False{}:
      ok

law check_ok:
  for a: A.Answer
  for +calls: Nat
  for +max: Nat
  for +rep: Bool
  for ok: {Nat.is_le(calls, max) == True{} : Bool}
  {Nat.is_le(A.Run.model_calls(A.check(a, calls, max, rep)), A.Run.max_turns(A.check(a, calls, max, rep))) == True{} : Bool}

def check_ok(a, calls, max, rep, ok):
  match a:
    case A.Answer.Pass{}:
      ok
    case A.Answer.CheckFail{}:
      check_failed_ok(Bool.and(Bool.not(rep), Nat.is_lt(calls, max)), calls, max, rep, ok)
    case A.Answer.Fatal{}:
      ok
    case A.Answer.Budget{}:
      ok
    case A.Answer.Text{}:
      ok
    case A.Answer.ToolCalls{}:
      ok
    case A.Answer.PauseTurn{}:
      ok
    case A.Answer.Retry{}:
      ok

def Laws.step_keeps_model_calls_within_max_turns(r, a, ok):
  match r:
    case A.Run{state, +calls, +max, +rep}:
      match state:
        case A.State.CallModel{}:
          call_model_ok(Nat.is_ge(calls, max), a, calls, max, rep, {==}, ok)
        case A.State.CallTools{}:
          ok
        case A.State.Check{}:
          check_ok(a, calls, max, rep, ok)
        case A.State.Repair{}:
          ok
        case A.State.Done{}:
          ok
        case A.State.Failed{}:
          ok
        case A.State.Paused{resume}:
          ok
