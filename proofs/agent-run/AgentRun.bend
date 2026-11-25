import Base

type State is Data:
  State.CallModel{}
  State.CallTools{}
  State.Check{}
  State.Repair{}
  State.Done{}
  State.Failed{}
  State.Paused{resume: State}

type Answer is Data:
  Answer.Text{}
  Answer.ToolCalls{}
  Answer.PauseTurn{}
  Answer.Fatal{}
  Answer.Retry{}
  Answer.Budget{}
  Answer.Pass{}
  Answer.CheckFail{}

type Run is Data:
  Run{state: State, model_calls: Nat, max_turns: Nat, repaired: Bool}

def Run.model_calls(r: Run) -> Nat:
  match r:
    case Run{_, model_calls, _, _}:
      model_calls

def Run.max_turns(r: Run) -> Nat:
  match r:
    case Run{_, _, max_turns, _}:
      max_turns

def new(max_turns: Nat) -> Run:
  Run{State.CallModel{}, 0n, max_turns, False{}}

def model_reply(a: Answer, +calls: Nat, +max: Nat, rep: Bool) -> Run:
  match a:
    case Answer.Text{}:
      Run{State.Check{}, 1n+calls, max, rep}
    case Answer.ToolCalls{}:
      Run{State.CallTools{}, 1n+calls, max, rep}
    case Answer.PauseTurn{}:
      Run{State.CallModel{}, 1n+calls, max, rep}
    case Answer.Fatal{}:
      Run{State.Failed{}, calls, max, rep}
    case Answer.Retry{}:
      Run{State.CallModel{}, calls, max, rep}
    case Answer.Budget{}:
      Run{State.Paused{State.CallModel{}}, calls, max, rep}
    case Answer.Pass{}:
      Run{State.CallModel{}, calls, max, rep}
    case Answer.CheckFail{}:
      Run{State.CallModel{}, calls, max, rep}

def call_model(over: Bool, a: Answer, +calls: Nat, +max: Nat, rep: Bool) -> Run:
  match over:
    case True{}:
      Run{State.Failed{}, calls, max, rep}
    case False{}:
      model_reply(a, calls, max, rep)

def check_failed(can_repair: Bool, +calls: Nat, +max: Nat, rep: Bool) -> Run:
  match can_repair:
    case True{}:
      Run{State.Repair{}, calls, max, True{}}
    case False{}:
      Run{State.Failed{}, calls, max, rep}

def check(a: Answer, +calls: Nat, +max: Nat, +rep: Bool) -> Run:
  match a:
    case Answer.Pass{}:
      Run{State.Done{}, calls, max, rep}
    case Answer.CheckFail{}:
      check_failed(Bool.and(Bool.not(rep), Nat.is_lt(calls, max)), calls, max, rep)
    case Answer.Fatal{}:
      Run{State.Failed{}, calls, max, rep}
    case Answer.Budget{}:
      Run{State.Paused{State.Check{}}, calls, max, rep}
    case Answer.Text{}:
      Run{State.Check{}, calls, max, rep}
    case Answer.ToolCalls{}:
      Run{State.Check{}, calls, max, rep}
    case Answer.PauseTurn{}:
      Run{State.Check{}, calls, max, rep}
    case Answer.Retry{}:
      Run{State.Check{}, calls, max, rep}

def step_in(s: State, a: Answer, +calls: Nat, +max: Nat, rep: Bool) -> Run:
  match s:
    case State.CallModel{}:
      call_model(Nat.is_ge(calls, max), a, calls, max, rep)
    case State.CallTools{}:
      Run{State.CallModel{}, calls, max, rep}
    case State.Check{}:
      check(a, calls, max, rep)
    case State.Repair{}:
      Run{State.CallModel{}, calls, max, rep}
    case State.Done{}:
      Run{State.Done{}, calls, max, rep}
    case State.Failed{}:
      Run{State.Failed{}, calls, max, rep}
    case State.Paused{resume}:
      Run{State.Paused{resume}, calls, max, rep}

def step(r: Run, a: Answer) -> Run:
  match r:
    case Run{state, calls, max, rep}:
      step_in(state, a, calls, max, rep)
