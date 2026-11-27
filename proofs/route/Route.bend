import Base

type Target is Data:
  Target{on_device: Bool, eligible: Bool}

type Targets is Data:
  Targets.Nil{}
  Targets.Cons{head: Target, tail: Targets}

def Target.on_device(t: Target) -> Bool:
  match t:
    case Target{on_device, _}:
      on_device

def Target.eligible(t: Target) -> Bool:
  match t:
    case Target{_, eligible}:
      eligible

def cons_if(c: Bool, h: Target, t: Targets) -> Targets:
  match c:
    case True{}:
      Targets.Cons{h, t}
    case False{}:
      t

def only_on_device(ts: Targets) -> Targets:
  match ts:
    case Targets.Nil{}:
      Targets.Nil{}
    case Targets.Cons{+h, t}:
      cons_if(Target.on_device(h), h, only_on_device(t))

def only_eligible(ts: Targets) -> Targets:
  match ts:
    case Targets.Nil{}:
      Targets.Nil{}
    case Targets.Cons{+h, t}:
      cons_if(Target.eligible(h), h, only_eligible(t))

def drop_remote(when: Bool, ts: Targets) -> Targets:
  match when:
    case True{}:
      only_on_device(ts)
    case False{}:
      ts

def decide(ts: Targets, offline: Bool, private: Bool) -> Targets:
  only_eligible(drop_remote(private, drop_remote(offline, ts)))

def all_on_device(ts: Targets) -> Bool:
  match ts:
    case Targets.Nil{}:
      True{}
    case Targets.Cons{h, t}:
      Bool.and(Target.on_device(h), all_on_device(t))
