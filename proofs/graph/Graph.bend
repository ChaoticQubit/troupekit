import Base

type Slot is Data:
  Slot.Empty{}
  Slot.Full{value: Nat}

type Slots is Data:
  Slots.Nil{}
  Slots.Con{head: Slot, tail: Slots}

type Runs is Data:
  Runs.Nil{}
  Runs.Con{head: Nat, tail: Runs}

type Steps is Data:
  Steps.Nil{}
  Steps.Con{head: Nat, tail: Steps}

type Store is Data:
  Store.Nil{}
  Store.Con{head: Steps, tail: Store}

def fill(s: Slots, i: Nat, +v: Nat) -> Slots:
  match s i:
    case Slots.Nil{} _:
      Slots.Nil{}
    case Slots.Con{_, t} 0n:
      Slots.Con{Slot.Full{v}, t}
    case Slots.Con{h, t} 1n+p:
      Slots.Con{h, fill(t, p, v)}

def pick(acc: Slot, h: Slot) -> Slot:
  match h:
    case Slot.Empty{}:
      acc
    case Slot.Full{v}:
      Slot.Full{v}

def merge_from(s: Slots, acc: Slot) -> Slot:
  match s:
    case Slots.Nil{}:
      acc
    case Slots.Con{h, t}:
      merge_from(t, pick(acc, h))

def merge(s: Slots) -> Slot:
  merge_from(s, Slot.Empty{})

def finish_faulty(acc: Slot, v: Nat) -> Slot:
  Slot.Full{v}

def runs_of(h: Slot) -> Nat:
  match h:
    case Slot.Empty{}:
      1n
    case Slot.Full{_}:
      0n

def rerun(s: Slots) -> Runs:
  match s:
    case Slots.Nil{}:
      Runs.Nil{}
    case Slots.Con{h, t}:
      Runs.Con{runs_of(h), rerun(t)}

def rerun_faulty(s: Slots) -> Runs:
  match s:
    case Slots.Nil{}:
      Runs.Nil{}
    case Slots.Con{_, t}:
      Runs.Con{1n, rerun_faulty(t)}

def run_at(r: Runs, i: Nat) -> Nat:
  match r i:
    case Runs.Nil{} _:
      0n
    case Runs.Con{h, _} 0n:
      h
    case Runs.Con{_, t} 1n+p:
      run_at(t, p)

def prefix(s: Steps, n: Nat) -> Steps:
  match s n:
    case Steps.Nil{} _:
      Steps.Nil{}
    case Steps.Con{_, _} 0n:
      Steps.Nil{}
    case Steps.Con{h, t} 1n+p:
      Steps.Con{h, prefix(t, p)}

def get(s: Store, k: Nat) -> Steps:
  match s k:
    case Store.Nil{} _:
      Steps.Nil{}
    case Store.Con{h, _} 0n:
      h
    case Store.Con{_, t} 1n+p:
      get(t, p)

def size(s: Store) -> Nat:
  match s:
    case Store.Nil{}:
      0n
    case Store.Con{_, t}:
      1n+size(t)

def snoc(s: Store, x: Steps) -> Store:
  match s:
    case Store.Nil{}:
      Store.Con{x, Store.Nil{}}
    case Store.Con{h, t}:
      Store.Con{h, snoc(t, x)}

def fork(+s: Store, p: Nat, n: Nat) -> Store:
  snoc(s, prefix(get(s, p), n))

def put(s: Store, k: Nat, x: Steps) -> Store:
  match s k:
    case Store.Nil{} _:
      Store.Nil{}
    case Store.Con{_, t} 0n:
      Store.Con{x, t}
    case Store.Con{h, t} 1n+p:
      Store.Con{h, put(t, p, x)}

def fork_faulty(+s: Store, +p: Nat, n: Nat) -> Store:
  put(s, p, prefix(get(s, p), n))
