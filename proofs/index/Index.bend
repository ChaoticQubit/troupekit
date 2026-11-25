import Base

type Scores is Data:
  Scores.Nil{}
  Scores.Cons{head: Nat, tail: Scores}

type Ranks is Data:
  Ranks.Nil{}
  Ranks.Cons{id: Nat, tail: Ranks}

type Lists is Data:
  Lists.Nil{}
  Lists.Cons{head: Ranks, tail: Lists}

def len(xs: Scores) -> Nat:
  match xs:
    case Scores.Nil{}:
      0n
    case Scores.Cons{h, t}:
      1n+len(t)

def insert(+x: Nat, xs: Scores) -> Scores:
  match xs:
    case Scores.Nil{}:
      Scores.Cons{x, Scores.Nil{}}
    case Scores.Cons{+h, t}:
      insert_at(Nat.is_ge(x, h), x, h, t)

def insert_at(ge: Bool, +x: Nat, +h: Nat, t: Scores) -> Scores:
  match ge:
    case True{}:
      Scores.Cons{x, Scores.Cons{h, t}}
    case False{}:
      Scores.Cons{h, insert(x, t)}

def sort(xs: Scores) -> Scores:
  match xs:
    case Scores.Nil{}:
      Scores.Nil{}
    case Scores.Cons{+h, t}:
      insert(h, sort(t))

def take(xs: Scores, k: Nat) -> Scores:
  match xs k:
    case Scores.Nil{} _:
      Scores.Nil{}
    case Scores.Cons{h, t} 0n:
      Scores.Nil{}
    case Scores.Cons{h, t} 1n+p:
      Scores.Cons{h, take(t, p)}

def top_k(k: Nat, xs: Scores) -> Scores:
  take(sort(xs), k)

def sorted(xs: Scores) -> Bool:
  match xs:
    case Scores.Nil{}:
      True{}
    case Scores.Cons{+h, t}:
      sorted_from(h, t)

def sorted_from(+h: Nat, t: Scores) -> Bool:
  match t:
    case Scores.Nil{}:
      True{}
    case Scores.Cons{+n, rest}:
      Bool.and(Nat.is_ge(h, n), sorted_from(n, rest))

def weight(rank: Nat) -> Nat:
  Nat.sub(1000n, rank)

def contrib(+id: Nat, ranks: Ranks, +rank: Nat) -> Nat:
  match ranks:
    case Ranks.Nil{}:
      0n
    case Ranks.Cons{+r, t}:
      contrib_at(Nat.is_eq(id, r), id, t, rank)

def contrib_at(hit: Bool, +id: Nat, t: Ranks, +rank: Nat) -> Nat:
  match hit:
    case True{}:
      weight(rank)
    case False{}:
      contrib(id, t, 1n+rank)

def rrf(+id: Nat, lists: Lists) -> Nat:
  match lists:
    case Lists.Nil{}:
      0n
    case Lists.Cons{h, t}:
      Nat.add(contrib(id, h, 1n), rrf(id, t))
