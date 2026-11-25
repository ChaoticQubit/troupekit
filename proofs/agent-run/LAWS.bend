import Base
import ./AgentRun.bend as A

law step_keeps_model_calls_within_max_turns:
  for r: A.Run
  for a: A.Answer
  for ok: {Nat.is_le(A.Run.model_calls(r), A.Run.max_turns(r)) == True{} : Bool}
  {Nat.is_le(A.Run.model_calls(A.step(r, a)), A.Run.max_turns(A.step(r, a))) == True{} : Bool}
