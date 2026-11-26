import Base
import ./Route.bend as R

law private_plan_has_only_on_device_targets:
  for ts: R.Targets
  for offline: Bool
  {R.all_on_device(R.decide(ts, offline, True{})) == True{} : Bool}

law offline_plan_has_no_remote_targets:
  for ts: R.Targets
  for private: Bool
  {R.all_on_device(R.decide(ts, True{}, private)) == True{} : Bool}
