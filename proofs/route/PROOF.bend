import Base
import ./Route.bend as R
import ./LAWS.bend as Laws

law eligible_on_device:
  for ts: R.Targets
  {R.all_on_device(R.only_eligible(R.only_on_device(ts))) == True{} : Bool}

def eligible_on_device(ts):
  match ts:
    case R.Targets.Nil{}:
      {==}
    case R.Targets.Cons{h, t}:
      match h:
        case R.Target{od, el}:
          match od:
            case True{}:
              match el:
                case True{}:
                  eligible_on_device(t)
                case False{}:
                  eligible_on_device(t)
            case False{}:
              eligible_on_device(t)

law eligible_on_device_twice:
  for ts: R.Targets
  {R.all_on_device(R.only_eligible(R.only_on_device(R.only_on_device(ts)))) == True{} : Bool}

def eligible_on_device_twice(ts):
  match ts:
    case R.Targets.Nil{}:
      {==}
    case R.Targets.Cons{h, t}:
      match h:
        case R.Target{od, el}:
          match od:
            case True{}:
              match el:
                case True{}:
                  eligible_on_device_twice(t)
                case False{}:
                  eligible_on_device_twice(t)
            case False{}:
              eligible_on_device_twice(t)

def Laws.private_plan_has_only_on_device_targets(ts, offline):
  match offline:
    case True{}:
      eligible_on_device_twice(ts)
    case False{}:
      eligible_on_device(ts)

def Laws.offline_plan_has_no_remote_targets(ts, private):
  match private:
    case True{}:
      eligible_on_device_twice(ts)
    case False{}:
      eligible_on_device(ts)
