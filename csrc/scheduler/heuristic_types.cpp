#include <exceptions.h>
#include <scheduler/heuristic_types.h>
namespace nvfuser {

std::string toString(ScheduleHeuristic sh) {
  switch (sh) {
    case ScheduleHeuristic::NoOp:
      return "no-op";
    case ScheduleHeuristic::PointWise:
      return "pointwise";
    case ScheduleHeuristic::Reduction:
      return "reduction";
    case ScheduleHeuristic::Persistent:
      return "persistent";
    case ScheduleHeuristic::Transpose:
      return "transpose";
    case ScheduleHeuristic::Matmul:
      return "matmul";
    default:
      NVF_ERROR(false, "undefined schedule");
  }
  return "";
}

std::ostream& operator<<(std::ostream& os, ScheduleHeuristic sh) {
  os << toString(sh);
  return os;
}

} // namespace nvfuser
