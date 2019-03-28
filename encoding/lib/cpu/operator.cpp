#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aggregate_forward", &Aggregate_Forward_CPU, "Aggregate forward (CPU)");
  m.def("aggregate_backward", &Aggregate_Backward_CPU, "Aggregate backward (CPU)");
  m.def("scaled_l2_forward", &ScaledL2_Forward_CPU, "ScaledL2 forward (CPU)");
  m.def("scaled_l2_backward", &ScaledL2_Backward_CPU, "ScaledL2 backward (CPU)");
  m.def("batchnorm_forward", &BatchNorm_Forward_CPU, "BatchNorm forward (CPU)");
  m.def("batchnorm_backward", &BatchNorm_Backward_CPU, "BatchNorm backward (CPU)");
  m.def("sumsquare_forward", &Sum_Square_Forward_CPU, "SumSqu forward (CPU)");
  m.def("sumsquare_backward", &Sum_Square_Backward_CPU, "SumSqu backward (CPU)");
}
