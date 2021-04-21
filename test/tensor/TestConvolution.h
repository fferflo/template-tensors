#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(convolution_single_result)
{
  tt::TensorT<double, tt::RowMajor, 1, 1, 4, 4> in(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  tt::TensorT<double, tt::RowMajor, 1, 1, 4, 4> filter(3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10, 15, 16, 13, 14);
  tt::TensorT<double, tt::RowMajor, 1, 1, 1, 1> out(5);

  out = tt::conv(in, filter);

  CHECK(out() == tt::dot(in, filter));
}
