#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_TEST_CASE(tensor_device_array_for_each)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> h1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::MatrixXXd<3, 4, tt::ColMajor> h2;
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d1(3, 4);
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d2(3, 4);

  d1 = h1;
  tt::op::DeviceArrayForEach<>::copy(d2, d1);
  h2 = d2;
  CHECK(eq(h1, h2));

  tt::op::DeviceArrayForEach<>::map([]__device__(double d){return d + 1;}, d2, d1);
  tt::op::DeviceArrayForEach<>::map([]__device__(double d){return d - 1;}, d1, d2);
  h2 = d1;
  CHECK(eq(h1, h2));


  tt::Vector3s h3;
  tt::AllocVectors<mem::alloc::device> d3(3);

  tt::op::DeviceArrayForEach<>::for_each<1>([]__device__(tt::Vector1s pos, size_t& el){el = pos();}, d3);
  h3 = d3;
  CHECK(eq(h3, tt::Vector3s(0, 1, 2)));
}

HOST_TEST_CASE(tensor_device_for_each)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> h1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::MatrixXXd<3, 4, tt::ColMajor> h2;
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d1(3, 4);
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d2(3, 4);

  d1 = h1;
  tt::op::DeviceForEach<>::copy(d2, d1);
  h2 = d2;
  CHECK(tt::eq(h1, h2));

  tt::op::DeviceForEach<>::map([]__device__(double d){return d + 1;}, d2, d1);
  tt::op::DeviceForEach<>::map([]__device__(double d){return d - 1;}, d1, d2);
  h2 = d1;
  CHECK(tt::eq(h1, h2));


  tt::Vector3s h3;
  tt::AllocVectors<mem::alloc::device> d3(3);

  tt::op::DeviceForEach<>::for_each<1>([]__device__(tt::Vector1s pos, size_t& el){el = pos();}, d3);
  h3 = d3;
  CHECK(tt::eq(h3, tt::Vector3s(0, 1, 2)));
}

HOST_TEST_CASE(tensor_device_for_each_by_decider)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> h1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::MatrixXXd<3, 4, tt::ColMajor> h2;
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d1(3, 4);
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d2(3, 4);

  d1 = h1;
  tt::map([]__device__(double d){return d + 1;}, d2, d1 - 1);
  h2 = d2;
  CHECK(tt::eq(h1, h2));
}

HOST_TEST_CASE(tensor_device_for_each_elwise_assign)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> h1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::MatrixXXd<3, 4, tt::ColMajor> h2;
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d1(3, 4);

  d1 = h1;
  d1 += d1;
  h2 = d1;
  CHECK(tt::eq(h1 + h1, h2));

  d1 = h1;
  d1 += 1;
  h2 = d1;
  CHECK(tt::eq(h1, h2 - 1));
}
