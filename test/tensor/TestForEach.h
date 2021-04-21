#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(tensor_local_for_each)
{
  tt::Vector3s v;
  tt::op::LocalForEach::for_each<1>([]__host__ __device__(tt::Vector1s pos, size_t& el){el = pos();}, v);

  CHECK(tt::eq(v, tt::Vector3s(0, 1, 2)));
}

HOST_TEST_CASE(tensor_host_array_for_each)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> h1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::MatrixXXd<3, 4, tt::ColMajor> h2;

  tt::op::LocalArrayForEach<>::copy(h2, h1);
  CHECK(tt::eq(h1, h2));

  tt::MatrixXXd<3, 4, tt::ColMajor> h3;
  tt::op::LocalArrayForEach<>::map([]__host__(double d){return d + 1;}, h3, h1);
  tt::op::LocalArrayForEach<>::map([]__host__(double d){return d - 1;}, h2, h3);
  CHECK(tt::eq(h1, h2));

  tt::Vector3s h4;
  tt::op::LocalArrayForEach<>::for_each<1>([]__host__(tt::Vector1s pos, size_t& el){el = pos();}, h4);
  CHECK(tt::eq(h4, tt::Vector3s(0, 1, 2)));
}

#ifdef __CUDACC__
DEVICE_TEST_CASE(tensor_device_array_for_each)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> h1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::MatrixXXd<3, 4, tt::ColMajor> h2;

  tt::op::LocalArrayForEach<>::copy(h2, h1);
  CHECK(tt::eq(h1, h2));

  tt::MatrixXXd<3, 4, tt::ColMajor> h3;
  tt::op::LocalArrayForEach<>::map([]__device__(double d){return d + 1;}, h3, h1);
  tt::op::LocalArrayForEach<>::map([]__device__(double d){return d - 1;}, h2, h3);
  CHECK(eq(h1, h2));

  tt::Vector3s h4;
  tt::op::LocalArrayForEach<>::for_each<1>([]__device__(tt::Vector1s pos, size_t& el){el = pos();}, h4);
  CHECK(tt::eq(h4, tt::Vector3s(0, 1, 2)));
}
#endif
