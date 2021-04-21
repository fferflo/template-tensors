#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(tensor_reduction_and_broadcasting)
{
  tt::Vector3ui vui(1, 2, 3);
  CHECK(tt::sum(vui) == 6);
  CHECK(tt::prod(vui) == 6);
  CHECK(tt::mean(vui) == 2);
  CHECK(tt::min_el(vui) == 1);
  CHECK(tt::min_el<uint32_t>(vui) == 1);
  CHECK(tt::min_el<float>(vui) == 1);
  CHECK(tt::max_el(vui) == 3);

  tt::MatrixXXT<double, 3, 4, tt::ColMajor> md(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  CHECK(sum(md) == 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0 + 10.0 + 11.0 + 12.0);
  CHECK(prod(md) == 1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0 * 8.0 * 9.0 * 10.0 * 11.0 * 12.0);
  CHECK(tt::eq(tt::reduce<1>(md, aggregator::sum<double>(0)), tt::Vector3d(22, 26, 30)));
  CHECK(tt::eq(tt::reduce<0>(md, aggregator::sum<double>(0)), tt::MatrixXXd<1, 4, tt::ColMajor>(6, 15, 24, 33)));

  CHECK(tt::eq(tt::broadcast<3, 3>(vui), tt::MatrixXXui<3, 3, tt::ColMajor>(1, 2, 3, 1, 2, 3, 1, 2, 3)));
  CHECK(tt::eq(tt::broadcast<3, tt::DYN>(vui, 3, 3), tt::MatrixXXui<3, 3, tt::ColMajor>(1, 2, 3, 1, 2, 3, 1, 2, 3)));
  CHECK(tt::eq(tt::broadcast(vui, 3, 3), tt::MatrixXXui<3, 3, tt::ColMajor>(1, 2, 3, 1, 2, 3, 1, 2, 3)));

  CHECK(tt::count(tt::MatrixXXT<bool, 3, 3>(true, false, false, false, true, true, true, false, true)) == 5);

  tt::fill(md, 0);
  CHECK(tt::sum(md) == 0);
}
