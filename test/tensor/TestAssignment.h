#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

#ifdef __CUDACC__
HOST_TEST_CASE(test_complex_assignment)
{
  std::vector<int> vec;
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  tt::AllocVectorT<int, mem::alloc::host_heap> points(vec.size());
  points = tt::elwise([](const int& in){return in * 2;}, tt::fromStdVector(vec));
  CHECK(tt::eq(points, tt::fromStdVector(vec) * 2));

  std::vector<int> vec2(3);
  tt::fromStdVector(vec2) = tt::elwise([](const int& in){return in / 2;}, points);
  CHECK(vec == vec2);

  std::vector<int> vec3(3);
  tt::fromStdVector(vec3) = mem::toHost(mem::toDevice(tt::fromStdVector(vec)));
  CHECK(vec == vec3);

  std::vector<int> vec4(3);
  tt::fromStdVector(vec4) = mem::toHost(mem::toDevice(mem::toHost(mem::toDevice(tt::fromStdVector(vec)))));
  CHECK(vec == vec4);
}

HOST_TEST_CASE(test_assignment_with_1temp)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> h1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::MatrixXXd<3, 4, tt::ColMajor> h2;
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d1(3, 4);

  d1 = h1;
  d1 += 1;
  h2 = d1 - 1;
  CHECK(tt::eq(h1, h2));
}

HOST_TEST_CASE(test_assignment_with_2temp)
{
  tt::MatrixXXd<3, 3, tt::ColMajor> h1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
  tt::MatrixXXd<3, 3, tt::ColMajor> h2;
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d1(3, 3);

  d1 = h1;
  d1 += 1;
  tt::transpose<2>(h2) = d1 - 1;
  CHECK(tt::eq(tt::transpose<2>(h1), h2));
}
#endif

HOST_DEVICE_TEST_CASE(test_assign_transposed)
{
  tt::AllocMatrixi<mem::alloc::heap, tt::ColMajor> m(3, 3);
  tt::for_each<2>([](tt::Vector2s pos, int32_t& el){el = pos(0) * 5 + pos(1);}, m);

  tt::AllocMatrixi<mem::alloc::heap, tt::ColMajor> m2(3, 3);
  tt::transpose<2>(m2) = m;
  CHECK(tt::eq(tt::transpose<2>(m), m2));
}
