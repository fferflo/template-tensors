#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

#include <random>

HOST_TEST_CASE(large_random_unequal_on_host)
{
  std::random_device r;

  tt::AllocVectorT<int, mem::alloc::host_heap> v1(1000);
  tt::AllocVectorT<int, mem::alloc::host_heap> v2(v1.dims());

  v1 = tt::random(std::default_random_engine(r()), std::uniform_int_distribution<uint32_t>(), v1.dims());
  v2 = tt::random(std::default_random_engine(r()), std::uniform_int_distribution<uint32_t>(), v2.dims());

  CHECK(tt::any(v1 != v2));
}

#ifdef __CUDACC__
DEVICE_TEST_CASE(large_random_real_unequal_on_device)
{
  tt::AllocVectorT<float, mem::alloc::device_heap> v1(1000);
  tt::AllocVectorT<float, mem::alloc::device_heap> v2(v1.dims());

  auto generator = cuda::device::XORWOW_generator();

  float min = 10.0;
  float max = 20.0;
  // TODO: add static "ADAPT" dimension size constant, so that tt::random, tt::elwise?, tt::broadcast etc dont require dimension size when assigned
  v1 = tt::random(generator, cuda::device::uniform_real_distribution<float>(min, max), v1.dims());
  v2 = tt::random(generator, cuda::device::uniform_real_distribution<float>(min, max), v2.dims());

  CHECK(tt::any(v1 != v2));
  CHECK(tt::all(min <= v1 && v1 < max));
  CHECK(tt::all(min <= v2 && v2 < max));
}

DEVICE_TEST_CASE(large_random_int_unequal_on_device)
{
  tt::AllocVectorT<int, mem::alloc::device_heap> v1(1000);
  tt::AllocVectorT<int, mem::alloc::device_heap> v2(v1.dims());

  auto generator = cuda::device::XORWOW_generator();

  int min = 10;
  int max = 20;
  // TODO: add static "ADAPT" dimension size constant, so that tt::random, tt::elwise?, tt::broadcast etc dont require dimension size when assigned
  v1 = tt::random(generator, cuda::device::uniform_int_distribution<int>(min, max), v1.dims());
  v2 = tt::random(generator, cuda::device::uniform_int_distribution<int>(min, max), v2.dims());

  CHECK(tt::any(v1 != v2));
  CHECK(tt::all(min <= v1 && v1 < max));
  CHECK(tt::all(min <= v2 && v2 < max));
}
#endif
