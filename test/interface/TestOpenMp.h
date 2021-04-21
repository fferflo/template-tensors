#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

#include <atomic>

BOOST_AUTO_TEST_CASE(openmp_for_each)
{
  size_t array[100];
  for (size_t i = 0; i < 100; i++)
  {
    array[i] = i;
  }
  openmp::ForEach::for_each(std::begin(array), std::end(array), [](size_t& i){i += 1;});
  for (size_t i = 0; i < 100; i++)
  {
    BOOST_CHECK(array[i] == i + 1);
  }
}

BOOST_AUTO_TEST_CASE(openmp_enabled)
{
  std::atomic<size_t> i;
  i = 0;
  #pragma omp parallel num_threads(4)
  {
    i += 1;
  }
  BOOST_CHECK(i == 4);
}