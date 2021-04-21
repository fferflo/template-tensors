#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(tbb_for_each)
{
  size_t array[100];
  for (size_t i = 0; i < 100; i++)
  {
    array[i] = i;
  }
  tbb::ForEach::for_each(std::begin(array), std::end(array), [](size_t& i){i += 1;});
  for (size_t i = 0; i < 100; i++)
  {
    BOOST_CHECK(array[i] == i + 1);
  }
}
