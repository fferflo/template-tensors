#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(tensor_std_interface)
{
  std::vector<int> els;
  for (size_t i = 0; i < 10; i++)
  {
    els.push_back(i);
  }

  CHECK(tt::eq(tt::fromStdVector(els), tt::fromStdVector(tt::toStdVector(tt::fromStdVector(els)))));
  CHECK(tt::eq(2 * tt::fromStdVector(els), tt::fromStdVector(tt::toStdVector(2 * tt::fromStdVector(els)))));

  CHECK(tt::eq(tt::Vector3i(1, 2, 3), tt::fromStdVector(std::vector<int32_t>({1, 2, 3}))));
}

BOOST_AUTO_TEST_CASE(sort_by_key)
{
  std::vector<int> v1{4, 3, 6, 2, 33, -6};
  std::vector<int> v2 = v1;

  std::sort_by_key(v1.begin(), v1.end(), v2.begin());

  for (size_t i = 0; i < v1.size(); i++)
  {
    if (i < v1.size() - 1)
    {
      CHECK(v1[i] < v1[i + 1]);
    }
    CHECK(v1[i] == v2[i]);
  }
}
