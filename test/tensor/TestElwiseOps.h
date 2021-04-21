#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(elwise_operations)
{
  CHECK(tt::eq(tt::Vector3ui(1, 2, 3) + tt::Vector3ui(1, 2, 3), tt::Vector3ui(2, 4, 6)));
  CHECK(tt::eq(tt::Vector3ui(1, 2, 3) + 1, tt::Vector3ui(2, 3, 4)));
  CHECK(tt::eq(2 + tt::Vector3ui(1, 2, 3), tt::Vector3ui(3, 4, 5)));
  CHECK(tt::eq(tt::static_cast_to<uint32_t>(tt::Vector3f(1.2, 2.2, 3.2)), tt::Vector3ui(1, 2, 3)));
  CHECK(tt::eq(tt::fmod(tt::Vector3f(2, 4, 6), tt::Vector3f(3, 3, 3)), tt::Vector3ui(2, 1, 0)));
  CHECK(tt::eq(tt::max(tt::Vector3ui(3, 2, 1), tt::Vector3ui(1, 2, 3)), tt::Vector3ui(3, 2, 3)));
  CHECK(tt::eq(tt::min(tt::Vector3ui(3, 2, 1), tt::Vector3ui(1, 2, 3)), tt::Vector3ui(1, 2, 1)));
  CHECK(tt::eq(tt::max(tt::Vector3i(3, 2, 1), 2), tt::Vector3i(3, 2, 2)));
  CHECK(tt::eq(tt::min(tt::Vector3i(3, 2, 1), 2), tt::Vector3i(2, 2, 1)));
  CHECK(tt::eq(tt::elwise<1>([]__host__ __device__(tt::Vector1s pos, size_t el){return pos() + el;}, tt::Vector3s(1)), tt::Vector3s(1, 2, 3)));

  {
    tt::Vector3i v(1, 2, 3);
    v += 1;
    CHECK(tt::eq(v, tt::Vector3i(2, 3, 4)));
    v += v;
    CHECK(tt::eq(v, tt::Vector3i(4, 6, 8)));
    v -= 2;
    CHECK(tt::eq(v, tt::Vector3i(2, 4, 6)));
    v *= 2;
    CHECK(tt::eq(v, tt::Vector3i(4, 8, 12)));
  }
  {
    tt::Vector3i v(3, 6, 7);
    v /= 2;
    CHECK(tt::eq(v, tt::Vector3i(1, 3, 3)));
  }

  CHECK(tt::eq(tt::clamp(0UL, tt::Vector3s(1, 2, 3), tt::Vector3s(4, 5, 6)), tt::Vector3s(1, 2, 3)));
  CHECK(tt::eq(tt::clamp(tt::Vector3s(1, 2, 3), 2UL, 2UL), tt::Vector3s(2, 2, 2)));
  CHECK(tt::eq(tt::clamp(tt::Vector3s(1, 2, 3), 4UL, 5UL), tt::Vector3s(4, 4, 4)));
  CHECK(tt::eq(tt::clamp(3UL, 0UL, tt::Vector3s(1, 2, 3)), tt::Vector3s(1, 2, 3)));
}
