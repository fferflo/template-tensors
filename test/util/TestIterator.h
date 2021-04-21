#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(filter_iterator)
{
  {
    tt::Vector4s in(1, 2, 3, 4);
    auto even = iterable::filter(in, [](size_t i){return i / 2 * 2 == i;});

    tt::Vector2s out;
    iterable::copy(out, even);

    CHECK(tt::eq(out, tt::Vector2s(2, 4)));
  }
}
