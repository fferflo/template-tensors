#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(tensor_tuple_conversion)
{
  tt::Vector3s v(1, 2, 3);
  CHECK(tt::eq(v, tt::fromTuple(tt::toTuple(v))));
}
