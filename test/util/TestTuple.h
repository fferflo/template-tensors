#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(tuple_get)
{
  ::tuple::Tuple<int, float, char> tuple(1, 2.0f, 'c');
  CHECK(::tuple::get<0>(tuple) == 1);
  CHECK(::tuple::get<1>(tuple) == 2.0f);
  CHECK(::tuple::get<2>(tuple) == 'c');
  CHECK(::tuple::get<int>(tuple) == 1);
  CHECK(::tuple::get<float>(tuple) == 2.0f);
  CHECK(::tuple::get<char>(tuple) == 'c');

  ::tuple::get<1>(tuple) = 3.0f;
  CHECK(::tuple::get<1>(tuple) == 3.0f);
}

HOST_DEVICE_TEST_CASE(tuple_funcs)
{
  ::tuple::Tuple<int, float, double> tuple(1, 2.0f, 3.0);
  CHECK(::tuple::for_all([](int a, float b, double c){return a + b + c;}, tuple) == 6);
}

HOST_DEVICE_TEST_CASE(tuple_eq)
{
  ::tuple::Tuple<int, float, double> tuple1(1, 2.0f, 3.0);
  ::tuple::Tuple<int, float, double> tuple2(2, 2.0f, 3.0);
  CHECK(::tuple::eq(tuple1, tuple1));
  CHECK(!::tuple::eq(tuple1, tuple2));
}

HOST_DEVICE_TEST_CASE(tuple_shape)
{
  ::tuple::Tuple<int, float, double> tuple1(1, 2.0f, 3.0);
  ::tuple::Tuple<float, double> tuple2(2.0f, 3.0);
  ::tuple::Tuple<float, double, int> tuple3(2.0f, 3.0, 2);
  ::tuple::pop_front(tuple1);
  CHECK(::tuple::eq(tuple2, ::tuple::pop_front(tuple1)));
  CHECK(::tuple::eq(::tuple::append<int>(tuple2, 2), tuple3));
}
