#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

struct DispatchTester
{
  template <typename T>
  void operator()(T)
  {
    is_float = std::is_same<typename std::decay<T>::type, float>::value;
    is_string = std::is_same<typename std::decay<T>::type, std::string>::value;
  }

  bool is_float = false;
  bool is_string = false;
};

HOST_TEST_CASE(test_dispatch_union)
{
  DispatchTester tester;

  dispatch::Union<float, std::string> u;

  u = 1.0f;
  u(tester);
  CHECK(tester.is_float);

  u = std::string("asd");
  dispatch::all(u)(tester);
  CHECK(tester.is_string);
}
