#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(test_logprob)
{
  numeric::LogProb<float> p;
  p = 0;
  double d = static_cast<double>(p);

  CHECK(p == d);
  CHECK(p == 0);

  p = 0.5;
  CHECK(p > 0.25);
  CHECK(p < 0.75);
}

HOST_DEVICE_TEST_CASE(test_fixed_floatingpoint)
{
  numeric::FixedFloatingPoint<int, 8> f;
  f = 0;
  double d = static_cast<double>(f);

  CHECK(f == d);
  CHECK(f == 0);

  f = 0.5;
  CHECK(f > 0.25);
  CHECK(f < 0.75);

  f += 0.5;
  CHECK(f == 1);

  f -= 0.5;
  CHECK(f == 0.5);

  f += 0.5;
  f *= 0.5;
  CHECK(f * 2 == 1);

  f /= 2;
  CHECK(f * 4 == 1);

  f = -1;
  CHECK(f == -1);

  f *= 0.5;
  CHECK(f == -0.5);
}
