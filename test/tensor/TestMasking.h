#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(boolean_mask_assignment)
{
  tt::MatrixXXT<bool, 2, 2> mask(true, false, false, true);

  tt::MatrixXXT<int, 2, 2> matrix(1, 1, 1, 1);
  matrix[mask] = 0;
  CHECK(tt::eq(matrix, tt::MatrixXXT<int, 2, 2>(0, 1, 1, 0)));
}

HOST_DEVICE_TEST_CASE(iterable_mask_assignment)
{
  tt::VectorXT<tt::Vector2s, 2> mask(tt::Vector2s(0, 0), tt::Vector2s(1, 1));

  tt::MatrixXXT<int, 2, 2> matrix(1, 1, 1, 1);
  matrix[mask] = 0;
  CHECK(tt::eq(matrix, tt::MatrixXXT<int, 2, 2>(0, 1, 1, 0)));
}

HOST_DEVICE_TEST_CASE(line)
{
  tt::MatrixXXT<int, 5, 5> matrix(0);
  matrix[tt::geometry::discrete::line(tt::Vector2f(0.7, 0.7), tt::Vector2f(4.2, 4.2))] = 1;
  CHECK(tt::eq(matrix, tt::eye<float, 5>(), math::functor::eq_real<double>(1e-6)));
}

HOST_DEVICE_TEST_CASE(box)
{
  tt::MatrixXXT<int, 5, 5> matrix(0);
  matrix[tt::geometry::discrete::box(tt::Vector2f(1.7, 1.7), tt::Vector2f(3.2, 3.2))] = 1;
  CHECK(tt::eq(matrix, tt::pad(tt::MatrixXXT<double, 3, 3>(1), tt::Vector2s(1, 1)), math::functor::eq_real<double>(1e-6)));
}

HOST_DEVICE_TEST_CASE(sphere)
{
  {
    tt::MatrixXXT<int, 5, 5> m1(0);
    m1[tt::geometry::discrete::sphere(tt::Vector2f(2.5, 2.5), 2.5)] = 1;
    tt::MatrixXXT<int, 5, 5> m2(1);
    m2(0, 0) = m2(4, 0) = m2(0, 4) = m2(4, 4) = 0;
    CHECK(tt::eq(m1, m2, math::functor::eq_real<double>(1e-6)));
  }
}
