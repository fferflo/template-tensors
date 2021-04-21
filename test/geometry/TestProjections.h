#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(project_unproject)
{
  tt::Vector3f v3(3, 4, 2);
  tt::Vector4f v4(3, 4, 5, 2);

  auto p1 = tt::geometry::projection::fromSymmetricFov<float>(1, math::to_rad(45.0));
  CHECK(tt::eq(v3, p1.unproject(p1.project(v3)) * v3(2), math::functor::eq_real<float>(1e-5)));

  tt::geometry::projection::PinholeK<tt::Matrix3f> p1_as_matrix(p1);
  CHECK(tt::eq(v3, p1_as_matrix.unproject(p1_as_matrix.project(v3)) * v3(2), math::functor::eq_real<float>(1e-5)));

  tt::geometry::projection::Orthographic<float, 3> p2(tt::Vector2f(-10, -5), tt::Vector2f(20, 30));
  CHECK(tt::eq(tt::head<2>(v3), tt::head<2>(p2.unproject(p2.project(v3))), math::functor::eq_real<float>(1e-5)));

  tt::geometry::projection::Orthographic<float, 4> p3(tt::Vector3f(-20, -3, 0), tt::Vector3f(10, 20, 30));
  CHECK(tt::eq(tt::head<3>(v4), tt::head<3>(p3.unproject(p3.project(v4))), math::functor::eq_real<float>(1e-5)));
}

HOST_DEVICE_TEST_CASE(matrix_vs_project)
{
  tt::Vector3f v3(3, 4, 2);
  tt::Vector4f v4(3, 4, 5, 2);

  auto p1 = tt::geometry::projection::fromSymmetricFov<float>(1, math::to_rad(45.0));
  tt::geometry::projection::PinholeK<tt::Matrix3f> p1_as_matrix(p1);
  CHECK(tt::eq(p1_as_matrix.project(v3), p1.project(v3), math::functor::eq_real<float>(1e-5)));

  tt::geometry::projection::Orthographic<float, 3> p2(tt::Vector2f(-10, -5), tt::Vector2f(20, 30));
  CHECK(tt::eq(tt::head<2>(tt::dehomogenize(matmul(p2.matrix(-1, -100), tt::homogenize(v3)))), p2.project(v3), math::functor::eq_real<float>(1e-5)));

  tt::geometry::projection::Orthographic<float, 4> p3(tt::Vector3f(-20, -3, 0), tt::Vector3f(10, 20, 30));
  CHECK(tt::eq(tt::head<3>(tt::dehomogenize(matmul(p3.matrix(-1, -100), tt::homogenize(v4)))), p3.project(v4), math::functor::eq_real<float>(1e-5)));
}
