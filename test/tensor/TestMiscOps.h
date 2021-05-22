#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(tensor_misc_ops)
{
  CHECK(tt::isSymmetric(tt::MatrixXXd<3, 3, tt::ColMajor>(1.0, 2.0, 1.0, 2.0, 5.0, 4.0, 1.0, 4.0, 3.0)));
  CHECK(!tt::isSymmetric(tt::MatrixXXd<3, 3, tt::ColMajor>(1.0, 2.0, 1.0, 2.0, 5.0, 4.0, 1.0, 5.0, 3.0)));
  CHECK(tt::all(tt::normalize(13 * tt::Vector2d(0.6, 0.8)) == tt::Vector2d(0.6, 0.8)));
  CHECK(tt::dot(tt::Vector2s(1, 2), tt::Vector2s(-2, 1)) == 0);
  CHECK(tt::length(tt::Vector2f(3, 4)) == 5);
  CHECK(tt::distance(tt::Vector2f(4, 6), tt::Vector2f(1, 2)) == 5);

  CHECK(tt::angle(tt::Vector3f(1, 0, 0), tt::Vector3f(0,  1, 0))       - math::consts<float>::PI / 2 <= 1e-4);
  CHECK(tt::angle(tt::Vector3f(1, 0, 0), tt::Vector3f(-1, 0, 0))       - math::consts<float>::PI     <= 1e-4);
  CHECK(tt::acute_angle(tt::Vector3f(1, 0, 0), tt::Vector3f(-1, 0, 0)) - 0                           <= 1e-4);

  array::LocalArray<uint32_t, 4> array(0, 1, 2, 3);
  CHECK(tt::eq(tt::ref<tt::ColMajor, 4>(array), tt::ref<tt::ColMajor>(array, 4)));
  CHECK(tt::eq(tt::ref<tt::ColMajor, 2, 2>(array), tt::ref<tt::ColMajor>(array, 2, 2)));

  CHECK(tt::eq(tt::cross(tt::Vector3ui(1, 0, 0), tt::Vector3ui(0, 1, 0)), tt::Vector3ui(0, 0, 1)));
  CHECK(tt::eq(tt::matmul(tt::cross_matrix(tt::Vector3ui(1, 0, 0)), tt::Vector3ui(0, 1, 0)), tt::Vector3ui(0, 0, 1)));

  CHECK(tt::eq(tt::Vector3s(0, 1, 2), tt::fromSupplier<3>([](size_t r){return r;})));
  CHECK(tt::eq(tt::Vector3s(0, 1, 2), tt::fromSupplier<tt::DimSeq<3>>([](size_t r){return r;})));
  CHECK(tt::eq(tt::Vector3s(0, 1, 2), tt::fromSupplier([](size_t r){return r;}, 3)));
  CHECK(tt::eq(tt::partial<>(tt::Vector3s(0, 1, 2)), tt::coordinates<3>()));
  CHECK(tt::eq(tt::partial<>(tt::Vector3s(0, 1, 2)), tt::coordinates<tt::DimSeq<3>>()));
  CHECK(tt::eq(tt::partial<>(tt::Vector3s(0, 1, 2)), tt::coordinates(3)));
  CHECK(tt::eq(tt::MatrixXXs<2, 2, tt::ColMajor>(0, 1, 2, 3), tt::fromSupplier<2, 2>([](tt::Vector2s pos){return pos(0) + 2 * pos(1);})));

  tt::Vector2f v(1.0, 3.0);
  CHECK(tt::eq(v, tt::toCartesian(tt::toPolar(v)), math::functor::eq_real<float>(1e-5)));

  CHECK(tt::all(tt::argmax<1>(tt::Vector3s(0, 1, 2)) == tt::Vector1s(2)));
  CHECK(tt::all(tt::argmin<2>(tt::MatrixXXi<2, 2, tt::ColMajor>(3, 1, 2, -1)) == tt::Vector2s(1, 1)));

  CHECK(tt::all(tt::VectorXT<int8_t, 1024>(0) == 0));
}

HOST_DEVICE_TEST_CASE(tensor_streamop)
{
  tt::MatrixXXui<3, 4, tt::ColMajor> m(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

  tt::stringstream<char> str;
  str << m;
  CHECK(tt::eq(tt::to_string<char>("[[1 2 3][4 5 6][7 8 9][10 11 12]]"), str.str()));
}

HOST_DEVICE_TEST_CASE(tensor_ordering)
{
  tt::AllocVectorT<size_t, mem::alloc::heap> v(2);

  CHECK(tt::lt(tt::Vector1s(), tt::Vector2s()));
  CHECK(!tt::lt(tt::Vector3s(), tt::Vector2s()));
  CHECK(tt::lt(tt::Vector1s(), v));
  CHECK(!tt::lt(tt::Vector3s(), v));

  CHECK(tt::lt(tt::Vector2s(1, 2), tt::Vector2s(2, 2)));
  CHECK(tt::lt(tt::Vector2s(2, 2), tt::Vector2s(2, 3)));
  CHECK(!tt::lt(tt::Vector2s(3, 2), tt::Vector2s(2, 2)));
  CHECK(!tt::lt(tt::Vector2s(2, 4), tt::Vector2s(2, 3)));

  v = tt::Vector2s(2, 2);
  CHECK(tt::lt(tt::Vector2s(1, 2), v));
  v = tt::Vector2s(2, 3);
  CHECK(tt::lt(tt::Vector2s(2, 2), v));
  v = tt::Vector2s(2, 2);
  CHECK(!tt::lt(tt::Vector2s(3, 2), v));
  v = tt::Vector2s(2, 3);
  CHECK(!tt::lt(tt::Vector2s(2, 4), v));
}

HOST_DEVICE_TEST_CASE(tensor_iterating)
{
  tt::Vector3i v(0, 1, 2);

  int i1 = 0;
  for (int i2 : v)
  {
    CHECK(i1 == i2);
    i1++;
  }
}
