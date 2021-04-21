#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(matrix_product)
{
  tt::MatrixXXd<2, 3, tt::ColMajor> md1(1, 4, 2, 5, 3, 6);
  tt::MatrixXXd<3, 2, tt::ColMajor> md2(7, 9, 11, 8, 10, 12);
  tt::MatrixXXd<2, 2, tt::ColMajor> md3(58, 139, 64, 154);
  CHECK(tt::eq(tt::matmul(md1, md2), md3));

  tt::MatrixXXd<3, 3, tt::ColMajor> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

  CHECK(tt::eq(tt::matmul(m, tt::IdentityMatrix<double, 3>()), m));
  CHECK(tt::eq(tt::matmul(tt::IdentityMatrix<double, 3>(), m), m));

  tt::MatrixXXd<3, 2, tt::ColMajor> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  tt::MatrixXXd<2, 3, tt::ColMajor> m2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  tt::MatrixXXd<3, 3, tt::ColMajor> m3(9, 12, 15, 19, 26, 33, 29, 40, 51);

  CHECK(tt::eq(tt::matmul(m1, m2), m3));
  CHECK(tt::isSymmetric(tt::matmul(tt::transpose<2>(m), m)));
}

HOST_DEVICE_TEST_CASE(identity_matrix_unit_vector)
{
  CHECK(tt::eq(tt::IdentityMatrix<double, 3>(), tt::MatrixXXd<3, 3, tt::ColMajor>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)));
  CHECK(tt::eq(tt::IdentityMatrix<double, tt::DYN>(3), tt::MatrixXXd<3, 3, tt::ColMajor>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)));
  CHECK(tt::eq(tt::IdentityMatrix<double, 3>(), tt::fromSupplier<3, 3>([](size_t r, size_t c){return r == c ? 1 : 0;})));

  CHECK(tt::eq(tt::IdentityMatrix<double, 3>(), tt::total<1>(tt::UnitVectors<double, 3>())));
}

HOST_DEVICE_TEST_CASE(basic_rotation)
{
  CHECK(tt::eq(tt::BasicRotationMatrix<double, 0, 1, 5>(math::consts<double>::PI), tt::BasicRotationMatrix<double, 0, 1, 5>(-math::consts<double>::PI), math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(tt::matmul(tt::BasicRotationMatrix<double, 0, 1, 5>(5), tt::BasicRotationMatrix<double, 0, 1, 5>(-5)), tt::IdentityMatrix<double, 5>(), math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(tt::matmul(tt::BasicRotationMatrix<double, 0, 1, 3>(math::consts<double>::PI / 2), tt::Vector3d(1, 0, 0)), tt::Vector3d(0, 1, 0), math::functor::eq_real<double>(1e-6)));
}

HOST_DEVICE_TEST_CASE(tensor_homogenize_dehomogenize)
{
  CHECK(tt::eq(tt::dehomogenize(tt::homogenize(tt::Vector3i(3, 6, 9))), tt::Vector3i(3, 6, 9)));
  CHECK(tt::eq(tt::dehomogenize(tt::homogenize(tt::Vector3i(3, 6, 9)) * 3), tt::Vector3i(3, 6, 9)));
}

HOST_DEVICE_TEST_CASE(tensor_homogeneous_transformation)
{
  CHECK(tt::eq(tt::matmul(tt::homogenizeRotation(tt::BasicRotationMatrix<double, 0, 1, 3>(3)), tt::homogenizeRotation(tt::BasicRotationMatrix<double, 0, 1, 3>(-3))), tt::IdentityMatrix<double, 4>(), math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(tt::matmul(tt::homogenizeTranslation(tt::Vector3d(1, 2, 3)), tt::homogenizeTranslation(tt::Vector3d(-1, -2, -3))), tt::IdentityMatrix<double, 4>(), math::functor::eq_real<double>(1e-6)));
}

HOST_DEVICE_TEST_CASE(diag_trace)
{
  tt::Matrix2i md3(58, 139, 64, 154);

  CHECK(tt::eq(tt::diag(md3), tt::Vector2i(58, 154)));
  CHECK(tt::trace(md3) == 58 + 154);
}

HOST_DEVICE_TEST_CASE(tensor_quaternion)
{
  CHECK(tt::eq(
    tt::quaternion::fromAxisAngle(tt::Vector3f(1.0, 0.0, 0.0), 1.3),
    tt::quaternion::inverse(tt::quaternion::fromAxisAngle(tt::Vector3f(1.0, 0.0, 0.0), -1.3)),
    math::functor::eq_real<float>(1e-6)));

  CHECK(tt::eq(
    tt::matmul(tt::quaternion::toMatrix(tt::quaternion::fromAxisAngle(tt::Vector3f(1.0, 0.0, 0.0), 1.3)), tt::quaternion::toMatrix(tt::quaternion::fromAxisAngle(tt::Vector3f(1.0, 0.0, 0.0), -1.3))),
    tt::IdentityMatrix<float, 3>(),
    math::functor::eq_real<float>(1e-6)));

  CHECK(tt::eq(
    tt::quaternion::toMatrix(tt::quaternion::fromAxisAngle(tt::Vector3f(1.0, 0.0, 0.0), 1.3)),
    tt::transpose<2>(tt::quaternion::toMatrix(tt::quaternion::fromAxisAngle(tt::Vector3f(1.0, 0.0, 0.0), -1.3))),
    math::functor::eq_real<float>(1e-6)));

  auto q = tt::quaternion::fromAxisAngle(tt::Vector3f(1.0, 0.0, 0.0), 1.3);
  CHECK(tt::eq(
    tt::quaternion::fromMatrix(tt::quaternion::toMatrix(q)),
    q,
    math::functor::eq_real<float>(1e-6)));
}

HOST_DEVICE_TEST_CASE(closed_form_inverse_2)
{
  tt::MatrixXXd<2, 2, tt::ColMajor> m(1, 2, 3, 2);
  tt::MatrixXXd<2, 2, tt::ColMajor> mi;
  tt::MatrixXXd<2, 2, tt::ColMajor> mii;

  CHECK(tt::op::ClosedFormInverse<2>()(mi, m));
  CHECK(tt::op::ClosedFormInverse<2>()(mii, mi));

  CHECK(tt::eq(mii, m, math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(tt::matmul(m, mi), tt::IdentityMatrix<double, 2>(), math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(tt::matmul(mi, mii), tt::IdentityMatrix<double, 2>(), math::functor::eq_real<double>(1e-6)));
}

HOST_DEVICE_TEST_CASE(closed_form_inverse_3)
{
  tt::MatrixXXd<3, 3, tt::ColMajor> m(1, 2, 3, 2, 4, 1, 4, 3, 1);
  tt::MatrixXXd<3, 3, tt::ColMajor> mi;
  tt::MatrixXXd<3, 3, tt::ColMajor> mii;

  CHECK(tt::op::ClosedFormInverse<3>()(mi, m));
  CHECK(tt::op::ClosedFormInverse<3>()(mii, mi));

  CHECK(tt::eq(mii, m, math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(matmul(m, mi), tt::IdentityMatrix<double, 3>(), math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(matmul(mi, mii), tt::IdentityMatrix<double, 3>(), math::functor::eq_real<double>(1e-6)));
}
