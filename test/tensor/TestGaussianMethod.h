#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(gauss_full_rank)
{
  tt::MatrixXXd<3, 3, tt::ColMajor> A(1, 2, 3, 4, 4, 3, 2, 1, 2);
  tt::Vector3d b(4, 3, 1);
  tt::MatrixXXd<3, 4, tt::ColMajor> Ab = tt::concat<1>(A, b);

  tt::Vector3d x;
  CHECK(tt::op::GaussSolver<float>(1e-6)(x, Ab));
  tt::eq(tt::matmul(A, x), b, math::functor::eq_real<double>(1e-6));

  CHECK(tt::op::GaussSolver<float>(1e-6)(x, A, b));
  tt::eq(tt::matmul(A, x), b, math::functor::eq_real<double>(1e-6));
}

HOST_DEVICE_TEST_CASE(gauss_solution_not_unique)
{
  tt::MatrixXXd<3, 3, tt::ColMajor> A(1, 1, 3, 4, 4, 3, 2, 2, 2);
  tt::Vector3d b(4, 4, 1);
  tt::MatrixXXd<3, 4, tt::ColMajor> Ab = tt::concat<1>(A, b);

  tt::Vector3d x;
  CHECK(!tt::op::GaussSolver<float>(1e-6)(x, Ab));
  CHECK(!tt::op::GaussSolver<float>(1e-6)(x, A, b));
}

HOST_DEVICE_TEST_CASE(gauss_no_solution)
{
  tt::MatrixXXd<3, 3, tt::ColMajor> A(1, 1, 3, 4, 4, 3, 2, 2, 2);
  tt::Vector3d b(4, 3, 1);
  tt::MatrixXXd<3, 4, tt::ColMajor> Ab = tt::concat<1>(A, b);

  tt::Vector3d x;
  CHECK(!tt::op::GaussSolver<float>(1e-6)(x, Ab));
  CHECK(!tt::op::GaussSolver<float>(1e-6)(x, A, b));
}

HOST_DEVICE_TEST_CASE(gauss_inverse)
{
  tt::MatrixXXd<3, 3, tt::ColMajor> m(1, 2, 3, 2, 4, 1, 4, 3, 1);
  tt::MatrixXXd<3, 3, tt::ColMajor> mi;
  tt::MatrixXXd<3, 3, tt::ColMajor> mii;

  CHECK(tt::op::GaussInverse<double>(1e-6)(mi, m));
  CHECK(tt::op::GaussInverse<double>(1e-6)(mii, mi));

  CHECK(tt::eq(mii, m, math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(tt::matmul(m, mi), tt::IdentityMatrix<double, 3>(), math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(tt::matmul(mi, mii), tt::IdentityMatrix<double, 3>(), math::functor::eq_real<double>(1e-6)));
}
