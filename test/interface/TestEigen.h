#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(storage_interface)
{
  {
    tt::AllocMatrixi<mem::alloc::host_heap, tt::ColMajor> m(2, 2);
    tt::for_each<2>([]__host__ __device__(tt::Vector2s pos, int32_t& el){el = pos(0) * 5 + pos(1);}, m);

    auto eigen_map = tt::toEigen(m);
    auto eigen_matrix = tt::toEigen(decltype(m)(m));
    auto m2_ref = tt::fromEigen(eigen_map);
    auto m2_val = tt::fromEigen(std::move(eigen_matrix));

    BOOST_CHECK(eigen_map.data() == &m());
    BOOST_CHECK(eigen_matrix.data() != &m());
    BOOST_CHECK(&m2_ref() == &m());
    BOOST_CHECK(&m2_val() != &m());

    BOOST_CHECK(tt::eq(m, m2_ref));
    BOOST_CHECK(tt::eq(m, m2_val));
    BOOST_CHECK(tt::eq(m, tt::fromEigen(eigen_map)));
    BOOST_CHECK(eigen_map.isApprox(eigen_matrix));
    BOOST_CHECK(eigen_map.isApprox(tt::toEigen(tt::fromEigen(eigen_map))));

    BOOST_CHECK(eigen_map.isApprox(tt::toEigen(1 * m)));
  }

  {
    const tt::VectorXT<uint32_t, 3, tt::ColMajor> m(1, 2, 3);

    auto eigen_matrix = tt::toEigen(std::move(m));
    auto m2_val = tt::fromEigen(std::move(eigen_matrix));

    BOOST_CHECK(eigen_matrix.data() != &m());
    BOOST_CHECK(&m2_val() != &m());

    BOOST_CHECK(tt::eq(m, m2_val));
    BOOST_CHECK(tt::eq(m, tt::fromEigen(eigen_matrix)));
    BOOST_CHECK(eigen_matrix.isApprox(tt::toEigen(tt::fromEigen(eigen_matrix))));

    BOOST_CHECK(eigen_matrix.isApprox(tt::toEigen(1 * m)));
  }
}

BOOST_AUTO_TEST_CASE(matrix_inverse)
{
  tt::MatrixXXf<3, 3, tt::ColMajor> m(1, 3, 2, 6, 4, 1, 7, 4, 0);
  tt::Matrix3f inv;
  BOOST_CHECK(tt::op::GaussInverse<float>(1e-6)(inv, m));

  tt::Matrix3f eigen_inv1 = tt::fromEigen(tt::toEigen(m).inverse());
  tt::Matrix3f eigen_inv2;
  BOOST_CHECK(tt::EigenColPivHouseholderQRInverse<float>()(eigen_inv2, m));

  BOOST_CHECK(tt::eq(inv, eigen_inv1, math::functor::eq_real<double>(1e-6)));
  BOOST_CHECK(tt::eq(inv, eigen_inv2, math::functor::eq_real<double>(1e-6)));
}

BOOST_AUTO_TEST_CASE(solve_full_rank)
{
  tt::MatrixXXd<3, 3, tt::ColMajor> A(1, 2, 3, 4, 4, 3, 2, 1, 2);
  tt::Vector3d b(4, 3, 1);
  tt::MatrixXXd<3, 4, tt::ColMajor> Ab = tt::concat<1>(A, b);

  tt::Vector3d x;
  BOOST_CHECK(tt::EigenColPivHouseholderQRSolver<float>()(x, Ab));
  tt::eq(matmul(A, x), b, math::functor::eq_real<double>(1e-6));

  BOOST_CHECK(tt::EigenColPivHouseholderQRSolver<float>()(x, A, b));
  tt::eq(matmul(A, x), b, math::functor::eq_real<double>(1e-6));
}

BOOST_AUTO_TEST_CASE(solve_solution_not_unique)
{
  tt::MatrixXXd<3, 3, tt::ColMajor> A(1, 1, 3, 4, 4, 3, 2, 2, 2);
  tt::Vector3d b(4, 4, 1);
  tt::MatrixXXd<3, 4, tt::ColMajor> Ab = tt::concat<1>(A, b);

  tt::Vector3d x;
  BOOST_CHECK(!tt::EigenColPivHouseholderQRSolver<float>()(x, Ab));
  BOOST_CHECK(!tt::EigenColPivHouseholderQRSolver<float>()(x, A, b));
}

BOOST_AUTO_TEST_CASE(solve_no_solution)
{
  tt::MatrixXXd<3, 3, tt::ColMajor> A(1, 1, 3, 4, 4, 3, 2, 2, 2);
  tt::Vector3d b(4, 3, 1);
  tt::MatrixXXd<3, 4, tt::ColMajor> Ab = tt::concat<1>(A, b);

  tt::Vector3d x;
  BOOST_CHECK(!tt::EigenColPivHouseholderQRSolver<float>()(x, Ab));
  BOOST_CHECK(!tt::EigenColPivHouseholderQRSolver<float>()(x, A, b));
}
