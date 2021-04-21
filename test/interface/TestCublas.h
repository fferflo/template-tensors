#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(tensor_cublasgemm)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> h1cm(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::MatrixXXd<4, 2, tt::ColMajor> h2cm(13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0);
  tt::MatrixXXd<3, 2, tt::ColMajor> h3cm;
  tt::MatrixXXd<3, 4, tt::RowMajor> h1rm;
  tt::MatrixXXd<4, 2, tt::RowMajor> h2rm;
  tt::MatrixXXd<3, 2, tt::RowMajor> h3rm;
  h1rm = h1cm;
  h2rm = h2cm;

  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d1cm(3, 4);
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d2cm(4, 2);
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d3cm(3, 2);
  tt::AllocMatrixd<mem::alloc::device, tt::RowMajor> d1rm(3, 4);
  tt::AllocMatrixd<mem::alloc::device, tt::RowMajor> d2rm(4, 2);
  tt::AllocMatrixd<mem::alloc::device, tt::RowMajor> d3rm(3, 2);

  d1cm = h1cm;
  d2cm = h2cm;
  d1rm = h1rm;
  d2rm = h2rm;


  tt::op::CublasGemm().gemm<double>(1, d1cm, d2cm, 0, d3cm);
  h3cm = d3cm;
  BOOST_CHECK(tt::eq(matmul(h1cm, h2cm), h3cm, math::functor::eq_real<double>(1e-6)));

  tt::op::CublasGemm().gemm<double>(1, d1rm, d2cm, 0, d3cm);
  h3cm = d3cm;
  BOOST_CHECK(tt::eq(matmul(h1cm, h2cm), h3cm, math::functor::eq_real<double>(1e-6)));

  tt::op::CublasGemm().gemm<double>(1, d1cm, d2rm, 0, d3cm);
  h3cm = d3cm;
  BOOST_CHECK(tt::eq(matmul(h1cm, h2cm), h3cm, math::functor::eq_real<double>(1e-6)));

  tt::op::CublasGemm().gemm<double>(1, d1rm, d2rm, 0, d3cm);
  h3cm = d3cm;
  BOOST_CHECK(tt::eq(matmul(h1cm, h2cm), h3cm, math::functor::eq_real<double>(1e-6)));

  tt::op::CublasGemm().gemm<double>(1, d1cm, d2cm, 0, d3rm);
  h3rm = d3rm;
  BOOST_CHECK(tt::eq(matmul(h1rm, h2rm), h3rm, math::functor::eq_real<double>(1e-6)));

  tt::op::CublasGemm().gemm<double>(1, d1rm, d2cm, 0, d3rm);
  h3rm = d3rm;
  BOOST_CHECK(tt::eq(matmul(h1rm, h2rm), h3rm, math::functor::eq_real<double>(1e-6)));

  tt::op::CublasGemm().gemm<double>(1, d1cm, d2rm, 0, d3rm);
  h3rm = d3rm;
  BOOST_CHECK(tt::eq(matmul(h1rm, h2rm), h3rm, math::functor::eq_real<double>(1e-6)));

  tt::op::CublasGemm().gemm<double>(1, d1rm, d2rm, 0, d3rm);
  h3rm = d3rm;
  BOOST_CHECK(tt::eq(matmul(h1rm, h2rm), h3rm, math::functor::eq_real<double>(1e-6)));
}
