#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(tensor_colmajor_rowmajor)
{
  CHECK(tt::ColMajor().toIndex<11, 4, 8>(5, 3, 7) == tt::RowMajor().toIndex<8, 4, 11>(7, 3, 5));

  CHECK(tt::RowMajor().toIndex<8, 4, 11, 1, 1, 1>(7, 3, 5) == tt::RowMajor().toIndex<8, 4, 11>(7, 3, 5, 0, 0, 0));
  CHECK(tt::ColMajor().toIndex<8, 4, 11, 1, 1, 1>(7, 3, 5) == tt::ColMajor().toIndex<8, 4, 11>(7, 3, 5, 0, 0, 0));

  CHECK(tt::RowMajor().toIndex(1, 0, 0, 0) == tt::RowMajor().toIndex<1, 1, 1>());
  CHECK(tt::ColMajor().toIndex(1, 0, 0, 0) == tt::ColMajor().toIndex<1, 1, 1>());

  CHECK(tt::ColMajor().toIndex(tt::Vector3s(11, 4, 8), 5, 3, 7) == tt::RowMajor().toIndex(tt::Vector3s(8, 4, 11), 7, 3, 5));

  CHECK(tt::RowMajor().toIndex(tt::VectorXs<6>(11, 4, 8, 1, 1, 1), 5, 3, 7) == tt::RowMajor().toIndex(tt::VectorXs<3>(11, 4, 8), 5, 3, 7, 0, 0, 0));
  CHECK(tt::ColMajor().toIndex(tt::VectorXs<6>(11, 4, 8, 1, 1, 1), 5, 3, 7) == tt::ColMajor().toIndex(tt::VectorXs<3>(11, 4, 8), 5, 3, 7, 0, 0, 0));

  CHECK(tt::RowMajor().toIndex(tt::VectorXs<0>(), 0, 0, 0) == tt::RowMajor().toIndex(tt::VectorXs<3>(1, 1, 1)));
  CHECK(tt::ColMajor().toIndex(tt::VectorXs<0>(), 0, 0, 0) == tt::ColMajor().toIndex(tt::VectorXs<3>(1, 1, 1)));


  CHECK(tt::eq(tt::ColMajor().fromIndex(tt::ColMajor().toIndex<11, 4, 8>(5, 3, 7), tt::Vector3s(11, 4, 8)), tt::Vector3s(5, 3, 7)));
  CHECK(tt::eq(tt::ColMajor().fromIndex(tt::ColMajor().toIndex<11, 4, 8>(5, 3, 7), tt::Vector4s(11, 4, 8, 1)), tt::Vector4s(5, 3, 7, 0)));

  CHECK(tt::eq(tt::RowMajor().fromIndex(tt::RowMajor().toIndex<11, 4, 8>(5, 3, 7), tt::Vector3s(11, 4, 8)), tt::Vector3s(5, 3, 7)));
  CHECK(tt::eq(tt::RowMajor().fromIndex(tt::RowMajor().toIndex<11, 4, 8>(5, 3, 7), tt::Vector4s(11, 4, 8, 1)), tt::Vector4s(5, 3, 7, 0)));


  CHECK(tt::eq(tt::ColMajor().fromIndex<11, 4, 8, 1>(tt::ColMajor().toIndex<11, 4, 8>(5, 3, 7)), tt::Vector4s(5, 3, 7, 0)));
  CHECK(tt::eq(tt::RowMajor().fromIndex<11, 4, 8, 1>(tt::RowMajor().toIndex<11, 4, 8>(5, 3, 7)), tt::Vector4s(5, 3, 7, 0)));

  CHECK(tt::eq(tt::ColMajor().fromIndex(0), tt::VectorXs<0>()));
  CHECK(tt::eq(tt::RowMajor().fromIndex(0), tt::VectorXs<0>()));
}

HOST_DEVICE_TEST_CASE(tensor_index_strategy_strides)
{
#define TEST_TENSOR(...) \
  { \
    __VA_ARGS__; \
    using DimSeq1 = tt::dimseq_t<decltype(m1)>; \
    const size_t RANK = metal::size<DimSeq1>::value; \
    tt::VectorXs<RANK> strides = m1.getIndexStrategy().toStride(m1.dims()); \
    tt::AllocTensorT<double, mem::alloc::heap, tt::Stride<RANK>, RANK> m2(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, tt::Stride<RANK>(strides), m1.dims()); \
    mem::copy<mem::LOCAL, mem::LOCAL>(m2.getArray().data(), m1.getArray().data(), m1.size()); \
    CHECK(tt::eq(m1, m2)); \
  }

  TEST_TENSOR(tt::TensorT<double, tt::ColMajor, 12> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
  TEST_TENSOR(tt::TensorT<double, tt::ColMajor, 3, 4> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
  TEST_TENSOR(tt::TensorT<double, tt::ColMajor, 3, 2, 2> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
  TEST_TENSOR(tt::TensorT<double, tt::RowMajor, 12> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
  TEST_TENSOR(tt::TensorT<double, tt::RowMajor, 3, 4> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
  TEST_TENSOR(tt::TensorT<double, tt::RowMajor, 3, 2, 2> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))

#undef TEST_TENSOR

  CHECK(tt::eq(tt::Stride<3>(1, 2, 3).template toStride<7>(), tt::Stride<5>(1, 2, 3, 0, 0).template toStride<7>()));
}

HOST_DEVICE_TEST_CASE(strided_tensor)
{
  tt::MatrixXXui<3, 4, tt::ColMajor> m(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

  array::AllocArray<uint32_t, mem::alloc::heap> array(12);
  for (size_t i = 0; i < 12; i++)
  {
    array[i] = i + 1;
  }

  tt::AllocMatrixui<mem::alloc::heap, tt::Stride<2>> strided_matrix(tt::Stride<2>(tt::Vector2s(1, 3)), 3, 4);
  CHECK(tt::eq(strided_matrix.dims(), m.dims()));
  mem::copy<mem::LOCAL, mem::LOCAL>(strided_matrix.getArray().data(), array.data(), m.size());

  CHECK(tt::eq(m, strided_matrix));

  CHECK(tt::Stride<2>(1, 3).getSize(3, 4) == 12);
  CHECK(tt::Stride<2>(1, 3).getSize(tt::Vector2s(3, 4)) == 12);
}

HOST_DEVICE_TEST_CASE(symmetric_matrix)
{
  const size_t ROWSCOLS = 5;

  tt::MatrixXXd<ROWSCOLS, ROWSCOLS, tt::SymmetricMatrixUpperTriangleRowMajor> m1;
  size_t i = 0;
  for (size_t r = 0; r < m1.dim(0); r++)
  {
    for (size_t c = r; c < m1.dim(1); c++)
    {
      CHECK(tt::SymmetricMatrixUpperTriangleRowMajor().template toIndex<ROWSCOLS, ROWSCOLS>(r, c) == i);
      i++;
      m1(r, c) = r * r + c * c + r * c;
    }
  }

  tt::MatrixXXd<ROWSCOLS, ROWSCOLS, tt::SymmetricMatrixLowerTriangleRowMajor> m2;
  i = 0;
  for (size_t r = 0; r < m2.dim(0); r++)
  {
    for (size_t c = 0; c <= r; c++)
    {
      CHECK(tt::SymmetricMatrixLowerTriangleRowMajor().template toIndex<ROWSCOLS, ROWSCOLS>(r, c) == i);
      i++;
      m2(r, c) = r * r + c * c + r * c;
    }
  }

  CHECK(tt::eq(m1, m2));
}

HOST_DEVICE_TEST_CASE(tensor_test_morton_for_loop_by_tensor_assign)
{
  tt::AllocTensorT<int32_t, mem::alloc::heap, tt::ColMajor, 3> m1(5, 4, 3);
  tt::AllocTensorT<int32_t, mem::alloc::heap, tt::MortonForLoop<3>, 3> m2(m1.dims());
  tt::AllocTensorT<int32_t, mem::alloc::heap, tt::RowMajor, 3> m3(m2.dims());

  tt::for_each<3>([]__host__ __device__(tt::Vector3s pos, int32_t& el){el = pos(0) * 5 + 15 * pos(1) - pos(0);}, m1);
  m2 = m1;
  m3 = m1;

  CHECK(tt::eq(m1, m2));
  CHECK(tt::eq(m1, m3));
}

HOST_DEVICE_TEST_CASE(tensor_morton_for_loop)
{
  CHECK(tt::eq(tt::MortonForLoop<3>().fromIndex(tt::MortonForLoop<3>().toIndex<11, 4, 8>(5, 3, 7), tt::Vector3s(11, 4, 8)), tt::Vector3s(5, 3, 7)));
  CHECK(tt::eq(tt::MortonForLoop<3>().fromIndex<4>(tt::MortonForLoop<3>().toIndex<11, 4, 8>(5, 3, 7), tt::Vector4s(11, 4, 8, 1)), tt::Vector4s(5, 3, 7, 0)));
}

HOST_DEVICE_TEST_CASE(tensor_morton_divide_and_conquer)
{
  CHECK(tt::eq(tt::MortonDivideAndConquer<3>().fromIndex(tt::MortonDivideAndConquer<3>().toIndex<11, 4, 8>(5, 3, 7), tt::Vector3s(11, 4, 8)), tt::Vector3s(5, 3, 7)));
  CHECK(tt::eq(tt::MortonDivideAndConquer<3>().fromIndex<4>(tt::MortonDivideAndConquer<3>().toIndex<11, 4, 8>(5, 3, 7), tt::Vector4s(11, 4, 8, 1)), tt::Vector4s(5, 3, 7, 0)));

  CHECK(tt::MortonForLoop<3>().toIndex<11, 4, 8>(5, 3, 7) == tt::MortonDivideAndConquer<3>().toIndex<11, 4, 8>(5, 3, 7));
  if (sizeof(size_t) * 8 >= 64)
  {
    CHECK(tt::MortonForLoop<2>()         .toIndex(tt::Vector2s((2 << 19) + 1, (2 << 19) + 1), 2 << 19, 2 << 19)
       == tt::MortonDivideAndConquer<2>().toIndex(tt::Vector2s((2 << 19) + 1, (2 << 19) + 1), 2 << 19, 2 << 19));
  }
}

HOST_DEVICE_TEST_CASE(tensor_test_morton_divide_and_conquer_by_tensor_assign)
{
  tt::AllocTensorT<int32_t, mem::alloc::heap, tt::ColMajor, 3> m1(5, 4, 3);
  tt::AllocTensorT<int32_t, mem::alloc::heap, tt::MortonDivideAndConquer<3>, 3> m2(m1.dims());
  tt::AllocTensorT<int32_t, mem::alloc::heap, tt::RowMajor, 3> m3(m2.dims());

  tt::for_each<3>([]__host__ __device__(tt::Vector3s pos, int32_t& el){el = pos(0) * 5 + 15 * pos(1) - pos(0);}, m1);
  m2 = m1;
  m3 = m1;

  CHECK(tt::eq(m1, m2));
  CHECK(tt::eq(m1, m3));
}
