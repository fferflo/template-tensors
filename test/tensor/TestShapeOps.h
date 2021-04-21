#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(tensor_headtail)
{
  tt::MatrixXXd<3, 2, tt::ColMajor> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  tt::AllocMatrixd<mem::alloc::heap, tt::ColMajor> m2(m.dims());
  m2 = m;

  CHECK(tt::eq(tt::MatrixXXd<2, 1, tt::ColMajor>(1.0, 2.0), tt::head<2>(m)));
  CHECK(tt::eq(tt::MatrixXXd<2, 2, tt::ColMajor>(1.0, 2.0, 4.0, 5.0), tt::head<2, 2>(m)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(1.0, 2.0, 3.0), tt::head<3, 1>(m)));
  CHECK(tt::eq(tt::MatrixXXd<2, 1, tt::ColMajor>(1.0, 2.0), tt::head<2>(m2)));
  CHECK(tt::eq(tt::MatrixXXd<2, 2, tt::ColMajor>(1.0, 2.0, 4.0, 5.0), tt::head<2, 2>(m2)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(1.0, 2.0, 3.0), tt::head<3, 1>(m2)));

  CHECK(tt::eq(tt::MatrixXXd<2, 1, tt::ColMajor>(1.0, 2.0), tt::head(m, 2)));
  CHECK(tt::eq(tt::MatrixXXd<2, 2, tt::ColMajor>(1.0, 2.0, 4.0, 5.0), tt::head(m, 2, 2)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(1.0, 2.0, 3.0), tt::head(m, 3, 1)));
  CHECK(tt::eq(tt::MatrixXXd<2, 1, tt::ColMajor>(1.0, 2.0), tt::head(m2, 2)));
  CHECK(tt::eq(tt::MatrixXXd<2, 2, tt::ColMajor>(1.0, 2.0, 4.0, 5.0), tt::head(m2, 2, 2)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(1.0, 2.0, 3.0), tt::head(m2, 3, 1)));


  CHECK(tt::eq(tt::MatrixXXd<1, 2, tt::ColMajor>(3.0, 6.0), tt::offset<2>(m)));
  CHECK(tt::eq(tt::MatrixXXd<1, 2, tt::ColMajor>(3.0, 6.0), tt::offset<2, 0>(m)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(4.0, 5.0, 6.0), tt::offset<0, 1>(m)));
  CHECK(tt::eq(tt::MatrixXXd<1, 2, tt::ColMajor>(3.0, 6.0), tt::offset<2>(m2)));
  CHECK(tt::eq(tt::MatrixXXd<1, 2, tt::ColMajor>(3.0, 6.0), tt::offset<2, 0>(m2)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(4.0, 5.0, 6.0), tt::offset<0, 1>(m2)));

  CHECK(tt::eq(tt::MatrixXXd<1, 2, tt::ColMajor>(3.0, 6.0), tt::offset(m, 2)));
  CHECK(tt::eq(tt::MatrixXXd<1, 2, tt::ColMajor>(3.0, 6.0), tt::offset(m, 2, 0)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(4.0, 5.0, 6.0), tt::offset(m, 0, 1)));
  CHECK(tt::eq(tt::MatrixXXd<1, 2, tt::ColMajor>(3.0, 6.0), tt::offset(m2, 2)));
  CHECK(tt::eq(tt::MatrixXXd<1, 2, tt::ColMajor>(3.0, 6.0), tt::offset(m2, 2, 0)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(4.0, 5.0, 6.0), tt::offset(m2, 0, 1)));


  CHECK(tt::eq(tt::MatrixXXd<2, 1, tt::ColMajor>(5.0, 6.0), tt::tail<2>(m)));
  CHECK(tt::eq(tt::MatrixXXd<2, 2, tt::ColMajor>(2.0, 3.0, 5.0, 6.0), tt::tail<2, 2>(m)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(4.0, 5.0, 6.0), tt::tail<3, 1>(m)));
  CHECK(tt::eq(tt::MatrixXXd<2, 1, tt::ColMajor>(5.0, 6.0), tt::tail<2>(m2)));
  CHECK(tt::eq(tt::MatrixXXd<2, 2, tt::ColMajor>(2.0, 3.0, 5.0, 6.0), tt::tail<2, 2>(m2)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(4.0, 5.0, 6.0), tt::tail<3, 1>(m2)));

  CHECK(tt::eq(tt::MatrixXXd<2, 1, tt::ColMajor>(5.0, 6.0), tt::tail(m, 2)));
  CHECK(tt::eq(tt::MatrixXXd<2, 2, tt::ColMajor>(2.0, 3.0, 5.0, 6.0), tt::tail(m, 2, 2)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(4.0, 5.0, 6.0), tt::tail(m, 3, 1)));
  CHECK(tt::eq(tt::MatrixXXd<2, 1, tt::ColMajor>(5.0, 6.0), tt::tail(m2, 2)));
  CHECK(tt::eq(tt::MatrixXXd<2, 2, tt::ColMajor>(2.0, 3.0, 5.0, 6.0), tt::tail(m2, 2, 2)));
  CHECK(tt::eq(tt::MatrixXXd<3, 1, tt::ColMajor>(4.0, 5.0, 6.0), tt::tail(m2, 3, 1)));

  tt::MatrixXXi<3, 3, tt::ColMajor> m3(1, 2, 3, 4, 5, 6, 7, 8, 9);
  tt::AllocMatrixd<mem::alloc::heap, tt::ColMajor> m4(m3.dims());
  m4 = m3;

  CHECK(tt::eq(tt::MatrixXXi<3, 1>(4, 5, 6), tt::col(m3, 1)));
  CHECK(tt::eq(tt::MatrixXXi<3, 1>(4, 5, 6), tt::col<1>(m3)));
  CHECK(tt::eq(tt::MatrixXXi<1, 3>(2, 5, 8), tt::row(m3, 1)));
  CHECK(tt::eq(tt::MatrixXXi<1, 3>(2, 5, 8), tt::row<1>(m3)));

  CHECK(tt::eq(tt::MatrixXXi<3, 1>(4, 5, 6), tt::col(m4, 1)));
  CHECK(tt::eq(tt::MatrixXXi<3, 1>(4, 5, 6), tt::col<1>(m4)));
  CHECK(tt::eq(tt::MatrixXXi<1, 3>(2, 5, 8), tt::row(m4, 1)));
  CHECK(tt::eq(tt::MatrixXXi<1, 3>(2, 5, 8), tt::row<1>(m4)));
}

HOST_DEVICE_TEST_CASE(tensor_row_assign)
{
  tt::MatrixXXi<3, 2, tt::ColMajor> m1(1, 2, 3, 4, 5, 6);
  tt::MatrixXXi<3, 2, tt::ColMajor> m2(3, 2, 3, 6, 5, 6);

  tt::row<0>(m1) += tt::transpose<2>(tt::Vector2i(1, 1));
  tt::row<0>(m1) += 1;

  CHECK(tt::eq(m1, m2));
}

HOST_DEVICE_TEST_CASE(tensor_concat)
{
  CHECK(tt::eq(tt::MatrixXXd<3, 4, tt::ColMajor>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0),
                      tt::concat<1>(tt::MatrixXXd<3, 2, tt::ColMajor>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                        tt::MatrixXXd<3, 2, tt::ColMajor>(7.0, 8.0, 9.0, 10.0, 11.0, 12.0))));

  CHECK(tt::eq(tt::concat<1>(tt::concat<1>(tt::Vector3d(1, 2, 3), tt::Vector3d(4, 5, 6)), tt::Vector3d(7, 8, 9)),
                      tt::concat<1>(tt::Vector3d(1, 2, 3), tt::concat<1>(tt::Vector3d(4, 5, 6), tt::Vector3d(7, 8, 9)))));
  CHECK(tt::eq(tt::concat<1>(tt::Vector3d(1, 2, 3), tt::Vector3d(4, 5, 6), tt::Vector3d(7, 8, 9)),
                      tt::concat<1>(tt::Vector3d(1, 2, 3), tt::concat<1>(tt::Vector3d(4, 5, 6), tt::Vector3d(7, 8, 9)))));

  CHECK(tt::eq(tt::concat<0>(tt::Vector1s(0), tt::Vector1s(1)), tt::Vector2s(0, 1)));

  tt::MatrixXXd<3, 4, tt::ColMajor> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::AllocMatrixd<mem::alloc::heap> m2(3, 4);
  m2 = m1;

  CHECK(tt::eq(tt::concat<0>(m1, m2), tt::concat<0>(m2, m1)));
  CHECK(tt::eq(tt::concat<0>(m1, m1), tt::concat<0>(m2, m2)));
}

HOST_DEVICE_TEST_CASE(tensor_flip)
{
  CHECK(tt::eq(tt::flip<0>(tt::Vector3i(1, 2, 3)), tt::Vector3i(3, 2, 1)));
  CHECK(tt::eq(tt::flip<1>(tt::flip<1>(tt::Matrix2i(1, 2, 3, 4))), tt::Matrix2i(1, 2, 3, 4)));
}

HOST_DEVICE_TEST_CASE(tensor_transpose)
{
  CHECK(tt::eq(tt::transpose<2>(tt::transpose<2>(tt::Matrix2i(1, 2, 3, 4))), tt::Matrix2i(1, 2, 3, 4)));
  CHECK(tt::eq(tt::transpose<2>(tt::MatrixXXi<2, 2, tt::ColMajor>(1, 2, 3, 4)), tt::MatrixXXi<2, 2, tt::RowMajor>(1, 2, 3, 4)));
  CHECK(tt::eq(tt::transpose<2>(tt::Vector3i(1, 2, 3)), tt::MatrixXXi<1, 3, tt::ColMajor>(1, 2, 3)));
}

HOST_DEVICE_TEST_CASE(tensor_partial_total)
{
  {
    const tt::MatrixXXi<2, 3, tt::ColMajor> m(1, 2, 3, 4, 5, 6);

    auto rows = tt::partial<1>(m);
    auto cols = tt::partial<0>(m);

    CHECK(tt::eq(tt::transpose<2>(tt::concat<1>(rows(0), rows(1))), m));
    CHECK(tt::eq(tt::concat<1>(cols(0), cols(1), cols(2)), m));
  }

  {
    tt::VectorXT<tt::Vector2s, 2> v;
    v(0) = tt::Vector2s(1, 2);
    v(1) = tt::Vector2s(3, 4);

    const tt::MatrixXXs<2, 2, tt::RowMajor> m(1, 2, 3, 4);

    CHECK(tt::eq(tt::total<1>(v), m));
  }

  {
    tt::MatrixXXd<3, 4, tt::ColMajor> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);

    CHECK(tt::eq(m, tt::total<0>(tt::partial<0>(m))));
    CHECK(tt::eq(m, tt::total<1>(tt::partial<1>(m))));
  }

  {
    tt::VectorXT<tt::VectorXT<int, 1>, 3> v(tt::VectorXT<int, 1>(3), tt::VectorXT<int, 1>(5), tt::VectorXT<int, 1>(7));
    CHECK(tt::all(tt::total<1>(v) == tt::Vector3i(3, 5, 7)));
  }

  {
    tt::AllocTensorT<tt::VectorXT<int, 2>, mem::alloc::heap, tt::RowMajor, 2> probs(2, 2);
    probs(0, 0) = tt::VectorXT<int, 2>(1, 2);
    probs(0, 1) = tt::VectorXT<int, 2>(1, 2);
    probs(1, 0) = tt::VectorXT<int, 2>(2, 1);
    probs(1, 1) = tt::VectorXT<int, 2>(2, 1);

    auto argmax = tt::elwise(tt::functor::argmax<1>(), probs);
    CHECK(tt::all(tt::total<2>(argmax) == tt::elwise([](tt::Vector1s v){return v();}, argmax)));
  }
}

HOST_DEVICE_TEST_CASE(tensor_dilate)
{
  tt::MatrixXXi<2, 2, tt::ColMajor> m(1, 1, 1, 1);

  CHECK(tt::eq(tt::dilate<2>(m), tt::MatrixXXi<3, 3, tt::ColMajor>(1, 0, 1, 0, 0, 0, 1, 0, 1)));
  CHECK(tt::eq(tt::dilate<3>(m), tt::MatrixXXi<4, 4, tt::ColMajor>(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1)));

  CHECK(tt::eq(tt::dilate<tt::DimSeq<2, 2>>(m), tt::MatrixXXi<3, 3, tt::ColMajor>(1, 0, 1, 0, 0, 0, 1, 0, 1)));
  CHECK(tt::eq(tt::dilate<tt::DimSeq<3, 3>>(m), tt::MatrixXXi<4, 4, tt::ColMajor>(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1)));

  CHECK(tt::eq(tt::dilate(m, 2), tt::MatrixXXi<3, 3, tt::ColMajor>(1, 0, 1, 0, 0, 0, 1, 0, 1)));
  CHECK(tt::eq(tt::dilate(m, 3), tt::MatrixXXi<4, 4, tt::ColMajor>(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1)));

  CHECK(tt::eq(tt::dilate(m, tt::Vector2s(2, 2)), tt::MatrixXXi<3, 3, tt::ColMajor>(1, 0, 1, 0, 0, 0, 1, 0, 1)));
  CHECK(tt::eq(tt::dilate(m, tt::Vector2s(3, 3)), tt::MatrixXXi<4, 4, tt::ColMajor>(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1)));
}

HOST_DEVICE_TEST_CASE(tensor_repeat)
{
  tt::MatrixXXi<2, 2, tt::ColMajor> m(1, 2, 3, 4);

  CHECK(tt::eq(tt::repeat<tt::DimSeq<2, 2>>(m), tt::MatrixXXi<4, 4, tt::ColMajor>(1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4)));
  CHECK(tt::eq(tt::repeat<tt::DimSeq<1, 2>>(m), tt::MatrixXXi<2, 4, tt::ColMajor>(1, 2, 1, 2, 3, 4, 3, 4)));

  CHECK(tt::eq(tt::repeat(m, 2), tt::MatrixXXi<4, 4, tt::ColMajor>(1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4)));

  CHECK(tt::eq(tt::repeat(m, tt::Vector2s(2, 2)), tt::MatrixXXi<4, 4, tt::ColMajor>(1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4)));
  CHECK(tt::eq(tt::repeat(m, tt::Vector2s(1, 2)), tt::MatrixXXi<2, 4, tt::ColMajor>(1, 2, 1, 2, 3, 4, 3, 4)));
}

HOST_DEVICE_TEST_CASE(tensor_pad)
{
  tt::MatrixXXi<2, 2, tt::ColMajor> m(1, 1, 1, 1);

  CHECK(tt::eq(tt::pad<tt::CoordSeq<1, 1>, tt::CoordSeq<1, 1>>(m), tt::MatrixXXi<4, 4, tt::ColMajor>(0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0)));
  CHECK(tt::eq(tt::pad<tt::CoordSeq<1, 1>>(m), tt::MatrixXXi<4, 4, tt::ColMajor>(0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0)));
  CHECK(tt::eq(tt::pad_front<tt::CoordSeq<1, 1>>(m), tt::MatrixXXi<3, 3, tt::ColMajor>(0, 0, 0, 0, 1, 1, 0, 1, 1)));
  CHECK(tt::eq(tt::pad_back<tt::CoordSeq<1, 1>>(m), tt::MatrixXXi<3, 3, tt::ColMajor>(1, 1, 0, 1, 1, 0, 0, 0, 0)));

  CHECK(tt::eq(tt::pad<1>(m), tt::MatrixXXi<4, 4, tt::ColMajor>(0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0)));
  CHECK(tt::eq(tt::pad_front<1>(m), tt::MatrixXXi<3, 3, tt::ColMajor>(0, 0, 0, 0, 1, 1, 0, 1, 1)));
  CHECK(tt::eq(tt::pad_back<1>(m), tt::MatrixXXi<3, 3, tt::ColMajor>(1, 1, 0, 1, 1, 0, 0, 0, 0)));

  CHECK(tt::eq(tt::pad(m, tt::Vector2s(1, 1)), tt::MatrixXXi<4, 4, tt::ColMajor>(0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0)));
  CHECK(tt::eq(tt::pad_front(m, tt::Vector2s(1, 1)), tt::MatrixXXi<3, 3, tt::ColMajor>(0, 0, 0, 0, 1, 1, 0, 1, 1)));
  CHECK(tt::eq(tt::pad_back(m, tt::Vector2s(1, 1)), tt::MatrixXXi<3, 3, tt::ColMajor>(1, 1, 0, 1, 1, 0, 0, 0, 0)));

  CHECK(tt::eq(tt::pad(m, 1), tt::MatrixXXi<4, 4, tt::ColMajor>(0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0)));
  CHECK(tt::eq(tt::pad_front(m, 1), tt::MatrixXXi<3, 3, tt::ColMajor>(0, 0, 0, 0, 1, 1, 0, 1, 1)));
  CHECK(tt::eq(tt::pad_back(m, 1), tt::MatrixXXi<3, 3, tt::ColMajor>(1, 1, 0, 1, 1, 0, 0, 0, 0)));
}
