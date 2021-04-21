#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(array_tensor)
{
  tt::Vector3ui vui(1, 2, 3);
  CHECK(vui(0) == 1);
  CHECK(vui(1) == 2);
  CHECK(vui(2) == 3);

  tt::MatrixXXd<3, 4, tt::ColMajor> md(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  CHECK(md(0, 0) == 1.0);
  CHECK(md(2, 0) == 3.0);
  CHECK(md(2, 2) == 9.0);

  tt::AllocVectorui<mem::alloc::heap> vui2(3);
  vui2 = vui;
  CHECK(vui2(0) == 1);
  CHECK(vui2(1) == 2);
  CHECK(vui2(2) == 3);

  tt::AllocMatrixd<mem::alloc::heap> md2(3, 4);
  md2 = md;

  CHECK(md2(0, 0) == 1.0);
  CHECK(md2(2, 0) == 3.0);
  CHECK(md2(2, 2) == 9.0);
}

HOST_DEVICE_TEST_CASE(tensor_equality)
{
  tt::Vector3ui vui1(1, 2, 3);
  tt::Vector3ui vui2(1, 2, 4);
  CHECK(tt::all(vui1 == vui1));
  CHECK(tt::any(vui1 != vui2));

  tt::Matrix34d md1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::Matrix34d md2(1.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  CHECK(tt::all(md1 == md1));
  CHECK(tt::any(md1 != md2));

  CHECK(tt::eq(vui1, vui1));
  CHECK(tt::neq(vui1, vui2));
  CHECK(tt::neq(vui1, md1));
}

HOST_DEVICE_TEST_CASE(tensor_of_tensors)
{
  tt::VectorXT<tt::Vector2i, 2> v1(tt::Vector2i(1, 2), tt::Vector2i(3, 4));
  tt::VectorXT<tt::Vector2i, 2> v2(tt::Vector2i(2, 4), tt::Vector2i(6, 8));

  CHECK(tt::sum(tt::sum<tt::Vector2i>(v1 * 2 - v2)) == 0);
}

HOST_DEVICE_TEST_CASE(tensor_volatile)
{
  tt::Vector3i v1(1, 2, 3);
  volatile tt::Vector3i v2;
  v2 = v1;

  CHECK(tt::all(v1 == v2));
}

HOST_DEVICE_TEST_CASE(tensor_eval)
{
  CHECK(tt::eq(tt::eval(tt::Vector3ui(1, 2, 3) + tt::Vector3ui(1, 2, 3)), tt::Vector3ui(2, 4, 6)));
  CHECK(tt::eq(tt::eval(tt::static_cast_to<uint32_t>(tt::Vector3f(1.2, 2.2, 3.2))), tt::Vector3ui(1, 2, 3)));

  tt::MatrixXXd<3, 4, tt::ColMajor> md(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  tt::AllocMatrixd<mem::alloc::heap> md2(3, 4);
  md2 = md;
  tt::AllocMatrixd<mem::alloc::heap> md3(md2);
  CHECK(tt::eq(eval(md2), md3));
}

HOST_DEVICE_TEST_CASE(tensor_assign_elementtype)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  m = 1.0;
  CHECK(tt::all(m == 1.0));

  tt::VectorXT<tt::Vector2i, 2> v(tt::Vector2i(2, 4), tt::Vector2i(6, 8));
  v = 1;
  CHECK(tt::sum(tt::sum<tt::Vector2i>(v)) == 4);
}

HOST_DEVICE_TEST_CASE(tensor_iterator)
{
  tt::MatrixXXi<3, 4, tt::ColMajor> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  auto m2 = 2 * m;
  int sum = 0;
  for (int d : m2)
  {
    sum += d;
  }
  CHECK(sum == tt::sum(m) * 2);

  sum = 0;
  for (int& d : m)
  {
    sum += d;
  }
  CHECK(sum * 2 == tt::sum(m2));

  sum = 0;
  for (int d : 2 * m)
  {
    sum += d;
  }
  CHECK(sum == tt::sum(m) * 2);
}
