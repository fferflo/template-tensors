#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_TEST_CASE(tf_vec_mat_interface)
{
  tt::AllocMatrixi<mem::alloc::host_heap, tt::ColMajor> m(3, 3);
  tt::for_each<2>([]__host__(tt::Vector2s pos, int32_t& el){el = pos(0) * 5 + pos(1);}, m);
  tt::Vector3f v(1, 2, 3);

  CHECK(tt::eq(m, tt::fromTfMat(tt::toTfMat(m)), math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(m, tt::fromTfMat(tt::toTfMat(tt::fromTfMat(tt::toTfMat(m)))), math::functor::eq_real<double>(1e-6)));
  CHECK(tt::eq(v, tt::fromTfVec(tt::toTfVec(v)), math::functor::eq_real<double>(1e-6)));
}

HOST_TEST_CASE(tf_transform_interface)
{
  tt::geometry::transform::Rigid<float, 3> t = tt::geometry::transform::lookAt(
      tt::Vector3f(3, 0, 10),
      tt::Vector3f(0, 2, 0),
      tt::Vector3f(0, 1, 0)
    );

  CHECK(tt::eq(t, tt::geometry::fromTf<float>(tt::geometry::toTf(t)), math::functor::eq_real<float>(1e-5)));
}
