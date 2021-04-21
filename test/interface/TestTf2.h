#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(tf2_vec_mat_interface)
{
  tt::AllocMatrixi<mem::alloc::host_heap, tt::ColMajor> m(3, 3);
  tt::for_each<2>([]__host__(tt::Vector2s pos, int32_t& el){el = pos(0) * 5 + pos(1);}, m);
  tt::Vector3f v(1, 2, 3);

  BOOST_CHECK(tt::eq(m, tt::fromTf2Mat(tt::toTf2Mat(m)), math::functor::eq_real<double>(1e-6)));
  BOOST_CHECK(tt::eq(m, tt::fromTf2Mat(tt::toTf2Mat(tt::fromTf2Mat(tt::toTf2Mat(m)))), math::functor::eq_real<double>(1e-6)));
  BOOST_CHECK(tt::eq(v, tt::fromTf2Vec(tt::toTf2Vec(v)), math::functor::eq_real<double>(1e-6)));
}

BOOST_AUTO_TEST_CASE(tf2_transform_interface)
{
  tt::geometry::transform::Rigid<float, 3> t = tt::geometry::transform::lookAt(
      tt::Vector3f(3, 0, 10),
      tt::Vector3f(0, 2, 0),
      tt::Vector3f(0, 1, 0)
    );

  BOOST_CHECK(tt::eq(t, tt::geometry::fromTf2<float>(tt::geometry::toTf2(t)), math::functor::eq_real<float>(1e-5)));
}
