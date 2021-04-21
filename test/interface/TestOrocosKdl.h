#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(kdl_conversion)
{
  tt::geometry::transform::Rigid<double, 3> transform;
  transform.getRotation() = tt::euler_rotation_3d<double, 2, 0>(0.1, 0.2);
  transform.getTranslation() = tt::Vector3f(1, 2, 3);

  BOOST_CHECK(tt::eq(transform, tt::fromKdl(tt::toKdlFrame(transform)), math::functor::eq_real<double>(1e-4)));
}
