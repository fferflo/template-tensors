#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(consecutive_transform)
{
  // Rigid
  {
    tt::geometry::transform::Rigid<float, 3> t1(// TODO: pose from rpy and offset
      tt::euler_rotation_3d<float, 2, 0, 1>(math::to_rad(5.0f), math::to_rad(10.0f), math::to_rad(150.0f)),
      tt::Vector3f(1, 5, 2)
    );
    tt::geometry::transform::Rigid<float, 3> t2(
      tt::euler_rotation_3d<float, 2, 0, 1>(math::to_rad(3.0f), math::to_rad(-23.0f), math::to_rad(-14.0f)),
      tt::Vector3f(-31, 25, 12)
    );
    tt::geometry::transform::Rigid<float, 3> t12 = t1 * t2;

    tt::Vector3f v(3, 4, 5);
    CHECK(tt::eq(t1(t2(v)), t12(v), math::functor::eq_real<float>(1e-5)));
  }

  // ScaledRigid
  {
    tt::geometry::transform::ScaledRigid<float, 3> t1(// TODO: pose from rpy and offset
      tt::euler_rotation_3d<float, 2, 0, 1>(math::to_rad(5.0f), math::to_rad(10.0f), math::to_rad(150.0f)),
      tt::Vector3f(1, 5, 2),
      -2
    );
    tt::geometry::transform::ScaledRigid<float, 3> t2(
      tt::euler_rotation_3d<float, 2, 0, 1>(math::to_rad(3.0f), math::to_rad(-23.0f), math::to_rad(-14.0f)),
      tt::Vector3f(-31, 25, 12),
      13
    );
    tt::geometry::transform::ScaledRigid<float, 3> t12 = t1 * t2;

    tt::Vector3f v(3, 4, 5);
    CHECK(tt::eq(t1(t2(v)), t12(v), math::functor::eq_real<float>(1e-4)));
  }
}

HOST_DEVICE_TEST_CASE(inverse_times_self)
{
  // Rigid
  {
    tt::geometry::transform::Rigid<float, 3> t(
      tt::euler_rotation_3d<float, 2, 0, 1>(math::to_rad(5.0f), math::to_rad(10.0f), math::to_rad(150.0f)),
      tt::Vector3f(1, 5, 2)
    );
    CHECK(tt::eq(t * t.inverse(), tt::geometry::transform::Rigid<float, 3>(), math::functor::eq_real<float>(1e-5)));
    CHECK(tt::eq(t.inverse() * t, tt::geometry::transform::Rigid<float, 3>(), math::functor::eq_real<float>(1e-5)));
  }

  // ScaledRigid
  {
    tt::geometry::transform::ScaledRigid<float, 3> t(
      tt::euler_rotation_3d<float, 2, 0, 1>(math::to_rad(5.0f), math::to_rad(10.0f), math::to_rad(150.0f)),
      tt::Vector3f(1, 5, 2),
      4.2
    );
    CHECK(tt::eq(t * t.inverse(), tt::geometry::transform::ScaledRigid<float, 3>(), math::functor::eq_real<float>(1e-4)));
    CHECK(tt::eq(t.inverse() * t, tt::geometry::transform::ScaledRigid<float, 3>(), math::functor::eq_real<float>(1e-4)));
  }
}

HOST_DEVICE_TEST_CASE(multiply)
{
  tt::geometry::transform::Rigid<float, 3> t1(// TODO: pose from rpy and offset
    tt::euler_rotation_3d<float, 2, 0, 1>(math::to_rad(5.0f), math::to_rad(10.0f), math::to_rad(150.0f)),
    tt::Vector3f(1, 5, 2)
  );
  tt::geometry::transform::Rigid<float, 3> t2(
    tt::euler_rotation_3d<float, 2, 0, 1>(math::to_rad(3.0f), math::to_rad(-23.0f), math::to_rad(-14.0f)),
    tt::Vector3f(-31, 25, 12)
  );
  tt::geometry::transform::Rigid<float, 3> t3 = t1;
  t3 *= t2;

  CHECK(tt::eq(t3, t1 * t2, math::functor::eq_real<float>(1e-5)));
}

HOST_DEVICE_TEST_CASE(homogeneous)
{
  tt::geometry::transform::Rigid<float, 3> t(// TODO: pose from rpy and offset
    tt::euler_rotation_3d<float, 2, 0, 1>(math::to_rad(5.0f), math::to_rad(10.0f), math::to_rad(150.0f)),
    tt::Vector3f(1, 5, 2)
  );

  tt::Vector3f v(3, 4, 5);
  CHECK(tt::eq(t(v), tt::dehomogenize(matmul(t.matrix(), tt::homogenize(v))), math::functor::eq_real<float>(1e-5)));
}

HOST_DEVICE_TEST_CASE(look_at)
{
  tt::Vector3f v(3, 2, -5);
  float camera_distance = 10;
  size_t num = 16;
  for (float angle = 0; angle <= 2 * math::consts<float>::PI; angle += 2 * math::consts<float>::PI / num)
  {
    tt::geometry::transform::Rigid<float, 3> t = tt::geometry::transform::lookAt(
      v + tt::Vector3f(math::cos(angle) * camera_distance, 0, math::sin(angle) * camera_distance),
      v,
      tt::Vector3f(0, 1, 0)
    );
    CHECK(tt::eq(t(v), tt::Vector3f(0, 0, camera_distance), math::functor::eq_real<float>(1e-5)));
  }
}
