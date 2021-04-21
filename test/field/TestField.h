#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(interpolate_linear_vector)
{
  auto in_field = field::fromSupplier<1>(tt::Vector3i(0, 1, 2));

  auto interpolated = field::interpolate(in_field, interpolate::Separable<interpolate::Linear>());
  auto out_field = field::transform(interpolated, [](tt::Vector1f in) -> tt::Vector1f {return in / 2.0;});
  auto out_tensor = tt::fromSupplier<4>(out_field);

  CHECK(tt::eq(out_tensor, tt::Vector4f(0.0, 0.5, 1.0, 1.5), math::functor::eq_real<float>(1e-6)));
}

HOST_DEVICE_TEST_CASE(interpolate_linear_matrix)
{
  {
    auto in_field = field::fromSupplier<2>(tt::Matrix3i(0, 1, 2, 1, 2, 3, 2, 3, 4));

    auto interpolated = field::interpolate(in_field, interpolate::Separable<interpolate::Linear>());
    auto out_field = field::transform(interpolated, [](tt::Vector2f in) -> tt::Vector2f {return in / 2.0;});
    auto out_tensor = tt::fromSupplier<4, 4>(out_field);

    CHECK(tt::eq(out_tensor, tt::Matrix4f(0.0, 0.5, 1.0, 1.5, 0.5, 1, 1.5, 2.0, 1, 1.5, 2, 2.5,1.5, 2, 2.5, 3), math::functor::eq_real<float>(1e-6)));
  }

  {
    auto in_field = field::fromSupplier<2>(tt::Matrix2i(0, 1, 1, 2));

    auto interpolated = field::interpolate(in_field, interpolate::Separable<interpolate::Linear>());
    auto out_field = field::transform(interpolated, [](tt::Vector2f in) -> tt::Vector2f {return in / 4.0;});
    auto out_tensor = tt::fromSupplier<4, 4>(out_field);

    CHECK(tt::eq(out_tensor, tt::Matrix4f(0.0, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 1.0, 0.5, 0.75, 1.0, 1.25, 0.75, 1.0, 1.25, 1.5), math::functor::eq_real<float>(1e-6)));
  }
}

HOST_DEVICE_TEST_CASE(interpolate_nearest_vector)
{
  auto in_field = field::fromSupplier<1>(tt::Vector3i(0, 1, 2));

  auto interpolated = field::interpolate(in_field, interpolate::Separable<interpolate::Nearest>());
  auto out_field = field::transform(interpolated, [](tt::Vector1f in) -> tt::Vector1f {return in / 3.0;});
  auto out_tensor = tt::fromSupplier<6>(out_field);

  CHECK(tt::eq(out_tensor, tt::VectorXT<int32_t, 6>(0, 0, 1, 1, 1, 2)));
}

HOST_DEVICE_TEST_CASE(wrap_repeat)
{
  tt::Vector3i in_tensor(0, 1, 2);

  auto in_field = field::wrap::repeat<1>(in_tensor);
  auto out_field = field::transform(in_field, [](tt::Vector1f in) -> tt::Vector1f {return in - 3;});

  auto out_tensor = tt::fromSupplier<9>(out_field);

  CHECK(tt::eq(out_tensor, tt::VectorXT<int32_t, 9>(0, 1, 2, 0, 1, 2, 0, 1, 2)));
}

HOST_DEVICE_TEST_CASE(wrap_clamp)
{
  tt::Vector3i in_tensor(0, 1, 2);

  auto in_field = field::wrap::clamp<1>(in_tensor);
  auto out_field = field::transform(in_field, [](tt::Vector1f in) -> tt::Vector1f {return in - 3;});

  auto out_tensor = tt::fromSupplier<9>(out_field);

  CHECK(tt::eq(out_tensor, tt::VectorXT<int32_t, 9>(0, 0, 0, 0, 1, 2, 2, 2, 2)));
}

HOST_DEVICE_TEST_CASE(wrap_constant)
{
  tt::Vector3i in_tensor(0, 1, 2);

  auto in_field = field::wrap::constant<1>(in_tensor, -1);
  auto out_field = field::transform(in_field, [](tt::Vector1f in) -> tt::Vector1f {return in - 3;});

  auto out_tensor = tt::fromSupplier<9>(out_field);

  CHECK(tt::eq(out_tensor, tt::VectorXT<int32_t, 9>(-1, -1, -1, 0, 1, 2, -1, -1, -1)));
}

HOST_DEVICE_TEST_CASE(differentiate)
{
  auto in_field = field::wrap::clamp<1>(tt::Vector3f(0, 1, 2));

  auto diff_field = field::differentiate(in_field);
  auto out_field = field::transform(diff_field, [](tt::Vector1f in) -> tt::Vector1f {return in - 1;});

  auto out_tensor = tt::fromSupplier<5>(out_field);

  CHECK(tt::eq(tt::total<1>(out_tensor), tt::VectorXT<float, 5>(0, 0.5, 1.0, 0.5, 0), math::functor::eq_real<float>(1e-6)));
}
