#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(tinyply_test)
{
  boost::filesystem::path test_dir(TEST_DIR);

  tt::VectorXT<tt::VectorXT<float, 3>, 3> rgb;
  rgb(0) = tt::VectorXT<float, 3>(1, 2, 3);
  rgb(1) = tt::VectorXT<float, 3>(4, 5, 6);
  rgb(2) = tt::VectorXT<float, 3>(7, 8, 9);

  tt::VectorXT<tt::VectorXT<int32_t, 3>, 3> indices;
  indices(0) = tt::VectorXT<int32_t, 3>(-1, -2, -3);
  indices(1) = tt::VectorXT<int32_t, 3>(-4, -5, -6);
  indices(2) = tt::VectorXT<int32_t, 3>(-7, -8, -9);

  {
    tt::tinyply::WriteProperty<float, 3> tinyply_rgb("element1", {"r", "g", "b"}, rgb);
    tt::tinyply::WriteProperty<int32_t, 3> tinyply_indices("element2", "indices", indices);

    tt::tinyply::write(test_dir / "tinyply_test1.ply", true, tinyply_rgb, tinyply_indices);
    tt::tinyply::write(test_dir / "tinyply_test2.ply", false, tinyply_rgb, tinyply_indices);
  }

  {
    tt::tinyply::ReadProperty<float, 3> tinyply_rgb("element1", {"r", "g", "b"});
    tt::tinyply::ReadProperty<int32_t, 3> tinyply_indices("element2", "indices");

    tt::tinyply::read(test_dir / "tinyply_test1.ply", tinyply_rgb, tinyply_indices);

    BOOST_CHECK(tt::eq(tt::total<1>(tinyply_rgb.data()), tt::total<1>(rgb)));
    BOOST_CHECK(tt::eq(tt::total<1>(tinyply_indices.data()), tt::total<1>(indices)));
  }

  {
    tt::tinyply::ReadProperty<float, 3> tinyply_rgb("element1", {"r", "g", "b"});
    tt::tinyply::ReadProperty<int32_t, 3> tinyply_indices("element2", "indices");

    tt::tinyply::read(test_dir / "tinyply_test2.ply", tinyply_rgb, tinyply_indices);

    BOOST_CHECK(tt::eq(tt::total<1>(tinyply_rgb.data()), tt::total<1>(rgb)));
    BOOST_CHECK(tt::eq(tt::total<1>(tinyply_indices.data()), tt::total<1>(indices)));
  }
}
