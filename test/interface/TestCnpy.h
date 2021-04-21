#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(cnpy_test)
{
  boost::filesystem::path test_dir(TEST_DIR);

  tt::MatrixXXT<float, 8, 8> m1;
  tt::for_each<2>([](tt::Vector2s pos, float& val){val = pos(0) + 1982 * pos(1);}, m1);

  tt::cnpy::save(test_dir / "cnpy_test.npy", m1);
  tt::cnpy::save(test_dir / "cnpy_test.npz", m1);

  auto m2 = tt::cnpy::load<float, 2>(test_dir / "cnpy_test.npy");
  auto m3 = tt::cnpy::load<float, 2>(test_dir / "cnpy_test.npz");

  BOOST_CHECK(tt::eq(m1, m2));
  BOOST_CHECK(tt::eq(m1, m3));
}
