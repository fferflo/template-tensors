#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(soil_loading)
{
  soil::Image soil = soil::load(IMAGE_PATH, SOIL_LOAD_RGB);
  auto soil_image = tt::fromSoil<tt::VectorXT<uint8_t, 3>>(soil);

  BOOST_CHECK(soil_image.rows() == IMAGE_HEIGHT && soil_image.cols() == IMAGE_WIDTH);

#ifdef OPENCV_INCLUDED
  auto cv_image_bgr = tt::opencv::load<tt::VectorXT<uint8_t, 3>>(IMAGE_PATH, cv::IMREAD_COLOR);
  auto cv_image_rgb = tt::elwise(tt::functor::flip<0>(), cv_image_bgr);

  BOOST_CHECK(tt::eq(soil_image, cv_image_rgb));
#endif
}
