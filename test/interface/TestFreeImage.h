#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(freeimage_conversions)
{
  freeimage::Library freeimage_lib;

  freeimage::FreeImage freeimage = freeimage::load(IMAGE_PATH);

  auto image_24 = tt::eval<tt::RowMajor, mem::alloc::host_heap>(tt::fromFreeImageRgb<uint8_t>(freeimage));
  auto image_32 = tt::eval<tt::RowMajor, mem::alloc::host_heap>(tt::fromFreeImageRgba<uint8_t>(freeimage));

  BOOST_CHECK(image_24.rows() == IMAGE_HEIGHT && image_24.cols() == IMAGE_WIDTH);
  BOOST_CHECK(image_24.rows() == IMAGE_HEIGHT && image_24.cols() == IMAGE_WIDTH);

  BOOST_CHECK(tt::all(tt::elwise(tt::functor::eq<>(), image_24, tt::fromFreeImageRgb<uint8_t>(tt::toFreeImage(image_24)))));
  BOOST_CHECK(tt::all(tt::elwise(tt::functor::eq<>(), image_32, tt::fromFreeImageRgba<uint8_t>(tt::toFreeImage(image_32)))));

  #ifdef OPENCV_INCLUDED
    auto cv_image_bgr = tt::opencv::load<tt::VectorXT<uint8_t, 3>>(IMAGE_PATH, cv::IMREAD_COLOR);
    auto cv_image_rgb = tt::elwise(tt::functor::flip<0>(), cv_image_bgr);

    BOOST_CHECK(tt::eq(image_24, cv_image_rgb));
  #endif
}
