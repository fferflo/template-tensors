#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_TEST_CASE(cv_tensor_interface)
{
  tt::AllocMatrixi<mem::alloc::host_heap, tt::RowMajor> m(50, 50);
  tt::for_each<2>([]__host__ __device__(tt::Vector2s pos, int32_t& el){el = pos(0) * 5 + pos(1);}, m);

  ::cv::Mat cv1 = tt::toCv(m);
  CHECK((void*) cv1.data == (void*) &m());
  CHECK(tt::eq(m, tt::fromCv<int32_t>(cv1)));

  const ::cv::Mat cv2 = tt::toCv(m);
  CHECK(tt::eq(m, tt::fromCv<int32_t>(cv2)));

  CHECK(tt::eq(m, tt::fromCv<int32_t>(tt::toCv(m))));
  CHECK(tt::eq(m, tt::fromCv<int32_t>(const_cast<const ::cv::Mat&&>(tt::toCv(m)))));

  CHECK(::cv::countNonZero(tt::toCv(const_cast<const decltype(m)&>(m)) != cv2) == 0);
  CHECK(::cv::countNonZero(tt::toCv(2 * m / 2) != cv2) == 0);
  CHECK(::cv::countNonZero(tt::toCv(tt::eval(m)) != cv2) == 0);
}

BOOST_AUTO_TEST_CASE(opencv_loading)
{
  auto cv_image_bgr = tt::opencv::load<tt::VectorXT<uint8_t, 3>>(IMAGE_PATH, cv::IMREAD_COLOR);

  BOOST_CHECK(cv_image_bgr.rows() == IMAGE_HEIGHT && cv_image_bgr.cols() == IMAGE_WIDTH);
}
