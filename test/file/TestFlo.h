#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

#ifdef OPENCV_INCLUDED
#include <opencv2/video.hpp>
#include <opencv2/optflow.hpp>
#endif

#define DATA_NAME "Venus"

BOOST_AUTO_TEST_CASE(load_flo)
{
  boost::filesystem::path file = boost::filesystem::path(MIDDLEBURY_FLOW_PATH) / "gt" / DATA_NAME / "flow10.flo";
  auto flow = tt::readFlo(file);

#ifdef OPENCV_INCLUDED
#if CV_VERSION_MAJOR <= 3
  #define CV_READ_FLO cv::optflow::readOpticalFlow
#else
#define CV_READ_FLO cv::readOpticalFlow
#endif
  auto cv_flow = tt::fromCv<tt::Vector2f>(CV_READ_FLO(file.string()));
  BOOST_CHECK(tt::eq(flow, cv_flow));
#endif
}

#ifdef OPENCV_INCLUDED
BOOST_AUTO_TEST_CASE(warp_by_flo)
{
  auto flow = tt::readFlo(boost::filesystem::path(MIDDLEBURY_FLOW_PATH) / "gt" / DATA_NAME / "flow10.flo");
  auto image1 = tt::opencv::load<tt::VectorXT<uint8_t, 3>>(boost::filesystem::path(MIDDLEBURY_FLOW_PATH) / "color" / DATA_NAME / "frame10.png", cv::IMREAD_COLOR);
  auto image2 = tt::opencv::load<tt::VectorXT<uint8_t, 3>>(boost::filesystem::path(MIDDLEBURY_FLOW_PATH) / "color" / DATA_NAME / "frame11.png", cv::IMREAD_COLOR);

  auto image2_field = field::wrap::constant(
    field::interpolate(field::fromSupplier<2>(image2), interpolate::Separable<interpolate::Linear>()),
    image2.template dims<2>(),
    tt::VectorXT<uint8_t, 3>(0, 0, 255)
  );
  auto image2_warped_to_image1 = tt::elwise<2>([&](tt::Vector2s image1_pos, tt::Vector2f flow){
    return image2_field(image1_pos + tt::flip<0>(flow));
  }, flow);

  float orig_error = tt::mean(tt::static_cast_to<float>(tt::abs(tt::total<2>(image2) - tt::total<2>(image1))));
  float warped_error = tt::mean(tt::static_cast_to<float>(tt::abs(tt::total<2>(image2_warped_to_image1) - tt::total<2>(image1))));
  BOOST_CHECK(warped_error < orig_error / 2);
}
#endif
