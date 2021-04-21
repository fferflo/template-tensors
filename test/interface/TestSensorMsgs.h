#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(camera_info_conversion)
{
  float fx = 3;
  float fy = 5;
  float cx = 7;
  float cy = 9;

  sensor_msgs::CameraInfo camera_info;
  camera_info.K[0] = fx;
  camera_info.K[1] = 0;
  camera_info.K[2] = cx;
  camera_info.K[3] = 0;
  camera_info.K[4] = fy;
  camera_info.K[5] = cy;
  camera_info.K[6] = 0;
  camera_info.K[7] = 0;
  camera_info.K[8] = 1;

  tt::geometry::projection::Pinhole<float, 3> p;
  p = tt::fromSensorMsgs<float>(camera_info);

  BOOST_CHECK(p.getFocalLengths()(0) == fx);
  BOOST_CHECK(p.getFocalLengths()(1) == fy);
  BOOST_CHECK(p.getPrincipalPoint()(0) == cx);
  BOOST_CHECK(p.getPrincipalPoint()(1) == cy);
}
