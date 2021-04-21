#ifdef SENSOR_MSGS_INCLUDED

#include <sensor_msgs/CameraInfo.h>

namespace template_tensors { // TODO: this should be geometry namespace? or move all interfaces to interface namespace to avoid confusion between tensor, geometry and other packages?

using SensorMsgsScalar = typename std::decay<decltype(std::declval<sensor_msgs::CameraInfo>().K[0])>::type;

template <typename TScalar = SensorMsgsScalar>
__host__
geometry::projection::Pinhole<TScalar, 3> fromSensorMsgs(const sensor_msgs::CameraInfo& camera_info)
{
  ASSERT(camera_info.K[1] == 0, "Invalid camera_info");
  ASSERT(camera_info.K[3] == 0, "Invalid camera_info");
  ASSERT(camera_info.K[6] == 0, "Invalid camera_info");
  ASSERT(camera_info.K[7] == 0, "Invalid camera_info");
  ASSERT(camera_info.K[8] == 1, "Invalid camera_info");
  return geometry::projection::Pinhole<TScalar, 3>(template_tensors::VectorXT<TScalar, 2>(camera_info.K[0], camera_info.K[4]), template_tensors::VectorXT<TScalar, 2>(camera_info.K[2], camera_info.K[5]));
}

template <typename TScalar = SensorMsgsScalar>
__host__
geometry::projection::Pinhole<TScalar, 3> fromSensorMsgs(const sensor_msgs::CameraInfoConstPtr& camera_info)
{
  return fromSensorMsgs<TScalar>(*camera_info);
}

} // end of ns tensor

#endif