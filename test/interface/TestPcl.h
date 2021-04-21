#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

BOOST_AUTO_TEST_CASE(load_pcl)
{
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  BOOST_REQUIRE(pcl::io::loadPCDFile<pcl::PointXYZ>(POINT_CLOUD_FILE, pcl_cloud) != -1);

  auto cloud = tt::fromPcl(pcl_cloud);
  auto cloud_eval = tt::eval(cloud);
  BOOST_CHECK(tt::all(tt::elwise([](const pcl::PointXYZ& p1, const pcl::PointXYZ& p2){
    return p1.x == p2.x && p1.y == p2.y && p1.z == p2.z;
  }, cloud, cloud_eval)));
}
