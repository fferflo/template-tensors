#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(thrust_interface_device_vector)
{
  thrust::device_vector<size_t> thrust;
  thrust.push_back(1);
  thrust.push_back(2);
  thrust.push_back(3);

  BOOST_CHECK(tt::eq(mem::toHost(tt::fromThrust(thrust)), tt::Vector3s(1, 2, 3)));

  const thrust::device_vector<size_t> thrust2(thrust);
  BOOST_CHECK(tt::eq(mem::toHost(tt::fromThrust(thrust2)), tt::Vector3s(1, 2, 3)));

  BOOST_CHECK(tt::eq(mem::toHost(tt::fromThrust(thrust::device_vector<size_t>(thrust))), tt::Vector3s(1, 2, 3)));
  BOOST_CHECK(tt::eq(
    mem::toHost(tt::fromThrust(const_cast<const thrust::device_vector<size_t>&&>(thrust::device_vector<size_t>(thrust)))),
    tt::Vector3s(1, 2, 3)
  ));

  BOOST_CHECK(tt::eq(
    mem::toHost(tt::fromThrust(tt::toThrust(tt::fromThrust(thrust)))),
    mem::toHost(tt::fromThrust(tt::toThrust(tt::VectorXs<3>(1, 2, 3))))
  ));
}

BOOST_AUTO_TEST_CASE(thrust_interface_host_vector)
{
  thrust::host_vector<size_t> thrust;
  thrust.push_back(1);
  thrust.push_back(2);
  thrust.push_back(3);

  BOOST_CHECK(tt::eq(mem::toHost(tt::fromThrust(thrust)), tt::Vector3s(1, 2, 3)));

  const thrust::host_vector<size_t> thrust2(thrust);
  BOOST_CHECK(tt::eq(mem::toHost(tt::fromThrust(thrust2)), tt::Vector3s(1, 2, 3)));

  BOOST_CHECK(tt::eq(mem::toHost(tt::fromThrust(thrust::host_vector<size_t>(thrust))), tt::Vector3s(1, 2, 3)));
  BOOST_CHECK(tt::eq(
    mem::toHost(tt::fromThrust(const_cast<const thrust::host_vector<size_t>&&>(thrust::host_vector<size_t>(thrust)))),
    tt::Vector3s(1, 2, 3)
  ));

  BOOST_CHECK(tt::eq(
    mem::toHost(tt::fromThrust(tt::toThrust(tt::fromThrust(thrust)))),
    mem::toHost(tt::fromThrust(tt::toThrust(tt::VectorXs<3>(1, 2, 3))))
  ));
}
