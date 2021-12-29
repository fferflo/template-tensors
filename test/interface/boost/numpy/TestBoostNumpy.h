#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(numpy_conversions)
{
  Py_Initialize(); // TODO: wrap these in class
  ::boost::python::numpy::initialize();

  tt::AllocTensorT<int, mem::alloc::host_heap, tt::RowMajor, 3> tensor1(4, 5, 6);
  tt::for_each<3>([](tt::Vector3s pos, int& value){
    value = pos(0) + 81723 * pos(1) + 182723523 * pos(2);
  }, tensor1);

  auto python = tt::python::boost::toNumpy(tensor1);
  auto tensor2 = tt::python::boost::fromNumpy<int, 3>(python, tensor1.template dims<3>());

  BOOST_CHECK(tt::eq(tensor1, tensor2));
}

BOOST_AUTO_TEST_CASE(numpy_moments)
{
  Py_Initialize(); // TODO: wrap these in class
  ::boost::python::numpy::initialize();

  tt::AllocTensorT<float, mem::alloc::host_heap, tt::RowMajor, 3> tensor1(4, 5, 3);
  tt::for_each<3>([](tt::Vector3s pos, float& value){
    value = pos(0) + 81723 * pos(1) + 182723523 * pos(2);
  }, tensor1);
  ::boost::python::numpy::ndarray python = tt::python::boost::toNumpy(tensor1);
  auto from_python = tt::python::boost::fromNumpy<float, 3>(python);

  auto image = tt::partial<2>(from_python);

  tt::Vector3f mean = tt::sum<tt::Vector3f>(image) / (image.rows() * image.cols());

  auto diff = image - tt::broadcast(tt::SingletonT<tt::Vector3f>(mean), image.dims());

  tt::Vector3f variance = tt::sum<tt::Vector3f>(diff * diff) / (image.rows() * image.cols());
  tt::Vector3f stddev = tt::sqrt(variance);
  BOOST_CHECK(!tt::any(tt::elwise(math::functor::isnan(), stddev)));
  BOOST_CHECK(!tt::any(tt::elwise(math::functor::isinf(), stddev)));
  BOOST_CHECK(tt::all(stddev > 0));
}
