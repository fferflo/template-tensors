#include <template_tensors/TemplateTensors.h>

struct Storage
{
  tt::AllocMatrixT<float, mem::alloc::host_heap, tt::ColMajor> data;

  Storage(int rows, int cols)
    : data(rows, cols)
  {
  }

  void store(::boost::python::object input)
  {
    auto result = tt::python::boost::dispatch::FromNumpy<
      metal::list<float>,
      metal::numbers<2>
    >(input)(util::functor::assign_to(data));
    if (!result)
    {
      throw std::invalid_argument(result.error());
    }
  }

  ::boost::python::numpy::ndarray load()
  {
    return tt::python::boost::toNumpy(data);
  }
};

BOOST_PYTHON_MODULE(${TEST_MODULE})
{
  Py_Initialize(); // TODO: wrap these in class
  ::boost::python::numpy::initialize();

  ::boost::python::class_<Storage>("Storage", boost::python::init<int, int>())
    .def("store", &Storage::store)
    .def("load", &Storage::load)
  ;
}
