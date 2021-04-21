#include <template_tensors/TemplateTensors.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

struct Storage
{
  tt::AllocMatrixT<float, mem::alloc::host_heap, tt::ColMajor> data;

  Storage(int rows, int cols)
    : data(rows, cols)
  {
  }

  void store(::boost::python::object input)
  {
    data = tt::boost::python::fromCupy<float, 2>(input);
  }

  ::boost::python::object load()
  {
    return tt::boost::python::toCupy(data);
  }
};

BOOST_PYTHON_MODULE(${TEST_MODULE})
{
  Py_Initialize(); // TODO: wrap these in class

  boost::python::class_<Storage>("Storage", boost::python::init<int, int>())
    .def("store", &Storage::store)
    .def("load", &Storage::load)
  ;
}
