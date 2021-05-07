#include <template_tensors/TemplateTensors.h>
#include <boost/python.hpp>

template <typename TAllocator>
struct Storage
{
  tt::AllocMatrixT<float, TAllocator, tt::RowMajor> data; // Tensorflow requires row-major data

  Storage(int rows, int cols)
    : data(rows, cols)
  {
  }

  void store(::boost::python::object input)
  {
    auto result = tt::boost::python::dispatch::FromTensorflow<
      metal::list<float>,
      metal::numbers<2>,
      metal::numbers<TAllocator::MEMORY_TYPE>
    >(input)(util::functor::assign_to(data));
    if (!result)
    {
      throw std::invalid_argument(result.error());
    }
  }

  ::boost::python::object load()
  {
    return tt::boost::python::toTensorflow(data);
  }
};

BOOST_PYTHON_MODULE(${TEST_MODULE})
{
  Py_Initialize(); // TODO: wrap these in class

  boost::python::class_<Storage<mem::alloc::host_heap>>("StorageCpu", boost::python::init<int, int>())
    .def("store", &Storage<mem::alloc::host_heap>::store)
    .def("load", &Storage<mem::alloc::host_heap>::load)
  ;

#ifdef __CUDACC__
  boost::python::class_<Storage<mem::alloc::device>>("StorageGpu", boost::python::init<int, int>())
    .def("store", &Storage<mem::alloc::device>::store)
    .def("load", &Storage<mem::alloc::device>::load)
  ;
#endif
}
