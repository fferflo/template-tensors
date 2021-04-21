#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

template <typename TElementType>
using StdVector = std::vector<TElementType>;

HOST_TEST_CASE(test_sparse_iterable)
{
  /* TODO: refactor this
  sparse::Iterable<float, StdVector, 100, 100> list(0.0);
  list.push_back(1.0, 3, 5);
  list.push_back(2.5, 1, 3);

  for (const sparse::CoordinateElement<float, 2>& f : list.getIterable())
  {
    CHECK(f.getElement() > 0);
  }

  TensorT<float, RowMajor, 100, 100> dense;
  tt::fill(dense, 0.0);
  dense(3, 5) = 1.0;
  dense(1, 3) = 2.5;

  CHECK(tt::all(list == dense));*/
}
