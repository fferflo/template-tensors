#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(cppflow_conversions_host)
{
  tt::AllocTensorT<int, mem::alloc::host_aligned_alloc<64>, tt::RowMajor, 3> tensor1(4, 5, 6);
  tt::for_each<3>([](tt::Vector3s pos, int& value){
    value = pos(0) + 6 * pos(1) + 3 * pos(2);
  }, tensor1);

  auto cppflow = tt::toCppFlow(tensor1);
  cppflow = 2 * cppflow;
  auto tensor2 = tt::fromCppFlow<int, 3, mem::HOST>(util::move(cppflow));

  BOOST_CHECK(tt::eq(2 * tensor1, tensor2));
}

#ifdef __CUDACC__
BOOST_AUTO_TEST_CASE(cppflow_conversions_device)
{
  tt::AllocTensorT<int, mem::alloc::device, tt::RowMajor, 3> tensor1(1000, 1000, 30);
  tt::for_each<3>([]__host__ __device__(tt::Vector3s pos, int& value){
    value = pos(0) + 81723 * pos(1) + 182723523 * pos(2);
  }, tensor1);

  auto cppflow = tt::toCppFlow(tensor1);
  cppflow = 2 * cppflow;
  auto tensor2 = tt::fromCppFlow<int, 3, mem::HOST>(cppflow); // TODO: why is this copied to host by cppflow

  BOOST_CHECK(tt::eq(mem::toHost(2 * tensor1), mem::toHost(tensor2)));
}
#endif
