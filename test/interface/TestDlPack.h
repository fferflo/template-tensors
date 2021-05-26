#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

BOOST_AUTO_TEST_CASE(dlpack_conversions_host)
{
  tt::AllocTensorT<int, mem::alloc::host_heap, tt::RowMajor, 3> tensor1(4, 5, 6);
  tt::for_each<3>([](tt::Vector3s pos, int& value){
    value = pos(0) + 81723 * pos(1) + 182723523 * pos(2);
  }, tensor1);

  auto dl = tt::toDlPack(tensor1);
  auto tensor2 = tt::fromDlPack<int, 3, mem::HOST>(std::move(dl));

  BOOST_CHECK(tt::eq(tensor1, tensor2));
}

#ifdef __CUDACC__
BOOST_AUTO_TEST_CASE(dlpack_conversions_device)
{
  tt::AllocTensorT<int, mem::alloc::device, tt::RowMajor, 3> tensor1(4, 5, 6);
  tt::for_each<3>([]__host__ __device__(tt::Vector3s pos, int& value){
    value = pos(0) + 81723 * pos(1) + 182723523 * pos(2);
  }, tensor1);

  auto dl = tt::toDlPack(tensor1);
  auto tensor2 = tt::fromDlPack<int, 3, mem::DEVICE>(std::move(dl));

  BOOST_CHECK(tt::eq(mem::toHost(tensor1), mem::toHost(tensor2)));
}
#endif
