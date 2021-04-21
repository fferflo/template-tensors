#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

template <typename TTensorDest, typename TTensorSrc>
__global__
void kernel_test_tensor_to_kernel(TTensorDest dest, TTensorSrc src)
{
  dest = src;
}

HOST_TEST_CASE(tensor_to_kernel)
{
  tt::TensorT<double, tt::RowMajor, 4, 4> src_h(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  tt::AllocTensorT<double, mem::alloc::device, tt::RowMajor, 2> src_d(src_h.dims());
  src_d = src_h;

  tt::TensorT<double, tt::RowMajor, 4, 4> dest_h;
  tt::AllocTensorT<double, mem::alloc::device, tt::RowMajor, 2> dest_d(dest_h.dims());


  CUDA_CHECK_CALL(kernel_test_tensor_to_kernel<<<1, 1>>>(mem::toKernel(dest_d), mem::toKernel(src_d + src_d)));
  CUDA_CHECK_CALL(cudaDeviceSynchronize());
  dest_h = dest_d;
  CHECK(tt::eq(2 * src_h, dest_h));

  CUDA_CHECK_CALL(kernel_test_tensor_to_kernel<<<1, 1>>>(mem::toKernel(dest_d), mem::toKernel(2 * src_d)));
  CUDA_CHECK_CALL(cudaDeviceSynchronize());
  tt::copy(dest_h, dest_d);
  CHECK(tt::eq(2 * src_h, dest_h));


  const auto v = 2 * src_d;
  CUDA_CHECK_CALL(kernel_test_tensor_to_kernel<<<1, 1>>>(mem::toKernel(dest_d), mem::toKernel(v)));
  CUDA_CHECK_CALL(cudaDeviceSynchronize());
  dest_h = dest_d;
  CHECK(tt::eq(2 * src_h, dest_h));
}

HOST_TEST_CASE(tensor_copying_transfer_array)
{
  tt::MatrixXXd<3, 4, tt::ColMajor> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);

  tt::MatrixXXd<3, 4, tt::ColMajor> h;
  h = m;
  tt::AllocMatrixd<mem::alloc::device, tt::ColMajor> d(3, 4);
  d = h;
  h = tt::broadcast<3, 4>(tt::SingletonT<double>(-1.2));
  h = d;

  CHECK(tt::eq(m, h));
}
