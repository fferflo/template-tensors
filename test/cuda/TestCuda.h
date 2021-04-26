#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

#include <template_tensors/cuda/CudaReduction.h>
#include <template_tensors/cuda/CudaMutex.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__
void kernel_test_warp_reduce()
{
  int value = threadIdx.x + 1;
  cuda::warp_reduce(math::functor::addassign(), value);
  if (threadIdx.x == 0)
  {
    CHECK(value == cuda::WARP_SIZE * (cuda::WARP_SIZE + 1) / 2);
  }
}

HOST_TEST_CASE(cuda_warp_reduce)
{
  CUDA_CHECK_CALL(kernel_test_warp_reduce<<<1, cuda::WARP_SIZE>>>());
  CUDA_CHECK_CALL(cudaDeviceSynchronize());
  CUDA_CHECK_CALL(kernel_test_warp_reduce<<<16, cuda::WARP_SIZE>>>());
  CUDA_CHECK_CALL(cudaDeviceSynchronize());
}

__global__
void kernel_test_block_reduce()
{
  int value = threadIdx.x + 1;
  cuda::block_reduce(math::functor::addassign(), value, 0.0);
  if (threadIdx.x == 0)
  {
    CHECK(value == blockDim.x * (blockDim.x + 1) / 2);
  }
}

HOST_TEST_CASE(cuda_block_reduce)
{
  CUDA_CHECK_CALL(kernel_test_block_reduce<<<1, cuda::grid::MAX_BLOCK_SIZE_1D_X / 1>>>());
  CUDA_CHECK_CALL(cudaDeviceSynchronize());
  CUDA_CHECK_CALL(kernel_test_block_reduce<<<1, cuda::grid::MAX_BLOCK_SIZE_1D_X / 2>>>());
  CUDA_CHECK_CALL(cudaDeviceSynchronize());
  CUDA_CHECK_CALL(kernel_test_block_reduce<<<2, cuda::grid::MAX_BLOCK_SIZE_1D_X / 4>>>());
  CUDA_CHECK_CALL(cudaDeviceSynchronize());
  CUDA_CHECK_CALL(kernel_test_block_reduce<<<128, 96>>>());
  CUDA_CHECK_CALL(cudaDeviceSynchronize());
}

__global__
void kernel_test_lock(volatile int* val, cuda::Mutex* mutex)
{
  bool done = false;
  while (!done)
  {
    if (auto lock = mutex::UniqueTryLock<cuda::Mutex>(mutex))
    {
      (*val)++;
      done = true;
    }
  }
}

HOST_TEST_CASE(cuda_lock)
{
  const size_t GRID_SIZE = 4;
  const size_t BLOCK_SIZE = 16;

  thrust::device_vector<int> val_d;
  val_d.push_back(0);
  thrust::device_vector<cuda::Mutex> mutex_d;
  mutex_d.push_back(cuda::Mutex());

  CUDA_CHECK_CALL(kernel_test_lock<<<GRID_SIZE, BLOCK_SIZE>>>(val_d.data().get(), mutex_d.data().get()));
  CUDA_CHECK_CALL(cudaDeviceSynchronize());

  thrust::host_vector<int> val_h(val_d);
  CHECK(val_h[0] == GRID_SIZE * BLOCK_SIZE);
}

__global__
void kernel_test_conditional_lock(volatile int* val, cuda::Mutex* mutex)
{
  bool done = false;
  while (!done)
  {
    if (auto lock = mutex::ConditionalUniqueTryLock<cuda::Mutex>(mutex, [&](){return (*val) / gridDim.x == threadIdx.x;}))
    {
      (*val)++;
      done = true;
    }
  }
}

HOST_TEST_CASE(cuda_conditional_lock)
{
  const size_t GRID_SIZE = 4;
  const size_t BLOCK_SIZE = 16;

  thrust::device_vector<int> val_d;
  val_d.push_back(0);
  thrust::device_vector<cuda::Mutex> mutex_d;
  mutex_d.push_back(cuda::Mutex());

  CUDA_CHECK_CALL(kernel_test_conditional_lock<<<GRID_SIZE, BLOCK_SIZE>>>(val_d.data().get(), mutex_d.data().get()));
  CUDA_CHECK_CALL(cudaDeviceSynchronize());

  thrust::host_vector<int> val_h(val_d);
  CHECK(val_h[0] == GRID_SIZE * BLOCK_SIZE);
}

HOST_TEST_CASE(cuda_to_host)
{
  static const int val = 13;

  thrust::device_vector<int> val_d;
  val_d.push_back(val);

  BOOST_CHECK(mem::toHost(val_d.data()) == val);
  BOOST_CHECK(mem::toHost(*val_d.data()) == val);
}
