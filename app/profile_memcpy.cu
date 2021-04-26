#include <template_tensors/TemplateTensors.h>
#include <profiler/Profiler.h>

__global__
void kernel_single_forloop(uint8_t* dest, uint8_t* src, size_t num)
{
  for (size_t i = 0; i < num; i++)
  {
    dest[i] = src[i];
  }
}

__global__
void kernel_single_memcpy(uint8_t* dest, uint8_t* src, size_t num)
{
  memcpy(dest, src, num);
}

__global__
void kernel_gridstride_forloop(uint8_t* dest, uint8_t* src, size_t num, size_t batch_size)
{
  num /= batch_size;
  size_t step = gridDim.x * blockDim.x;
  for (size_t index = cuda::grid::thread_id_in_grid<1>()(); index < num; index += step)
  {
    for (size_t i = index * batch_size; i < index * batch_size + batch_size; i++)
    {
      dest[i] = src[i];
    }
  }
}

__global__
void kernel_gridstride_memcpy(uint8_t* dest, uint8_t* src, size_t num, size_t batch_size)
{
  num /= batch_size;
  size_t step = gridDim.x * blockDim.x;
  for (size_t index = cuda::grid::thread_id_in_grid<1>()(); index < num; index += step)
  {
    memcpy(&dest[index * batch_size], &src[index * batch_size], batch_size);
  }
}

int main(int argc, char** argv)
{
  size_t bytes = 2UL << 28;

  {
    array::AllocArray<uint8_t, mem::alloc::heap> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::heap> dest(bytes);
    PROFILE("host memmove");
    memmove(dest.data(), src.data(), src.size());
  }

  {
    array::AllocArray<uint8_t, mem::alloc::heap> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::heap> dest(bytes);
    PROFILE("host memcpy");
    memcpy(dest.data(), src.data(), src.size());
  }

  {
    array::AllocArray<uint8_t, mem::alloc::heap> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::heap> dest(bytes);
    PROFILE("host for-loop");
    for (size_t i = 0; i < src.size(); i++)
    {
      dest[i] = src[i];
    }
  }

  {
    array::AllocArray<uint8_t, mem::alloc::heap> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::heap> dest(bytes);
    PROFILE("host std::copy");
    std::copy(src.begin(), src.end(), dest.begin());
  }

  bytes = 2 << 26;

  for (size_t block_size = 256; block_size <= 256; block_size *= 2)
  {
    for (size_t grid_size = 256; grid_size <= 256; grid_size *= 2)
    {
      for (size_t batch_size = 1; batch_size <= 2 << 5; batch_size *= 2)
      {
        array::AllocArray<uint8_t, mem::alloc::device> src(bytes);
        array::AllocArray<uint8_t, mem::alloc::device> dest(bytes);

        dim3 block, grid;
        block = dim3(block_size);
        grid = dim3(grid_size);
        PROFILE("device grid-stride forloop (grid=" << std::setw(3) << grid_size << " block=" << std::setw(3) << block_size <<
                " batch_size=" << std::setw(3) << batch_size << ")");
        TT_CUDA_SAFE_CALL((kernel_gridstride_forloop<<<grid, block>>>(dest.data(), src.data(), src.size(), batch_size)));
        TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
    }
  }

  for (size_t block_size = 256; block_size <= 256; block_size *= 2)
  {
    for (size_t grid_size = 256; grid_size <= 256; grid_size *= 2)
    {
      for (size_t batch_size = 1; batch_size <= 2 << 5; batch_size *= 2)
      {
        array::AllocArray<uint8_t, mem::alloc::device> src(bytes);
        array::AllocArray<uint8_t, mem::alloc::device> dest(bytes);

        dim3 block, grid;
        block = dim3(256);
        grid = dim3(256);
        PROFILE("device grid-stride memcpy (grid=" << std::setw(3) << grid_size << " block=" << std::setw(3) << block_size <<
                " batch_size=" << std::setw(3) << batch_size << ")");
        TT_CUDA_SAFE_CALL((kernel_gridstride_memcpy<<<grid, block>>>(dest.data(), src.data(), src.size(), batch_size)));
        TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
    }
  }

  bytes = 2 << 20;

  {
    array::AllocArray<uint8_t, mem::alloc::device> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::device> dest(bytes);

    dim3 block, grid;
    block = dim3(1);
    grid = dim3(1);
    PROFILE("device single forloop");
    TT_CUDA_SAFE_CALL((kernel_single_forloop<<<grid, block>>>(dest.data(), src.data(), src.size())));
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }

  for (size_t batch_size = 1; batch_size <= 2 << 5; batch_size *= 2)
  {
    array::AllocArray<uint8_t, mem::alloc::device> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::device> dest(bytes);

    dim3 block, grid;
    block = dim3(1);
    grid = dim3(1);
    PROFILE("device single memcpy (batch_size=" << std::setw(3) << batch_size << ")");
    TT_CUDA_SAFE_CALL((kernel_gridstride_memcpy<<<grid, block>>>(dest.data(), src.data(), src.size(), batch_size)));
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }

  {
    array::AllocArray<uint8_t, mem::alloc::device> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::device> dest(bytes);

    dim3 block, grid;
    block = dim3(1);
    grid = dim3(1);
    PROFILE("device single memcpy");
    TT_CUDA_SAFE_CALL((kernel_single_memcpy<<<grid, block>>>(dest.data(), src.data(), src.size())));
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }

  profiler::print();
}
