#include <template_tensors/TemplateTensors.h>

#include <nanobench.h>

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
  ankerl::nanobench::Bench bench;
  size_t bytes = 2UL << 28;

  {
    array::AllocArray<uint8_t, mem::alloc::heap> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::heap> dest(bytes);
    bench.run("host memmove", [&]{
        memmove(dest.data(), src.data(), src.size());
    });
  }

  {
    array::AllocArray<uint8_t, mem::alloc::heap> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::heap> dest(bytes);
    bench.run("host memmove", [&]{
        memcpy(dest.data(), src.data(), src.size());
    });
  }

  {
    array::AllocArray<uint8_t, mem::alloc::heap> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::heap> dest(bytes);
    bench.run("host for-loop", [&]{
      for (size_t i = 0; i < src.size(); i++)
      {
        ankerl::nanobench::doNotOptimizeAway(dest[i] = src[i]);
      }
      ankerl::nanobench::doNotOptimizeAway(dest);
    });
  }

  {
    array::AllocArray<uint8_t, mem::alloc::heap> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::heap> dest(bytes);
    bench.run("host std::copy", [&]{
      std::copy(src.begin(), src.end(), dest.begin());
    });
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
        bench.run("device grid-stride forloop (grid=" + std::to_string(grid_size) + " block=" + std::to_string(block_size) + " batch_size=" + std::to_string(batch_size) + ")",
          [&]{
            TT_CUDA_SAFE_CALL((kernel_gridstride_forloop<<<grid, block>>>(dest.data(), src.data(), src.size(), batch_size)));
            TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
          }
        );
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
        bench.run("device grid-stride memcpy (grid=" + std::to_string(grid_size) + " block=" + std::to_string(block_size) + " batch_size=" + std::to_string(batch_size) + ")",
          [&]{
            TT_CUDA_SAFE_CALL((kernel_gridstride_memcpy<<<grid, block>>>(dest.data(), src.data(), src.size(), batch_size)));
            TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
          }
        );
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
    bench.run("device single forloop",
      [&]{
        TT_CUDA_SAFE_CALL((kernel_single_forloop<<<grid, block>>>(dest.data(), src.data(), src.size())));
        TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
    );
  }

  for (size_t batch_size = 1; batch_size <= 2 << 5; batch_size *= 2)
  {
    array::AllocArray<uint8_t, mem::alloc::device> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::device> dest(bytes);

    dim3 block, grid;
    block = dim3(1);
    grid = dim3(1);
    bench.run("device single memcpy (batch_size=" + std::to_string(batch_size) + ")",
      [&]{
        TT_CUDA_SAFE_CALL((kernel_gridstride_memcpy<<<grid, block>>>(dest.data(), src.data(), src.size(), batch_size)));
        TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
    );
  }

  {
    array::AllocArray<uint8_t, mem::alloc::device> src(bytes);
    array::AllocArray<uint8_t, mem::alloc::device> dest(bytes);

    dim3 block, grid;
    block = dim3(1);
    grid = dim3(1);
    bench.run("device single memcpy",
      [&]{
        TT_CUDA_SAFE_CALL((kernel_single_memcpy<<<grid, block>>>(dest.data(), src.data(), src.size())));
        TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
    );
  }
}
