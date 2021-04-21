#if defined(__CUDACC__)

#include <memory>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>

namespace template_tensors {

namespace geometry {

namespace render {

template <size_t TThreadsPerPrimitive, typename TMutexMap, typename TDestMap, typename TPrimitiveIterator, typename TShader, typename... TArgs>
__global__
void kernel_DeviceMutexRasterizer_nthreadsperprimitive(
  TMutexMap mutex_map,
  TDestMap dest,
  TPrimitiveIterator primitives_begin,
  TPrimitiveIterator primitives_end,
  TShader shader,
  TArgs... args)
{
  using Pixel = template_tensors::decay_elementtype_t<TDestMap>;
  using Primitive = typename std::decay<decltype(thrust::raw_reference_cast(*primitives_begin))>::type;
  using Scalar = typename Primitive::Scalar;

  const size_t thread_id_in_grid = cuda::grid::thread_id_in_grid<1>()();
  const size_t offset = thread_id_in_grid / TThreadsPerPrimitive;
  const size_t step = cuda::grid::thread_num_in_grid() / TThreadsPerPrimitive;

  for (TPrimitiveIterator primitive_it = primitives_begin + offset; primitive_it < primitives_end; primitive_it += step)
  {
    auto& primitive = thrust::raw_reference_cast(*primitive_it);

    auto handler = [&](Vector2s pos, const typename Primitive::Intersection& intersection){
      volatile Pixel& pixel = dest(pos);
      cuda::Mutex& mutex = mutex_map(pos);

      // Depth test
      if (auto lock = mutex::ConditionalUniqueTryLock<cuda::Mutex>(mutex, [&](){return pixel.z > intersection.z;})) // TODO: include epsilon, at which points contribute to picture?
      {
        // Shading
        pixel.z = intersection.z;
        shader(pixel, primitive, intersection);
      }
      else if (lock.getState() == mutex::LOCK_FAILED)
      {
        // Repeat
        return false;
      }
      return true;
    };

    primitive.template rasterize<TThreadsPerPrimitive>(handler, dest.template dims<2>(), args...);
  }
}

template <size_t TThreadsPerPrimitive = 16>
class DeviceMutexRasterizer
{
public:
  __host__
  DeviceMutexRasterizer(template_tensors::Vector2s resolution)
    : DeviceMutexRasterizer(template_tensors::prod(resolution))
  {
  }

  __host__
  DeviceMutexRasterizer(size_t mutex_memory_size)
    : DeviceMutexRasterizer(std::make_shared<thrust::device_vector<cuda::Mutex>>(mutex_memory_size))
  {
    reset();
  }

  __host__
  DeviceMutexRasterizer()
    : DeviceMutexRasterizer(0)
  {
    reset();
  }

  __host__
  DeviceMutexRasterizer(std::shared_ptr<thrust::device_vector<cuda::Mutex>> mutex_memory)
    : m_mutex_memory(mutex_memory)
  {
  }

  __host__
  void reset()
  {
    thrust::for_each(m_mutex_memory->begin(), m_mutex_memory->end(), []__device__(cuda::Mutex& mutex){mutex = cuda::Mutex();});
  }

  template <typename TDestMap, typename TPrimitiveIterator, typename TShader, typename... TArgs>
  __host__
  void operator()(
    TDestMap&& dest,
    TPrimitiveIterator primitives_begin,
    TPrimitiveIterator primitives_end,
    TShader shader,
    TArgs&&... args)
  {
    ASSERT(template_tensors::prod(dest.template dims<2>()) <= m_mutex_memory->size(), "Not enough memory");

    dim3 block, grid;

    // TODO: take index strategy from dest?
    auto mutex_map = template_tensors::ref<template_tensors::ColMajor, mem::DEVICE>(thrust::raw_pointer_cast(m_mutex_memory->data()), dest.template dims<2>());
    block = 96;
    grid  = 128;
    CUDA_SAFE_CALL(kernel_DeviceMutexRasterizer_nthreadsperprimitive<TThreadsPerPrimitive><<<grid, block>>>
      (mem::toKernel(mutex_map), mem::toKernel(dest), primitives_begin, primitives_end, shader,
          static_cast<typename std::decay<TArgs&&>::type>(args)...));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }

  std::shared_ptr<thrust::device_vector<cuda::Mutex>> getMutexMemory()
  {
    return m_mutex_memory;
  }

private:
  std::shared_ptr<thrust::device_vector<cuda::Mutex>> m_mutex_memory;
};

} // end of ns render

} // end of ns geometry

} // end of ns tensor

#endif
