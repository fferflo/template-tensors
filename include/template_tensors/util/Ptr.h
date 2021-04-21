#pragma once

#include <template_tensors/cuda/Cuda.h>
#include <template_tensors/util/Util.h>

#include <memory>
#include <type_traits>

#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#endif

namespace ptr {

template <typename TPtr>
struct ptr_defines;

template <typename T>
struct ptr_defines<T*>
{
  using value_t = T;

  __host__ __device__
  static value_t* toRawPointer(T* ptr)
  {
    return ptr;
  }

  template <typename T2>
  __host__ __device__
  static T2* static_cast_to(T* ptr)
  {
    return static_cast<T2*>(ptr);
  }
};

template <typename T>
struct ptr_defines<std::shared_ptr<T>>
{
  using value_t = T;

  HD_WARNING_DISABLE
  __host__ __device__
  static value_t* toRawPointer(const std::shared_ptr<T>& ptr)
  {
    return ptr.get();
  }

  HD_WARNING_DISABLE
  template <typename T2>
  __host__ __device__
  static std::shared_ptr<T2> static_cast_to(const std::shared_ptr<T>& ptr)
  {
    return std::static_pointer_cast<T2>(ptr);
  }
};

#ifdef __CUDACC__
template <typename T>
struct ptr_defines<thrust::device_ptr<T>>
{
  using value_t = T;

  HD_WARNING_DISABLE
  __host__ __device__
  static value_t* toRawPointer(const thrust::device_ptr<T>& ptr)
  {
    return thrust::raw_pointer_cast(ptr);
  }

  HD_WARNING_DISABLE
  template <typename T2>
  __host__ __device__
  static thrust::device_ptr<T2> static_cast_to(const thrust::device_ptr<T>& ptr)
  {
    return thrust::device_ptr<T2>(static_cast<T2*>(thrust::raw_pointer_cast(ptr)));
  }
};
#endif

template <typename TPtr>
using value_t = typename ptr_defines<typename std::decay<TPtr>::type>::value_t;

template <typename TPtr>
__host__ __device__
auto toRawPointer(const TPtr& ptr)
RETURN_AUTO(ptr_defines<typename std::decay<TPtr>::type>::toRawPointer(ptr))

template <typename TDestType, typename TPtr>
__host__ __device__
auto static_cast_to(const TPtr& ptr)
RETURN_AUTO(ptr_defines<typename std::decay<TPtr>::type>::template static_cast_to<TDestType>(ptr))

} // end of ns ptr
