#pragma once

#include <iostream>
#include <cstring>
#include <type_traits>
#include <stdlib.h>
#include <metal.hpp>

#include <template_tensors/util/Util.h>
#include <template_tensors/util/Constexpr.h>
#include <template_tensors/util/Assert.h>
#include <template_tensors/util/Math.h>
#include <template_tensors/cuda/Cuda.h>
#include <template_tensors/tmp/Deduce.h>

#if CUDA_RUNTIME_FUNCTIONS_AVAILABLE
#include <thrust/device_reference.h>
#include <thrust/device_ptr.h>
#endif

#include <boost/preprocessor/facilities/overload.hpp>

#ifdef __CUDACC__
#include <cuda.h>
#define USE_THRUST_TRIVIAL_RELOCATION (CUDA_VERSION > 10000)
#else
#define USE_THRUST_TRIVIAL_RELOCATION false
#endif

#if USE_THRUST_TRIVIAL_RELOCATION
#include <thrust/type_traits/is_trivially_relocatable.h>
#endif

namespace mem {

static const metal::int_ DYN = static_cast<metal::int_>(-1);

enum MemoryType
{
  LOCAL = 0,
  DEVICE = 1,
  HOST = 2,
  UNKNOWN = 3
  // TODO: Add unified memory (how to decide where to run stuff like tensor<um> = tensor<um>)? Just locally?
};

inline std::ostream& operator<<(std::ostream& ostream, MemoryType mem)
{
  switch (mem)
  {
    case LOCAL:
    {
      ostream << "local memory";
      break;
    }
    case DEVICE:
    {
      ostream << "device memory";
      break;
    }
    case HOST:
    {
      ostream << "host memory";
      break;
    }
    case UNKNOWN:
    {
      ostream << "unknown memory";
      break;
    }
    default:
    {
      ASSERT_(false, "Invalid memory type");
      break;
    }
  }
  return ostream;
}

template <MemoryType T1>
__host__ __device__
constexpr MemoryType combine()
{
  return T1;
}

template <MemoryType T1, MemoryType T2>
__host__ __device__
constexpr MemoryType combine()
{
  static_assert(T1 == LOCAL || T2 == LOCAL || T1 == T2, "Invalid combination of memory types");
  return T1 == LOCAL ? T2 : T1;
}

template <MemoryType T1, MemoryType T2, MemoryType T3, MemoryType... TRest>
__host__ __device__
constexpr MemoryType combine()
{
  return combine<combine<T1, T2>, T3, TRest...>();
}

template <MemoryType T, bool TIsOnHost = TT_IS_ON_HOST>
__host__ __device__
constexpr bool isOnHost()
{
  return T == HOST || (T == LOCAL && TIsOnHost);
}

template <MemoryType T, bool TIsOnHost = TT_IS_ON_HOST>
__host__ __device__
constexpr bool isOnDevice()
{
  return T == DEVICE || (T == LOCAL && !TIsOnHost);
}

template <MemoryType T, bool TIsOnHost = TT_IS_ON_HOST>
__host__ __device__
constexpr bool isOnLocal()
{
  return (T == LOCAL) || (T == HOST && TIsOnHost) || (T == DEVICE && !TIsOnHost);
}





template <typename TThisType, MemoryType TMemoryType>
class HasMemoryType;

namespace detail {

template <typename TArg>
struct ToHasMemoryType
{
  template <typename TThisType, MemoryType TMemoryType>
  TMP_IF(const HasMemoryType<TThisType, TMemoryType>&)
  TMP_RETURN_VALUE(TMemoryType)

  TMP_DEDUCE_VALUE(typename std::decay<TArg>::type);
};

template <typename TArg>
struct CheckIfHasMemoryType
{
  template <typename TThisType, MemoryType TMemoryType>
  TMP_IF(const HasMemoryType<TThisType, TMemoryType>&)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(typename std::decay<TArg>::type);
};

} // end of ns detail

template <typename TType>
TVALUE(bool, has_memorytype_v, detail::CheckIfHasMemoryType<TType>::value)

template <typename TType, ENABLE_IF(has_memorytype_v<TType>::value)>
TVALUE(mem::MemoryType, memorytype_v, detail::ToHasMemoryType<TType>::value)



template <typename TThisType, MemoryType TMemoryType>
class HasMemoryType
{
public:
  template <typename TType, ENABLE_IF(!std::is_same<TType, void>::value
    && (TMemoryType == mem::HOST || TMemoryType == mem::LOCAL))>
  __host__
  static auto toHost(TType&& t)
  RETURN_AUTO(TType(std::forward<TType>(t)))

  template <typename TType, ENABLE_IF(!std::is_same<TType, void>::value
    && (TMemoryType == mem::DEVICE))>
  __host__
  static auto toDevice(TType&& t)
  RETURN_AUTO(TType(std::forward<TType>(t)))

  template <typename TType, ENABLE_IF(!std::is_same<TType, void>::value
    && (TMemoryType == mem::DEVICE || TMemoryType == mem::LOCAL))>
  __host__
  static typename std::decay<TType&&>::type toDevice(TType&& t)
  {
    return typename std::decay<TType&&>::type(std::forward<TType>(t));
  }
};

template <typename TType, ENABLE_IF(has_memorytype_v<TType>::value)>
__host__
auto toHost(TType&& t)
RETURN_AUTO(t.toHost(std::forward<TType>(t)))
template <typename TType, ENABLE_IF(!has_memorytype_v<TType>::value)>
__host__
auto toHost(TType&& t)
RETURN_AUTO(typename std::decay<TType&&>::type(std::forward<TType>(t)))
FUNCTOR(toHost, mem::toHost)
// TODO: replace decay constructors with util::decay_rvalue
#ifdef __CUDACC__

template <typename TType, ENABLE_IF(has_memorytype_v<TType>::value)>
__host__
auto toDevice(TType&& t)
RETURN_AUTO(t.toDevice(std::forward<TType>(t)))
template <typename TType, ENABLE_IF(!has_memorytype_v<TType>::value)>
__host__
auto toDevice(TType&& t)
RETURN_AUTO(typename std::decay<TType&&>::type(std::forward<TType>(t)))
FUNCTOR(toDevice, mem::toDevice)

template <typename TType, ENABLE_IF(has_memorytype_v<TType>::value)>
__host__
auto toKernel(TType&& t)
RETURN_AUTO(t.toKernel(std::forward<TType>(t)))
template <typename TType, ENABLE_IF(!has_memorytype_v<TType>::value)>
__host__
auto toKernel(TType&& t)
RETURN_AUTO(typename std::decay<TType&&>::type(std::forward<TType>(t)))
FUNCTOR(toKernel, mem::toKernel)

#endif

namespace detail {

template <bool TTDataOnHost, bool TExecutionOnHost>
struct toFunctor;

template <bool TSame>
struct toFunctor<TSame, TSame>
{
  HD_WARNING_DISABLE
  template <typename T>
  __host__  __device__
  static auto get(T&& t)
  RETURN_AUTO(T(std::forward<T>(t)))
};

#ifdef __CUDACC__
template <>
struct toFunctor<false, true>
{
  HD_WARNING_DISABLE
  template <typename T>
  __host__  __device__
  static auto get(T&& t)
  RETURN_AUTO(mem::toKernel(std::forward<T>(t)))
};

template <>
struct toFunctor<true, false>
{
  HD_WARNING_DISABLE
  template <typename T>
  __host__  __device__
  static auto get(T&& t) -> decltype(T(std::forward<T>(t)))
  {
    ASSERT_(false, "Cannot pass host memory to functor on device");
    return T(std::forward<T>(t));
  }
};
#endif

} // end of ns detail

template <mem::MemoryType TMemoryType, bool TIsOnHost = TT_IS_ON_HOST, typename TType>
__host__  __device__
auto toFunctor(TType&& t)
RETURN_AUTO(detail::toFunctor<mem::isOnHost<TMemoryType, TIsOnHost>(), TIsOnHost>::get(std::forward<TType>(t)))
// TODO: naming toFunctor: toLambda?
namespace functor {
template <mem::MemoryType TMemoryType, bool TIsOnHost = TT_IS_ON_HOST>
struct toFunctor
{
  template <typename TType>
  __host__  __device__
  auto operator()(TType&& t)
  RETURN_AUTO(mem::toFunctor<TMemoryType, TIsOnHost>(std::forward<TType>(t)))
};
} // end of ns functor





namespace alloc {

/*!
 * \defgroup MemoryAllocators Memory Allocators
 * \brief Memory allocators for allocating types of linear memory
 * @{
 */

#if CUDA_RUNTIME_FUNCTIONS_AVAILABLE

template <uint32_t TFlags = cudaHostAllocDefault>
struct host_pinned
{
  static const MemoryType MEMORY_TYPE = HOST;

#ifdef __CUDACC__
  #pragma hd_warning_disable
#endif
  template <typename TType>
  __host__ __device__
  static TType* allocate(size_t size)
  {
    TType* data;
    TT_CUDA_SAFE_CALL(cudaHostAlloc(&data, size * sizeof(TType), TFlags));
    return data;
  }

#ifdef __CUDACC__
  #pragma hd_warning_disable
#endif
  template <typename TType>
  __host__ __device__
  static void free(TType* data)
  {
    TT_CUDA_SAFE_CALL(cudaFreeHost(data));
  }
};

#endif

namespace detail {

template <bool THostHeap>
struct heap
{
  static const MemoryType MEMORY_TYPE = THostHeap ? HOST : DEVICE;

  HD_WARNING_DISABLE
  template <typename TType>
  __host__ __device__
  static TType* allocate(size_t size)
  {
    return new TType[size];
  }

  HD_WARNING_DISABLE
  template <typename TType>
  __host__ __device__
  static void free(TType* data)
  {
    delete[] data;
  }
};

} // end of ns detail

#if defined(ALWAYS_USE_PINNED_MEMORY) && CUDA_RUNTIME_FUNCTIONS_AVAILABLE
using host_heap = host_pinned<>;
#else
using host_heap = detail::heap<true>;
#endif
using device_heap = detail::heap<false>;

using heap = typename std::conditional<TT_IS_ON_HOST, host_heap, device_heap>::type;



template <metal::int_ TAlignmentBytes>
struct host_aligned_alloc
{
  static const MemoryType MEMORY_TYPE = HOST;

  HD_WARNING_DISABLE
  template <typename TType>
  __host__
  static TType* allocate(size_t size)
  {
    return reinterpret_cast<TType*>(aligned_alloc(TAlignmentBytes, size * sizeof(TType)));
  }

  HD_WARNING_DISABLE
  template <typename TType>
  __host__
  static void free(TType* data)
  {
    ::free(data);
  }
};



#if CUDA_RUNTIME_FUNCTIONS_AVAILABLE

/*!
 * \brief An allocator that allocates memory on the device using cudaMalloc and cudaFree.
 */
struct device
{
  static const MemoryType MEMORY_TYPE = DEVICE;

#ifdef __CUDACC__
  #pragma hd_warning_disable
#endif
  template <typename TType>
  __host__ __device__
  static TType* allocate(size_t size)
  {
    TType* data;
    cudaMalloc(&data, size * sizeof(TType));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("\ncudaMalloc failed to allocate %u * %u bytes \nCuda Error String: %s\n", (uint32_t) size, (uint32_t) sizeof(TType), ::cudaGetErrorString(err));
      TT_EXIT;
    }
    // printf("Allocated device memory at %p\n", (void*) data);
    return data;
  }

#ifdef __CUDACC__
  #pragma hd_warning_disable
#endif
  template <typename TType>
  __host__ __device__
  static void free(TType* data)
  {
    TT_CUDA_SAFE_CALL(cudaFree(data));
    // printf("Freed device memory at %p\n", (void*) data);
  }
};

#endif

namespace detail {

struct host_from_device
{
  static const MemoryType MEMORY_TYPE = mem::HOST;

  template <typename TType>
  __host__ __device__
  static TType* allocate(size_t size)
  {
    ASSERT_(false, "Cannot allocate host memory from device");
    return nullptr;
  }

  template <typename TType>
  __host__ __device__
  static void free(TType* data)
  {
    ASSERT_(false, "Cannot allocate host memory from device");
  }
};

} // end of ns detail

template <MemoryType TMemoryType, bool TIsOnHost = TT_IS_ON_HOST>
using default_for =
#ifdef __CUDACC__
  typename std::conditional<mem::isOnHost<TMemoryType, TIsOnHost>(),
    typename std::conditional<TIsOnHost,
      mem::alloc::host_heap,
      mem::alloc::detail::host_from_device
    >::type,
    typename std::conditional<TIsOnHost,
      mem::alloc::device,
      mem::alloc::device_heap
    >::type
  >::type;
#else
  host_heap;
#endif
/*!
 * @}
 */

} // end of ns alloc


#if !USE_THRUST_TRIVIAL_RELOCATION
namespace detail {
template <typename TArg>
struct proclaim_trivially_relocatable
{
  static const bool value = false;
};
} // end of ns detail
#endif

template <typename TArg>
struct is_trivially_relocatable_v
{
  static const bool value =
#if USE_THRUST_TRIVIAL_RELOCATION
    ::thrust::is_trivially_relocatable<typename std::decay<TArg>::type>::value;
#else
    detail::proclaim_trivially_relocatable<typename std::decay<TArg>::type>::value;
#endif
};

#if USE_THRUST_TRIVIAL_RELOCATION
#define TT_PROCLAIM_TRIVIALLY_RELOCATABLE_2(TYPE, ...) \
  struct thrust::proclaim_trivially_relocatable<ESC TYPE> \
    : std::conditional<__VA_ARGS__, ::thrust::true_type, ::thrust::false_type>::type \
  { \
  }
#else
#define TT_PROCLAIM_TRIVIALLY_RELOCATABLE_2(TYPE, ...) \
  struct mem::detail::proclaim_trivially_relocatable<ESC TYPE> \
  { \
    static const bool value = __VA_ARGS__; \
  }
#endif

#define TT_PROCLAIM_TRIVIALLY_RELOCATABLE_1(TYPE) TT_PROCLAIM_TRIVIALLY_RELOCATABLE_2(TYPE, true)
#define TT_PROCLAIM_TRIVIALLY_RELOCATABLE(...) BOOST_PP_OVERLOAD(TT_PROCLAIM_TRIVIALLY_RELOCATABLE_,__VA_ARGS__)(__VA_ARGS__)
#define TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE(...) \
  template <> \
  TT_PROCLAIM_TRIVIALLY_RELOCATABLE(__VA_ARGS__)



namespace detail {

template <bool TCopyByAssign>
struct AssignOrMemcpy;

template <>
struct AssignOrMemcpy<true>
{
  template <mem::MemoryType TDestMemoryType, mem::MemoryType TSrcMemoryType, typename T1, typename T2>
  static void copy(T1* dest, const T2* src, size_t num)
  {
    static const bool dest_volatile = std::is_volatile<T1>::value;
    static const bool src_volatile = std::is_volatile<T2>::value;

    using DestByte = typename std::conditional<dest_volatile, volatile uint8_t, uint8_t>::type;
    using SrcByte = typename std::conditional<src_volatile, const volatile uint8_t, const uint8_t>::type;
    DestByte* dest_bytes = reinterpret_cast<DestByte*>(dest);
    SrcByte* src_bytes = reinterpret_cast<SrcByte*>(src);
    num *= sizeof(T1);
    for (size_t i = 0; i < num; i++)
    {
      dest_bytes[i] = src_bytes[i];
    }
  }
};

template <>
struct AssignOrMemcpy<false>
{
  template <mem::MemoryType TDestMemoryType, mem::MemoryType TSrcMemoryType, typename T1, typename T2>
  static void copy(T1* dest, const T2* src, size_t num)
  {
#if CUDA_RUNTIME_FUNCTIONS_AVAILABLE
    cuda::cudaMemcpy<isOnHost<TDestMemoryType>(), isOnHost<TSrcMemoryType>()>(dest, src, num);
#else
    static_assert((isOnHost<TDestMemoryType>() && isOnHost<TSrcMemoryType>()), "Cannot copy device memory without CUDA");

    void* dest_void = reinterpret_cast<void*>(dest);
    const void* src_void = reinterpret_cast<const void*>(src);
    memcpy(dest_void, src_void, num * sizeof(T1));
#endif
  }
};

template <bool TVolatile>
struct VolatileCopyOnHost;

template <>
struct VolatileCopyOnHost<false>
{
  template <mem::MemoryType TDestMemoryType, mem::MemoryType TSrcMemoryType, metal::int_ TNum = DYN, typename T1, typename T2>
  static void copy(T1* dest, const T2* src, size_t num)
  {
    static const bool can_assign =
         TNum != DYN
      && math::lt(TNum * sizeof(T1), static_cast<size_t>(COPY_BY_ASSIGN_BELOW_NUMBER_OF_BYTES))
      && mem::isOnLocal<TDestMemoryType>()
      && mem::isOnLocal<TSrcMemoryType>();
    static const bool must_memcpy = mem::isOnLocal<TDestMemoryType>() != mem::isOnLocal<TSrcMemoryType>();

    static_assert(!(must_memcpy && !(is_trivially_relocatable_v<T1>::value && is_trivially_relocatable_v<T2>::value)), "Type must be trivially copyable");

    AssignOrMemcpy<!must_memcpy && can_assign>::template copy<TDestMemoryType, TSrcMemoryType>(dest, src, num);
  }
};

template <>
struct VolatileCopyOnHost<true>
{
  template <MemoryType TDestMemoryType, MemoryType TSrcMemoryType, metal::int_ TNum = DYN, typename T1, typename T2>
  static void copy(T1* dest, const T2* src, size_t num)
  {
    static_assert((isOnHost<TDestMemoryType>() && isOnHost<TSrcMemoryType>()), "Cannot copy volatile device memory from host");

    static const bool dest_volatile = std::is_volatile<T1>::value;
    static const bool src_volatile = std::is_volatile<T2>::value;

    using DestByte = typename std::conditional<dest_volatile, volatile uint8_t, uint8_t>::type;
    using SrcByte = typename std::conditional<src_volatile, const volatile uint8_t, const uint8_t>::type;
    DestByte* dest_bytes = reinterpret_cast<DestByte*>(dest);
    SrcByte* src_bytes = reinterpret_cast<SrcByte*>(src);
    num *= sizeof(T1);
    for (size_t i = 0; i < num; i++)
    {
      dest_bytes[i] = src_bytes[i];
    }
  }
};

template <MemoryType TDestMemoryType, MemoryType TSrcMemoryType>
struct Copy
{
  template <metal::int_ TNum = DYN, typename T1, typename T2>
  __host__ __device__
  static void copy(T1* dest, T2* src, size_t num)
  {
    static const bool dest_volatile = std::is_volatile<T1>::value;
    static const bool src_volatile = std::is_volatile<T2>::value;

#if TT_IS_ON_HOST
    VolatileCopyOnHost<dest_volatile || src_volatile>::template copy<TDestMemoryType, TSrcMemoryType, TNum>(dest, src, num);
#else
    util::constexpr_if<!(isOnDevice<TDestMemoryType>() && isOnDevice<TSrcMemoryType>())>([]__host__ __device__(){
      ASSERT_(false, "Can only copy device memory from device");
    });
    using DestByte = typename std::conditional<dest_volatile, volatile uint8_t, uint8_t>::type;
    using SrcByte = typename std::conditional<src_volatile, const volatile uint8_t, const uint8_t>::type;
    // This is faster than a regular memcpy() on device
    num *= sizeof(T1);
    DestByte* dest_bytes = reinterpret_cast<DestByte*>(dest);
    SrcByte* src_bytes = reinterpret_cast<SrcByte*>(src);
    for (size_t i = 0; i < num; i++)
    {
      dest_bytes[i] = src_bytes[i];
    }
#endif
  }
};

} // detail

template <MemoryType TDestMemoryType, MemoryType TSrcMemoryType, metal::int_ TNum = DYN, typename T1, typename T2,
  ENABLE_IF(std::is_same<typename std::decay<T1>::type, typename std::decay<T2>::type>::value)>
__host__ __device__
void copy(T1* dest, T2* src, size_t num)
{
  static_assert(std::is_same<typename std::decay<T1>::type, typename std::decay<T2>::type>::value, "Types must be the same");
  detail::Copy<TDestMemoryType, TSrcMemoryType>::template copy<TNum>(dest, src, num);
}



#if CUDA_RUNTIME_FUNCTIONS_AVAILABLE
template <typename T>
__host__
T toHost(thrust::device_ptr<T> device_ptr)
{
  T result;
  copy<TT_IS_ON_HOST ? HOST : DEVICE, DEVICE, 1>(&result, thrust::raw_pointer_cast(device_ptr), 1);
  return result;
}

template <typename T>
__host__
T toHost(thrust::device_reference<T> device_ref)
{
  return toHost(&device_ref);
}
#endif

} // end of ns mem

TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((bool));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((float));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((double));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((char));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((uint8_t));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((uint16_t));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((uint32_t));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((uint64_t));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((int8_t));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((int16_t));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((int32_t));
TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((int64_t));
