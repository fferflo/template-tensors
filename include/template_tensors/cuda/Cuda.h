#pragma once

#ifndef __CUDACC__
#  define IS_ON_HOST true
#  define IS_ON_DEVICE false
#  ifndef __host__
#    define __host__
#    define __device__
#    define __constant__
#    define CUDA_RUNTIME_FUNCTIONS_AVAILABLE false
#  else
#    define CUDA_RUNTIME_FUNCTIONS_AVAILABLE true
#  endif
#else
#  include <cuda_runtime.h> // TODO: should this not be done by the user to switch cuda tensors on?
#  ifdef __CUDA_ARCH__
#    define IS_ON_HOST false
#    define IS_ON_DEVICE true
#  else
#    define IS_ON_HOST true
#    define IS_ON_DEVICE false
#  endif
#  define CUDA_RUNTIME_FUNCTIONS_AVAILABLE true
#endif

#ifdef __CUDACC__
#  ifdef _MSC_VER
#    define HD_WARNING_DISABLE __pragma("hd_warning_disable")
#  else
#    define HD_WARNING_DISABLE _Pragma("hd_warning_disable")
#  endif
#else
#  define HD_WARNING_DISABLE
#endif

#if IS_ON_HOST
#  define HD_NAME "Host"
#else
#  define HD_NAME "Device"
#endif



#ifdef __CUDACC__

#include <template_tensors/util/Assert.h>
#include <type_traits>
#include <stdio.h>

#define CUDA_SAFE_CALL(...) \
  do \
  { \
    __VA_ARGS__; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
    { \
      printf("\nCuda safe call '" #__VA_ARGS__ "' failed in " __FILE__ ": %u!\nCuda Error Code: %u\nCuda Error String: %s\n", \
        (unsigned int) __LINE__, (unsigned int) err, ::cudaGetErrorString(err)); \
      EXIT; \
    } \
  } while(false)

#define CUDA_DRIVER_SAFE_CALL(...) \
  do \
  { \
    CUresult err = __VA_ARGS__; \
    if (err != CUDA_SUCCESS) \
    { \
      const char* err_text = nullptr; \
      cuGetErrorName(err, &err_text); \
      printf("\nCuda driver safe call '" #__VA_ARGS__ "' failed in " __FILE__ ": %u!\nCuda Error Code: %u\nCuda Error String: %s\n", \
        (unsigned int) __LINE__, (unsigned int) err, err_text); \
      EXIT; \
    } \
  } while(false)

namespace cuda {

template <typename T>
__host__
bool isDevicePtr(T* ptr)
{
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, ptr);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess && attributes.type == cudaMemoryTypeDevice;
}



template <bool TDestOnHost, bool TSrcOnHost>
struct CudaMemcpyKind;

template <>
struct CudaMemcpyKind<true, true>
{
  static const cudaMemcpyKind value = cudaMemcpyHostToHost;
};

template <>
struct CudaMemcpyKind<true, false>
{
  static const cudaMemcpyKind value = cudaMemcpyDeviceToHost;
};

template <>
struct CudaMemcpyKind<false, true>
{
  static const cudaMemcpyKind value = cudaMemcpyHostToDevice;
};

template <>
struct CudaMemcpyKind<false, false>
{
  static const cudaMemcpyKind value = cudaMemcpyDeviceToDevice;
};

template <bool TDestOnHost, bool TSrcOnHost, typename T>
__host__ __device__
void cudaMemcpy(T* dest, const T* src, size_t num)
{
  ASSERT(TDestOnHost || isDevicePtr(dest), "cudaMemcpy destination pointer is invalid");
  ASSERT(TSrcOnHost || isDevicePtr(src), "cudaMemcpy source pointer is invalid");
  CUDA_SAFE_CALL(cudaMemcpy(dest, src, num * sizeof(T), CudaMemcpyKind<TDestOnHost, TSrcOnHost>::value));
}

} // end of ns cuda

#endif
