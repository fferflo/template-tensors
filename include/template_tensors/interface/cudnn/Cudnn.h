#pragma once

#include <template_tensors/cuda/Cuda.h>

#if defined(__CUDACC__) && defined(CUDNN_INCLUDED)

#include <cudnn.h>

#include <template_tensors/util/Assert.h>
#include <type_traits>
// TODO: split assertions into different types to find out which ones are slowing down ctest the most
namespace cudnn {

#define CUDNN_SAFE_CALL(...) \
  do \
  { \
    cudnnStatus_t err = __VA_ARGS__; \
    if (err != CUDNN_STATUS_SUCCESS) \
    { \
      printf("\nCudnn safe call '" #__VA_ARGS__ "' failed in " __FILE__ ": %u!\nCudnn Error Code: %u\nCudnn Error String: %s\n", \
        (unsigned int) __LINE__, (unsigned int) err, cudnnGetErrorString(err)); \
      EXIT; \
    } \
  } while(false)

class CudnnContext final
{
public:
  __host__
  CudnnContext()
  {
    CUDNN_SAFE_CALL(cudnnCreate(&m_handle));
  }

  __host__
  ~CudnnContext()
  {
    CUDNN_SAFE_CALL(cudnnDestroy(m_handle));
  }

  __host__
  cudnnHandle_t& getHandle()
  {
    return m_handle;
  }

private:
  cudnnHandle_t m_handle;
};

__host__
CudnnContext& getContext();



template <typename T>
struct CudnnDataType;

template <>
struct CudnnDataType<float>
{
  static const cudnnDataType_t value = CUDNN_DATA_FLOAT;
};

template <>
struct CudnnDataType<double>
{
  static const cudnnDataType_t value = CUDNN_DATA_DOUBLE;
};

template <>
struct CudnnDataType<int8_t>
{
  static const cudnnDataType_t value = CUDNN_DATA_INT8;
};

template <>
struct CudnnDataType<int32_t>
{
  static const cudnnDataType_t value = CUDNN_DATA_INT32;
};

template <>
struct CudnnDataType<uint8_t>
{
  static const cudnnDataType_t value = CUDNN_DATA_UINT8;
};

} // end of ns cudnn

#endif