#pragma once

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <template_tensors/cuda/Cuda.h>
#include <template_tensors/util/Assert.h>

#ifndef __CUDACC__
#define HOST_DEVICE_TEST_CASE(NAME) BOOST_AUTO_TEST_CASE(NAME)
#else
#define DEVICE_TEST_CASE(NAME) \
  __device__ void NAME##_test(); \
  __global__ void kernel_##NAME##_test() \
  { \
    NAME##_test(); \
  } \
  BOOST_AUTO_TEST_CASE(NAME##_device) \
  { \
    CUDA_SAFE_CALL(kernel_##NAME##_test<<<1, 1>>>()); \
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); \
  } \
  __device__ void NAME##_test()
#define HOST_DEVICE_TEST_CASE(NAME) \
  __host__ __device__ void NAME##_test(); \
  __global__ void kernel_##NAME##_test() \
  { \
    NAME##_test(); \
  } \
  BOOST_AUTO_TEST_CASE(NAME##_device) \
  { \
    CUDA_SAFE_CALL(kernel_##NAME##_test<<<1, 1>>>()); \
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); \
  } \
  BOOST_AUTO_TEST_CASE(NAME##_host) \
  { \
    NAME##_test(); \
  } \
  __host__ __device__ void NAME##_test()
#endif
#define HOST_TEST_CASE(NAME) \
  __host__ void NAME##_test(); \
  BOOST_AUTO_TEST_CASE(NAME##_host) \
  { \
    NAME##_test(); \
  } \
  __host__ void NAME##_test()



#define CUDA_CHECK_CALL(...) \
  do \
  { \
    __VA_ARGS__; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
    { \
      printf("\nCuda safe call '" #__VA_ARGS__ "' failed in " __FILE__ ": %u!\nCuda Error Code: %u\nCuda Error String: %s\n", \
        (unsigned int) __LINE__, (unsigned int) err, ::cudaGetErrorString(err)); \
      BOOST_CHECK(err != cudaSuccess); \
    } \
  } while(false)



#if IS_ON_HOST
#define CHECK(...) BOOST_CHECK((__VA_ARGS__))
#else
#define CHECK(...) \
  do \
  { \
    if (!(__VA_ARGS__)) \
    { \
      printf("\nCUDA device test failed: '%s' in %s:%u\n", #__VA_ARGS__, __FILE__, __LINE__); \
      EXIT; \
    } \
  } while(0)
#endif
