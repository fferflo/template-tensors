#pragma once

#include <template_tensors/cuda/Cuda.h>

#if defined(__CUDACC__) && defined(CUBLAS_INCLUDED)

#include <cublas_v2.h>

#include <template_tensors/util/Assert.h>
#include <type_traits>

namespace cublas {

__host__ __device__
const char* cublasGetErrorString(cublasStatus_t err);

__host__ __device__
const char* cublasGetOpString(cublasOperation_t op);

#define TT_CUBLAS_SAFE_CALL(...) \
    do \
    { \
      cublasStatus_t err = __VA_ARGS__; \
      if (err != CUBLAS_STATUS_SUCCESS) \
      { \
        printf("\nCublas safe call '" #__VA_ARGS__ "' failed in " __FILE__ ": %u!\nCublas Error Code: %u\nCublas Error String: %s\n", \
          (unsigned int) __LINE__, (unsigned int) err, cublas::cublasGetErrorString(err)); \
        TT_EXIT; \
      } \
    } while(false)

class CublasContext final
{
public:
  __host__
  CublasContext()
  {
    TT_CUBLAS_SAFE_CALL(cublasCreate(&m_handle));
  }

  __host__
  ~CublasContext()
  {
    TT_CUBLAS_SAFE_CALL(cublasDestroy(m_handle));
  }

  __host__
  cublasHandle_t& getHandle()
  {
    return m_handle;
  }

private:
  cublasHandle_t m_handle;
};

__host__
CublasContext& getContext();



template <typename T>
struct CublasTgemm;

#define DEFINE_CUBLAS_T_GEMM(TYPE, NAME) \
  template <> \
  struct CublasTgemm<TYPE> \
  { \
    __host__ \
    static cublasStatus_t cublasTgemm(cublasOperation_t transa, cublasOperation_t transb, \
                               int m, int n, int k, \
                               const TYPE* alpha, \
                               const TYPE* A, int lda, \
                               const TYPE* B, int ldb, \
                               const TYPE* beta, \
                               TYPE* C, int ldc) \
    { \
      ASSERT(m >= 1, "Invalid m. m=%i", m); \
      ASSERT(n >= 1, "Invalid n. n=%i", n); \
      ASSERT(k >= 1, "Invalid k. k=%i", k); \
      ASSERT(lda >= (transa == CUBLAS_OP_N ? m : k), "Invalid lda. transa=%s lda=%i m=%i k=%i", cublasGetOpString(transa), lda, m, k); \
      ASSERT(ldb >= (transb == CUBLAS_OP_N ? k : n), "Invalid ldb. transb=%s ldb=%i k=%i n=%i", cublasGetOpString(transb), ldb, k, n); \
      ASSERT(ldc >= m, "Invalid ldc. ldc=%i m=%i", ldc, m); \
      return NAME(cublas::getContext().getHandle(), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); \
    } \
  };

DEFINE_CUBLAS_T_GEMM(float, cublasSgemm)
DEFINE_CUBLAS_T_GEMM(double, cublasDgemm)
DEFINE_CUBLAS_T_GEMM(cuComplex, cublasCgemm)
DEFINE_CUBLAS_T_GEMM(cuDoubleComplex, cublasZgemm)
DEFINE_CUBLAS_T_GEMM(__half, cublasHgemm)

#undef DEFINE_CUBLAS_T_GEMM

template <typename T>
__host__
cublasStatus_t cublasTgemm(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const T* alpha,
                           const T* A, int lda,
                           const T* B, int ldb,
                           const T* beta,
                           T* C, int ldc)
{
  return CublasTgemm<T>::cublasTgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

} // end of ns cublas

#endif
