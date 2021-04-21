#include <template_tensors/interface/cublas/Cublas.h>

#ifdef __CUDACC__

namespace cublas {

__host__ __device__
const char* cublasGetErrorString(cublasStatus_t err)
{
  switch (err)
  {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "Unknown CUBLAS error";
}

__host__ __device__
const char* cublasGetOpString(cublasOperation_t op)
{
  switch (op)
  {
    case CUBLAS_OP_N: return "CUBLAS_OP_N";
    case CUBLAS_OP_T: return "CUBLAS_OP_T";
    case CUBLAS_OP_C: return "CUBLAS_OP_C";
  }
  return "Unknown CUBLAS op";
}

thread_local CublasContext context;

__host__
CublasContext& getContext()
{
  return context;
}

} // end of ns cublas

#endif