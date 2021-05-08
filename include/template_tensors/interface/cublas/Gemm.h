#include "Cublas.h"

#if defined(__CUDACC__) && defined(CUBLAS_INCLUDED)

namespace template_tensors {

namespace op {

struct CublasGemm
{
  // C = alpha * A * B + beta * C
  template <typename TElementType, typename TTensorType1, typename TTensorType2, typename TTensorType3>
  __host__
  void gemm(TElementType alpha, const TTensorType1& A, const TTensorType2& B, TElementType beta, TTensorType3& C)
  {
    static_assert(are_compatible_dimseqs_v<TTensorType1, TTensorType2, TTensorType3>::value, "Incompatible static dimensions");
    ASSERT(areSameDimensions((A * B).dims(), C.dims()), "Incompatible runtime dimensions");
    static_assert(std::is_same<decay_elementtype_t<TTensorType1>, decay_elementtype_t<TTensorType2>>::value, "Inconsistent element types");
    static_assert(std::is_same<decay_elementtype_t<TTensorType1>, decay_elementtype_t<TTensorType3>>::value, "Inconsistent element types");
    static_assert(mem::isOnDevice<mem::memorytype_v<TTensorType1>::value>() && mem::isOnDevice<mem::memorytype_v<TTensorType2>::value>() && mem::isOnDevice<mem::memorytype_v<TTensorType3>::value>(), "Invalid memory types");

    static_assert(std::is_same<indexstrategy_t<TTensorType1>, RowMajor>::value || std::is_same<indexstrategy_t<TTensorType1>, ColMajor>::value, "cublasTgemm only works on RowMajor or ColMajor matrices");
    static_assert(std::is_same<indexstrategy_t<TTensorType2>, RowMajor>::value || std::is_same<indexstrategy_t<TTensorType2>, ColMajor>::value, "cublasTgemm only works on RowMajor or ColMajor matrices");
    static_assert(std::is_same<indexstrategy_t<TTensorType3>, RowMajor>::value || std::is_same<indexstrategy_t<TTensorType3>, ColMajor>::value, "cublasTgemm only works on RowMajor or ColMajor matrices");

    const bool TA = std::is_same<indexstrategy_t<TTensorType1>, RowMajor>::value;
    const bool TB = std::is_same<indexstrategy_t<TTensorType2>, RowMajor>::value;
    const bool TC = std::is_same<indexstrategy_t<TTensorType3>, RowMajor>::value;

    decay_elementtype_t<TTensorType1> alpha2 = alpha;
    decay_elementtype_t<TTensorType1> beta2 = beta;

    if (!TC)
    {
      TT_CUBLAS_SAFE_CALL(cublas::cublasTgemm<decay_elementtype_t<TTensorType1>>(
        TA ? CUBLAS_OP_T : CUBLAS_OP_N,
        TB ? CUBLAS_OP_T : CUBLAS_OP_N,
        C.template dim<0>(), // = A.template dim<0>() = m
        C.template dim<1>(), // = B.template dim<1>() = n
        A.template dim<1>(), // = B.template dim<0>() = k
        &alpha2,
        A.data(),
        TA ? A.template dim<1>() : A.template dim<0>(),
        B.data(),
        TB ? B.template dim<1>() : B.template dim<0>(),
        &beta2,
        C.data(),
        C.template dim<0>()
      ));
    }
    else
    {
      TT_CUBLAS_SAFE_CALL(cublas::cublasTgemm<decay_elementtype_t<TTensorType1>>(
        !TB ? CUBLAS_OP_T : CUBLAS_OP_N,
        !TA ? CUBLAS_OP_T : CUBLAS_OP_N,
        C.template dim<1>(), // = B.template dim<1>() = m
        C.template dim<0>(), // = A.template dim<0>() = n
        A.template dim<1>(), // = B.template dim<0>() = k
        &alpha2,
        B.data(),
        TB ? B.template dim<1>() : B.template dim<0>(),
        A.data(),
        TA ? A.template dim<1>() : A.template dim<0>(),
        &beta2,
        C.data(),
        C.template dim<1>()
      ));
    }
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize()); // TODO: when to synchronize
  }
};

} // end of ns op

} // end of ns template_tensors

#endif
