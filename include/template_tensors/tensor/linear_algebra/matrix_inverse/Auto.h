namespace template_tensors {

namespace op {

namespace detail {

template <metal::int_ TRank>
struct AutoInverseHelper;

template <>
struct AutoInverseHelper<2>
{
  template <typename TMatrixTypeDest, typename TMatrixTypeSrc>
  __host__ __device__
  bool operator()(TMatrixTypeDest&& dest, TMatrixTypeSrc&& src)
  {
    TT_MATRIX_INVERSE_CHECK_DIMS

    return template_tensors::op::ClosedFormInverse<2>()(util::forward<TMatrixTypeDest>(dest), util::forward<TMatrixTypeSrc>(src));
  }
};
// TODO: add this to Dispatch.h
template <>
struct AutoInverseHelper<3>
{
  template <typename TMatrixTypeDest, typename TMatrixTypeSrc>
  __host__ __device__
  bool operator()(TMatrixTypeDest&& dest, TMatrixTypeSrc&& src)
  {
    TT_MATRIX_INVERSE_CHECK_DIMS

    return template_tensors::op::ClosedFormInverse<3>()(util::forward<TMatrixTypeDest>(dest), util::forward<TMatrixTypeSrc>(src));
  }
};

} // end of ns detail

class AutoInverse
{
public:
  template <typename TMatrixTypeDest, typename TMatrixTypeSrc>
  __host__ __device__
  bool operator()(TMatrixTypeDest&& dest, TMatrixTypeSrc&& src)
  {
    TT_MATRIX_INVERSE_CHECK_DIMS

    return detail::AutoInverseHelper<RANK>()(util::forward<TMatrixTypeDest>(dest), util::forward<TMatrixTypeSrc>(src));
  }
};

} // end of ns op

template <typename TIndexStrategy = ColMajor, typename TAllocatorIn = util::EmptyDefaultType, typename TMatrixType,
  typename TAllocator = TT_WITH_DEFAULT_TYPE(TAllocatorIn, mem::alloc::default_for<mem::memorytype_v<TMatrixType>::value>),
  typename TResultType = LocalOrAllocTensorT<decay_elementtype_t<TMatrixType>, TAllocator, TIndexStrategy, dimseq_t<TMatrixType>>>
__host__ __device__
TResultType inverse(TMatrixType&& matrix, TIndexStrategy index_strategy = TIndexStrategy())
{
  TResultType result(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, index_strategy, matrix.dims());
  op::AutoInverse()(result, util::forward<TMatrixType>(matrix));
  return result;
}

} // end of ns tensor
