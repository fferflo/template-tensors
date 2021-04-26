namespace template_tensors {

// TODO: get efficient-access index strategy
template <typename TIndexStrategy = ColMajor, typename TAllocatorIn = util::EmptyDefaultType, typename TTensorType,
  typename TAllocator = TT_WITH_DEFAULT_TYPE(TAllocatorIn, mem::alloc::default_for<mem::memorytype_v<TTensorType>::value>),
  typename TResultType = LocalOrAllocTensorT<decay_elementtype_t<TTensorType>, TAllocator, TIndexStrategy, dimseq_t<TTensorType>>>
__host__ __device__
TResultType eval(TTensorType&& tensor, TIndexStrategy index_strategy = TIndexStrategy())
{
  TResultType result(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, index_strategy, tensor.dims());
  result = util::forward<TTensorType>(tensor);
  return result;
}

} // end of ns tensor
