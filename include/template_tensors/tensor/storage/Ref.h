namespace template_tensors {

template <typename TPointerType, mem::MemoryType TMemoryType, size_t TSize>
using RefArrayType = ::array::ReferenceArray<ptr::value_t<TPointerType>, TMemoryType, TSize, typename std::decay<TPointerType>::type>;
template <typename TPointerType, mem::MemoryType TMemoryType, typename TIndexStrategy, typename TDimSeq>
using RefArrayTypeEx = ::array::ReferenceArray<ptr::value_t<TPointerType>, TMemoryType, indexed_size_ex_v<TIndexStrategy, TDimSeq>::value, typename std::decay<TPointerType>::type>;
template <typename TPointerType, mem::MemoryType TMemoryType, typename TIndexStrategy, typename TDimSeq>
using RefTensorType = IndexedArrayTensor<
        RefArrayTypeEx<TPointerType, TMemoryType, TIndexStrategy, TDimSeq>,
        ptr::value_t<TPointerType>, TIndexStrategy, TDimSeq>;



HD_WARNING_DISABLE
template <typename TIndexStrategy, mem::MemoryType TMemoryType, size_t... TDims, typename TPointerType,
  ENABLE_IF(is_static_v<DimSeq<TDims...>>::value)>
__host__ __device__
auto ref(const TPointerType& data)
RETURN_AUTO(RefTensorType<TPointerType, TMemoryType, TIndexStrategy, DimSeq<TDims...>>(
    RefArrayTypeEx<TPointerType, TMemoryType, TIndexStrategy, DimSeq<TDims...>>(data, TIndexStrategy().getSize(DimSeq<TDims...>()))
  ))

HD_WARNING_DISABLE
template <typename TIndexStrategy, mem::MemoryType TMemoryType, typename... TDimArgTypes, typename TPointerType,
  ENABLE_IF(math::gt(sizeof...(TDimArgTypes), 0UL))>
__host__ __device__
auto ref(const TPointerType& data, TDimArgTypes&&... dim_args)
RETURN_AUTO(RefTensorType<TPointerType, TMemoryType, TIndexStrategy, dyn_dimseq_t<dimension_num_v<TDimArgTypes...>::value>>(
    RefArrayType<TPointerType, TMemoryType, ::array::DYN>(data, TIndexStrategy().getSize(util::forward<TDimArgTypes>(dim_args)...)), util::forward<TDimArgTypes>(dim_args)...
  ))

HD_WARNING_DISABLE
template <mem::MemoryType TMemoryType, typename TIndexStrategy, typename... TDimArgTypes, typename TPointerType,
  ENABLE_IF(math::gt(sizeof...(TDimArgTypes), 0UL) && !are_dim_args_v<TIndexStrategy>::value)>
__host__ __device__
auto ref(const TPointerType& data, TIndexStrategy index_strategy, TDimArgTypes&&... dim_args)
RETURN_AUTO(RefTensorType<TPointerType, TMemoryType, typename std::decay<TIndexStrategy>::type, dyn_dimseq_t<dimension_num_v<TDimArgTypes...>::value>>(
    TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, RefArrayType<TPointerType, TMemoryType, ::array::DYN>(data, index_strategy.getSize(util::forward<TDimArgTypes>(dim_args)...)), index_strategy, util::forward<TDimArgTypes>(dim_args)...
  ))



template <typename TDimSeq, typename TIndexStrategy, typename... TDimArgTypes, typename TArray, ENABLE_IF(::array::is_array_type_v<TArray&&>::value),
  typename TArrayRef = decltype(::array::ref(std::declval<TArray&&>()))>
__host__ __device__
auto refEx(TArray&& array, TIndexStrategy index_strategy, TDimArgTypes&&... dim_args)
RETURN_AUTO(
    IndexedArrayTensor<TArrayRef, ::array::elementtype_t<TArrayRef>, typename std::decay<TIndexStrategy>::type, TDimSeq>
      (TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, ::array::ref(util::forward<TArray>(array)),
        index_strategy, util::forward<TDimArgTypes>(dim_args)...)
  )

template <size_t... TDims, typename TIndexStrategy, typename TArray, ENABLE_IF(::array::is_array_type_v<TArray>::value),
  typename TArrayRef = decltype(::array::ref(std::declval<TArray&&>()))>
__host__ __device__
auto ref(TArray&& array, TIndexStrategy index_strategy)
RETURN_AUTO(template_tensors::refEx<DimSeq<TDims...>>(util::forward<TArray>(array), index_strategy, DimSeq<TDims...>()))

template <typename TIndexStrategy, size_t... TDims, typename TArray, ENABLE_IF(::array::is_array_type_v<TArray>::value)>
__host__ __device__
auto ref(TArray&& array)
RETURN_AUTO(template_tensors::ref<TDims...>(util::forward<TArray>(array), TIndexStrategy()))

template <typename TIndexStrategy, typename... TDimArgTypes, typename TArray, ENABLE_IF(::array::is_array_type_v<TArray&&>::value && sizeof...(TDimArgTypes) != 0UL),
  typename TDimSeq = template_tensors::dyn_dimseq_t<dimension_num_v<TDimArgTypes...>::value>>
__host__ __device__
auto ref(TArray&& array, TIndexStrategy index_strategy, TDimArgTypes&&... dim_args)
RETURN_AUTO(template_tensors::refEx<TDimSeq>(util::forward<TArray>(array), index_strategy, util::forward<TDimArgTypes>(dim_args)...))

template <typename TIndexStrategy, typename... TDimArgTypes, typename TArray, ENABLE_IF(::array::is_array_type_v<TArray&&>::value && sizeof...(TDimArgTypes) != 0UL)>
__host__ __device__
auto ref(TArray&& array, TDimArgTypes&&... dim_args)
RETURN_AUTO(template_tensors::ref(util::forward<TArray>(array), TIndexStrategy(), util::forward<TDimArgTypes>(dim_args)...))



template <typename TTensorType, ENABLE_IF(is_indexed_array_tensor_v<TTensorType>::value)>
__host__ __device__
auto ref(TTensorType&& tensor)
RETURN_AUTO(template_tensors::refEx<dimseq_t<TTensorType&&>>(
  ::array::ref(util::forward<TTensorType>(tensor).getArray()),
  tensor.getIndexStrategy(),
  tensor.dims()
))

template <typename TTensorType, ENABLE_IF(!is_indexed_array_tensor_v<TTensorType>::value && is_indexed_pointer_tensor_v<TTensorType>::value)>
__host__ __device__
auto ref(TTensorType&& tensor)
RETURN_AUTO(template_tensors::refEx<dimseq_t<TTensorType&&>>(
  ::array::ref<mem::memorytype_v<TTensorType>::value, indexed_size_v<TTensorType>::value>(util::forward<TTensorType>(tensor).data(), tensor.size()),
  tensor.getIndexStrategy(),
  tensor.dims()
))

} // end of ns tensor
