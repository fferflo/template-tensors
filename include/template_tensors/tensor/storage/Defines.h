namespace template_tensors {

// TODO: remove all of these for being ugly:
struct ExplicitConstructWithDynDims
{
};
#define TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS (::template_tensors::ExplicitConstructWithDynDims())
struct ExplicitConstructWithStorageArgs
{
};
#define TT_EXPLICIT_CONSTRUCT_WITH_STORAGE_ARGS (::template_tensors::ExplicitConstructWithStorageArgs())

class IsIndexedPointerTensor
{
};
template <typename TType>
TVALUE(bool, is_indexed_pointer_tensor_v, std::is_base_of<IsIndexedPointerTensor, typename std::decay<TType>::type>::value)

class IsIndexedArrayTensor
{
};
template <typename TType>
TVALUE(bool, is_indexed_array_tensor_v, std::is_base_of<IsIndexedArrayTensor, typename std::decay<TType>::type>::value)

namespace detail {

template <typename TArg>
struct IndexStrategyGetter
{
  template <typename TThisType, typename TElementType, typename TIndexStrategy, mem::MemoryType TMemoryType, typename TDimSeq>
  TMP_IF(const IndexedPointerTensor<TThisType, TElementType, TIndexStrategy, TMemoryType, TDimSeq>&)
  TMP_RETURN_TYPE(TIndexStrategy)

  TMP_DEDUCE_TYPE(typename std::decay<TArg>::type);
};

} // end of ns detail

template <typename TTensorRefType, ENABLE_IF(is_indexed_pointer_tensor_v<TTensorRefType>::value)>
using indexstrategy_t = typename detail::IndexStrategyGetter<TTensorRefType>::type;

template <typename TTensorRefType, ENABLE_IF(is_indexed_array_tensor_v<TTensorRefType>::value)>
using array_t = typename std::decay<TTensorRefType>::type::ThisType::ArrayType;



namespace detail {

template <bool TStatic, typename TIndexStrategy, typename TDimSeq>
struct IndexedStaticLength;

template <typename TIndexStrategy, typename TDimSeq>
struct IndexedStaticLength<true, TIndexStrategy, TDimSeq>
{
  static const metal::int_ value = TIndexStrategy().getSize(TDimSeq());
};

template <typename TIndexStrategy, typename TDimSeq>
struct IndexedStaticLength<false, TIndexStrategy, TDimSeq>
{
  static const metal::int_ value = mem::DYN;
};

} // end of ns detail

template <typename TIndexStrategy, typename TDimSeq>
TVALUE(metal::int_, indexed_size_ex_v, detail::IndexedStaticLength<
  TIndexStrategy::IS_STATIC && template_tensors::is_static_v<TDimSeq>::value, TIndexStrategy, TDimSeq>::value)
template <typename TTensorType, ENABLE_IF(is_indexed_pointer_tensor_v<TTensorType>::value)>
TVALUE(metal::int_, indexed_size_v, indexed_size_ex_v<indexstrategy_t<TTensorType>, dimseq_t<TTensorType>>::value)



template <typename TArg, typename TIndexStrategy>
struct has_indexstrategy_v
{
  template <typename TTensorType, ENABLE_IF(std::is_same<indexstrategy_t<TTensorType>, TIndexStrategy>::value)>
  TMP_IF(TTensorType&&)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

template <typename... TArgs>
struct have_same_indexstrategy_v
{
  template <typename TTensorType1, typename... TTensorTypesRest,
    ENABLE_IF(is_indexed_pointer_tensor_v<TTensorType1>::value && math::land(is_indexed_pointer_tensor_v<TTensorTypesRest>::value...))>
  TMP_IF(TTensorType1&&, TTensorTypesRest&&...)
  TMP_RETURN_VALUE(math::land(std::is_same<indexstrategy_t<TTensorTypesRest>, indexstrategy_t<TTensorType1>>::value...))

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArgs...);
};

template <>
struct have_same_indexstrategy_v<>
{
  static const bool value = true;
};

} // end of ns template_tensors
