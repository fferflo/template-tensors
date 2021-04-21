namespace template_tensors {

#define INDEXSTRATEGY_TO_INDEX_2 \
  template <size_t... TDims, typename... TCoordArgTypes, ENABLE_IF(are_coord_args_v<TCoordArgTypes&&...>::value && sizeof...(TDims) != 0)> \
  __host__ __device__ \
  size_t toIndex(TCoordArgTypes&&... coords) const \
  { \
    return toIndex(DimSeq<TDims...>(), util::forward<TCoordArgTypes>(coords)...); \
  } \
  template <size_t... TDims, typename... TCoordArgTypes, ENABLE_IF(are_coord_args_v<TCoordArgTypes&&...>::value && sizeof...(TDims) != 0)> \
  __host__ __device__ \
  size_t toIndex(TCoordArgTypes&&... coords) const volatile \
  { \
    return toIndex(DimSeq<TDims...>(), util::forward<TCoordArgTypes>(coords)...); \
  }

#define INDEXSTRATEGY_FROM_INDEX_2 \
  template <size_t... TDims> \
  __host__ __device__ \
  auto fromIndex(size_t index) const \
  RETURN_AUTO(fromIndex(index, DimSeq<TDims...>())) \
  template <size_t... TDims> \
  __host__ __device__ \
  auto fromIndex(size_t index) const volatile \
  RETURN_AUTO(fromIndex(index, DimSeq<TDims...>()))

template <typename TArg>
struct indexstrategy_can_convert_from_index_v
{
  template <typename TIndexStrategy>
  TMP_IF(TIndexStrategy&&, decltype(std::declval<typename std::decay<TIndexStrategy>::value>().fromIndex(0))* dummy = nullptr)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

} // end of ns tensor
