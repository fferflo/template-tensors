namespace template_tensors {

namespace detail {

template <metal::int_ I, metal::int_ N>
struct RowMajorToIndexHelper
{
  template <typename TDimArgType, typename... TCoordArgTypes>
  __host__ __device__
  static size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords)
  {
    return getNthCoordinate<N - I - 1>(util::forward<TCoordArgTypes>(coords)...) + getNthDimension<N - I - 1>(util::forward<TDimArgType>(dims)) * RowMajorToIndexHelper<I + 1, N>
      ::toIndex(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...);
  }

  template <typename TDimArgType>
  __host__ __device__
  static size_t toIndex(TDimArgType&& dims)
  {
    return 0;
  }
};

template <metal::int_ N>
struct RowMajorToIndexHelper<N, N>
{
  template <typename TDimArgType, typename... TCoordArgTypes>
  __host__ __device__
  static size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords)
  {
    return 0;
  }
};



template <metal::int_ I, metal::int_ N>
struct RowMajorFromIndexHelper
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static size_t fromIndex(VectorXs<TRank>& dest, size_t index, TDimArgTypes&&... dims)
  {
    const dim_t dim = getNthDimension<I - 1>(util::forward<TDimArgTypes>(dims)...);
    index = RowMajorFromIndexHelper<I + 1, N>::fromIndex(dest, index, util::forward<TDimArgTypes>(dims)...);
    dest(I - 1) = index % dim;
    return index / dim;
  }
};

template <metal::int_ I>
struct RowMajorFromIndexHelper<I, I>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static size_t fromIndex(VectorXs<TRank>& dest, size_t index, TDimArgTypes&&... dims)
  {
    const dim_t dim = getNthDimension<I - 1>(util::forward<TDimArgTypes>(dims)...);
    dest(I - 1) = index % dim;
    return index / dim;
  }
};

template <metal::int_ N>
struct RowMajorFromIndexHelper<1, N>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static size_t fromIndex(VectorXs<TRank>& dest, size_t index, TDimArgTypes&&... dims)
  {
    const dim_t dim = getNthDimension<0>(util::forward<TDimArgTypes>(dims)...);
    index = RowMajorFromIndexHelper<2, N>::fromIndex(dest, index, util::forward<TDimArgTypes>(dims)...);
    dest(0) = index % dim;
    return 0; // This return value is not used
  }
};

template <>
struct RowMajorFromIndexHelper<1, 1>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static size_t fromIndex(VectorXs<TRank>& dest, size_t index, TDimArgTypes&&... dims)
  {
    dest(0) = index;
    return 0; // This return value is not used
  }
};

template <>
struct RowMajorFromIndexHelper<1, 0>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static size_t fromIndex(VectorXs<TRank>& dest, size_t index, TDimArgTypes&&... dims)
  {
    return 0; // This return value is not used
  }
};



template <metal::int_ I, metal::int_ N>
struct RowMajorToStrideHelper
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
    const dim_t dim = getNthDimension<N - I>(util::forward<TDimArgTypes>(dims)...);
    dest(N - I - 1) = dest(N - I) * dim;
    RowMajorToStrideHelper<I + 1, N>::toStride(dest, util::forward<TDimArgTypes>(dims)...);
  }
};

template <metal::int_ N>
struct RowMajorToStrideHelper<N, N>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
  }
};

template <metal::int_ N>
struct RowMajorToStrideHelper<1, N>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
    const dim_t dim = getNthDimension<N - 1>(util::forward<TDimArgTypes>(dims)...);
    dest(N - 2) = dim;
    RowMajorToStrideHelper<2, N>::toStride(dest, util::forward<TDimArgTypes>(dims)...);
  }
};

template <metal::int_ N>
struct RowMajorToStrideHelper<0, N>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
    dest(N - 1) = 1;
    RowMajorToStrideHelper<1, N>::toStride(dest, util::forward<TDimArgTypes>(dims)...);
  }
};

template <>
struct RowMajorToStrideHelper<1, 1>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
  }
};

template <>
struct RowMajorToStrideHelper<0, 0>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
  }
};

} // end of ns detail





/*!
 * \brief An indexing strategy for row-major dense tensors. The last coordinate lays contiguous in memory.
 */
struct RowMajor
{
  static const bool IS_STATIC = true;

  template <typename TDimArgType, typename... TCoordArgTypes, ENABLE_IF(are_dim_args_v<TDimArgType&&>::value)>
  __host__ __device__
  size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords) const volatile
  {
    ASSERT(coordsAreInRange(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...), "Coordinates are out of range");
    return detail::RowMajorToIndexHelper<0, dimension_num_v<TDimArgType>::value>::toIndex(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...);
  }

  TT_INDEXSTRATEGY_TO_INDEX_2

  template <metal::int_ TDimsArg = DYN, typename... TDimArgTypes, metal::int_ TDims = TDimsArg == DYN ? dimension_num_v<TDimArgTypes&&...>::value : TDimsArg>
  __host__ __device__
  VectorXs<TDims> fromIndex(size_t index, TDimArgTypes&&... dims) const volatile
  {
    VectorXs<TDims> result;
    detail::RowMajorFromIndexHelper<1, TDims>::fromIndex(result, index, util::forward<TDimArgTypes>(dims)...);
    return result;
  }

  TT_INDEXSTRATEGY_FROM_INDEX_2

  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  constexpr size_t getSize(TDimArgTypes&&... dim_args) const volatile
  {
    return multiplyDimensions(util::forward<TDimArgTypes>(dim_args)...);
  }

  template <metal::int_ TDimsArg = DYN, typename... TDimArgTypes, metal::int_ TDims = TDimsArg == DYN ? dimension_num_v<TDimArgTypes&&...>::value : TDimsArg>
  __host__ __device__
  VectorXs<TDims> toStride(TDimArgTypes&&... dims) const volatile
  {
    VectorXs<TDims> result;
    detail::RowMajorToStrideHelper<0, TDims>::toStride(result, util::forward<TDimArgTypes>(dims)...);
    return result;
  }
};

__host__ __device__
inline bool operator==(const volatile RowMajor&, const volatile RowMajor&)
{
  return true;
}

HD_WARNING_DISABLE
template <typename TStreamType>
__host__ __device__
TStreamType&& operator<<(TStreamType&& stream, const RowMajor& index_strategy)
{
  stream << "RowMajor";
  return util::forward<TStreamType>(stream);
}

#ifdef CEREAL_INCLUDED
template <typename TArchive>
void save(TArchive& archive, const RowMajor& m)
{
}

template <typename TArchive>
void load(TArchive& archive, RowMajor& m)
{
}
#endif

} // end of ns tensor

TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((template_tensors::RowMajor));
