namespace template_tensors {

namespace detail {

template <metal::int_ I, metal::int_ N>
struct ColMajorToIndexHelper
{
  template <typename TDimArgType, typename... TCoordArgTypes>
  __host__ __device__
  static size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords)
  {
    return getNthCoordinate<I>(util::forward<TCoordArgTypes>(coords)...) + getNthDimension<I>(util::forward<TDimArgType>(dims)) * ColMajorToIndexHelper<I + 1, N>
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
struct ColMajorToIndexHelper<N, N>
{
  template <typename TDimArgType, typename... TCoordArgTypes>
  __host__ __device__
  static size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords)
  {
    return 0;
  }
};



template <metal::int_ I, metal::int_ N>
struct ColMajorFromIndexHelper
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void fromIndex(VectorXs<TRank>& dest, size_t index, TDimArgTypes&&... dims)
  {
    const dim_t dim = getNthDimension<I - 1>(util::forward<TDimArgTypes>(dims)...);
    dest(I - 1) = index % dim;
    ColMajorFromIndexHelper<I + 1, N>::fromIndex(dest, index / dim, util::forward<TDimArgTypes>(dims)...);
  }
};

template <metal::int_ I>
struct ColMajorFromIndexHelper<I, I>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void fromIndex(VectorXs<TRank>& dest, size_t index, TDimArgTypes&&... dims)
  {
    dest(I - 1) = index;
  }
};

template <>
struct ColMajorFromIndexHelper<1, 0>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void fromIndex(VectorXs<TRank>& dest, size_t index, TDimArgTypes&&... dims)
  {
  }
};



template <metal::int_ I, metal::int_ N>
struct ColMajorToStrideHelper
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
    dest(I) = dest(I - 1) * getNthDimension<I - 1>(util::forward<TDimArgTypes>(dims)...);
    ColMajorToStrideHelper<I + 1, N>::toStride(dest, util::forward<TDimArgTypes>(dims)...);
  }
};

template <metal::int_ N>
struct ColMajorToStrideHelper<N, N>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
  }
};

template <metal::int_ N>
struct ColMajorToStrideHelper<1, N>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
    dest(1) = getNthDimension<0>(util::forward<TDimArgTypes>(dims)...);
    ColMajorToStrideHelper<2, N>::toStride(dest, util::forward<TDimArgTypes>(dims)...);
  }
};

template <metal::int_ N>
struct ColMajorToStrideHelper<0, N>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
    dest(0) = 1;
    ColMajorToStrideHelper<1, N>::toStride(dest, util::forward<TDimArgTypes>(dims)...);
  }
};

template <>
struct ColMajorToStrideHelper<1, 1>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
  }
};

template <>
struct ColMajorToStrideHelper<0, 0>
{
  template <metal::int_ TRank, typename... TDimArgTypes>
  __host__ __device__
  static void toStride(VectorXs<TRank>& dest, TDimArgTypes&&... dims)
  {
  }
};

} // end of ns detail





/*!
 * \brief An indexing strategy for column-major dense tensors. The first coordinate lays contiguous in memory.
 */
struct ColMajor
{
  static const bool IS_STATIC = true;

  template <typename TDimArgType, typename... TCoordArgTypes, ENABLE_IF(are_dim_args_v<TDimArgType&&>::value)>
  __host__ __device__
  size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords) const volatile
  {
    ASSERT(coordsAreInRange(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...), "Coordinates are out of range");
    return detail::ColMajorToIndexHelper<0, dimension_num_v<TDimArgType>::value>::toIndex(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...);
  }

  TT_INDEXSTRATEGY_TO_INDEX_2

  template <metal::int_ TDimsArg = DYN, typename... TDimArgTypes, metal::int_ TDims = TDimsArg == DYN ? dimension_num_v<TDimArgTypes&&...>::value : TDimsArg>
  __host__ __device__
  VectorXs<TDims> fromIndex(size_t index, TDimArgTypes&&... dims) const volatile
  {
    VectorXs<TDims> result;
    detail::ColMajorFromIndexHelper<1, TDims>::fromIndex(result, index, util::forward<TDimArgTypes>(dims)...);
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
    detail::ColMajorToStrideHelper<0, TDims>::toStride(result, util::forward<TDimArgTypes>(dims)...);
    return result;
  }
};

__host__ __device__
inline bool operator==(const volatile ColMajor&, const volatile ColMajor&)
{
  return true;
}

HD_WARNING_DISABLE
template <typename TStreamType>
__host__ __device__
TStreamType&& operator<<(TStreamType&& stream, const ColMajor& index_strategy)
{
  stream << "ColMajor";
  return util::forward<TStreamType>(stream);
}

#ifdef CEREAL_INCLUDED
template <typename TArchive>
void save(TArchive& archive, const ColMajor& m)
{
}

template <typename TArchive>
void load(TArchive& archive, ColMajor& m)
{
}
#endif

} // end of ns tensor

TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((template_tensors::ColMajor));
