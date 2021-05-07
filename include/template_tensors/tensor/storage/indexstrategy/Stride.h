namespace template_tensors {

namespace detail {

template <metal::int_ I>
struct StrideToIndexHelper
{
  template <typename TStrideVector, typename... TCoordArgTypes>
  __host__ __device__
  static size_t toIndex(TStrideVector&& stride, TCoordArgTypes&&... coords)
  {
    return stride(I) * getNthCoordinate<I>(util::forward<TCoordArgTypes>(coords)...)
      + StrideToIndexHelper<I - 1>::toIndex(stride, util::forward<TCoordArgTypes>(coords)...);
  }
};

template <>
struct StrideToIndexHelper<0>
{
  template <typename TStrideVector, typename... TCoordArgTypes>
  __host__ __device__
  static size_t toIndex(TStrideVector&& stride, TCoordArgTypes&&... coords)
  {
    return stride(0) * getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...);
  }
};



template <metal::int_ I, metal::int_ N>
struct StrideGetSizeHelper
{
  template <metal::int_ TStrideRank, typename... TDimArgTypes>
  __host__ __device__
  static size_t getSize(const VectorXs<TStrideRank>& stride, TDimArgTypes&&... dims)
  {
    const dim_t dim = getNthDimension<I>(util::forward<TDimArgTypes>(dims)...);
    return stride(I) * (dim - 1) + StrideGetSizeHelper<I + 1, N>::getSize(stride, util::forward<TDimArgTypes>(dims)...);
  }
};

template <metal::int_ I>
struct StrideGetSizeHelper<I, I>
{
  template <metal::int_ TStrideRank, typename... TDimArgTypes>
  __host__ __device__
  static size_t getSize(const VectorXs<TStrideRank>& stride, TDimArgTypes&&... dims)
  {
    return 1;
  }
};

} // end of ns detail





/*!
 * \brief An indexing strategy that calculates indices using a vector of strides that determines how far apart consecutive elements in each direction lie.
 *
 * Strides that are not given default to zero.
 */
template <metal::int_ TRank>
class Stride
{
public:
  static const bool IS_STATIC = false;

  template <typename... TStrideArgs, ENABLE_IF(std::is_constructible<VectorXs<TRank>, TStrideArgs...>::value)>
  __host__ __device__
  Stride(TStrideArgs&&... strides)
    : m_stride(util::forward<TStrideArgs>(strides)...)
  {
  }

  template <typename TDimArgType, typename... TCoordArgTypes, ENABLE_IF(are_dim_args_v<TDimArgType&&>::value)>
  __host__ __device__
  size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords) const
  {
    ASSERT(coordsAreInRange(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...), "Coordinates are out of range");
    return detail::StrideToIndexHelper<TRank - 1>::toIndex(m_stride, util::forward<TCoordArgTypes>(coords)...);
  }

  template <typename TDimArgType, typename... TCoordArgTypes, ENABLE_IF(are_dim_args_v<TDimArgType&&>::value)>
  __host__ __device__
  size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords) const volatile
  {
    ASSERT(coordsAreInRange(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...), "Coordinates are out of range");
    return detail::StrideToIndexHelper<TRank - 1>::toIndex(m_stride, util::forward<TCoordArgTypes>(coords)...);
  }

  TT_INDEXSTRATEGY_TO_INDEX_2

  template <typename... TDimArgTypes>
  __host__ __device__
  size_t getSize(TDimArgTypes&&... dims) const // template_tensors::dot(m_stride, dims - 1) + 1
  {
    return detail::StrideGetSizeHelper<0, TRank>::getSize(m_stride, util::forward<TDimArgTypes>(dims)...);
  }

  template <typename... TDimArgTypes>
  __host__ __device__
  size_t getSize(TDimArgTypes&&... dims) const volatile // template_tensors::dot(m_stride, dims - 1) + 1
  {
    return detail::StrideGetSizeHelper<0, TRank>::getSize(m_stride, util::forward<TDimArgTypes>(dims)...);
  }

  template <metal::int_ TRank2>
  __host__ __device__
  VectorXs<TRank2> toStride() const
  {
    return toCoordVector<TRank2>(m_stride);
  }

  template <metal::int_ TRank2>
  __host__ __device__
  VectorXs<TRank2> toStride() const volatile
  {
    return toCoordVector<TRank2>(m_stride);
  }

  template <metal::int_ TDimsArg = DYN, typename... TDimArgTypes, metal::int_ TDims = TDimsArg == DYN ? dimension_num_v<TDimArgTypes&&...>::value : TDimsArg>
  __host__ __device__
  VectorXs<TDims> toStride(TDimArgTypes&&... dims) const
  {
    return toStride<TDims>();
  }

  template <metal::int_ TDimsArg = DYN, typename... TDimArgTypes, metal::int_ TDims = TDimsArg == DYN ? dimension_num_v<TDimArgTypes&&...>::value : TDimsArg>
  __host__ __device__
  VectorXs<TDims> toStride(TDimArgTypes&&... dims) const volatile
  {
    return toStride<TDims>();
  }

  template <metal::int_ TRank2>
  __host__ __device__
  friend bool operator==(const Stride<TRank2>& left, const Stride<TRank2>& right);
  template <metal::int_ TRank2>
  __host__ __device__
  friend bool operator==(const Stride<TRank2>& left, const volatile Stride<TRank2>& right);
  template <metal::int_ TRank2>
  __host__ __device__
  friend bool operator==(const volatile Stride<TRank2>& left, const Stride<TRank2>& right);
  template <typename TStreamType, metal::int_ TRank2>
  __host__ __device__
  friend TStreamType&& operator<<(TStreamType&& stream, const Stride<TRank2>& index_strategy);

  template <typename TArchive, metal::int_ TRank2>
  friend void save(TArchive& archive, const Stride<TRank2>& m);
  template <typename TArchive, metal::int_ TRank2>
  friend void load(TArchive& archive, Stride<TRank2>& m);

private:
  VectorXs<TRank> m_stride;
};

template <metal::int_ TRank>
__host__ __device__
bool operator==(const Stride<TRank>& left, const Stride<TRank>& right)
{
  for (auto i = 0; i < TRank; i++)
  {
    if (left.m_stride(i) != right.m_stride(i))
    {
      return false;
    }
  }
  return true;
}

template <metal::int_ TRank>
__host__ __device__
bool operator==(const Stride<TRank>& left, const volatile Stride<TRank>& right)
{
  for (auto i = 0; i < TRank; i++)
  {
    if (left.m_stride(i) != right.m_stride(i))
    {
      return false;
    }
  }
  return true;
}

template <metal::int_ TRank>
__host__ __device__
bool operator==(const volatile Stride<TRank>& left, const Stride<TRank>& right)
{
  for (auto i = 0; i < TRank; i++)
  {
    if (left.m_stride(i) != right.m_stride(i))
    {
      return false;
    }
  }
  return true;
}

HD_WARNING_DISABLE
template <typename TStreamType, metal::int_ TRank>
__host__ __device__
TStreamType&& operator<<(TStreamType&& stream, const Stride<TRank>& index_strategy)
{
  stream << "Stride(" << index_strategy.m_stride << ")";
  return util::forward<TStreamType>(stream);
}

#ifdef CEREAL_INCLUDED
template <typename TArchive, metal::int_ TRank>
void save(TArchive& archive, const Stride<TRank>& m)
{
  archive(m.getArray());
  archive(m.getIndexStrategy());
}

template <typename TArchive, metal::int_ TRank>
void load(TArchive& archive, Stride<TRank>& m)
{
  archive(m.template toStride<TRank>);
}
#endif

} // end of ns tensor

template <metal::int_ TRank>
TT_PROCLAIM_TRIVIALLY_RELOCATABLE((template_tensors::Stride<TRank>));
