namespace template_tensors {

/*!
 * \brief An indexing strategy for a symmetric matrix where the upper triangular portion is stored in row major order.
 */
struct SymmetricMatrixUpperTriangleRowMajor
{
  static const bool IS_STATIC = true;

  template <typename TDimArgType, typename... TCoordArgTypes, ENABLE_IF(are_dim_args_v<TDimArgType&&>::value)>
  __host__ __device__
  size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords) const
  {
    ASSERT(getNonTrivialDimensionsNum(util::forward<TDimArgType>(dims)) <= 2, "Not a matrix");
    ASSERT(coordsAreInRange(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...), "Coordinates are out of range");
    const dim_t rows = getNthDimension<0>(util::forward<TDimArgType>(dims));
    const dim_t row = getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...);
    const dim_t col = getNthCoordinate<1>(util::forward<TCoordArgTypes>(coords)...);
    if (col >= row)
    {
      return row * (rows - 1) - ((row - 1) * row >> 1) + col;
    }
    else
    {
      return col * (rows - 1) - ((col - 1) * col >> 1) + row;
    }
  }

  TT_INDEXSTRATEGY_TO_INDEX_2

  template <typename... TDimArgTypes>
  __host__ __device__
  constexpr size_t getSize(TDimArgTypes&&... dims) const
  {
    // TODO: ASSERT is symmetric and rest of dimension are 1
    const dim_t dim0 = template_tensors::getNthDimension<0>(util::forward<TDimArgTypes>(dims)...);
    return (dim0 * dim0 + dim0) >> 1;
  }

  /*template <metal::int_... TDims> // TODO: implement
  __host__ __device__
  auto fromIndex(size_t index) const -> decltype(TGetSizeStaticDims::template getSize<TDims...>(index));*/

  /*template <typename TVectorType, typename TElementType, metal::int_ TRank>
  __host__ __device__
  VectorXs<TRank> fromIndex(const Vector<TVectorType, TElementType, TRank>& dims, size_t index) const
  {

  }*/
};

/*!
 * \brief An indexing strategy for a symmetric matrix where the lower triangular portion is stored in column major order.
 */
using SymmetricMatrixLowerTriangleColMajor = SymmetricMatrixUpperTriangleRowMajor;

__host__ __device__
inline bool operator==(const SymmetricMatrixUpperTriangleRowMajor&, const SymmetricMatrixUpperTriangleRowMajor&)
{
  return true;
}

HD_WARNING_DISABLE
template <typename TStreamType>
__host__ __device__
TStreamType&& operator<<(TStreamType&& stream, const SymmetricMatrixUpperTriangleRowMajor& index_strategy)
{
  stream << "SymmetricMatrixUpperTriangleRowMajor";
  return util::forward<TStreamType>(stream);
}

#ifdef CEREAL_INCLUDED
template <typename TArchive>
void save(TArchive& archive, const SymmetricMatrixUpperTriangleRowMajor& m)
{
}

template <typename TArchive>
void load(TArchive& archive, SymmetricMatrixUpperTriangleRowMajor& m)
{
}
#endif

} // end of ns template_tensors

TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((template_tensors::SymmetricMatrixUpperTriangleRowMajor));
