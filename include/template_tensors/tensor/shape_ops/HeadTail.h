namespace template_tensors {

namespace detail {

template <size_t TStaticDim>
struct HeadTensorDimHelper
{
  __host__ __device__
  static size_t get(size_t dynamic_dim)
  {
    return TStaticDim;
  }
};

template <>
struct HeadTensorDimHelper<template_tensors::DYN>
{
  __host__ __device__
  static size_t get(size_t dynamic_dim)
  {
    return dynamic_dim;
  }
};

} // end of ns detail

#define ThisType HeadTensor<TTensorTypeIn, TNewDimSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        TNewDimSeq \
                              >
template <typename TTensorTypeIn, typename TNewDimSeq>
class HeadTensor : public SuperType, public StoreDimensions<TNewDimSeq>
{
public:
  static_assert(
    !is_static_v<TNewDimSeq>::value ||
    !is_static_v<TTensorTypeIn>::value ||
    tmp::vs::all_v<
      tmp::vs::map2_t<
        math::functor::lte,
        dimseq_make_length_t<TNewDimSeq             , math::max(tmp::vs::length_v<dimseq_t<TTensorTypeIn>>::value, tmp::vs::length_v<TNewDimSeq>::value)>,
        dimseq_make_length_t<dimseq_t<TTensorTypeIn>, math::max(tmp::vs::length_v<dimseq_t<TTensorTypeIn>>::value, tmp::vs::length_v<TNewDimSeq>::value)>
      >
    >::value,
    "New dimensions must be smaller or equal to original dimensions");

  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor"); // TODO: remove all of these assertions

  static const size_t NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  HeadTensor(TTensorTypeIn tensor)
    : SuperType(dims_helper(tmp::vs::ascending_numbers_t<NON_TRIVIAL_DIMENSIONS_NUM>(), tensor))
    , StoreDimensions<TNewDimSeq>(dims_helper(tmp::vs::ascending_numbers_t<NON_TRIVIAL_DIMENSIONS_NUM>(), tensor))
    , m_tensor(tensor)
  {
  }

  template <typename... TDimArgTypes>
  __host__ __device__
  HeadTensor(TTensorTypeIn tensor, TDimArgTypes&&... dim_args)
    : SuperType(util::forward<TDimArgTypes>(dim_args)...)
    , StoreDimensions<TNewDimSeq>(util::forward<TDimArgTypes>(dim_args)...)
    , m_tensor(tensor)
  {
    ASSERT(areCompatibleDimensions<TNewDimSeq>(util::forward<TDimArgTypes>(dim_args)...), "Incompatible run-time and compile-time dimensions");
  }

  TENSOR_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_tensor(util::forward<TCoordArgTypes>(coords)...)
  )
  TENSOR_FORWARD_ELEMENT_ACCESS(getElement)

private:
  TTensorTypeIn m_tensor;

  template <size_t... TIndices, typename TTensorType>
  __host__ __device__
  auto dims_helper(tmp::vs::Sequence<size_t, TIndices...>, const TTensorType& tensor)
  RETURN_AUTO(VectorXs<sizeof...(TIndices)>(
      detail::HeadTensorDimHelper<
                      nth_dimension_v<TIndices, TNewDimSeq>::value
                >::get(tensor.template dim<TIndices>())...
    ))

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(HeadTensor<decltype(transform(m_tensor)), TNewDimSeq>
    (transform(m_tensor), this->dims())
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(HeadTensor<decltype(transform(m_tensor)), TNewDimSeq>
    (transform(m_tensor), this->dims())
  )
};
#undef SuperType
#undef ThisType



namespace detail {

struct OffsetDimHelper
{
  constexpr size_t operator()(size_t dim, size_t offset) const
  {
    return (dim == DYN || offset == DYN) ? DYN : dim - offset;
  }
};

} // end of ns detail

#define ThisType StaticOffsetTensor<TTensorTypeIn, TOffsetCoordSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        tmp::vs::map2_t< \
                                          detail::OffsetDimHelper, \
                                          dimseq_make_length_t<dimseq_t<TTensorTypeIn>, math::max(tmp::vs::length_v<dimseq_t<TTensorTypeIn>>::value, tmp::vs::length_v<TOffsetCoordSeq>::value)>, \
                                          coordseq_make_length_t<TOffsetCoordSeq      , math::max(tmp::vs::length_v<dimseq_t<TTensorTypeIn>>::value, tmp::vs::length_v<TOffsetCoordSeq>::value)> \
                                        > \
                              >
template <typename TTensorTypeIn, typename TOffsetCoordSeq>
class StaticOffsetTensor : public SuperType
{
public:
  static_assert(is_static_v<TOffsetCoordSeq>::value, "Offset must be static");
  static_assert(
    !is_static_v<TTensorTypeIn>::value ||
    tmp::vs::all_v<
      tmp::vs::map2_t<
        math::functor::lt,
        coordseq_make_length_t<TOffsetCoordSeq      , math::max(tmp::vs::length_v<dimseq_t<TTensorTypeIn>>::value, tmp::vs::length_v<TOffsetCoordSeq>::value)>,
        dimseq_make_length_t<dimseq_t<TTensorTypeIn>, math::max(tmp::vs::length_v<dimseq_t<TTensorTypeIn>>::value, tmp::vs::length_v<TOffsetCoordSeq>::value)>
      >
    >::value,
    "Offset must be strictly smaller than dimension");
  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  static const size_t MAX_RANK = math::max(non_trivial_dimensions_num_v<dimseq_t<TTensorTypeIn>>::value, non_trivial_coordinates_num_v<TOffsetCoordSeq>::value);

  __host__ __device__
  StaticOffsetTensor(TTensorTypeIn tensor)
    : SuperType(tensor.template dims<MAX_RANK>() - toCoordVector<MAX_RANK>(TOffsetCoordSeq()))
    , m_tensor(tensor)
  {
  }

  template <typename... TOffsetArgTypes>
  __host__ __device__
  StaticOffsetTensor(TTensorTypeIn tensor, TOffsetArgTypes&&... offset_args)
    : SuperType(tensor.template dims<MAX_RANK>() - toCoordVector<MAX_RANK>(TOffsetCoordSeq()))
    , m_tensor(tensor)
  {
    ASSERT(areCompatibleCoordinates<TOffsetCoordSeq>(util::forward<TOffsetArgTypes>(offset_args)...), "Incompatible run-time and compile-time offset");
  }

  TENSOR_ASSIGN(ThisType)

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    return m_tensor.template dim<TIndex>() - nth_coordinate_v<TIndex, TOffsetCoordSeq>::value;
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    return m_tensor.dim(index) - tmp::vs::getByIterating<TOffsetCoordSeq>(index, 0);
  }

private:
  TTensorTypeIn m_tensor;

public:
  HD_WARNING_DISABLE
  template <typename TThisType, size_t... TIndices, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, tmp::vs::Sequence<size_t, TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    util::forward<TThisType>(self).m_tensor((getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...) + nth_coordinate_v<TIndices, TOffsetCoordSeq>::value)...)
  )
  TENSOR_FORWARD_ELEMENT_ACCESS_SEQ_N(getElement, non_trivial_dimensions_num_v<dimseq_t<TTensorTypeIn>>::value)

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(StaticOffsetTensor<decltype(transform(m_tensor)), TOffsetCoordSeq>
    (transform(m_tensor))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(StaticOffsetTensor<decltype(transform(m_tensor)), TOffsetCoordSeq>
    (transform(m_tensor))
  )
};
#undef SuperType
#undef ThisType



#define ThisType DynamicOffsetTensor<TTensorTypeIn, TOffsetRank>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        dyn_dimseq_t<non_trivial_dimensions_num_v<TTensorTypeIn>::value> \
                              >

template <typename TTensorTypeIn, size_t TOffsetRank>
class DynamicOffsetTensor : public SuperType
{
public:
  static const size_t MAX_RANK = math::max(non_trivial_dimensions_num_v<SuperType>::value, TOffsetRank);

  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  template <typename... TOffsetArgTypes>
  __host__ __device__
  DynamicOffsetTensor(TTensorTypeIn tensor, TOffsetArgTypes&&... offset_args)
    : SuperType(tensor.template dims<MAX_RANK>() - toCoordVector<MAX_RANK>(util::forward<TOffsetArgTypes>(offset_args)...))
    , m_tensor(tensor)
    , m_offset(util::forward<TOffsetArgTypes>(offset_args)...)
  {
  }

  TENSOR_ASSIGN(ThisType)

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    return m_tensor.template dim<TIndex>() - getNthCoordinate<TIndex>(m_offset);
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    return m_tensor.dim(index) - (math::lt(index, TOffsetRank) ? m_offset(index) : 0);
  }

private:
  TTensorTypeIn m_tensor;
  VectorXs<TOffsetRank> m_offset;



public:
  HD_WARNING_DISABLE
  template <typename TThisType, size_t... TIndices, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, tmp::vs::Sequence<size_t, TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_tensor((getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...) + getNthCoordinate<TIndices>(self.m_offset))...)
  )
  TENSOR_FORWARD_ELEMENT_ACCESS_SEQ_N(getElement, non_trivial_dimensions_num_v<dimseq_t<TTensorTypeIn>>::value)

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(DynamicOffsetTensor<decltype(transform(m_tensor)), TOffsetRank>
    (transform(m_tensor), m_offset)
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(DynamicOffsetTensor<decltype(transform(m_tensor)), TOffsetRank>
    (transform(m_tensor), m_offset)
  )
};
#undef SuperType
#undef ThisType



template <typename TTensorTypeIn, typename TOffsetCoordSeq>
using OffsetTensor = typename std::conditional<is_static_v<TOffsetCoordSeq>::value,
  StaticOffsetTensor<TTensorTypeIn, TOffsetCoordSeq>,
  DynamicOffsetTensor<TTensorTypeIn, tmp::vs::length_v<TOffsetCoordSeq>::value>
>::type;









/*!
 * \defgroup HeadTail Sub-Tensors
 * \ingroup TensorOperations
 * \brief Sub-tensors of other tensors
 *
 * @{
 */

/*!
 * \brief Returns the sub-tensor at location (0 ... 0) with size (THeadDimSeq...)
 *
 * @param tensor the tensor that contains the sub-tensor
 * @tparam THeadDimSeq the new dimension sequence
 * @return the sub-tensor
 */
template <size_t... THeadDimSeq, typename TOtherTensorType>
__host__ __device__
auto head(TOtherTensorType&& tensor)
RETURN_AUTO(HeadTensor<util::store_member_t<TOtherTensorType&&>, DimSeq<THeadDimSeq...>>
  (util::forward<TOtherTensorType>(tensor))
);

/*!
 * \brief Returns the sub-tensor at location (0 ... 0) with size (dim_args...)
 *
 * @param tensor the tensor that contains the sub-tensor
 * @param dim_args... the new dimensions
 * @return the sub-tensor
 */
template <typename TOtherTensorType, typename... TDimArgTypes, ENABLE_IF(sizeof...(TDimArgTypes) != 0)>
__host__ __device__
auto head(TOtherTensorType&& tensor, TDimArgTypes&&... dim_args)
RETURN_AUTO(HeadTensor<util::store_member_t<TOtherTensorType&&>, dyn_dimseq_t<dimension_num_v<TDimArgTypes...>::value>>
  (util::forward<TOtherTensorType>(tensor), util::forward<TDimArgTypes>(dim_args)...)
);

template <typename THeadDimSeq, typename TOtherTensorType, typename... TDimArgTypes>
__host__ __device__
auto headEx(TOtherTensorType&& tensor, TDimArgTypes&&... dim_args)
RETURN_AUTO(HeadTensor<util::store_member_t<TOtherTensorType&&>, THeadDimSeq>
  (util::forward<TOtherTensorType>(tensor), util::forward<TDimArgTypes>(dim_args)...)
);

/*!
 * \brief Returns the sub-tensor at location (TOffsetCoordSeq...) ending at the original tensor's dimensions
 *
 * @param tensor the tensor that contains the sub-tensor
 * @tparam TOffsetCoordSeq the offset coordinate sequence
 * @return the sub-tensor
 */
template <size_t... TOffsetCoordSeq, ENABLE_IF(is_static_v<CoordSeq<TOffsetCoordSeq...>>::value), typename TOtherTensorType>
__host__ __device__
auto offset(TOtherTensorType&& tensor)
RETURN_AUTO(StaticOffsetTensor<util::store_member_t<TOtherTensorType&&>, CoordSeq<TOffsetCoordSeq...>>
  (util::forward<TOtherTensorType>(tensor))
);

/*!
 * \brief Returns the sub-tensor at location (offset_args...) ending at the original tensor's dimensions
 *
 * @param tensor the tensor that contains the sub-tensor
 * @param offset_args the offset
 * @return the sub-tensor
 */
template <typename TOtherTensorType, typename... TOffsetArgTypes, ENABLE_IF(sizeof...(TOffsetArgTypes) != 0)>
__host__ __device__
auto offset(TOtherTensorType&& tensor, TOffsetArgTypes&&... offset_args)
RETURN_AUTO(DynamicOffsetTensor<util::store_member_t<TOtherTensorType&&>, coordinate_num_v<TOffsetArgTypes...>::value>
  (util::forward<TOtherTensorType>(tensor), util::forward<TOffsetArgTypes>(offset_args)...)
);

template <typename TOffsetCoordSeq, typename TOtherTensorType, typename... TOffsetArgTypes>
__host__ __device__
auto offsetEx(TOtherTensorType&& tensor, TOffsetArgTypes&&... offset_args)
RETURN_AUTO(OffsetTensor<util::store_member_t<TOtherTensorType&&>, TOffsetCoordSeq>
  (util::forward<TOtherTensorType>(tensor), util::forward<TOffsetArgTypes>(offset_args)...)
);



template <typename TOtherTensorType, size_t... TTailDimSeq>
using TailReturnType1 = OffsetTensor<util::store_member_t<TOtherTensorType&&>,
    tmp::vs::map2_t<
      detail::OffsetDimHelper,
      dimseq_make_length_t<dimseq_t<TOtherTensorType>, math::max(tmp::vs::length_v<dimseq_t<TOtherTensorType>>::value, sizeof...(TTailDimSeq))>,
      dimseq_make_length_t<DimSeq<TTailDimSeq...>    , math::max(tmp::vs::length_v<dimseq_t<TOtherTensorType>>::value, sizeof...(TTailDimSeq))>
    >
  >;
/*!
 * \brief Returns the sub-tensor with size (TTailDimSeq...) ending at the original tensor's dimensions
 *
 * @param tensor the tensor that contains the sub-tensor
 * @tparam TTailDimSeq the new dimension sequence
 * @return the sub-tensor
 */
template <size_t... TTailDimSeq, typename TOtherTensorType, size_t TMaxRank = math::max(tmp::vs::length_v<dimseq_t<TOtherTensorType>>::value, sizeof...(TTailDimSeq)), ENABLE_IF(is_static_v<DimSeq<TTailDimSeq...>>::value)>
__host__ __device__
TailReturnType1<TOtherTensorType, TTailDimSeq...> tail(TOtherTensorType&& tensor)
{
  return TailReturnType1<TOtherTensorType, TTailDimSeq...>(
    util::forward<TOtherTensorType>(tensor),
    tensor.template dims<TMaxRank>() - toDimVector<TMaxRank>(DimSeq<TTailDimSeq...>()));
}

/*!
 * \brief Returns the sub-tensor with size (tail_args...) ending at the original tensor's dimensions
 *
 * @param tensor the tensor that contains the sub-tensor
 * @param tail_args the new dimensions
 * @return the sub-tensor
 */
template <typename TOtherTensorType, typename... TTailArgTypes,
  size_t TMaxRank = math::max(dimension_num_v<TTailArgTypes...>::value, non_trivial_dimensions_num_v<dimseq_t<TOtherTensorType>>::value)>
__host__ __device__
auto tail(TOtherTensorType&& tensor, TTailArgTypes&&... tail_args)
RETURN_AUTO(
  DynamicOffsetTensor<TOtherTensorType, TMaxRank>(util::forward<TOtherTensorType>(tensor),
    tensor.template dims<TMaxRank>() - toDimVector<TMaxRank>(util::forward<TTailArgTypes>(tail_args)...))
);



template <typename TMatrixType>
__host__ __device__
auto row(TMatrixType&& matrix, size_t row)
RETURN_AUTO(template_tensors::head<1, cols_v<TMatrixType>::value>(template_tensors::offset(util::forward<TMatrixType>(matrix), row)));

template <typename TMatrixType>
__host__ __device__
auto col(TMatrixType&& matrix, size_t col)
RETURN_AUTO(template_tensors::head<rows_v<TMatrixType>::value, 1>(template_tensors::offset(util::forward<TMatrixType>(matrix), 0, col)));

template <size_t TRow, typename TMatrixType>
__host__ __device__
auto row(TMatrixType&& matrix)
RETURN_AUTO(template_tensors::head<1, cols_v<TMatrixType>::value>(template_tensors::offset<TRow>(util::forward<TMatrixType>(matrix))));

template <size_t TCol, typename TMatrixType>
__host__ __device__
auto col(TMatrixType&& matrix)
RETURN_AUTO(template_tensors::head<rows_v<TMatrixType>::value, 1>(template_tensors::offset<0, TCol>(util::forward<TMatrixType>(matrix))));

/*!
 * @}
 */

} // end of ns tensor
