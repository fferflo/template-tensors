namespace template_tensors {

namespace detail {

template <metal::int_ N, typename TDimSeq1, typename TDimSeq2, metal::int_ TConcatDim>
struct NthConcatDim
{
  static_assert(N != TConcatDim, "This should never happen");
  static_assert(nth_dimension_v<N, TDimSeq1>::value == DYN || nth_dimension_v<N, TDimSeq2>::value == DYN
    || nth_dimension_v<N, TDimSeq1>::value == nth_dimension_v<N, TDimSeq2>::value, "Incompatible concat dimensions");

  static const metal::int_ value = nth_dimension_v<N, TDimSeq1>::value == DYN ?
    nth_dimension_v<N, TDimSeq2>::value : nth_dimension_v<N, TDimSeq1>::value;
};

template <typename TDimSeq1, typename TDimSeq2, metal::int_ TConcatDim>
struct NthConcatDim<TConcatDim, TDimSeq1, TDimSeq2, TConcatDim>
{
  static const metal::int_ value = (nth_dimension_v<TConcatDim, TDimSeq1>::value == DYN || nth_dimension_v<TConcatDim, TDimSeq2>::value == DYN) ?
      DYN : (nth_dimension_v<TConcatDim, TDimSeq1>::value + nth_dimension_v<TConcatDim, TDimSeq2>::value);
};

template <typename TDimSeq1, typename TDimSeq2, metal::int_ TConcatDim, typename TIndexSeq>
struct ConcatDimsHelper;

template <typename TDimSeq1, typename TDimSeq2, metal::int_ TConcatDim, metal::int_... TIndices>
struct ConcatDimsHelper<TDimSeq1, TDimSeq2, TConcatDim, metal::numbers<TIndices...>>
{
  using type = DimSeq<NthConcatDim<TIndices, TDimSeq1, TDimSeq2, TConcatDim>::value...>;
};

template <typename TDimSeq1, typename TDimSeq2, metal::int_ TConcatDim>
using ConcatDimSeq = typename ConcatDimsHelper<TDimSeq1, TDimSeq2, TConcatDim,
                                  metal::iota<
                                    metal::number<0>, metal::number<math::max(non_trivial_dimensions_num_v<TDimSeq1>::value, non_trivial_dimensions_num_v<TDimSeq2>::value, TConcatDim + 1)>
                                  >
                              >::type;

static_assert(std::is_same<ConcatDimSeq<DimSeq<2, 3, 4>, DimSeq<2, 3, 4>, 1>, DimSeq<2, 6, 4>>::value, "ConcatDimSeq not working");
static_assert(std::is_same<ConcatDimSeq<DimSeq<2, 3, 4, 1>, DimSeq<2, 3, 4, 1>, 3>, DimSeq<2, 3, 4, 2>>::value, "ConcatDimSeq not working");



template <metal::int_ I, typename TDimSeq1, typename TDimSeq2, metal::int_ TConcatDim>
struct DynamicConcatDimHelper
{
  template <typename TTensorType1, typename TTensorType2>
  __host__ __device__
  static dim_t get(const TTensorType1& tensor1, const TTensorType2& tensor2)
  {
    return tensor1.template dim<I>();
  }
};

template <typename TDimSeq1, typename TDimSeq2, metal::int_ TConcatDim>
struct DynamicConcatDimHelper<TConcatDim, TDimSeq1, TDimSeq2, TConcatDim>
{
  template <typename TTensorType1, typename TTensorType2>
  __host__ __device__
  static dim_t get(const TTensorType1& tensor1, const TTensorType2& tensor2)
  {
    return tensor1.template dim<TConcatDim>() + tensor2.template dim<TConcatDim>();
  }
};



template <metal::int_ I, metal::int_ TConcatDim>
struct ConcatDimsForTensor2
{
  template <typename TTensorType1, typename... TCoordArgTypes>
  __host__ __device__
  static dim_t getCoord(TTensorType1& tensor1, TCoordArgTypes&&... coords)
  {
    return getNthCoordinate<I>(util::forward<TCoordArgTypes>(coords)...);
  }
};

template <metal::int_ TConcatDim>
struct ConcatDimsForTensor2<TConcatDim, TConcatDim>
{
  template <typename TTensorType1, typename... TCoordArgTypes>
  __host__ __device__
  static dim_t getCoord(TTensorType1& tensor1, TCoordArgTypes&&... coords)
  {
    return getNthCoordinate<TConcatDim>(util::forward<TCoordArgTypes>(coords)...) - tensor1.template dim<TConcatDim>();
  }
};

} // end of ns detail



#define ThisType ConcatTensor<TTensorTypeIn1, TTensorTypeIn2, TConcatDim>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::combine<mem::memorytype_v<TTensorTypeIn1>::value, mem::memorytype_v<TTensorTypeIn2>::value>(), \
                                        detail::ConcatDimSeq<dimseq_t<TTensorTypeIn1>, dimseq_t<TTensorTypeIn2>, TConcatDim> \
                              >

template <typename TTensorTypeIn1, typename TTensorTypeIn2, metal::int_ TConcatDim>
class ConcatTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeIn1>::value && is_tensor_v<TTensorTypeIn2>::value, "TTensorTypeIn1 and TTensorTypeIn2 must be tensors");

  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  ConcatTensor(TTensorTypeIn1 tensor1, TTensorTypeIn2 tensor2)
    : SuperType(dims_helper(metal::iota<metal::number<0>, metal::number<NON_TRIVIAL_DIMENSIONS_NUM>>(), tensor1, tensor2))
    , m_tensor1(tensor1)
    , m_tensor2(tensor2)
  {
#ifdef DEBUG
    for (auto i = 0; i < NON_TRIVIAL_DIMENSIONS_NUM; i++)
    {
      ASSERT(i == TConcatDim || m_tensor1.dim(i) == m_tensor2.dim(i), "Incompatible dimensions");
    }
#endif
  }

  template <typename TDummy = uint32_t, // stupid hack to ensure this is only available with static tensors
    ENABLE_IF(is_static_v<TTensorTypeIn1>::value && is_static_v<TTensorTypeIn2>::value && std::is_integral<TDummy>::value)>
  __host__ __device__
  ConcatTensor(TDummy dummy = 1)
    : SuperType(toDimVector(dimseq_t<SuperType>())) // TODO: is toDimVector needed here? grep for toDimVector usages elsewhere
    , m_tensor1()
    , m_tensor2()
  {
#ifdef DEBUG
    for (auto i = 0; i < NON_TRIVIAL_DIMENSIONS_NUM; i++)
    {
      ASSERT(i == TConcatDim || m_tensor1.dim(i) == m_tensor2.dim(i), "Incompatible dimensions");
    }
#endif
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static auto getElement(TThisType&& self, metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
    -> common_elementtype_t<decltype(self.m_tensor1), decltype(self.m_tensor2)>
  {
    if (getNthCoordinate<TConcatDim>(util::forward<TCoordArgTypes>(coords)...) < self.m_tensor1.template dim<TConcatDim>())
    {
      return self.m_tensor1(util::forward<TCoordArgTypes>(coords)...);
    }
    else
    {
      return self.m_tensor2(detail::ConcatDimsForTensor2<TIndices, TConcatDim>::getCoord(self.m_tensor1, util::forward<TCoordArgTypes>(coords)...)...);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ(getElement)
  // TODO: self.m_tensorX should possibly be an rvalue here? Also everywhere else

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return detail::DynamicConcatDimHelper<
                      TIndex,
                      dimseq_t<TTensorTypeIn1>,
                      dimseq_t<TTensorTypeIn2>,
                      TConcatDim
                >::get(m_tensor1, m_tensor2);
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index == TConcatDim ? m_tensor1.dim(index) + m_tensor2.dim(index) : m_tensor1.dim(index);
  }

private: // TODO: why are there extra bytes here? size of ConcatTensor does not equal sum of size of members -> try with clang?
  TTensorTypeIn1 m_tensor1;
  TTensorTypeIn2 m_tensor2;

  template <metal::int_... TIndices, typename TTensorType1, typename TTensorType2>
  __host__ __device__
  auto dims_helper(metal::numbers<TIndices...>, const TTensorType1& tensor1, const TTensorType2& tensor2)
  RETURN_AUTO(VectorXs<sizeof...(TIndices)>(
      detail::DynamicConcatDimHelper<
                      TIndices,
                      dimseq_t<TTensorType1>,
                      dimseq_t<TTensorType2>,
                      TConcatDim
                >::get(tensor1, tensor2)...
    ))

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(ConcatTensor<decltype(transform(m_tensor1)), decltype(transform(m_tensor2)), TConcatDim>
    (transform(m_tensor1), transform(m_tensor2))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(ConcatTensor<decltype(transform(m_tensor1)), decltype(transform(m_tensor2)), TConcatDim>
    (transform(m_tensor1), transform(m_tensor2))
  )
};
#undef SuperType
#undef ThisType


namespace detail {

template <metal::int_ TConcatDim, typename TTensorTypeIn,
  ENABLE_IF(is_tensor_v<TTensorTypeIn>::value)>
__host__ __device__
auto concat(TTensorTypeIn&& tensor)
RETURN_AUTO(util::forward_lvalue<TTensorTypeIn>(tensor))

template <metal::int_ TConcatDim, typename TNonTensorTypeIn,
  ENABLE_IF(!is_tensor_v<TNonTensorTypeIn>::value)>
__host__ __device__
auto concat(TNonTensorTypeIn&& tensor)
RETURN_AUTO(template_tensors::singleton(util::forward<TNonTensorTypeIn>(tensor)))

template <metal::int_ TConcatDim, typename TTensorTypeIn1, typename TTensorTypeIn2,
  ENABLE_IF(is_tensor_v<TTensorTypeIn1>::value && is_tensor_v<TTensorTypeIn2>::value)>
__host__ __device__
auto concat(TTensorTypeIn1&& tensor1, TTensorTypeIn2&& tensor2)
RETURN_AUTO(
  ConcatTensor<
                util::store_member_t<TTensorTypeIn1&&>,
                util::store_member_t<TTensorTypeIn2&&>,
                TConcatDim
              >(util::forward<TTensorTypeIn1>(tensor1),
                util::forward<TTensorTypeIn2>(tensor2))
)

template <metal::int_ TConcatDim, typename TNonTensorTypeIn1, typename TTensorTypeIn2,
  ENABLE_IF(!is_tensor_v<TNonTensorTypeIn1>::value && is_tensor_v<TTensorTypeIn2>::value)>
__host__ __device__
auto concat(TNonTensorTypeIn1&& tensor1, TTensorTypeIn2&& tensor2)
RETURN_AUTO(
  detail::concat<TConcatDim>(template_tensors::singleton(util::forward<TNonTensorTypeIn1>(tensor1)), util::forward<TTensorTypeIn2>(tensor2))
)

template <metal::int_ TConcatDim, typename TTensorTypeIn1, typename TNonTensorTypeIn2,
  ENABLE_IF(is_tensor_v<TTensorTypeIn1>::value && !is_tensor_v<TNonTensorTypeIn2>::value)>
__host__ __device__
auto concat(TTensorTypeIn1&& tensor1, TNonTensorTypeIn2&& tensor2)
RETURN_AUTO(
  detail::concat<TConcatDim>(util::forward<TTensorTypeIn1>(tensor1), template_tensors::singleton(util::forward<TNonTensorTypeIn2>(tensor2)))
)

template <metal::int_ TConcatDim, typename TNonTensorTypeIn1, typename TNonTensorTypeIn2,
  ENABLE_IF(!is_tensor_v<TNonTensorTypeIn1>::value && !is_tensor_v<TNonTensorTypeIn2>::value)>
__host__ __device__
auto concat(TNonTensorTypeIn1&& tensor1, TNonTensorTypeIn2&& tensor2)
RETURN_AUTO(
  detail::concat<TConcatDim>(template_tensors::singleton(util::forward<TNonTensorTypeIn1>(tensor1)), template_tensors::singleton(util::forward<TNonTensorTypeIn2>(tensor2)))
)

template <metal::int_ TConcatDim, typename TTensorTypeIn1, typename TTensorTypeIn2, typename TTensorTypeIn3, typename... TTensorTypeInRest>
__host__ __device__
auto concat(TTensorTypeIn1&& tensor1, TTensorTypeIn2&& tensor2, TTensorTypeIn3&& tensor3, TTensorTypeInRest&&... rest)
RETURN_AUTO(
  detail::concat<TConcatDim>(
    detail::concat<TConcatDim>(util::forward<TTensorTypeIn1>(tensor1), util::forward<TTensorTypeIn2>(tensor2)),
    util::forward<TTensorTypeIn3>(tensor3),
    util::forward<TTensorTypeInRest>(rest)...
  )
)

} // end of ns detail

/*!
 * \defgroup ConcatTensor Concatenation
 * \ingroup TensorOperations
 * \brief Concatenate tensors in a given direction
 *
 * @{
 */

/*!
 * \brief Concatenates the given tensors in the given direction
 *
 * @param tensors... the tensors to be concatenated
 * @tparam TConcatDim the dimension in which the tensors will be concatenated
 * @return the concatenated tensor
 */
template <metal::int_ TConcatDim, typename... TTensorTypesIn>
__host__ __device__
auto concat(TTensorTypesIn&&... tensors)
RETURN_AUTO(
  detail::concat<TConcatDim>(util::forward<TTensorTypesIn>(tensors)...)
)

/*!
 * @}
 */

} // end of ns tensor
