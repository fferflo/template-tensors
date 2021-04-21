namespace template_tensors {

namespace detail {

template <size_t I, bool TIsCurrent>
struct ReductionCoordHelper
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static size_t get(size_t new_coord, TCoordArgTypes&&... old_coords)
  {
    return new_coord;
  }
};

template <size_t I>
struct ReductionCoordHelper<I, false>
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static size_t get(size_t new_coord, TCoordArgTypes&&... old_coords)
  {
    return getNthCoordinate<I>(util::forward<TCoordArgTypes>(old_coords)...);
  }
};

template <bool TIsReducedDimension, size_t I>
struct ReductionHelper
{
  template <typename TReducedDimsAsSeq, typename TAggregator, typename TTensorTypeIn, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static void reduce(tmp::vs::Sequence<size_t, TIndices...> seq, TAggregator& aggregator, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    ASSERT(getNthCoordinate<I - 1>(util::forward<TCoordArgTypes>(coords)...) == 0, "Reduced coordinate has to be zero");
    for (size_t i = 0; i < tensor.template dim<I - 1>(); i++)
    {
      ReductionHelper<tmp::vs::contains_v<size_t, TReducedDimsAsSeq, I - 2>::value, I - 1>
        ::template reduce<TReducedDimsAsSeq>(seq, aggregator, util::forward<TTensorTypeIn>(tensor),
            ReductionCoordHelper<TIndices, TIndices == I - 1>::get(i, util::forward<TCoordArgTypes>(coords)...)...
          );
    }
  }
};

template <size_t I>
struct ReductionHelper<false, I>
{
  template <typename TReducedDimsAsSeq, typename TAggregator, typename TTensorTypeIn, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static void reduce(tmp::vs::Sequence<size_t, TIndices...> seq, TAggregator& aggregator, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    ReductionHelper<tmp::vs::contains_v<size_t, TReducedDimsAsSeq, I - 2>::value, I - 1>
        ::template reduce<TReducedDimsAsSeq>(seq, aggregator, util::forward<TTensorTypeIn>(tensor),
            util::forward<TCoordArgTypes>(coords)...
          );
  }
};

template <>
struct ReductionHelper<true, 0>
{
  template <typename TReducedDimsAsSeq, typename TAggregator, typename TTensorTypeIn, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static void reduce(tmp::vs::Sequence<size_t, TIndices...> seq, TAggregator& aggregator, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    aggregator(tensor(util::forward<TCoordArgTypes>(coords)...));
  }
};

template <>
struct ReductionHelper<false, 0>
{
  template <typename TReducedDimsAsSeq, typename TAggregator, typename TTensorTypeIn, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static void reduce(tmp::vs::Sequence<size_t, TIndices...> seq, TAggregator& aggregator, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    aggregator(tensor(util::forward<TCoordArgTypes>(coords)...));
  }
};



template <size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq, bool TIsReducedDim = tmp::vs::contains_v<size_t, TReducedDimsAsSeq, I>::value>
struct StaticReducedDimHelper
{
  static const size_t value = nth_dimension_v<I, TOriginalDimSeq>::value;
};

template <size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq>
struct StaticReducedDimHelper<I, TOriginalDimSeq, TReducedDimsAsSeq, true>
{
  static const size_t value = 1;
};



template <typename TOriginalDimSeq, typename TReducedDimsAsSeq, typename TIndexSeq>
struct ReducedDimsHelper;

template <typename TOriginalDimSeq, typename TReducedDimsAsSeq, size_t... TIndices>
struct ReducedDimsHelper<TOriginalDimSeq, TReducedDimsAsSeq, tmp::vs::Sequence<size_t, TIndices...>>
{
  using type = DimSeq<StaticReducedDimHelper<TIndices, TOriginalDimSeq, TReducedDimsAsSeq>::value...>;
};

template <typename TOriginalDimSeq, typename TReducedDimsAsSeq>
using ReducedDimSeq = typename ReducedDimsHelper<TOriginalDimSeq, TReducedDimsAsSeq,
                                  tmp::vs::ascending_numbers_t<non_trivial_dimensions_num_v<TOriginalDimSeq>::value>>::type;

static_assert(std::is_same<ReducedDimSeq<DimSeq<2, 3, 4>, tmp::vs::Sequence<size_t, 1, 5>>, DimSeq<2, 1, 4>>::value, "ReducedDimSeq not working");



template <size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq, bool TIsReducedDim = tmp::vs::contains_v<size_t, TReducedDimsAsSeq, I>::value>
struct DynamicReducedDimHelper
{
  template <typename TOtherTensor>
  __host__ __device__
  static size_t get(const TOtherTensor& tensor)
  {
    return tensor.template dim<I>();
  }
};

template <size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq>
struct DynamicReducedDimHelper<I, TOriginalDimSeq, TReducedDimsAsSeq, true>
{
  template <typename TOtherTensor>
  __host__ __device__
  static size_t get(const TOtherTensor& tensor)
  {
    return 1;
  }
};



template <typename TReducedDimsAsSeq>
struct IsReducedDim;

template <size_t TDim0>
struct IsReducedDim<DimSeq<TDim0>>
{
  static bool is(size_t dim)
  {
    return dim == TDim0;
  }
};

template <size_t TDim0, size_t TDim1, size_t... TDimsRest>
struct IsReducedDim<DimSeq<TDim0, TDim1, TDimsRest...>>
{
  static bool is(size_t dim)
  {
    return dim == TDim0 || IsReducedDim<DimSeq<TDim1, TDimsRest...>>::is(dim);
  }
};

} // end of ns detail



#define ThisType ReductionTensor<TAggregator, TTensorTypeIn, TReducedDimsAsSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        detail::ReducedDimSeq<dimseq_t<TTensorTypeIn>, TReducedDimsAsSeq> \
                              >
template <typename TAggregator, typename TTensorTypeIn, typename TReducedDimsAsSeq>
class ReductionTensor : public SuperType
{
public:
  static const size_t ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<TTensorTypeIn>::value;
  static const size_t NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  __host__ __device__
  ReductionTensor(TTensorTypeIn tensor, TAggregator aggregator)
    : SuperType(dims_helper(tmp::vs::ascending_numbers_t<NON_TRIVIAL_DIMENSIONS_NUM>(), tensor))
    , m_tensor(tensor)
    , m_aggregator(aggregator)
  {
  }

  TENSOR_ASSIGN(ThisType)

  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static aggregator::resulttype_t<TAggregator> getElement(TThisType&& self, TCoordArgTypes&&... coords)
  {
    TAggregator aggregator = self.m_aggregator;
    detail::ReductionHelper<tmp::vs::contains_v<
                                                    size_t,
                                                    TReducedDimsAsSeq,
                                                    ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM - 1
                                                  >::value,
                    ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM>
        ::template reduce<TReducedDimsAsSeq>(
            tmp::vs::ascending_numbers_t<ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM>(),
            aggregator,
            self.m_tensor,
            util::forward<TCoordArgTypes>(coords)...
          );
    return aggregator.get();
  }
  TENSOR_FORWARD_ELEMENT_ACCESS(getElement)

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    return detail::DynamicReducedDimHelper<
                      TIndex,
                      dimseq_t<TTensorTypeIn>,
                      TReducedDimsAsSeq
                >::get(m_tensor);
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    return detail::IsReducedDim<TReducedDimsAsSeq>::is(index) ? 1 : m_tensor.dim(index);
  }

private:
  TTensorTypeIn m_tensor;
  TAggregator m_aggregator;

  template <size_t... TIndices, typename TTensorType>
  __host__ __device__
  auto dims_helper(tmp::vs::Sequence<size_t, TIndices...>, const TTensorType& tensor)
  RETURN_AUTO(VectorXs<sizeof...(TIndices)>(
      detail::DynamicReducedDimHelper<
                      TIndices,
                      dimseq_t<TTensorType>,
                      TReducedDimsAsSeq
                >::get(tensor)...
    ))

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(ReductionTensor<decltype(transform(m_aggregator)), decltype(transform(m_tensor)), TReducedDimsAsSeq>
    (transform(m_tensor), transform(m_aggregator))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(ReductionTensor<decltype(transform(m_aggregator)), decltype(transform(m_tensor)), TReducedDimsAsSeq>
    (transform(m_tensor), transform(m_aggregator))
  )
};
#undef SuperType
#undef ThisType


/*!
 * \defgroup Reduction Reduction
 * \ingroup TensorOperations
 * \brief Reduce dimensions by accumulating elements along those dimensions using a given operation.
 *
 * Reduced dimensions of the output tensor are equal to 1. Example: A matrix that is reduced in the second dimension using the sum-operation results
 * in a column vector containing the sums of the matrices rows.
 * @{
 */

template <typename TAggregator, typename TTensorType>
__host__ __device__
auto reduceAll(TTensorType&& tensor, TAggregator&& aggregator)
RETURN_AUTO(
  ReductionTensor<
              util::store_member_t<TAggregator&&>,
              util::store_member_t<TTensorType&&>,
              tmp::vs::ascending_numbers_t<non_trivial_dimensions_num_v<dimseq_t<TTensorType>>::value>
            >(util::forward<TTensorType>(tensor), util::forward<TAggregator>(aggregator))
);

/*!
 * \brief Reduces the given dimensions of a tensor using the given binary functor
 *
 * @param tensor the input tensor
 * @tparam TElementType the desired element type of the resulting tensor
 * @tparam TFunctor the binary accumulation operation
 * @tparam TReducedDims... a list of dimensions that will be reduced to 1
 */
template <size_t... TReducedDims, typename TTensorType, typename TAggregator>
__host__ __device__
auto reduce(TTensorType&& tensor, TAggregator&& aggregator)
RETURN_AUTO(
  ReductionTensor<
              util::store_member_t<TAggregator&&>,
              util::store_member_t<TTensorType&&>,
              tmp::vs::Sequence<size_t, TReducedDims...>
            >(util::forward<TTensorType>(tensor), util::forward<TAggregator>(aggregator))
);
// TODO: eval the elementtype of a tensor that is reduced if that elementtype is a tensor
/*!
 * \brief Returns the sum of all elements of the given tensor
 *
 * @param tensor the input tensor
 * @return the sum of all elements of the given tensor
 */
template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType,
  typename TElementType = WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto sum(TTensorType&& tensor, TElementType initial_value = 0)
RETURN_AUTO(reduceAll(
  util::forward<TTensorType>(tensor),
  aggregator::sum(initial_value)
)());
FUNCTOR(sum, template_tensors::sum)

/*!
 * \brief Returns the product of all elements of the given tensor
 *
 * @param tensor the input tensor
 * @return the product of all elements of the given tensor
 */
template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType,
  typename TElementType = WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto prod(TTensorType&& tensor, TElementType initial_value = 1)
RETURN_AUTO(reduceAll(
  util::forward<TTensorType>(tensor),
  aggregator::prod(initial_value)
)());
FUNCTOR(prod, template_tensors::prod)

template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType,
  typename TElementType = WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto mean(TTensorType&& tensor)
RETURN_AUTO(static_cast<TElementType>(template_tensors::sum(tensor)) / template_tensors::prod(tensor.dims()));
FUNCTOR(mean, template_tensors::mean)

template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType,
  typename TElementType = WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto min_el(TTensorType&& tensor)
RETURN_AUTO(reduceAll(
  util::forward<TTensorType>(tensor),
  aggregator::min(tensor())
)());
FUNCTOR(min_el, template_tensors::min_el)

template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType,
  typename TElementType = WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto max_el(TTensorType&& tensor)
RETURN_AUTO(reduceAll(
  util::forward<TTensorType>(tensor),
  aggregator::max(tensor())
)());
FUNCTOR(max_el, template_tensors::max_el)

template <typename TTensorType>
__host__ __device__
auto all(TTensorType&& tensor)
RETURN_AUTO(reduceAll(
  util::forward<TTensorType>(tensor),
  aggregator::all()
)());

template <typename TTensorType>
__host__ __device__
auto any(TTensorType&& tensor)
RETURN_AUTO(reduceAll(
  util::forward<TTensorType>(tensor),
  aggregator::any()
)());

namespace detail {

struct BoolTo01
{
  __host__ __device__
  size_t operator()(bool el) const
  {
    return el ? 1 : 0;
  }
};

} // end of ns detail

template <typename TTensorType>
__host__ __device__
size_t count(TTensorType&& t)
{
  return sum<size_t>(elwise(detail::BoolTo01(), util::forward<TTensorType>(t)));
}

/*!
 * @}
 */

} // end of ns tensor
