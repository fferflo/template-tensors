namespace template_tensors {

namespace detail {

template <metal::int_ I, bool TIsCurrent>
struct ReductionCoordHelper
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static dim_t get(dim_t new_coord, TCoordArgTypes&&... old_coords)
  {
    return new_coord;
  }
};

template <metal::int_ I>
struct ReductionCoordHelper<I, false>
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static dim_t get(dim_t new_coord, TCoordArgTypes&&... old_coords)
  {
    return getNthCoordinate<I>(std::forward<TCoordArgTypes>(old_coords)...);
  }
};

template <bool TIsReducedDimension, metal::int_ I>
struct ReductionHelper
{
  template <typename TReducedDimsAsSeq, typename TAggregator, typename TTensorTypeIn, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static void reduce(metal::numbers<TIndices...> seq, TAggregator& aggregator, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    ASSERT(getNthCoordinate<I - 1>(std::forward<TCoordArgTypes>(coords)...) == 0, "Reduced coordinate has to be zero");
    for (dim_t i = 0; i < tensor.template dim<I - 1>(); i++)
    {
      ReductionHelper<metal::contains<TReducedDimsAsSeq, metal::number<I - 2>>::value, I - 1>
        ::template reduce<TReducedDimsAsSeq>(seq, aggregator, std::forward<TTensorTypeIn>(tensor),
            ReductionCoordHelper<TIndices, TIndices == I - 1>::get(i, std::forward<TCoordArgTypes>(coords)...)...
          );
    }
  }
};

template <metal::int_ I>
struct ReductionHelper<false, I>
{
  template <typename TReducedDimsAsSeq, typename TAggregator, typename TTensorTypeIn, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static void reduce(metal::numbers<TIndices...> seq, TAggregator& aggregator, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    ReductionHelper<metal::contains<TReducedDimsAsSeq, metal::number<I - 2>>::value, I - 1>
        ::template reduce<TReducedDimsAsSeq>(seq, aggregator, std::forward<TTensorTypeIn>(tensor),
            std::forward<TCoordArgTypes>(coords)...
          );
  }
};

template <>
struct ReductionHelper<true, 0>
{
  template <typename TReducedDimsAsSeq, typename TAggregator, typename TTensorTypeIn, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static void reduce(metal::numbers<TIndices...> seq, TAggregator& aggregator, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    aggregator(tensor(std::forward<TCoordArgTypes>(coords)...));
  }
};

template <>
struct ReductionHelper<false, 0>
{
  template <typename TReducedDimsAsSeq, typename TAggregator, typename TTensorTypeIn, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static void reduce(metal::numbers<TIndices...> seq, TAggregator& aggregator, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    aggregator(tensor(std::forward<TCoordArgTypes>(coords)...));
  }
};



template <metal::int_ I, typename TOriginalDimSeq, typename TReducedDimsAsSeq, bool TIsReducedDim = metal::contains<TReducedDimsAsSeq, metal::number<I>>::value>
struct StaticReducedDimHelper
{
  static const metal::int_ value = nth_dimension_v<I, TOriginalDimSeq>::value;
};

template <metal::int_ I, typename TOriginalDimSeq, typename TReducedDimsAsSeq>
struct StaticReducedDimHelper<I, TOriginalDimSeq, TReducedDimsAsSeq, true>
{
  static const metal::int_ value = 1;
};



template <typename TOriginalDimSeq, typename TReducedDimsAsSeq, typename TIndexSeq>
struct ReducedDimsHelper;

template <typename TOriginalDimSeq, typename TReducedDimsAsSeq, metal::int_... TIndices>
struct ReducedDimsHelper<TOriginalDimSeq, TReducedDimsAsSeq, metal::numbers<TIndices...>>
{
  using type = DimSeq<StaticReducedDimHelper<TIndices, TOriginalDimSeq, TReducedDimsAsSeq>::value...>;
};

template <typename TOriginalDimSeq, typename TReducedDimsAsSeq>
using ReducedDimSeq = typename ReducedDimsHelper<TOriginalDimSeq, TReducedDimsAsSeq,
                                  metal::iota<metal::number<0>, metal::number<non_trivial_dimensions_num_v<TOriginalDimSeq>::value>>>::type;

static_assert(std::is_same<ReducedDimSeq<DimSeq<2, 3, 4>, metal::numbers<1, 5>>, DimSeq<2, 1, 4>>::value, "ReducedDimSeq not working");



template <metal::int_ I, typename TOriginalDimSeq, typename TReducedDimsAsSeq, bool TIsReducedDim = metal::contains<TReducedDimsAsSeq, metal::number<I>>::value>
struct DynamicReducedDimHelper
{
  template <typename TOtherTensor>
  __host__ __device__
  static dim_t get(const TOtherTensor& tensor)
  {
    return tensor.template dim<I>();
  }
};

template <metal::int_ I, typename TOriginalDimSeq, typename TReducedDimsAsSeq>
struct DynamicReducedDimHelper<I, TOriginalDimSeq, TReducedDimsAsSeq, true>
{
  template <typename TOtherTensor>
  __host__ __device__
  static dim_t get(const TOtherTensor& tensor)
  {
    return 1;
  }
};



template <typename TReducedDimsAsSeq>
struct IsReducedDim;

template <metal::int_ TDim0>
struct IsReducedDim<DimSeq<TDim0>>
{
  static bool is(dim_t dim)
  {
    return dim == TDim0;
  }
};

template <metal::int_ TDim0, metal::int_ TDim1, metal::int_... TDimsRest>
struct IsReducedDim<DimSeq<TDim0, TDim1, TDimsRest...>>
{
  static bool is(dim_t dim)
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
  static const metal::int_ ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<TTensorTypeIn>::value;
  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  __host__ __device__
  ReductionTensor(TTensorTypeIn tensor, TAggregator aggregator)
    : SuperType(dims_helper(metal::iota<metal::number<0>, metal::number<NON_TRIVIAL_DIMENSIONS_NUM>>(), tensor))
    , m_tensor(tensor)
    , m_aggregator(aggregator)
  {
  }

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static aggregator::resulttype_t<TAggregator> getElement(TThisType&& self, TCoordArgTypes&&... coords)
  {
    TAggregator aggregator = self.m_aggregator;
    detail::ReductionHelper<metal::contains<
                                            TReducedDimsAsSeq,
                                            metal::number<ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM - 1>
                                          >::value,
                    ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM>
        ::template reduce<TReducedDimsAsSeq>(
            metal::iota<metal::number<0>, metal::number<ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM>>(),
            aggregator,
            self.m_tensor,
            std::forward<TCoordArgTypes>(coords)...
          );
    return aggregator.get();
  }
  TT_TENSOR_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return detail::DynamicReducedDimHelper<
                      TIndex,
                      dimseq_t<TTensorTypeIn>,
                      TReducedDimsAsSeq
                >::get(m_tensor);
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return detail::IsReducedDim<TReducedDimsAsSeq>::is(index) ? 1 : m_tensor.dim(index);
  }

private:
  TTensorTypeIn m_tensor;
  TAggregator m_aggregator;

  template <metal::int_... TIndices, typename TTensorType>
  __host__ __device__
  auto dims_helper(metal::numbers<TIndices...>, const TTensorType& tensor)
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
              TAggregator,
              TTensorType,
              metal::iota<metal::number<0>, metal::number<non_trivial_dimensions_num_v<dimseq_t<TTensorType>>::value>>
            >(std::forward<TTensorType>(tensor), std::forward<TAggregator>(aggregator))
);

/*!
 * \brief Reduces the given dimensions of a tensor using the given binary functor
 *
 * @param tensor the input tensor
 * @tparam TElementType the desired element type of the resulting tensor
 * @tparam TFunctor the binary accumulation operation
 * @tparam TReducedDims... a list of dimensions that will be reduced to 1
 */
template <metal::int_... TReducedDims, typename TTensorType, typename TAggregator>
__host__ __device__
auto reduce(TTensorType&& tensor, TAggregator&& aggregator)
RETURN_AUTO(
  ReductionTensor<
              TAggregator,
              TTensorType,
              metal::numbers<TReducedDims...>
            >(std::forward<TTensorType>(tensor), std::forward<TAggregator>(aggregator))
);
// TODO: eval the elementtype of a tensor that is reduced if that elementtype is a tensor
/*!
 * \brief Returns the sum of all elements of the given tensor
 *
 * @param tensor the input tensor
 * @return the sum of all elements of the given tensor
 */
template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType,
  typename TElementType = TT_WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto sum(TTensorType&& tensor, TElementType initial_value = 0)
RETURN_AUTO(reduceAll(
  std::forward<TTensorType>(tensor),
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
  typename TElementType = TT_WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto prod(TTensorType&& tensor, TElementType initial_value = 1)
RETURN_AUTO(reduceAll(
  std::forward<TTensorType>(tensor),
  aggregator::prod(initial_value)
)());
FUNCTOR(prod, template_tensors::prod)

template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType,
  typename TElementType = TT_WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto mean(TTensorType&& tensor)
RETURN_AUTO(static_cast<TElementType>(template_tensors::sum(tensor)) / template_tensors::prod(tensor.dims()));
FUNCTOR(mean, template_tensors::mean)

template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType,
  typename TElementType = TT_WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto min_el(TTensorType&& tensor)
RETURN_AUTO(reduceAll(
  std::forward<TTensorType>(tensor),
  aggregator::min(tensor())
)());
FUNCTOR(min_el, template_tensors::min_el)

template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType,
  typename TElementType = TT_WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__ __device__
auto max_el(TTensorType&& tensor)
RETURN_AUTO(reduceAll(
  std::forward<TTensorType>(tensor),
  aggregator::max(tensor())
)());
FUNCTOR(max_el, template_tensors::max_el)

template <typename TTensorType>
__host__ __device__
auto all(TTensorType&& tensor)
RETURN_AUTO(reduceAll(
  std::forward<TTensorType>(tensor),
  aggregator::all()
)());

template <typename TTensorType>
__host__ __device__
auto any(TTensorType&& tensor)
RETURN_AUTO(reduceAll(
  std::forward<TTensorType>(tensor),
  aggregator::any()
)());

namespace detail {

struct BoolTo01
{
  __host__ __device__
  dim_t operator()(bool el) const
  {
    return el ? 1 : 0;
  }
};

} // end of ns detail

template <typename TTensorType>
__host__ __device__
dim_t count(TTensorType&& t)
{
  return sum<dim_t>(elwise(detail::BoolTo01(), std::forward<TTensorType>(t)));
}

/*!
 * @}
 */

} // end of ns template_tensors
