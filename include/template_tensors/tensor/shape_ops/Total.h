namespace template_tensors {

namespace detail {

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, size_t TKeepSeqOffset, size_t I, size_t N>
struct NthTotalDim2
{
  static const size_t current_keep_index = tmp::vs::get_optional_v<TKeepSeqOffset, TKeepSeq, (size_t) -1>::value;
  static const bool keep = current_keep_index == I;
  static const size_t value = NthTotalDim2<TKeepDimSeq, TDropDimSeq, TKeepSeq, keep ? TKeepSeqOffset + 1 : TKeepSeqOffset, I + 1, N>::value;
};

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, size_t TKeepSeqOffset, size_t N>
struct NthTotalDim2<TKeepDimSeq, TDropDimSeq, TKeepSeq, TKeepSeqOffset, N, N>
{
  static const size_t drop_seq_offset = N - TKeepSeqOffset;
  static const size_t current_keep_index = tmp::vs::get_optional_v<TKeepSeqOffset, TKeepSeq, (size_t) -1>::value;
  static const bool keep = current_keep_index == N;

  static const size_t value = keep ? nth_dimension_v<TKeepSeqOffset, TKeepDimSeq>::value : nth_dimension_v<drop_seq_offset, TDropDimSeq>::value;
};

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, size_t N>
struct nth_total_dim_v
{
  static const size_t value = NthTotalDim2<TKeepDimSeq, TDropDimSeq, TKeepSeq, 0, 0, N>::value;
};

static_assert(nth_total_dim_v<DimSeq<2>, DimSeq<3>, tmp::vs::IndexSequence<0>, 0>::value == 2, "nth_total_dim_v not working");
static_assert(nth_total_dim_v<DimSeq<2>, DimSeq<3>, tmp::vs::IndexSequence<0>, 1>::value == 3, "nth_total_dim_v not working");
static_assert(nth_total_dim_v<DimSeq<2, 3>, DimSeq<4, 5>, tmp::vs::IndexSequence<1, 3>, 0>::value == 4, "nth_total_dim_v not working");
static_assert(nth_total_dim_v<DimSeq<2, 3>, DimSeq<4, 5>, tmp::vs::IndexSequence<1, 3>, 1>::value == 2, "nth_total_dim_v not working");
static_assert(nth_total_dim_v<DimSeq<2, 3>, DimSeq<4, 5>, tmp::vs::IndexSequence<1, 3>, 2>::value == 5, "nth_total_dim_v not working");
static_assert(nth_total_dim_v<DimSeq<2, 3>, DimSeq<4, 5>, tmp::vs::IndexSequence<1, 3>, 3>::value == 3, "nth_total_dim_v not working");
static_assert(nth_total_dim_v<DimSeq<2, 3>, DimSeq<4, 5>, tmp::vs::IndexSequence<1, 3>, 4>::value == 1, "nth_total_dim_v not working");

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, typename TIndices>
struct TotalDims;

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, size_t... TIndices>
struct TotalDims<TKeepDimSeq, TDropDimSeq, TKeepSeq, tmp::vs::IndexSequence<TIndices...>>
{
  using type = DimSeq<nth_total_dim_v<TKeepDimSeq, TDropDimSeq, TKeepSeq, TIndices>::value...>;
};

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq>
using total_dimseq_t = typename TotalDims<TKeepDimSeq, TDropDimSeq, TKeepSeq, tmp::vs::ascending_numbers_t<
  math::max(non_trivial_dimensions_num_v<TKeepDimSeq>::value + non_trivial_dimensions_num_v<TDropDimSeq>::value, tmp::vs::max_v<TKeepSeq>::value + 1)
>>::type;

static_assert(std::is_same<total_dimseq_t<DimSeq<2, 3>, DimSeq<4, 5>, tmp::vs::IndexSequence<1, 3>>, tmp::vs::IndexSequence<4, 2, 5, 3>>::value, "total_dimseq_t not working");

static_assert(std::is_same<total_dimseq_t<DimSeq<2, 3>, DimSeq<4, 5, 6>, tmp::vs::IndexSequence<1, 3>>, tmp::vs::IndexSequence<4, 2, 5, 3, 6>>::value, "total_dimseq_t not working");




template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, size_t TKeepSeqOffset, size_t I, size_t N>
struct NthTotalDimRuntime2
{
  static const size_t current_keep_index = tmp::vs::get_optional_v<TKeepSeqOffset, TKeepSeq, (size_t) -1>::value;
  static const bool keep = current_keep_index == I;
  static const size_t value = NthTotalDim2<TKeepDimSeq, TDropDimSeq, TKeepSeq, keep ? TKeepSeqOffset + 1 : TKeepSeqOffset, I + 1, N>::value;

  template <typename TTensorTypeIn>
  __host__ __device__
  static size_t get(TTensorTypeIn&& tensor)
  {
    return NthTotalDimRuntime2<TKeepDimSeq, TDropDimSeq, TKeepSeq, keep ? TKeepSeqOffset + 1 : TKeepSeqOffset, I + 1, N>::get(util::forward<TTensorTypeIn>(tensor));
  }
};

template <bool TElmentIsStatic>
struct NthElementDimension;

template <>
struct NthElementDimension<true>
{
  template <size_t TIndex, typename TTensorType>
  __host__ __device__
  static size_t get(TTensorType&& tensor)
  {
    return nth_dimension_v<TIndex, decay_elementtype_t<TTensorType>>::value;
  }
};

template <>
struct NthElementDimension<false>
{
  template <size_t TIndex, typename TTensorType>
  __host__ __device__
  static size_t get(TTensorType&& tensor)
  {
    return tensor().template dim<TIndex>();
  }
};

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, size_t TKeepSeqOffset, size_t N>
struct NthTotalDimRuntime2<TKeepDimSeq, TDropDimSeq, TKeepSeq, TKeepSeqOffset, N, N>
{
  static const size_t drop_seq_offset = N - TKeepSeqOffset;
  static const size_t current_keep_index = tmp::vs::get_optional_v<TKeepSeqOffset, TKeepSeq, (size_t) -1>::value;
  static const bool keep = current_keep_index == N;

  template <typename TTensorTypeIn>
  __host__ __device__
  static size_t get(TTensorTypeIn&& tensor)
  {
    return keep ? tensor.template dim<TKeepSeqOffset>() : NthElementDimension<is_static_v<decay_elementtype_t<TTensorTypeIn>>::value>::template get<drop_seq_offset>(util::forward<TTensorTypeIn>(tensor));
  }
};

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, size_t N>
struct NthTotalDimRuntime
{
  template <typename TTensorTypeIn>
  __host__ __device__
  static size_t get(TTensorTypeIn&& tensor)
  {
    return NthTotalDimRuntime2<TKeepDimSeq, TDropDimSeq, TKeepSeq, 0, 0, N>::get(util::forward<TTensorTypeIn>(tensor));
  }
};





template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, typename TIndices>
struct TotalDimsRuntime2;

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, size_t... TIndices>
struct TotalDimsRuntime2<TKeepDimSeq, TDropDimSeq, TKeepSeq, tmp::vs::IndexSequence<TIndices...>>
{
  template <typename TTensorTypeIn>
  __host__ __device__
  static VectorXs<sizeof...(TIndices)> get(TTensorTypeIn&& tensor)
  {
    return VectorXs<sizeof...(TIndices)>(NthTotalDimRuntime<TKeepDimSeq, TDropDimSeq, TKeepSeq, TIndices>::get(util::forward<TTensorTypeIn>(tensor))...);
  }
};

template <typename TKeepDimSeq, typename TDropDimSeq, typename TKeepSeq, size_t TRank>
struct TotalDimsRuntime
{
  template <typename TTensorTypeIn>
  __host__ __device__
  static VectorXs<TRank> get(TTensorTypeIn&& tensor)
  {
    return TotalDimsRuntime2<TKeepDimSeq, TDropDimSeq, TKeepSeq, tmp::vs::ascending_numbers_t<TRank>>::get(util::forward<TTensorTypeIn>(tensor));
  }
};



template <typename TKeepSeq, typename TDropSeq>
struct TotalCoordinateHelper;

template <size_t... TKeepIndices, size_t... TDropIndices>
struct TotalCoordinateHelper<tmp::vs::IndexSequence<TKeepIndices...>, tmp::vs::IndexSequence<TDropIndices...>>
{
  template <typename TTensorTypeIn, typename... TCoordArgTypes>
  __host__ __device__
  static auto get(TTensorTypeIn&& tensor, TCoordArgTypes&&... coords) -> decltype(util::forward<TTensorTypeIn>(tensor)(getNthCoordinate<TKeepIndices>(util::forward<TCoordArgTypes>(coords)...)...)(getNthCoordinate<TDropIndices>(util::forward<TCoordArgTypes>(coords)...)...))
  {
    return util::forward<TTensorTypeIn>(tensor)(getNthCoordinate<TKeepIndices>(util::forward<TCoordArgTypes>(coords)...)...)(getNthCoordinate<TDropIndices>(util::forward<TCoordArgTypes>(coords)...)...);
  }
};

} // end of ns detail

#define ThisType TotalTensor<TTensorTypeIn, TKeepSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        detail::total_dimseq_t<dimseq_t<TTensorTypeIn>, dimseq_t<decay_elementtype_t<TTensorTypeIn>>, TKeepSeq> \
                              >
template <typename TTensorTypeIn, typename TKeepSeq>
class TotalTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");
  static_assert(is_tensor_v<decay_elementtype_t<TTensorTypeIn>>::value, "Elements of TTensorTypeIn must be tensors");

  using DropSeq = detail::invert_seq_t<math::max(
    tmp::vs::max_v<TKeepSeq>::value + (size_t) 1,
    non_trivial_dimensions_num_v<TTensorTypeIn>::value + non_trivial_dimensions_num_v<decay_elementtype_t<TTensorTypeIn>>::value
  ), TKeepSeq>;

  using KeepDimSeq = dimseq_t<TTensorTypeIn>;
  using DropDimSeq = dimseq_t<decay_elementtype_t<TTensorTypeIn>>;

  static const size_t NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  TotalTensor(TTensorTypeIn tensor)
    : SuperType(detail::TotalDimsRuntime<KeepDimSeq, DropDimSeq, TKeepSeq, NON_TRIVIAL_DIMENSIONS_NUM>::get(tensor))
    , m_tensor(tensor)
  {
  }

  TENSOR_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    detail::TotalCoordinateHelper<TKeepSeq, DropSeq>::get(self.m_tensor, util::forward<TCoordArgTypes>(coords)...)
  )
  TENSOR_FORWARD_ELEMENT_ACCESS(getElement)

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    return detail::NthTotalDimRuntime<KeepDimSeq, DropDimSeq, TKeepSeq, TIndex>::get(m_tensor);
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    return detail::TotalDimsRuntime<KeepDimSeq, DropDimSeq, TKeepSeq, NON_TRIVIAL_DIMENSIONS_NUM>::get(m_tensor)(index);
  }

private:
  TTensorTypeIn m_tensor;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(TotalTensor<decltype(transform(m_tensor)), TKeepSeq>
    (transform(m_tensor))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(TotalTensor<decltype(transform(m_tensor)), TKeepSeq>
    (transform(m_tensor))
  )
};
#undef SuperType
#undef ThisType

template <typename TDropSeq, typename TOtherTensorType, typename TKeepSeq = detail::invert_seq_t<
  math::max(non_trivial_dimensions_num_v<TOtherTensorType>::value + non_trivial_dimensions_num_v<decay_elementtype_t<TOtherTensorType>>::value, tmp::vs::max_v<TDropSeq>::value + 1),
TDropSeq>>
__host__ __device__
auto total(TOtherTensorType&& tensor)
RETURN_AUTO(TotalTensor<util::store_member_t<TOtherTensorType&&>, TKeepSeq>
  (util::forward<TOtherTensorType>(tensor))
);

namespace functor {
template <typename TDropSeq>
struct total_ex
{
  template <typename TTensorType>
  __host__ __device__
  auto operator()(TTensorType&& tensor)
  RETURN_AUTO(template_tensors::total<TDropSeq>(util::forward<TTensorType>(tensor)))
};
} // end of ns functor

template <size_t... TDropDims, typename TOtherTensorType>
__host__ __device__
auto total(TOtherTensorType&& tensor)
RETURN_AUTO(total<tmp::vs::IndexSequence<TDropDims...>>
  (util::forward<TOtherTensorType>(tensor))
);

namespace functor {
template <size_t... TDropDims>
struct total
{
  template <typename TTensorType>
  __host__ __device__
  auto operator()(TTensorType&& tensor)
  RETURN_AUTO(template_tensors::total<tmp::vs::IndexSequence<TDropDims...>>(util::forward<TTensorType>(tensor)))
};
} // end of ns functor

} // end of ns tensor
