namespace template_tensors {

namespace detail {

template <typename TDimSeq, typename TRepetitionsSeq, typename TIndexSequence
  = metal::iota<metal::number<0>, metal::number<math::max(non_trivial_dimensions_num_v<TDimSeq>::value, non_trivial_dimensions_num_v<TRepetitionsSeq>::value)>>>
struct StaticRepeatDimSeq;

template <typename TDimSeq, typename TRepetitionsSeq, metal::int_... TIndices>
struct StaticRepeatDimSeq<TDimSeq, TRepetitionsSeq, metal::numbers<TIndices...>>
{
  using type = DimSeq<(nth_dimension_v<TIndices, TDimSeq>::value == DYN ? DYN :
    nth_dimension_v<TIndices, TDimSeq>::value * nth_dimension_v<TIndices, TRepetitionsSeq>::value
  )...>;
};

static_assert(std::is_same<
  typename StaticRepeatDimSeq<DimSeq<1>, DimSeq<2>>::type,
  DimSeq<2>
>::value, "StaticRepeatDimSeq not working");

template <typename TDimSeq, typename TRepetitionsSeq, typename TIndexSequence
  = metal::iota<metal::number<0>, metal::number<math::max(non_trivial_dimensions_num_v<TDimSeq>::value, non_trivial_dimensions_num_v<TRepetitionsSeq>::value)>>>
struct StaticRepeatDims;

template <typename TDimSeq, typename TRepetitionsSeq, metal::int_... TIndices>
struct StaticRepeatDims<TDimSeq, TRepetitionsSeq, metal::numbers<TIndices...>>
{
  template <typename TTensorType>
  __host__ __device__
  static VectorXs<sizeof...(TIndices)> get(TTensorType&& tensor)
  {
    return VectorXs<sizeof...(TIndices)>((
      tensor.template dim<TIndices>() * nth_dimension_v<TIndices, TRepetitionsSeq>::value
    )...);
  }
};

} // end of ns detail

#define ThisType StaticRepeatTensor<TTensorTypeIn, TRepetitionsSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        typename detail::StaticRepeatDimSeq<dimseq_t<TTensorTypeIn>, TRepetitionsSeq>::type \
                              >
template <typename TTensorTypeIn, typename TRepetitionsSeq>
class StaticRepeatTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  StaticRepeatTensor(TTensorTypeIn tensor)
    : SuperType(detail::StaticRepeatDims<dimseq_t<SuperType>, TRepetitionsSeq>::get(std::forward<TTensorTypeIn>(tensor)))
    , m_tensor(tensor)
  {
  }

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static auto getElement(TThisType&& self, metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_tensor((coords / util::constant<metal::int_, nth_dimension_v<TIndices, TRepetitionsSeq>::value>())...)
  ) // TODO: std::forward coords
  TT_TENSOR_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ(getElement)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_tensor.template dim<TIndex>() * nth_dimension_v<TIndex, TRepetitionsSeq>::value;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    static const metal::int_ rows = NON_TRIVIAL_DIMENSIONS_NUM;
    return math::lt(index, rows) ? (m_tensor.dim(index) * toCoordVector<NON_TRIVIAL_DIMENSIONS_NUM>(TRepetitionsSeq())(index)) : 1;
  }

private:
  TTensorTypeIn m_tensor;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(StaticRepeatTensor<decltype(transform(m_tensor)), TRepetitionsSeq>
    (transform(m_tensor))
  )
  // TODO: allow non-tensortypes in transform(...) as well
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(StaticRepeatTensor<decltype(transform(m_tensor)), TRepetitionsSeq>
    (transform(m_tensor))
  )
};
#undef SuperType
#undef ThisType


#define ThisType DynamicRepeatTensor<TTensorTypeIn, TRepetitionsVector>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        template_tensors::dyn_dimseq_t<rows_v<TRepetitionsVector>::value> \
                              >
template <typename TTensorTypeIn, typename TRepetitionsVector>
class DynamicRepeatTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  DynamicRepeatTensor(TTensorTypeIn tensor, TRepetitionsVector repetitions)
    : SuperType(tensor.template dims<NON_TRIVIAL_DIMENSIONS_NUM>() * repetitions)
    , m_tensor(tensor)
    , m_repetitions(repetitions)
  {
  }

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static auto getElement(TThisType&& self, metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_tensor((coords / getNthDimension<TIndices>(self.m_repetitions))...)
  ) // TODO: std::forward coords?
  TT_TENSOR_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ(getElement)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_tensor.template dim<TIndex>() * getNthDimension<TIndex>(m_repetitions);
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    static const metal::int_ rows = NON_TRIVIAL_DIMENSIONS_NUM;
    return math::lt(index, static_cast<size_t>(rows)) ? (m_tensor.dim(index) * m_repetitions(index)) : 1;
  }

private:
  TTensorTypeIn m_tensor;
  TRepetitionsVector m_repetitions;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(DynamicRepeatTensor<decltype(transform(m_tensor)), decltype(transform(m_repetitions))>
    (transform(m_tensor), transform(m_repetitions))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(DynamicRepeatTensor<decltype(transform(m_tensor)), decltype(transform(m_repetitions))>
    (transform(m_tensor), transform(m_repetitions))
  )
};
#undef SuperType
#undef ThisType



template <typename TRepetitionsSeq, typename TOtherTensorType, ENABLE_IF(is_dimseq_v<TRepetitionsSeq>::value)>
__host__ __device__
auto repeat(TOtherTensorType&& tensor)
RETURN_AUTO(StaticRepeatTensor<TOtherTensorType, TRepetitionsSeq>
  (std::forward<TOtherTensorType>(tensor))
)


template <typename TDummy = void, typename TRepetitionsVector, typename TOtherTensorType, ENABLE_IF(is_tensor_v<TRepetitionsVector>::value && std::is_same<TDummy, void>::value)>
__host__ __device__
auto repeat(TOtherTensorType&& tensor, TRepetitionsVector&& factor)
RETURN_AUTO(DynamicRepeatTensor<TOtherTensorType, TRepetitionsVector>
  (std::forward<TOtherTensorType>(tensor), std::forward<TRepetitionsVector>(factor))
)

template <typename TDummy = void, typename TRepetitionsScalar, typename TOtherTensorType, bool TDummy2 = true, ENABLE_IF(!is_tensor_v<TRepetitionsScalar>::value && std::is_same<TDummy, void>::value)>
__host__ __device__
auto repeat(TOtherTensorType&& tensor, TRepetitionsScalar&& repetitions)
RETURN_AUTO(repeat(std::forward<TOtherTensorType>(tensor), broadcast<non_trivial_dimensions_num_v<TOtherTensorType>::value>(std::forward<TRepetitionsScalar>(repetitions))))

} // end of ns template_tensors
