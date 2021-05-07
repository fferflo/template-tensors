namespace template_tensors {

namespace detail {

template <typename TDimSeq, typename TFactorSeq, typename TIndexSequence = metal::iota<metal::number<0>, metal::number<non_trivial_dimensions_num_v<TDimSeq>::value>>>
struct StaticDilatedDimSeq;

template <typename TDimSeq, typename TFactorSeq, metal::int_... TIndices>
struct StaticDilatedDimSeq<TDimSeq, TFactorSeq, metal::numbers<TIndices...>>
{
  using type = DimSeq<(nth_dimension_v<TIndices, TDimSeq>::value == DYN ? DYN :
    nth_dimension_v<TIndices, TDimSeq>::value + (nth_dimension_v<TIndices, TDimSeq>::value - 1) * (nth_dimension_v<TIndices, TFactorSeq>::value - 1)
  )...>;
};

template <typename TDimSeq, typename TFactorSeq, typename TIndexSequence = metal::iota<metal::number<0>, metal::number<non_trivial_dimensions_num_v<TDimSeq>::value>>>
struct StaticDilatedDims;

template <typename TDimSeq, typename TFactorSeq, metal::int_... TIndices>
struct StaticDilatedDims<TDimSeq, TFactorSeq, metal::numbers<TIndices...>>
{
  template <typename TTensorType>
  __host__ __device__
  static VectorXs<sizeof...(TIndices)> get(TTensorType&& tensor)
  {
    return VectorXs<sizeof...(TIndices)>((
      tensor.template dim<TIndices>() + (tensor.template dim<TIndices>() - 1) * (nth_dimension_v<TIndices, TFactorSeq>::value - 1)
    )...);
  }
};

} // end of ns detail

#define ThisType StaticDilatedTensor<TTensorTypeOverlay, TBackgroundFunctor, TFactorSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeOverlay>::value, \
                                        typename detail::StaticDilatedDimSeq<dimseq_t<TTensorTypeOverlay>, TFactorSeq>::type \
                              >

template <typename TTensorTypeOverlay, typename TBackgroundFunctor, typename TFactorSeq>
class StaticDilatedTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeOverlay>::value, "TTensorTypeOverlay must be a tensor");
  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  StaticDilatedTensor(TTensorTypeOverlay overlay, TBackgroundFunctor background)
    : SuperType(detail::StaticDilatedDims<dimseq_t<SuperType>, TFactorSeq>::get(util::forward<TTensorTypeOverlay>(overlay)))
    , m_overlay(overlay)
    , m_background(background)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
    -> typename detail::CombineTensorMemberElementTypesHelper<decltype(self.m_overlay), decltype(self.m_background)>::type
  {
    VectorXs<NON_TRIVIAL_DIMENSIONS_NUM> dilated_coords = toCoordVector<NON_TRIVIAL_DIMENSIONS_NUM>(util::forward<TCoordArgTypes>(coords)...);
    VectorXs<NON_TRIVIAL_DIMENSIONS_NUM> factor = toCoordVector<NON_TRIVIAL_DIMENSIONS_NUM>(TFactorSeq());
    VectorXs<NON_TRIVIAL_DIMENSIONS_NUM> original_coords = dilated_coords / factor;
    if (template_tensors::all(original_coords * factor == dilated_coords))
    {
      return self.m_overlay(original_coords);
    }
    else
    {
      return self.m_background(util::forward<TCoordArgTypes>(coords)...);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_overlay.template dim<TIndex>() + (m_overlay.template dim<TIndex>() - 1) * (nth_dimension_v<TIndex, TFactorSeq>::value - 1);
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    static const metal::int_ rows = NON_TRIVIAL_DIMENSIONS_NUM;
    dim_t dim = m_overlay.dim(index);
    return math::lt(index, rows) ? (dim + (dim - 1) * (toCoordVector<NON_TRIVIAL_DIMENSIONS_NUM>(TFactorSeq())(index) - 1)) : 1;
  }

private:
  TTensorTypeOverlay m_overlay;
  TBackgroundFunctor m_background;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(StaticDilatedTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), TFactorSeq>
    (transform(m_overlay), transform(m_background))
  )
  // TODO: allow non-tensortypes in transform(...) as well
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(StaticDilatedTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), TFactorSeq>
    (transform(m_overlay), transform(m_background))
  )
};
#undef SuperType
#undef ThisType



#define ThisType DynamicDilatedTensor<TTensorTypeOverlay, TBackgroundFunctor, TFactorVector>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeOverlay>::value, \
                                        template_tensors::dyn_dimseq_t<rows_v<TFactorVector>::value> \
                              >

template <typename TTensorTypeOverlay, typename TBackgroundFunctor, typename TFactorVector>
class DynamicDilatedTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeOverlay>::value, "TTensorTypeOverlay must be a tensor");
  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  DynamicDilatedTensor(TTensorTypeOverlay overlay, TBackgroundFunctor background, TFactorVector factor)
    : SuperType(overlay.dims() + (overlay.dims() - 1) * (factor - 1))
    , m_overlay(overlay)
    , m_background(background)
    , m_factor(factor)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
    -> typename detail::CombineTensorMemberElementTypesHelper<decltype(self.m_overlay), decltype(self.m_background)>::type
  {
    VectorXs<NON_TRIVIAL_DIMENSIONS_NUM> dilated_coords = toCoordVector<NON_TRIVIAL_DIMENSIONS_NUM>(util::forward<TCoordArgTypes>(coords)...);
    VectorXs<NON_TRIVIAL_DIMENSIONS_NUM> original_coords = dilated_coords / self.m_factor;
    if (template_tensors::all(original_coords * self.m_factor == dilated_coords))
    {
      return self.m_overlay(original_coords);
    }
    else
    {
      return self.m_background(util::forward<TCoordArgTypes>(coords)...);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_overlay.template dim<TIndex>() + (m_overlay.template dim<TIndex>() - 1) * (getNthDimension<TIndex>(m_factor) - 1);
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    static const metal::int_ rows = NON_TRIVIAL_DIMENSIONS_NUM;
    dim_t dim = m_overlay.dim(index);
    return math::lt(index, static_cast<size_t>(rows)) ? (dim + (dim - 1) * (m_factor(index) - 1)) : dim;
  }

private:
  TTensorTypeOverlay m_overlay;
  TBackgroundFunctor m_background;
  TFactorVector m_factor;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(DynamicDilatedTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), decltype(transform(m_factor))>
    (transform(m_overlay), transform(m_background), transform(m_factor))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(DynamicDilatedTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), decltype(transform(m_factor))>
    (transform(m_overlay), transform(m_background), transform(m_factor))
  )
};
#undef SuperType
#undef ThisType


template <typename TFactorSeq, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, ENABLE_IF(is_dimseq_v<TFactorSeq>::value)>
__host__ __device__
auto dilate(TOtherTensorType&& tensor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(StaticDilatedTensor<util::store_member_t<TOtherTensorType&&>, util::store_member_t<TBackgroundFunctor&&>, TFactorSeq>
  (util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background))
)

template <metal::int_ TFactor, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>>
__host__ __device__
auto dilate(TOtherTensorType&& tensor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(dilate<repeat_dimseq_t<TFactor, non_trivial_dimensions_num_v<TOtherTensorType>::value>>(util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background)))


template <typename TDummy = void, typename TFactorVector, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, ENABLE_IF(is_tensor_v<TFactorVector>::value && std::is_same<TDummy, void>::value)>
__host__ __device__
auto dilate(TOtherTensorType&& tensor, TFactorVector&& factor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(DynamicDilatedTensor<util::store_member_t<TOtherTensorType&&>, util::store_member_t<TBackgroundFunctor&&>, util::store_member_t<TFactorVector&&>>
  (util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background), util::forward<TFactorVector>(factor))
)

template <typename TDummy = void, typename TFactorScalar, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, bool TDummy2 = true, ENABLE_IF(!is_tensor_v<TFactorScalar>::value && std::is_same<TDummy, void>::value)>
__host__ __device__
auto dilate(TOtherTensorType&& tensor, TFactorScalar&& scalar, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(dilate(util::forward<TOtherTensorType>(tensor), broadcast<non_trivial_dimensions_num_v<TOtherTensorType>::value>(util::forward<TFactorScalar>(scalar)), util::forward<TBackgroundFunctor>(background)))

} // end of ns tensor
