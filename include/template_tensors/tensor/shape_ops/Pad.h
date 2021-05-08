namespace template_tensors {

namespace detail {

template <typename TDimSeq, typename TPadFrontSeq, typename TPadBackSeq,
  typename TIndexSequence = metal::iota<metal::number<0>, metal::number<math::max(non_trivial_dimensions_num_v<TDimSeq>::value, non_trivial_coordinates_num_v<TPadFrontSeq>::value, non_trivial_coordinates_num_v<TPadBackSeq>::value)>>>
struct StaticPaddedDimSeq;

template <typename TDimSeq, typename TPadFrontSeq, typename TPadBackSeq, metal::int_... TIndices>
struct StaticPaddedDimSeq<TDimSeq, TPadFrontSeq, TPadBackSeq, metal::numbers<TIndices...>>
{
  using type = DimSeq<(nth_dimension_v<TIndices, TDimSeq>::value == DYN ? DYN :
    nth_dimension_v<TIndices, TDimSeq>::value + nth_coordinate_v<TIndices, TPadFrontSeq>::value + nth_coordinate_v<TIndices, TPadBackSeq>::value
  )...>;
};

template <typename TDimSeq, typename TPadFrontSeq, typename TPadBackSeq,
  typename TIndexSequence = metal::iota<metal::number<0>, metal::number<math::max(non_trivial_dimensions_num_v<TDimSeq>::value, non_trivial_coordinates_num_v<TPadFrontSeq>::value, non_trivial_coordinates_num_v<TPadBackSeq>::value)>>>
struct StaticPaddedDims;

template <typename TDimSeq, typename TPadFrontSeq, typename TPadBackSeq, metal::int_... TIndices>
struct StaticPaddedDims<TDimSeq, TPadFrontSeq, TPadBackSeq, metal::numbers<TIndices...>>
{
  template <typename TTensorType>
  __host__ __device__
  static VectorXs<sizeof...(TIndices)> get(TTensorType&& tensor)
  {
    return VectorXs<sizeof...(TIndices)>((
      tensor.template dim<TIndices>() + nth_coordinate_v<TIndices, TPadFrontSeq>::value + nth_coordinate_v<TIndices, TPadBackSeq>::value
    )...);
  }
};

template <metal::int_ TRank, typename TPadFrontSeq, typename TPadBackSeq,
  typename TIndexSequence = metal::iota<metal::number<0>, metal::number<TRank>>>
struct StaticPaddedElementAccess;

template <metal::int_ TRank, typename TPadFrontSeq, typename TPadBackSeq, metal::int_... TIndices>
struct StaticPaddedElementAccess<TRank, TPadFrontSeq, TPadBackSeq, metal::numbers<TIndices...>>
{
  template <typename TTensorType, typename... TCoordArgTypes>
  __host__ __device__
  static bool isInOriginal(TTensorType&& tensor, TCoordArgTypes&&... coords)
  {
    return math::landsc((
         (util::constant<metal::int_, nth_coordinate_v<TIndices, TPadFrontSeq>::value>() == 0 || math::gte(getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...), (size_t) util::constant<metal::int_, nth_coordinate_v<TIndices, TPadFrontSeq>::value>()))
      && (util::constant<metal::int_, nth_coordinate_v<TIndices, TPadBackSeq>::value>() == 0 || math::lt(getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...), tensor.template dim<TIndices>() + (size_t) util::constant<metal::int_, nth_coordinate_v<TIndices, TPadFrontSeq>::value>()))
    )...);
  }

  template <metal::int_... TCoordinateIndices, typename TTensorType, typename... TCoordArgTypes>
  __host__ __device__
  static auto get(TTensorType&& tensor, TCoordArgTypes&&... coords)
  RETURN_AUTO(tensor((coords - nth_coordinate_v<TCoordinateIndices, TPadFrontSeq>::value)...))
};

} // end of ns detail

#define ThisType StaticPaddedTensor<TTensorTypeOverlay, TBackgroundFunctor, TPadFrontSeq, TPadBackSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeOverlay>::value, \
                                        typename detail::StaticPaddedDimSeq<dimseq_t<TTensorTypeOverlay>, TPadFrontSeq, TPadBackSeq>::type \
                              >

template <typename TTensorTypeOverlay, typename TBackgroundFunctor, typename TPadFrontSeq, typename TPadBackSeq>
class StaticPaddedTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeOverlay>::value, "TTensorTypeOverlay must be a tensor");
  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  StaticPaddedTensor(TTensorTypeOverlay overlay, TBackgroundFunctor background)
    : SuperType(detail::StaticPaddedDims<dimseq_t<SuperType>, TPadFrontSeq, TPadBackSeq>::get(util::forward<TTensorTypeOverlay>(overlay)))
    , m_overlay(overlay)
    , m_background(background)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static auto getElement(TThisType&& self, metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
    -> typename detail::CombineTensorMemberElementTypesHelper<decltype(self.m_overlay), decltype(self.m_background)>::type
  {
    if (detail::StaticPaddedElementAccess<NON_TRIVIAL_DIMENSIONS_NUM, TPadFrontSeq, TPadBackSeq>::isInOriginal(self.m_overlay, util::forward<TCoordArgTypes>(coords)...))
    {
      return detail::StaticPaddedElementAccess<NON_TRIVIAL_DIMENSIONS_NUM, TPadFrontSeq, TPadBackSeq>::template get<TIndices...>(self.m_overlay, util::forward<TCoordArgTypes>(coords)...);
    }
    else
    {
      return self.m_background(util::forward<TCoordArgTypes>(coords)...);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ(getElement)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_overlay.template dim<TIndex>() + nth_coordinate_v<TIndex, TPadFrontSeq>::value + nth_coordinate_v<TIndex, TPadBackSeq>::value;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    static const metal::int_ rows = NON_TRIVIAL_DIMENSIONS_NUM;
    return index >= rows ? 1 : (m_overlay.dim(index) + toCoordVector<NON_TRIVIAL_DIMENSIONS_NUM>(TPadFrontSeq())(index) + toCoordVector<NON_TRIVIAL_DIMENSIONS_NUM>(TPadBackSeq())(index));
  }

private:
  TTensorTypeOverlay m_overlay;
  TBackgroundFunctor m_background;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(StaticPaddedTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), TPadFrontSeq, TPadBackSeq>
    (transform(m_overlay), transform(m_background))
  )
  // TODO: allow non-tensortypes in transform(...) as well
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(StaticPaddedTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), TPadFrontSeq, TPadBackSeq>
    (transform(m_overlay), transform(m_background))
  )
};
#undef SuperType
#undef ThisType


namespace detail {

template <metal::int_ TRank, typename TIndexSequence = metal::iota<metal::number<0>, metal::number<TRank>>>
struct DynamicPadFrontElementAccess;

template <metal::int_ TRank, metal::int_... TIndices>
struct DynamicPadFrontElementAccess<TRank, metal::numbers<TIndices...>>
{
  template <typename TPaddingVector, typename... TCoordArgTypes>
  __host__ __device__
  static bool isInOriginal(TPaddingVector&& padding, TCoordArgTypes&&... coords)
  {
    return math::landsc((
      math::gte(getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...), getNthCoordinate<TIndices>(padding))
    )...);
  }

  template <metal::int_... TCoordinateIndices, typename TTensorType, typename TPaddingVector, typename... TCoordArgTypes>
  __host__ __device__
  static auto get(TTensorType&& tensor, TPaddingVector&& padding, TCoordArgTypes&&... coords)
  RETURN_AUTO(tensor((coords - getNthCoordinate<TCoordinateIndices>(util::forward<TPaddingVector>(padding)))...))
};

} // end of ns detail

#define ThisType DynamicPadFrontTensor<TTensorTypeOverlay, TBackgroundFunctor, TPaddingVector>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeOverlay>::value, \
                                        dyn_dimseq_t<rows_v<TPaddingVector>::value> \
                              >
template <typename TTensorTypeOverlay, typename TBackgroundFunctor, typename TPaddingVector>
class DynamicPadFrontTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeOverlay>::value, "TTensorTypeOverlay must be a tensor");
  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  DynamicPadFrontTensor(TTensorTypeOverlay overlay, TBackgroundFunctor background, TPaddingVector padding)
    : SuperType(overlay.template dims<NON_TRIVIAL_DIMENSIONS_NUM>() + padding)
    , m_overlay(overlay)
    , m_background(background)
    , m_padding(padding)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static auto getElement(TThisType&& self, metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
    -> typename detail::CombineTensorMemberElementTypesHelper<decltype(self.m_overlay), decltype(self.m_background)>::type
  {
    if (detail::DynamicPadFrontElementAccess<NON_TRIVIAL_DIMENSIONS_NUM>::isInOriginal(self.m_padding, util::forward<TCoordArgTypes>(coords)...))
    {
      return detail::DynamicPadFrontElementAccess<NON_TRIVIAL_DIMENSIONS_NUM>::template get<TIndices...>(self.m_overlay, self.m_padding, util::forward<TCoordArgTypes>(coords)...);
    }
    else
    {
      return self.m_background(util::forward<TCoordArgTypes>(coords)...);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ(getElement)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_overlay.template dim<TIndex>() + getNthCoordinate<TIndex>(m_padding);
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    static const metal::int_ rows = NON_TRIVIAL_DIMENSIONS_NUM;
    return index >= rows ? 1 : (m_overlay.dim(index) + m_padding(index));
  }

private:
  TTensorTypeOverlay m_overlay;
  TBackgroundFunctor m_background;
  TPaddingVector m_padding;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(DynamicPadFrontTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), decltype(transform(m_padding))>
    (transform(m_overlay), transform(m_background), transform(m_padding))
  )
  // TODO: allow non-tensortypes in transform(...) as well
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(DynamicPadFrontTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), decltype(transform(m_padding))>
    (transform(m_overlay), transform(m_background), transform(m_padding))
  )
};
#undef SuperType
#undef ThisType



namespace detail {

template <metal::int_ TRank, typename TIndexSequence = metal::iota<metal::number<0>, metal::number<TRank>>>
struct DynamicPadBackElementAccess;

template <metal::int_ TRank, metal::int_... TIndices>
struct DynamicPadBackElementAccess<TRank, metal::numbers<TIndices...>>
{
  template <typename TTensorType, typename... TCoordArgTypes>
  __host__ __device__
  static bool isInOriginal(TTensorType&& overlay, TCoordArgTypes&&... coords)
  {
    return math::landsc((
      math::lt(getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...), overlay.template dim<TIndices>())
    )...);
  }
};

} // end of ns detail

#define ThisType DynamicPadBackTensor<TTensorTypeOverlay, TBackgroundFunctor, TPaddingVector>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeOverlay>::value, \
                                        dyn_dimseq_t<rows_v<TPaddingVector>::value> \
                              >

template <typename TTensorTypeOverlay, typename TBackgroundFunctor, typename TPaddingVector>
class DynamicPadBackTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeOverlay>::value, "TTensorTypeOverlay must be a tensor");
  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<SuperType>::value;

  __host__ __device__
  DynamicPadBackTensor(TTensorTypeOverlay overlay, TBackgroundFunctor background, TPaddingVector padding)
    : SuperType(overlay.template dims<NON_TRIVIAL_DIMENSIONS_NUM>() + padding)
    , m_overlay(overlay)
    , m_background(background)
    , m_padding(padding)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
    -> typename detail::CombineTensorMemberElementTypesHelper<decltype(self.m_overlay), decltype(self.m_background)>::type
  {
    if (detail::DynamicPadBackElementAccess<NON_TRIVIAL_DIMENSIONS_NUM>::isInOriginal(self.m_overlay, util::forward<TCoordArgTypes>(coords)...))
    {
      return self.m_overlay(util::forward<TCoordArgTypes>(coords)...);
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
    return m_overlay.template dim<TIndex>() + getNthCoordinate<TIndex>(m_padding);
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    static const metal::int_ rows = NON_TRIVIAL_DIMENSIONS_NUM;
    return index >= rows ? 1 : (m_overlay.dim(index) + m_padding(index));
  }

private:
  TTensorTypeOverlay m_overlay;
  TBackgroundFunctor m_background;
  TPaddingVector m_padding;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(DynamicPadBackTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), decltype(transform(m_padding))>
    (transform(m_overlay), transform(m_background), transform(m_padding))
  )
  // TODO: allow non-tensortypes in transform(...) as well
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(DynamicPadBackTensor<decltype(transform(m_overlay)), decltype(transform(m_background)), decltype(transform(m_padding))>
    (transform(m_overlay), transform(m_background), transform(m_padding))
  )
};
#undef SuperType
#undef ThisType



template <typename TPadFrontSeq, typename TPadBackSeq, bool TDummy = true, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, ENABLE_IF(is_coordseq_v<TPadFrontSeq>::value && is_coordseq_v<TPadBackSeq>::value)>
__host__ __device__
auto pad(TOtherTensorType&& tensor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(StaticPaddedTensor<util::store_member_t<TOtherTensorType&&>, util::store_member_t<TBackgroundFunctor&&>, TPadFrontSeq, TPadBackSeq>
  (util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background))
)

template <typename TPadBothSeq, bool TDummy = true, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, ENABLE_IF(is_coordseq_v<TPadBothSeq>::value)>
__host__ __device__
auto pad(TOtherTensorType&& tensor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad<TPadBothSeq, TPadBothSeq>(util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background)))

template <typename TPadFrontSeq, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, ENABLE_IF(is_coordseq_v<TPadFrontSeq>::value)>
__host__ __device__
auto pad_front(TOtherTensorType&& tensor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad<TPadFrontSeq, CoordSeq<>>(util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background)))

template <typename TPadBackSeq, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, ENABLE_IF(is_coordseq_v<TPadBackSeq>::value)>
__host__ __device__
auto pad_back(TOtherTensorType&& tensor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad<CoordSeq<>, TPadBackSeq>(util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background)))


template <metal::int_ TPadding, bool TDummy = true, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>>
__host__ __device__
auto pad(TOtherTensorType&& tensor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad<
  repeat_dimseq_t<TPadding, non_trivial_dimensions_num_v<TOtherTensorType>::value>,
  repeat_dimseq_t<TPadding, non_trivial_dimensions_num_v<TOtherTensorType>::value>>
  (util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background))
)

template <metal::int_ TPadding, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>>
__host__ __device__
auto pad_front(TOtherTensorType&& tensor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad_front<repeat_dimseq_t<TPadding, non_trivial_dimensions_num_v<TOtherTensorType>::value>>
  (util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background))
)

template <metal::int_ TPadding, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>>
__host__ __device__
auto pad_back(TOtherTensorType&& tensor, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad_back<repeat_dimseq_t<TPadding, non_trivial_dimensions_num_v<TOtherTensorType>::value>>
  (util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background))
)


template <typename TDummy = void, typename TPaddingVector, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, ENABLE_IF(std::is_same<TDummy, void>::value && is_tensor_v<TPaddingVector>::value)>
__host__ __device__
auto pad_front(TOtherTensorType&& tensor, TPaddingVector&& padding, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(DynamicPadFrontTensor<util::store_member_t<TOtherTensorType&&>, util::store_member_t<TBackgroundFunctor&&>, util::store_member_t<TPaddingVector&&>>
  (util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background), util::forward<TPaddingVector>(padding))
)

template <typename TDummy = void, typename TPaddingVector, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, ENABLE_IF(std::is_same<TDummy, void>::value && is_tensor_v<TPaddingVector>::value)>
__host__ __device__
auto pad_back(TOtherTensorType&& tensor, TPaddingVector&& padding, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(DynamicPadBackTensor<util::store_member_t<TOtherTensorType&&>, util::store_member_t<TBackgroundFunctor&&>, util::store_member_t<TPaddingVector&&>>
  (util::forward<TOtherTensorType>(tensor), util::forward<TBackgroundFunctor>(background), util::forward<TPaddingVector>(padding))
)

template <typename TDummy = void, typename TPaddingVector, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, ENABLE_IF(std::is_same<TDummy, void>::value && is_tensor_v<TPaddingVector>::value)>
__host__ __device__
auto pad(TOtherTensorType&& tensor, TPaddingVector&& padding, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad_front(
  pad_back(util::forward<TOtherTensorType>(tensor), util::forward<TPaddingVector>(padding), util::forward<TBackgroundFunctor>(background)),
  util::forward<TPaddingVector>(padding), util::forward<TBackgroundFunctor>(background)
))


template <typename TDummy = void, typename TPaddingScalar, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, bool TDummy2 = true, ENABLE_IF(std::is_same<TDummy, void>::value && !is_tensor_v<TPaddingScalar>::value)>
__host__ __device__
auto pad_front(TOtherTensorType&& tensor, TPaddingScalar&& padding, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad_front(util::forward<TOtherTensorType>(tensor), broadcast<non_trivial_dimensions_num_v<TOtherTensorType>::value>(util::forward<TPaddingScalar>(padding)), util::forward<TBackgroundFunctor>(background)))


template <typename TDummy = void, typename TPaddingScalar, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, bool TDummy2 = true, ENABLE_IF(std::is_same<TDummy, void>::value && !is_tensor_v<TPaddingScalar>::value)>
__host__ __device__
auto pad_back(TOtherTensorType&& tensor, TPaddingScalar&& padding, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad_back(util::forward<TOtherTensorType>(tensor), broadcast<non_trivial_dimensions_num_v<TOtherTensorType>::value>(util::forward<TPaddingScalar>(padding)), util::forward<TBackgroundFunctor>(background)))

template <typename TDummy = void, typename TPaddingScalar, typename TOtherTensorType, typename TBackgroundFunctor = util::functor::zero<decay_elementtype_t<TOtherTensorType>>, bool TDummy2 = true, ENABLE_IF(std::is_same<TDummy, void>::value && !is_tensor_v<TPaddingScalar>::value)>
__host__ __device__
auto pad(TOtherTensorType&& tensor, TPaddingScalar&& padding, TBackgroundFunctor&& background = TBackgroundFunctor())
RETURN_AUTO(pad_front(
  pad_back(util::forward<TOtherTensorType>(tensor), util::forward<TPaddingScalar>(padding), util::forward<TBackgroundFunctor>(background)),
  util::forward<TPaddingScalar>(padding), util::forward<TBackgroundFunctor>(background)
))

} // end of ns template_tensors
