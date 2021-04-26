namespace template_tensors {

namespace detail {

template <size_t TTargetOuterDim, size_t TOuterDim, size_t TInnerDim, typename TDropSeq,
  bool TDimIsDropped = tmp::vs::contains_v<size_t, TDropSeq, TInnerDim>::value>
struct KeepDim;

template <size_t TTargetOuterDim, size_t TOuterDim, size_t TInnerDim, typename TDropSeq>
struct KeepDim<TTargetOuterDim, TOuterDim, TInnerDim, TDropSeq, true>
{
  static const size_t value = KeepDim<TTargetOuterDim, TOuterDim, TInnerDim + 1, TDropSeq>::value;
};

template <size_t TTargetOuterDim, size_t TOuterDim, size_t TInnerDim, typename TDropSeq>
struct KeepDim<TTargetOuterDim, TOuterDim, TInnerDim, TDropSeq, false>
{
  static const size_t value = KeepDim<TTargetOuterDim, TOuterDim + 1, TInnerDim + 1, TDropSeq>::value;
};

template <size_t TTargetOuterDim, size_t TInnerDim, typename TDropSeq>
struct KeepDim<TTargetOuterDim, TTargetOuterDim, TInnerDim, TDropSeq, false>
{
  static const size_t value = TInnerDim;
};

template <size_t TInnerRank, typename TDropSeq, typename TIndices = tmp::vs::ascending_numbers_t<TInnerRank - tmp::vs::length_v<TDropSeq>::value>>
struct KeepSeq;

template <size_t TInnerRank, typename TDropSeq, size_t... TIndices>
struct KeepSeq<TInnerRank, TDropSeq, tmp::vs::IndexSequence<TIndices...>>
{
  using type = DimSeq<KeepDim<TIndices, 0, 0, TDropSeq>::value...>;
};

template <size_t TInnerRank, typename TDropSeq>
using invert_seq_t = typename KeepSeq<TInnerRank, TDropSeq>::type;

static_assert(std::is_same<tmp::vs::IndexSequence<0>, invert_seq_t<2, tmp::vs::IndexSequence<1>>>::value, "invert_seq_t not working");
static_assert(std::is_same<tmp::vs::IndexSequence<0, 2>, invert_seq_t<3, tmp::vs::IndexSequence<1>>>::value, "invert_seq_t not working");



template <typename TInnerDimSeq, typename TKeepSeq>
struct PickDimSeq;

template <typename TInnerDimSeq, size_t... TKeepDims>
struct PickDimSeq<TInnerDimSeq, tmp::vs::IndexSequence<TKeepDims...>>
{
  using type = DimSeq<nth_dimension_v<TKeepDims, TInnerDimSeq>::value...>;
};

template <typename TInnerDimSeq, typename TKeepSeq>
using pick_dimseq_t = typename PickDimSeq<TInnerDimSeq, TKeepSeq>::type;

static_assert(are_compatible_dimseqs_v<DimSeq<4>, pick_dimseq_t<DimSeq<3, 4>, tmp::vs::IndexSequence<1>>>::value, "pick_dimseq_t not working");



template <typename TKeepSeq>
struct PickDims;

template <size_t... TKeepDims>
struct PickDims<tmp::vs::IndexSequence<TKeepDims...>>
{
  template <typename TInnerTensor>
  __host__ __device__
  static VectorXs<sizeof...(TKeepDims)> pick(TInnerTensor&& inner_tensor)
  {
    return VectorXs<sizeof...(TKeepDims)>(inner_tensor.template dim<TKeepDims>()...);
  }
};





template <size_t N, size_t TDropDim, size_t TKeepDim, typename TDropSeq,
  bool TDimIsDropped = tmp::vs::contains_v<size_t, TDropSeq, TDropDim + TKeepDim>::value>
struct PartialCoordinateHelper;

template <size_t N, size_t TDropDim, size_t TKeepDim, typename TDropSeq>
struct PartialCoordinateHelper<N, TDropDim, TKeepDim, TDropSeq, true>
{
  template <typename TKeepCoordVector, typename... TCoordArgTypes>
  __host__ __device__
  static size_t coord(TKeepCoordVector&& keep_coords, TCoordArgTypes&&... drop_coords)
  {
    return PartialCoordinateHelper<N - 1, TDropDim + 1, TKeepDim, TDropSeq>::coord(util::forward<TKeepCoordVector>(keep_coords), util::forward<TCoordArgTypes>(drop_coords)...);
  }
};

template <size_t N, size_t TDropDim, size_t TKeepDim, typename TDropSeq>
struct PartialCoordinateHelper<N, TDropDim, TKeepDim, TDropSeq, false>
{
  template <typename TKeepCoordVector, typename... TCoordArgTypes>
  __host__ __device__
  static size_t coord(TKeepCoordVector&& keep_coords, TCoordArgTypes&&... drop_coords)
  {
    return PartialCoordinateHelper<N - 1, TDropDim, TKeepDim + 1, TDropSeq>::coord(util::forward<TKeepCoordVector>(keep_coords), util::forward<TCoordArgTypes>(drop_coords)...);
  }
};

template <size_t TDropDim, size_t TKeepDim, typename TDropSeq>
struct PartialCoordinateHelper<0, TDropDim, TKeepDim, TDropSeq, true>
{
  template <typename TKeepCoordVector, typename... TCoordArgTypes>
  __host__ __device__
  static size_t coord(TKeepCoordVector&& keep_coords, TCoordArgTypes&&... drop_coords)
  {
    return getNthCoordinate<TDropDim>(util::forward<TCoordArgTypes>(drop_coords)...);
  }
};

template <size_t TDropDim, size_t TKeepDim, typename TDropSeq>
struct PartialCoordinateHelper<0, TDropDim, TKeepDim, TDropSeq, false>
{
  template <typename TKeepCoordVector, typename... TCoordArgTypes>
  __host__ __device__
  static size_t coord(TKeepCoordVector&& keep_coords, TCoordArgTypes&&... drop_coords)
  {
    return getNthCoordinate<TKeepDim>(util::forward<TKeepCoordVector>(keep_coords));
  }
};





template <size_t TFirst, typename TRestSeq>
struct PartialMaxHelper;

template <size_t TFirst, size_t... TRest>
struct PartialMaxHelper<TFirst, tmp::vs::IndexSequence<TRest...>>
{
  static const size_t value = math::max(TFirst, (TRest + 1)...);
};

} // end of ns detail





#define ThisType PartialTensorElement<TTensorTypeIn, TDropSeq, TRValue>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        typename detail::pick_dimseq_t<dimseq_t<TTensorTypeIn>, TDropSeq> \
                              >
template <typename TTensorTypeIn, typename TDropSeq, bool TRValue>
class PartialTensorElement : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  using KeepSeq = detail::invert_seq_t<non_trivial_dimensions_num_v<TTensorTypeIn>::value, TDropSeq>;

  template <typename... TCoordArgTypes>
  __host__ __device__
  PartialTensorElement(TTensorTypeIn tensor, TCoordArgTypes&&... keep_coords)
    : SuperType(detail::PickDims<TDropSeq>::pick(tensor))
    , m_tensor(tensor)
    , m_keep_coords(toCoordVector<tmp::vs::length_v<KeepSeq>::value>(util::forward<TCoordArgTypes>(keep_coords)...))
  {
  }

  __host__ __device__
  PartialTensorElement(const PartialTensorElement<TTensorTypeIn, TDropSeq, TRValue>& other)
    : SuperType(static_cast<const SuperType&>(other))
    , m_tensor(other.m_tensor)
    , m_keep_coords(other.m_keep_coords)
  {
  }

  __host__ __device__
  PartialTensorElement(PartialTensorElement<TTensorTypeIn, TDropSeq, TRValue>&& other)
    : SuperType(static_cast<SuperType&&>(other))
    , m_tensor(other.m_tensor)
    , m_keep_coords(util::move(other.m_keep_coords))
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    return m_tensor.template dim<tmp::vs::get_v<TIndex, TDropSeq>::value>();
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    return m_tensor.dim(tmp::vs::getByIterating<TDropSeq>(index, 1UL));
  }

private:
  TTensorTypeIn m_tensor;
  VectorXs<tmp::vs::length_v<KeepSeq>::value> m_keep_coords;

public:
  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static auto getElement(TThisType&& self, tmp::vs::Sequence<size_t, TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    util::move_if<TRValue>(self.m_tensor)(detail::PartialCoordinateHelper<TIndices, 0, 0, TDropSeq>::coord(self.m_keep_coords, util::forward<TCoordArgTypes>(coords)...)...)
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ_N(getElement, non_trivial_dimensions_num_v<TTensorTypeIn>::value)

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(PartialTensorElement<decltype(transform(m_tensor)), TDropSeq, TRValue>
    (transform(m_tensor))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(PartialTensorElement<decltype(transform(m_tensor)), TDropSeq, TRValue>
    (transform(m_tensor))
  )
};
#undef SuperType
#undef ThisType

#define ThisType PartialTensor<TTensorTypeIn, TKeepSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        typename detail::pick_dimseq_t<dimseq_t<TTensorTypeIn>, TKeepSeq> \
                              >

template <typename TTensorTypeIn, typename TKeepSeq>
class PartialTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  using KeepSeq = TKeepSeq;

  __host__ __device__
  PartialTensor(TTensorTypeIn tensor)
    : SuperType(detail::PickDims<TKeepSeq>::pick(tensor))
    , m_tensor(static_cast<TTensorTypeIn&&>(tensor))
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    PartialTensorElement<
      TTensorTypeIn&,
      detail::invert_seq_t<non_trivial_dimensions_num_v<TTensorTypeIn>::value, TKeepSeq>,
      std::is_rvalue_reference<TThisType&&>::value && !std::is_reference<TTensorTypeIn>::value
    >(self.m_tensor, util::forward<TCoordArgTypes>(coords)...)
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)
  // TODO: self is not properly forwarded as rvalue

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    return m_tensor.template dim<tmp::vs::get_v<TIndex, TKeepSeq>::value>();
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    return m_tensor.dim(tmp::vs::getByIterating<TKeepSeq>(index, 1UL));
  }

private:
  TTensorTypeIn m_tensor;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(PartialTensor<decltype(transform(m_tensor)), TKeepSeq>
    (transform(m_tensor))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(PartialTensor<decltype(transform(m_tensor)), TKeepSeq>
    (transform(m_tensor))
  )
};
#undef SuperType
#undef ThisType



template <typename TDropSeq, typename TOtherTensorType, typename TKeepSeq = detail::invert_seq_t<
  detail::PartialMaxHelper<non_trivial_dimensions_num_v<TOtherTensorType>::value, TDropSeq>::value,
TDropSeq>>
__host__ __device__
auto partial(TOtherTensorType&& tensor)
RETURN_AUTO(PartialTensor<util::store_member_t<TOtherTensorType&&>, TKeepSeq>
  (util::forward<TOtherTensorType>(tensor))
);

template <size_t... TDropDims, typename TOtherTensorType>
__host__ __device__
auto partial(TOtherTensorType&& tensor)
RETURN_AUTO(partial<tmp::vs::IndexSequence<TDropDims...>>
  (util::forward<TOtherTensorType>(tensor))
);

template <typename TMatrixType>
__host__ __device__
auto rows(TMatrixType&& matrix)
RETURN_AUTO(partial<1>(util::forward<TMatrixType>(matrix)))

template <typename TMatrixType>
__host__ __device__
auto cols(TMatrixType&& matrix)
RETURN_AUTO(partial<0>(util::forward<TMatrixType>(matrix)))

} // end of ns tensor
