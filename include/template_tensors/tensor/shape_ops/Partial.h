namespace template_tensors {

namespace detail {

template <metal::int_ TTargetOuterDim, metal::int_ TOuterDim, metal::int_ TInnerDim, typename TDropSeq,
  bool TDimIsDropped = metal::contains<TDropSeq, metal::number<TInnerDim>>::value>
struct KeepDim;

template <metal::int_ TTargetOuterDim, metal::int_ TOuterDim, metal::int_ TInnerDim, typename TDropSeq>
struct KeepDim<TTargetOuterDim, TOuterDim, TInnerDim, TDropSeq, true>
{
  static const metal::int_ value = KeepDim<TTargetOuterDim, TOuterDim, TInnerDim + 1, TDropSeq>::value;
};

template <metal::int_ TTargetOuterDim, metal::int_ TOuterDim, metal::int_ TInnerDim, typename TDropSeq>
struct KeepDim<TTargetOuterDim, TOuterDim, TInnerDim, TDropSeq, false>
{
  static const metal::int_ value = KeepDim<TTargetOuterDim, TOuterDim + 1, TInnerDim + 1, TDropSeq>::value;
};

template <metal::int_ TTargetOuterDim, metal::int_ TInnerDim, typename TDropSeq>
struct KeepDim<TTargetOuterDim, TTargetOuterDim, TInnerDim, TDropSeq, false>
{
  static const metal::int_ value = TInnerDim;
};

template <metal::int_ TInnerRank, typename TDropSeq, typename TIndices = metal::iota<metal::number<0>, metal::number<TInnerRank - metal::size<TDropSeq>::value>>>
struct KeepSeq;

template <metal::int_ TInnerRank, typename TDropSeq, metal::int_... TIndices>
struct KeepSeq<TInnerRank, TDropSeq, metal::numbers<TIndices...>>
{
  using type = DimSeq<KeepDim<TIndices, 0, 0, TDropSeq>::value...>;
};

template <metal::int_ TInnerRank, typename TDropSeq>
using invert_seq_t = typename KeepSeq<TInnerRank, TDropSeq>::type;

static_assert(std::is_same<metal::numbers<0>, invert_seq_t<2, metal::numbers<1>>>::value, "invert_seq_t not working");
static_assert(std::is_same<metal::numbers<0, 2>, invert_seq_t<3, metal::numbers<1>>>::value, "invert_seq_t not working");



template <typename TInnerDimSeq, typename TKeepSeq>
struct PickDimSeq;

template <typename TInnerDimSeq, metal::int_... TKeepDims>
struct PickDimSeq<TInnerDimSeq, metal::numbers<TKeepDims...>>
{
  using type = DimSeq<nth_dimension_v<TKeepDims, TInnerDimSeq>::value...>;
};

template <typename TInnerDimSeq, typename TKeepSeq>
using pick_dimseq_t = typename PickDimSeq<TInnerDimSeq, TKeepSeq>::type;

static_assert(are_compatible_dimseqs_v<DimSeq<4>, pick_dimseq_t<DimSeq<3, 4>, metal::numbers<1>>>::value, "pick_dimseq_t not working");



template <typename TKeepSeq>
struct PickDims;

template <metal::int_... TKeepDims>
struct PickDims<metal::numbers<TKeepDims...>>
{
  template <typename TInnerTensor>
  __host__ __device__
  static VectorXs<sizeof...(TKeepDims)> pick(TInnerTensor&& inner_tensor)
  {
    return VectorXs<sizeof...(TKeepDims)>(inner_tensor.template dim<TKeepDims>()...);
  }
};





template <metal::int_ N, metal::int_ TDropDim, metal::int_ TKeepDim, typename TDropSeq,
  bool TDimIsDropped = metal::contains<TDropSeq, metal::number<TDropDim + TKeepDim>>::value>
struct PartialCoordinateHelper;

template <metal::int_ N, metal::int_ TDropDim, metal::int_ TKeepDim, typename TDropSeq>
struct PartialCoordinateHelper<N, TDropDim, TKeepDim, TDropSeq, true>
{
  template <typename TKeepCoordVector, typename... TCoordArgTypes>
  __host__ __device__
  static dim_t coord(TKeepCoordVector&& keep_coords, TCoordArgTypes&&... drop_coords)
  {
    return PartialCoordinateHelper<N - 1, TDropDim + 1, TKeepDim, TDropSeq>::coord(util::forward<TKeepCoordVector>(keep_coords), util::forward<TCoordArgTypes>(drop_coords)...);
  }
};

template <metal::int_ N, metal::int_ TDropDim, metal::int_ TKeepDim, typename TDropSeq>
struct PartialCoordinateHelper<N, TDropDim, TKeepDim, TDropSeq, false>
{
  template <typename TKeepCoordVector, typename... TCoordArgTypes>
  __host__ __device__
  static dim_t coord(TKeepCoordVector&& keep_coords, TCoordArgTypes&&... drop_coords)
  {
    return PartialCoordinateHelper<N - 1, TDropDim, TKeepDim + 1, TDropSeq>::coord(util::forward<TKeepCoordVector>(keep_coords), util::forward<TCoordArgTypes>(drop_coords)...);
  }
};

template <metal::int_ TDropDim, metal::int_ TKeepDim, typename TDropSeq>
struct PartialCoordinateHelper<0, TDropDim, TKeepDim, TDropSeq, true>
{
  template <typename TKeepCoordVector, typename... TCoordArgTypes>
  __host__ __device__
  static dim_t coord(TKeepCoordVector&& keep_coords, TCoordArgTypes&&... drop_coords)
  {
    return getNthCoordinate<TDropDim>(util::forward<TCoordArgTypes>(drop_coords)...);
  }
};

template <metal::int_ TDropDim, metal::int_ TKeepDim, typename TDropSeq>
struct PartialCoordinateHelper<0, TDropDim, TKeepDim, TDropSeq, false>
{
  template <typename TKeepCoordVector, typename... TCoordArgTypes>
  __host__ __device__
  static dim_t coord(TKeepCoordVector&& keep_coords, TCoordArgTypes&&... drop_coords)
  {
    return getNthCoordinate<TKeepDim>(util::forward<TKeepCoordVector>(keep_coords));
  }
};





template <metal::int_ TFirst, typename TRestSeq>
struct PartialMaxHelper;

template <metal::int_ TFirst, metal::int_... TRest>
struct PartialMaxHelper<TFirst, metal::numbers<TRest...>>
{
  static const metal::int_ value = math::max(TFirst, (TRest + 1)...);
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
    , m_keep_coords(toCoordVector<metal::size<KeepSeq>::value>(util::forward<TCoordArgTypes>(keep_coords)...))
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

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_tensor.template dim<metal::at<TDropSeq, metal::number<TIndex>>::value>();
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return m_tensor.dim(getNthDimension(index, TDropSeq()));
  }

private:
  TTensorTypeIn m_tensor;
  VectorXs<metal::size<KeepSeq>::value> m_keep_coords;

public:
  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static auto getElement(TThisType&& self, metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
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

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_tensor.template dim<metal::at<TKeepSeq, metal::number<TIndex>>::value>();
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return m_tensor.dim(getNthDimension(index, TKeepSeq()));
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

template <metal::int_... TDropDims, typename TOtherTensorType>
__host__ __device__
auto partial(TOtherTensorType&& tensor)
RETURN_AUTO(partial<metal::numbers<TDropDims...>>
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

} // end of ns template_tensors
