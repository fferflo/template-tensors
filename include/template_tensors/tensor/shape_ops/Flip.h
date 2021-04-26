namespace template_tensors {

namespace detail {

template <size_t I, size_t TFlipDim>
struct FlipIndexHelper
{
  __host__ __device__
  static size_t get(size_t index, size_t dim)
  {
    return index;
  }
};

template <size_t I>
struct FlipIndexHelper<I, I>
{
  __host__ __device__
  static size_t get(size_t index, size_t dim)
  {
    return dim - 1 - index;
  }
};

} // end of ns detail

#define ThisType FlipTensor<TTensorTypeIn, TFlipDim>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        dimseq_t<TTensorTypeIn> \
                              >
template <typename TTensorTypeIn, size_t TFlipDim>
class FlipTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  __host__ __device__
  FlipTensor(TTensorTypeIn tensor)
    : SuperType(tensor.dims())
    , m_tensor(tensor)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static auto getElement(TThisType&& self, tmp::vs::Sequence<size_t, TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_tensor(detail::FlipIndexHelper<TIndices, TFlipDim>::get(getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...), self.m_tensor.template dim<TIndices>())...)
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ(getElement)

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    return m_tensor.template dim<TIndex>();
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    return m_tensor.dim(index);
  }

private:
  TTensorTypeIn m_tensor;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(FlipTensor<decltype(transform(m_tensor)), TFlipDim>
    (transform(m_tensor))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(FlipTensor<decltype(transform(m_tensor)), TFlipDim>
    (transform(m_tensor))
  )
};
#undef SuperType
#undef ThisType


template <size_t TFlipDim, typename TOtherTensorType>
__host__ __device__
auto flip(TOtherTensorType&& tensor)
RETURN_AUTO(FlipTensor<util::store_member_t<TOtherTensorType&&>, TFlipDim>
  (util::forward<TOtherTensorType>(tensor))
);

namespace functor {

template <size_t TFlipDim> // TODO: add functors to all tensor class helpers, #define macro for FORWARD_FUNCTOR
struct flip
{
  template <typename TOtherTensorType>
  __host__ __device__
  auto operator()(TOtherTensorType&& tensor) const
  RETURN_AUTO(template_tensors::flip<TFlipDim>(util::forward<TOtherTensorType>(tensor)))
};

} // end of ns functor

} // end of ns tensor
