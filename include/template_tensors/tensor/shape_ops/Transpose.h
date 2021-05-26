namespace template_tensors {

#define ThisType TransposedTensor<TTensorTypeIn, TTransposeDims>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        metal::reverse<dimseq_make_length_t<TTensorTypeIn, TTransposeDims>> \
                              >
// TODO: convert this to a general dimension permutation tensor
template <typename TTensorTypeIn, metal::int_ TTransposeDims>
class TransposedTensor : public SuperType
{
public:
  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");
  static_assert(non_trivial_dimensions_num_v<TTensorTypeIn>::value <= TTransposeDims, "TTransposeDims too small");

  __host__ __device__
  TransposedTensor(TTensorTypeIn tensor)
    : SuperType(template_tensors::flip<0>(tensor.template dims<TTransposeDims>()))
    , m_tensor(tensor)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_tensor.template dim<math::lt(TIndex, TTransposeDims) ? TTransposeDims - 1 - TIndex : 1>();
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return m_tensor.dim(index < TTransposeDims ? TTransposeDims - 1 - index : 1);
  }

private:
  TTensorTypeIn m_tensor;

  template <typename TThisType, typename... TCoordArgTypes, metal::int_... TTransposeIndices>
  __host__ __device__
  static auto getHelper(TThisType&& self, metal::numbers<TTransposeIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_tensor(getNthCoordinate<TTransposeDims - 1 - TTransposeIndices>(std::forward<TCoordArgTypes>(coords)...)...)
  )

public:
  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    getHelper(std::forward<TThisType>(self), metal::iota<metal::number<0>, metal::number<TTransposeDims>>(), std::forward<TCoordArgTypes>(coords)...)
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(TransposedTensor<decltype(transform(m_tensor)), TTransposeDims>
    (transform(m_tensor))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(TransposedTensor<decltype(transform(m_tensor)), TTransposeDims>
    (transform(m_tensor))
  )
};
#undef SuperType
#undef ThisType

template <metal::int_ TTransposeDims, typename TOtherTensorType>
__host__ __device__
auto transpose(TOtherTensorType&& tensor)
RETURN_AUTO(TransposedTensor<TOtherTensorType, TTransposeDims>
  (std::forward<TOtherTensorType>(tensor))
);

} // end of ns template_tensors
