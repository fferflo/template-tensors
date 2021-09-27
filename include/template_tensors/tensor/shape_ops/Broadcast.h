namespace template_tensors {

#define ThisType BroadcastingTensor<TTensorTypeIn, TDimSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TTensorTypeIn>::value, \
                                        TDimSeq \
                              >

template <typename TTensorTypeIn, typename TDimSeq>
class BroadcastingTensor : public SuperType, public StoreDimensions<TDimSeq>
{
public:
  static_assert(is_tensor_v<TTensorTypeIn>::value, "TTensorTypeIn must be a tensor");

  template <typename... TDimArgTypes>
  __host__ __device__
  BroadcastingTensor(TTensorTypeIn tensor, TDimArgTypes&&... dim_args)
    : SuperType(std::forward<TDimArgTypes>(dim_args)...)
    , StoreDimensions<TDimSeq>(std::forward<TDimArgTypes>(dim_args)...)
    , m_tensor(tensor)
  {
  }

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes, metal::int_... TIndices>
  __host__ __device__
  static auto getElement(TThisType&& self, metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_tensor((getNthCoordinate<TIndices>(std::forward<TCoordArgTypes>(coords)...) % self.m_tensor.template dim<TIndices>())...)
  )
  TT_TENSOR_SUBCLASS_FORWARD_ELEMENT_ACCESS_SEQ(getElement)

private:
  TTensorTypeIn m_tensor;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(BroadcastingTensor<decltype(transform(m_tensor)), TDimSeq>(transform(m_tensor), this->dims()))

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(BroadcastingTensor<decltype(transform(m_tensor)), TDimSeq>(transform(m_tensor), this->dims()))
};
#undef SuperType
#undef ThisType



/*!
 * \defgroup BroadcastingTensor Broadcasting
 * \ingroup TensorOperations
 * \brief Replicate a given tensor in different directions.
 *
 * Broadcasting a tensor concatenates it with itself in the given dimensions. Example: A column vector that
 * is broadcast in the second dimension results in a matrix where every column is equal to that vector.
 * @{
 */

/*!
 * \brief Broadcasts the given tensor resulting in the new tensor dimensions.
 *
 * @param tensor the tensor to be broadcast
 * @param dim_args... the new run-time dimensions of the broadcast tensor
 * @tparam TBroadcastedDims... the new compile-time dimensions of the broadcast tensor
 * @return the broadcast tensor
 */
template <metal::int_... TBroadcastedDims, typename... TDimArgTypes, typename TOtherTensorType, ENABLE_IF(is_tensor_v<TOtherTensorType>::value && sizeof...(TBroadcastedDims) != 0)>
__host__ __device__
auto broadcast(TOtherTensorType&& tensor, TDimArgTypes&&... dim_args)
RETURN_AUTO(BroadcastingTensor<TOtherTensorType, DimSeq<TBroadcastedDims...>>
  (std::forward<TOtherTensorType>(tensor), std::forward<TDimArgTypes>(dim_args)...)
);

template <metal::int_... TBroadcastedDims, typename... TDimArgTypes, typename TOtherTensorType, ENABLE_IF(is_tensor_v<TOtherTensorType>::value && sizeof...(TBroadcastedDims) == 0)>
__host__ __device__
auto broadcast(TOtherTensorType&& tensor, TDimArgTypes&&... dim_args)
RETURN_AUTO(BroadcastingTensor<TOtherTensorType, dyn_dimseq_t<dimension_num_v<TDimArgTypes...>::value>>
  (std::forward<TOtherTensorType>(tensor), std::forward<TDimArgTypes>(dim_args)...)
);

template <typename TBroadcastedDimSeq, typename... TDimArgTypes, typename TOtherTensorType, ENABLE_IF(is_tensor_v<TOtherTensorType>::value)>
__host__ __device__
auto broadcast(TOtherTensorType&& tensor, TDimArgTypes&&... dim_args)
RETURN_AUTO(BroadcastingTensor<TOtherTensorType, TBroadcastedDimSeq>
  (std::forward<TOtherTensorType>(tensor), std::forward<TDimArgTypes>(dim_args)...)
)

/*!
 * \brief Broadcasts the given single element resulting in a tensor with the same value in all locations
 *
 * @param singleton the element to be broadcast
 * @param dim_args... the new run-time dimensions of the broadcast tensor
 * @tparam TBroadcastedDims... the new compile-time dimensions of the broadcast tensor
 * @return the broadcast tensor
 */
template <metal::int_... TBroadcastedDims, typename... TDimArgTypes, typename TSingletonType, ENABLE_IF(!is_tensor_v<TSingletonType>::value && sizeof...(TBroadcastedDims) != 0)>
__host__ __device__
auto broadcast(TSingletonType&& singleton, TDimArgTypes&&... dim_args)
RETURN_AUTO(BroadcastingTensor<SingletonT<typename std::decay<TSingletonType>::type>, DimSeq<TBroadcastedDims...>>
  (SingletonT<typename std::decay<TSingletonType>::type>(std::forward<TSingletonType>(singleton)), std::forward<TDimArgTypes>(dim_args)...)
)

template <metal::int_... TBroadcastedDims, typename... TDimArgTypes, typename TSingletonType, ENABLE_IF(!is_tensor_v<TSingletonType>::value && sizeof...(TBroadcastedDims) == 0)>
__host__ __device__
auto broadcast(TSingletonType&& singleton, TDimArgTypes&&... dim_args)
RETURN_AUTO(BroadcastingTensor<SingletonT<typename std::decay<TSingletonType>::type>, dyn_dimseq_t<dimension_num_v<TDimArgTypes...>::value>>
  (SingletonT<typename std::decay<TSingletonType>::type>(std::forward<TSingletonType>(singleton)), std::forward<TDimArgTypes>(dim_args)...)
)

template <typename TBroadcastedDimSeq, typename... TDimArgTypes, typename TSingletonType, ENABLE_IF(!is_tensor_v<TSingletonType>::value)>
__host__ __device__
auto broadcast(TSingletonType&& singleton, TDimArgTypes&&... dim_args)
RETURN_AUTO(BroadcastingTensor<SingletonT<typename std::decay<TSingletonType>::type>, TBroadcastedDimSeq>
  (SingletonT<typename std::decay<TSingletonType>::type>(std::forward<TSingletonType>(singleton)), std::forward<TDimArgTypes>(dim_args)...)
)
// TODO: singletons are decayed everywhere in this file, this shouldnt be the case
template <typename TTensorType, typename TElementType>
__host__ __device__
void fill(TTensorType&& tensor, TElementType&& fill)
{
  tensor = broadcast<dimseq_t<TTensorType>>(template_tensors::singleton(std::forward<TElementType>(fill)), tensor.dims());
}

/*!
 * @}
 */

} // end of ns template_tensors
