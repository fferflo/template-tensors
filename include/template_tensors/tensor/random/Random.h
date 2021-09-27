namespace template_tensors {

#define ThisType RandomTensor<TGenerator, TDistribution, TDimSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        TDimSeq \
                              >
// TODO: add ToDeviceTransform for this, or forbid ToDeviceTransform transformation
template <typename TGenerator, typename TDistribution, typename TDimSeq>
class RandomTensor : public SuperType, public StoreDimensions<TDimSeq>
{
public:
  template <typename... TDimArgTypes>
  __host__ __device__
  RandomTensor(TGenerator generator, TDistribution distribution, TDimArgTypes&&... dim_args)
    : SuperType(std::forward<TDimArgTypes>(dim_args)...)
    , StoreDimensions<TDimSeq>(std::forward<TDimArgTypes>(dim_args)...)
    , m_generator(generator)
    , m_distribution(distribution)
  {
  }

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_distribution(self.m_generator)
  )
  TT_TENSOR_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }

private:
  TGenerator m_generator;
  TDistribution m_distribution;
};
#undef SuperType
#undef ThisType

template <metal::int_... TDims, typename TGenerator, typename TDistribution>
__host__ __device__
auto random(TGenerator&& generator, TDistribution&& distribution)
RETURN_AUTO(RandomTensor<TGenerator, TDistribution, DimSeq<TDims...>>
  (std::forward<TGenerator>(generator), std::forward<TDistribution>(distribution)));

template <typename TGenerator, typename TDistribution, typename... TDimArgTypes, ENABLE_IF(sizeof...(TDimArgTypes) != 0 && are_dim_args_v<TDimArgTypes...>::value)>
__host__ __device__
auto random(TGenerator&& generator, TDistribution&& distribution, TDimArgTypes&&... dim_args)
RETURN_AUTO(RandomTensor<TGenerator, TDistribution, dyn_dimseq_t<dimension_num_v<TDimArgTypes...>::value>>
  (std::forward<TGenerator>(generator), std::forward<TDistribution>(distribution), std::forward<TDimArgTypes>(dim_args)...));

template <typename TDimSeq, typename TGenerator, typename TDistribution>
__host__ __device__
auto random(TGenerator&& generator, TDistribution&& distribution)
RETURN_AUTO(RandomTensor<TGenerator, TDistribution, TDimSeq>
  (std::forward<TGenerator>(generator), std::forward<TDistribution>(distribution)));

} // end of ns template_tensors
