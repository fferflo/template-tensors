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
    : SuperType(util::forward<TDimArgTypes>(dim_args)...)
    , StoreDimensions<TDimSeq>(util::forward<TDimArgTypes>(dim_args)...)
    , m_generator(generator)
    , m_distribution(distribution)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_distribution(self.m_generator)
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

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
RETURN_AUTO(RandomTensor<util::store_member_t<TGenerator&&>, util::store_member_t<TDistribution&&>, DimSeq<TDims...>>
  (util::forward<TGenerator>(generator), util::forward<TDistribution>(distribution)));

template <typename TGenerator, typename TDistribution, typename... TDimArgTypes, ENABLE_IF(sizeof...(TDimArgTypes) != 0 && are_dim_args_v<TDimArgTypes...>::value)>
__host__ __device__
auto random(TGenerator&& generator, TDistribution&& distribution, TDimArgTypes&&... dim_args)
RETURN_AUTO(RandomTensor<util::store_member_t<TGenerator&&>, util::store_member_t<TDistribution&&>, dyn_dimseq_t<dimension_num_v<TDimArgTypes...>::value>>
  (util::forward<TGenerator>(generator), util::forward<TDistribution>(distribution), util::forward<TDimArgTypes>(dim_args)...));

template <typename TDimSeq, typename TGenerator, typename TDistribution>
__host__ __device__
auto random(TGenerator&& generator, TDistribution&& distribution)
RETURN_AUTO(RandomTensor<util::store_member_t<TGenerator&&>, util::store_member_t<TDistribution&&>, TDimSeq>
  (util::forward<TGenerator>(generator), util::forward<TDistribution>(distribution)));

} // end of ns template_tensors
