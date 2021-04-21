namespace template_tensors {

template <size_t TRank2 = DYN, typename TTensorType, typename TCompareFunctor, size_t TRank = TRank2 != DYN ? TRank2 : non_trivial_dimensions_num_v<TTensorType>::value>
__host__ __device__
VectorXs<TRank> argcomp(TTensorType&& tensor, TCompareFunctor compare)
{
  VectorXs<TRank> best_pos(0);
  decay_elementtype_t<TTensorType> best_el = tensor();
  template_tensors::for_each<TRank>([&](VectorXs<TRank> pos, const decay_elementtype_t<TTensorType>& el){
    if (compare(el, best_el))
    {
      best_el = el;
      best_pos = pos;
    }
  }, util::forward<TTensorType>(tensor));
  return best_pos;
}

template <size_t TRank2, typename TTensorType, size_t TRank = TRank2 != DYN ? TRank2 : non_trivial_dimensions_num_v<TTensorType>::value>
__host__ __device__
VectorXs<TRank> argmax(TTensorType&& tensor)
{
  return argcomp<TRank>(util::forward<TTensorType>(tensor), math::functor::gt());
}

namespace functor {
  template <size_t TRank = DYN>
  struct argmax
  {
    template <typename TTensorType>
    __host__ __device__
    auto operator()(TTensorType&& t) const
    RETURN_AUTO(template_tensors::argmax<TRank>(util::forward<TTensorType>(t)))
  };
}

template <size_t TRank2, typename TTensorType, size_t TRank = TRank2 != DYN ? TRank2 : non_trivial_dimensions_num_v<TTensorType>::value>
__host__ __device__
VectorXs<TRank> argmin(TTensorType&& tensor)
{
  return argcomp<TRank>(util::forward<TTensorType>(tensor), math::functor::lt());
}

namespace functor {
  template <size_t TRank = DYN>
  struct argmin
  {
    template <typename TTensorType>
    __host__ __device__
    auto operator()(TTensorType&& t) const
    RETURN_AUTO(template_tensors::argmin<TRank>(util::forward<TTensorType>(t)))
  };
}

} // end of ns tensor