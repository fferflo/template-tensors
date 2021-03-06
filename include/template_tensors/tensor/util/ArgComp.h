namespace template_tensors {

template <metal::int_ TRank2 = DYN, typename TTensorType, typename TCompareFunctor, metal::int_ TRank = TRank2 != DYN ? TRank2 : non_trivial_dimensions_num_v<TTensorType>::value>
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
  }, std::forward<TTensorType>(tensor));
  return best_pos;
}

template <metal::int_ TRank2, typename TTensorType, metal::int_ TRank = TRank2 != DYN ? TRank2 : non_trivial_dimensions_num_v<TTensorType>::value>
__host__ __device__
VectorXs<TRank> argmax(TTensorType&& tensor)
{
  return argcomp<TRank>(std::forward<TTensorType>(tensor), math::functor::gt());
}

namespace functor {
  template <metal::int_ TRank = DYN>
  struct argmax
  {
    template <typename TTensorType>
    __host__ __device__
    auto operator()(TTensorType&& t) const
    RETURN_AUTO(template_tensors::argmax<TRank>(std::forward<TTensorType>(t)))
  };
}

template <metal::int_ TRank2, typename TTensorType, metal::int_ TRank = TRank2 != DYN ? TRank2 : non_trivial_dimensions_num_v<TTensorType>::value>
__host__ __device__
VectorXs<TRank> argmin(TTensorType&& tensor)
{
  return argcomp<TRank>(std::forward<TTensorType>(tensor), math::functor::lt());
}

namespace functor {
  template <metal::int_ TRank = DYN>
  struct argmin
  {
    template <typename TTensorType>
    __host__ __device__
    auto operator()(TTensorType&& t) const
    RETURN_AUTO(template_tensors::argmin<TRank>(std::forward<TTensorType>(t)))
  };
}

} // end of ns template_tensors
