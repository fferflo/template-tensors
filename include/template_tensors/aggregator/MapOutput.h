#pragma once

namespace aggregator {

namespace detail {

template <typename TFunctor, typename TAggregator>
class map_output : public aggregator::IsAggregator
{
private:
  ::tuple::CompressedPair<TFunctor, TAggregator> m_functor_and_aggregator;

public:
  template <typename TFunctor2, typename TAggregator2>
  __host__ __device__
  map_output(TFunctor2&& functor, TAggregator2&& aggregator)
    : m_functor_and_aggregator(util::forward<TFunctor2>(functor), util::forward<TAggregator2>(aggregator))
  {
  }

  template <typename TDummy = void>
  __host__ __device__
  map_output()
  {
  }

  template <typename... TInput>
  __host__ __device__
  void operator()(TInput&&... input)
  {
    m_functor_and_aggregator.second()(util::forward<TInput>(input)...);
  }

  template <typename TDummy = void>
  __host__ __device__
  auto get()
  RETURN_AUTO(m_functor_and_aggregator.first()(m_functor_and_aggregator.second().get()))

  template <typename TDummy = void>
  __host__ __device__
  auto get() const
  RETURN_AUTO(m_functor_and_aggregator.first()(m_functor_and_aggregator.second().get()))
};

} // end of ns detail

template <typename TFunctor, typename TAggregator>
__host__ __device__
auto map_output(TFunctor&& functor, TAggregator&& aggregator)
RETURN_AUTO(detail::map_output<util::store_member_t<TFunctor&&>, util::store_member_t<TAggregator&&>>(util::forward<TFunctor>(functor), util::forward<TAggregator>(aggregator)))

template <typename TFunctor, typename TAggregator1, typename TAggregator2, typename... TAggregatorRest>
__host__ __device__
auto map_output(TFunctor&& functor, TAggregator1&& aggregator1, TAggregator2&& aggregator2, TAggregatorRest&&... rest)
RETURN_AUTO(map_output(
  ::tuple::functor::for_all(util::forward<TFunctor>(functor)),
  multi(util::forward<TAggregator1>(aggregator1), util::forward<TAggregator2>(aggregator2), util::forward<TAggregatorRest>(rest)...)
))

} // end of ns aggregator
