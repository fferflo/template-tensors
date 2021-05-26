#pragma once

#include <jtuple/tuple.hpp>
#include <jtuple/tuple_utility.hpp>

namespace aggregator {

namespace detail {

template <typename TFunctor, typename TAggregator>
class map_input : public aggregator::IsAggregator
{
private:
  jtuple::tuple<TFunctor, TAggregator> m_functor_and_aggregator;

public:
  template <typename TFunctor2, typename TAggregator2>
  __host__ __device__
  map_input(TFunctor2&& functor, TAggregator2&& aggregator)
    : m_functor_and_aggregator(std::forward<TFunctor2>(functor), std::forward<TAggregator2>(aggregator))
  {
  }

  __host__ __device__
  map_input()
  {
  }

  template <typename... TInput>
  __host__ __device__
  void operator()(TInput&&... input)
  {
    jtuple::get<1>(m_functor_and_aggregator)(jtuple::get<0>(m_functor_and_aggregator)(std::forward<TInput>(input)...));
  }

  __host__ __device__
  auto get() const
  RETURN_AUTO(jtuple::get<1>(m_functor_and_aggregator).get())
};

} // end of ns detail

template <typename TFunctor, typename TAggregator>
__host__ __device__
auto map_input(TFunctor&& functor, TAggregator&& aggregator)
RETURN_AUTO(detail::map_input<util::store_member_t<TFunctor&&>, util::store_member_t<TAggregator&&>>(std::forward<TFunctor>(functor), std::forward<TAggregator>(aggregator)))

} // end of ns aggregator
