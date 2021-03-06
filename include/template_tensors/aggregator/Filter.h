#pragma once

#include <jtuple/tuple.hpp>
#include <jtuple/tuple_utility.hpp>

namespace aggregator {

namespace detail {

template <typename TPredicate, typename TAggregator>
class filter : public aggregator::IsAggregator
{
private:
  jtuple::tuple<TPredicate, TAggregator> m_predicate_and_aggregator;

public:
  template <typename TPredicate2, typename TAggregator2>
  __host__ __device__
  filter(TPredicate2&& predicate, TAggregator2&& aggregator)
    : m_predicate_and_aggregator(std::forward<TPredicate2>(predicate), std::forward<TAggregator2>(aggregator))
  {
  }

  __host__ __device__
  filter()
  {
  }

  HD_WARNING_DISABLE
  template <typename... TInput>
  __host__ __device__
  void operator()(TInput&&... input)
  {
    if (jtuple::get<0>(m_predicate_and_aggregator)(std::forward<TInput>(input)...))
    {
      jtuple::get<1>(m_predicate_and_aggregator)(std::forward<TInput>(input)...);
    }
  }

  __host__ __device__
  auto get() const
  RETURN_AUTO(jtuple::get<1>(m_predicate_and_aggregator).get())
};

} // end of ns detail

template <typename TPredicate, typename TAggregator>
__host__ __device__
auto filter(TPredicate&& predicate, TAggregator&& aggregator)
RETURN_AUTO(detail::filter<TPredicate, TAggregator>(std::forward<TPredicate>(predicate), std::forward<TAggregator>(aggregator)))


} // end of ns aggregator
