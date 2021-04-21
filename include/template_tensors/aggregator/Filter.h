#pragma once

namespace aggregator {

namespace detail {

template <typename TPredicate, typename TAggregator>
class filter : public aggregator::IsAggregator
{
private:
  ::tuple::CompressedPair<TPredicate, TAggregator> m_predicate_and_aggregator;

public:
  template <typename TPredicate2, typename TAggregator2>
  __host__ __device__
  filter(TPredicate2&& predicate, TAggregator2&& aggregator)
    : m_predicate_and_aggregator(util::forward<TPredicate2>(predicate), util::forward<TAggregator2>(aggregator))
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
    if (m_predicate_and_aggregator.first()(util::forward<TInput>(input)...))
    {
      m_predicate_and_aggregator.second()(util::forward<TInput>(input)...);
    }
  }

  __host__ __device__
  auto get() const
  RETURN_AUTO(m_predicate_and_aggregator.second().get())
};

} // end of ns detail

template <typename TPredicate, typename TAggregator>
__host__ __device__
auto filter(TPredicate&& predicate, TAggregator&& aggregator)
RETURN_AUTO(detail::filter<util::store_member_t<TPredicate&&>, util::store_member_t<TAggregator&&>>(util::forward<TPredicate>(predicate), util::forward<TAggregator>(aggregator)))


} // end of ns aggregator
