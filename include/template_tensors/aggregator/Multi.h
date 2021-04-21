#pragma once

namespace aggregator {

namespace detail {

template <typename... TAggregators>
class multi : public aggregator::IsAggregator
{
private:
  ::tuple::Tuple<TAggregators...> m_aggregators;

  struct Getter
  {
    template <typename TIn>
    __host__ __device__
    auto operator()(TIn&& in)
    RETURN_AUTO(in.get())
  };

public:
  __host__ __device__
  multi(TAggregators&&... aggregators)
    : m_aggregators(::tuple::Tuple<TAggregators...>(util::forward<TAggregators>(aggregators)...))
  {
  }

  __host__ __device__
  multi()
  {
  }

  template <typename... TInput>
  __host__ __device__
  void operator()(TInput&&... input)
  {
    ::tuple::for_each(util::functor::apply_to_all(util::forward<TInput>(input)...), m_aggregators);
  }

  __host__ __device__
  auto get() const
  RETURN_AUTO(::tuple::map(Getter(), m_aggregators))
};

} // end of ns detail

template <typename... TAggregators>
__host__ __device__
auto multi(TAggregators&&... aggregators)
RETURN_AUTO(detail::multi<util::store_member_t<TAggregators&&>...>(util::forward<TAggregators>(aggregators)...))

} // end of ns aggregator
