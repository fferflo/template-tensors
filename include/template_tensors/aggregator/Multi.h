#pragma once

#include <jtuple/tuple.hpp>
#include <jtuple/tuple_utility.hpp>

namespace aggregator {

namespace detail {

template <typename... TAggregators>
class multi : public aggregator::IsAggregator
{
private:
  jtuple::tuple<TAggregators...> m_aggregators;

  struct Getter // TODO: replace with bind
  {
    template <typename TIn>
    __host__ __device__
    auto operator()(TIn&& in)
    RETURN_AUTO(in.get())
  };

public:
  __host__ __device__
  multi(TAggregators&&... aggregators)
    : m_aggregators(jtuple::tuple<TAggregators...>(std::forward<TAggregators>(aggregators)...))
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
    jtuple::tuple_for_each([&](auto&& aggregator){aggregator(input...);}, m_aggregators);
  }

  __host__ __device__
  auto get() const
  RETURN_AUTO(jtuple::tuple_map(Getter(), m_aggregators))
};

} // end of ns detail

template <typename... TAggregators>
__host__ __device__
auto multi(TAggregators&&... aggregators)
RETURN_AUTO(detail::multi<util::store_member_t<TAggregators&&>...>(std::forward<TAggregators>(aggregators)...))

} // end of ns aggregator
