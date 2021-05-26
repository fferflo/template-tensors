#pragma once

#include <jtuple/tuple.hpp>
#include <jtuple/tuple_utility.hpp>

namespace aggregator {

namespace detail {

template <typename TResultType, typename TFunctor>
class assign : public aggregator::IsAggregator
{
public:
  static_assert(!std::is_reference<TResultType>::value, "Cannot aggregate to reference");

  __host__ __device__
  assign(TFunctor functor, TResultType value)
    : m_functor_and_value(functor, value)
  {
  }

  __host__ __device__
  assign()
  {
  }

  template <typename... TInput>
  __host__ __device__
  void operator()(TInput&&... input)
  {
    jtuple::get<0>(m_functor_and_value)(jtuple::get<1>(m_functor_and_value), std::forward<TInput>(input)...);
  }

  __host__ __device__
  TResultType get() const
  {
    return jtuple::get<1>(m_functor_and_value);
  }

  __host__ __device__
  TResultType get() const volatile
  {
    return jtuple::get<1>(m_functor_and_value);
  }

private:
  jtuple::tuple<TFunctor, TResultType> m_functor_and_value;
};

} // end of ns detail

template <typename TResultType, typename TFunctor>
__host__ __device__
auto assign(TFunctor&& functor, TResultType initial_value)
RETURN_AUTO(detail::assign<typename std::decay<TResultType>::type, TFunctor>(
  std::forward<TFunctor>(functor), initial_value))

} // end of ns aggregator
