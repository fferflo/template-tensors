#pragma once

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
    m_functor_and_value.first()(m_functor_and_value.second(), util::forward<TInput>(input)...);
  }

  __host__ __device__
  TResultType get() const
  {
    return m_functor_and_value.second();
  }

  __host__ __device__
  TResultType get() const volatile
  {
    return m_functor_and_value.second();
  }

private:
  ::tuple::CompressedPair<TFunctor, TResultType> m_functor_and_value;
};

} // end of ns detail

template <typename TResultType, typename TFunctor>
__host__ __device__
auto assign(TFunctor&& functor, TResultType initial_value)
RETURN_AUTO(detail::assign<typename std::decay<TResultType>::type, util::store_member_t<TFunctor&&>>(
  util::forward<TFunctor>(functor), initial_value))

} // end of ns aggregator
