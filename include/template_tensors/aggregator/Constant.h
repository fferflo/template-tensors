#pragma once

namespace aggregator {

template <typename TResultType>
class constant : public aggregator::IsAggregator
{
public:
  __host__ __device__
  constant(TResultType constant)
    : m_constant(constant)
  {
  }

  __host__ __device__
  constant()
  {
  }

  template <typename... TInput>
  __host__ __device__
  void operator()(TInput&&... input)
  {
  }

  __host__ __device__
  TResultType get() const
  {
    return m_constant;
  }

private:
  TResultType m_constant;
};

} // end of ns aggregator
