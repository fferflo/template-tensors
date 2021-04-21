#pragma once

namespace aggregator {

template <typename TResultType, typename TCounter = size_t>
__host__ __device__
auto mean_offline(TResultType initial_value = 0, TCounter initial_count = 0)
RETURN_AUTO(sum(util::forward<TResultType>(initial_value)) / count(util::forward<TCounter>(initial_count)))

namespace weighted {
template <typename TResultType, typename TCounter = TResultType>
__host__ __device__
auto mean_offline(TResultType initial_value = 0, TCounter initial_count = 0)
RETURN_AUTO(aggregator::weighted::sum<TResultType>(util::forward<TResultType>(initial_value))
  / aggregator::weighted::count<TCounter>(util::forward<TCounter>(initial_count)))
} // end of ns weighted

namespace detail {

template <typename TResultType, typename TCounter>
class mean_online : public aggregator::IsAggregator
{
public:
  __host__ __device__
  mean_online(TResultType initial_value, TCounter initial_counter)
    : m_mean(initial_value)
    , m_num(initial_counter)
  {
  }

  __host__ __device__
  mean_online()
  {
  }

  template <typename TInput>
  __host__ __device__
  void operator()(TInput&& in)
  {
    m_num++;
    m_mean = m_mean + (in - m_mean) / m_num;
  }

  __host__ __device__
  TResultType get() const
  {
    return m_mean;
  }

private:
  TResultType m_mean;
  TCounter m_num;
};

} // end of ns detail

template <typename TResultType, typename TCounter = size_t>
__host__ __device__
auto mean_online(TResultType initial_value = 0, TCounter initial_count = 0)
RETURN_AUTO(detail::mean_online<TResultType, TCounter>(
  initial_value, initial_count
))



namespace detail {

template <typename TResultType, typename TCounter = size_t>
class variance_online
{
public:
  __host__ __device__
  variance_online(TResultType value_zero = 0, TCounter counter_zero = 0)
    : m_mean(value_zero)
    , m_m2(value_zero)
    , m_num(counter_zero)
  {
  }

  template <typename TInput>
  __host__ __device__
  void operator()(TInput&& t)
  {
    m_num++;
    TResultType delta = t - m_mean;
    m_mean += delta / m_num;
    TResultType delta2 = t - m_mean;
    m_m2 += delta * delta2;
  }

  __host__ __device__
  TResultType get() const
  {
    return getVariance();
  }

  __host__ __device__
  TResultType getMean() const
  {
    return m_mean;
  }

  __host__ __device__
  TResultType getStandardDeviation() const
  {
    return math::sqrt(getVariance());
  }

  __host__ __device__
  TResultType getVariance() const
  {
    return m_m2 / m_num;
  }

private:
  TResultType m_mean;
  TResultType m_m2;
  TCounter m_num;
};



template <typename TResultType, typename TCounter = size_t>
class variance_offline
{
public:
  __host__ __device__
  variance_offline(TResultType mean, TResultType value_zero = 0, TCounter counter_zero = 0)
    : m_mean(mean)
    , m_sum(value_zero)
    , m_num(counter_zero)
  {
  }

  __host__ __device__
  variance_offline()
  {
  }

  template <typename TInput>
  __host__ __device__
  void operator()(TInput&& t)
  {
    m_sum += math::squared(t - m_mean);
    m_num++;
  }

  __host__ __device__
  TResultType get() const
  {
    return m_sum / m_num;
  }

private:
  TResultType m_mean;
  TResultType m_sum;
  TCounter m_num;
};

} // end of ns detail

template <typename TResultType, typename TCounter = size_t>
__host__ __device__
auto variance_online(TResultType initial_value = 0, TCounter initial_count = 0)
RETURN_AUTO(detail::variance_online<TResultType, TCounter>(
  initial_value, initial_count
))

template <typename TResultType, typename TCounter = size_t>
__host__ __device__
auto variance_offline(TResultType mean, TResultType initial_value = 0, TCounter initial_count = 0)
RETURN_AUTO(detail::variance_offline<TResultType, TCounter>(
  mean, initial_value, initial_count
))



namespace detail {

template <typename TResultType>
class exponential_moving_average
{
public:
  exponential_moving_average(TResultType next_weight, TResultType initial_value)
    : m_aggregated(initial_value)
    , m_next_weight(next_weight)
  {
  }

  exponential_moving_average()
  {
  }

  exponential_moving_average<TResultType>& operator=(TResultType new_value)
  {
    m_aggregated = new_value;
    return *this;
  }

  template <typename TInput>
  void operator()(TInput&& in)
  {
    m_aggregated = m_next_weight * in + (1 - m_next_weight) * m_aggregated;
  }

  __host__ __device__
  TResultType get() const
  {
    return m_aggregated;
  }

private:
  TResultType m_aggregated;
  TResultType m_next_weight;
};

} // end of ns detail

template <typename TResultType>
__host__ __device__
auto exponential_moving_average(TResultType next_weight, TResultType initial_value = 0)
RETURN_AUTO(detail::exponential_moving_average<TResultType>(
  next_weight, initial_value
))

} // end of ns aggregator
