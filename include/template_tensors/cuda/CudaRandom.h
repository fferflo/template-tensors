#pragma once

#ifdef __CUDACC__

#include "Cuda.h"
#include <curand_kernel.h>

namespace cuda {

namespace device {

template <typename TReal>
class uniform_real_distribution01;

template <>
class uniform_real_distribution01<float>
{
public:
  template <typename TGenerator>
  __device__
  float operator()(TGenerator&& generator)
  {
    return curand_uniform(&generator.getState());
  }
};

template <>
class uniform_real_distribution01<double>
{
public:
  template <typename TGenerator>
  __device__
  double operator()(TGenerator&& generator)
  {
    return curand_uniform_double(&generator.getState());
  }
};

template <typename TReal>
class uniform_real_distribution;

template <>
class uniform_real_distribution<float>
{
public:
  __host__ __device__
  uniform_real_distribution(float start, float end)
    : m_range(end - start)
    , m_end(end)
  {
  }

  template <typename TGenerator>
  __device__
  float operator()(TGenerator&& generator)
  {
    return m_end - m_range * curand_uniform(&generator.getState());
  }

private:
  float m_range;
  float m_end;
};

template <>
class uniform_real_distribution<double>
{
public:
  __host__ __device__
  uniform_real_distribution(double start, double end)
    : m_range(end - start)
    , m_end(end)
  {
  }

  template <typename TGenerator>
  __device__
  double operator()(TGenerator&& generator)
  {
    return m_end - m_range * curand_uniform(&generator.getState());
  }

private:
  double m_range;
  double m_end;
};

template <typename TInt, typename TGenerateFloatingType = float>
class uniform_int_distribution
{
public:
  __host__ __device__
  uniform_int_distribution(TInt start, TInt end = std::numeric_limits<TInt>::max())
    : m_uniform_real_distribution(start, end)
  {
  }

  template <typename TGenerator>
  __device__
  TInt operator()(TGenerator&& generator)
  {
    return static_cast<TInt>(m_uniform_real_distribution(std::forward<TGenerator>(generator)));
  }

private:
  uniform_real_distribution<TGenerateFloatingType> m_uniform_real_distribution;
};

template <typename TReal>
class normal_distribution;

template <>
class normal_distribution<float>
{
public:
  template <typename TGenerator>
  __device__
  float operator()(TGenerator&& generator)
  {
    return curand_normal(&generator.getState());
  }
};

template <>
class normal_distribution<double>
{
public:
  template <typename TGenerator>
  __device__
  double operator()(TGenerator&& generator)
  {
    return curand_normal_double(&generator.getState());
  }
};

template <typename TReal>
class log_normal_distribution;

template <>
class log_normal_distribution<float>
{
public:
  __host__ __device__
  log_normal_distribution(float mean = 0.0, float stddev = 1.0)
    : m_mean(mean)
    , m_stddev(stddev)
  {
  }

  template <typename TGenerator>
  __device__
  float operator()(TGenerator&& generator)
  {
    return curand_log_normal(&generator.getState(), m_mean, m_stddev);
  }

private:
  float m_mean;
  float m_stddev;
};

template <>
class log_normal_distribution<double>
{
public:
  __host__ __device__
  log_normal_distribution(double mean = 0.0, double stddev = 1.0)
    : m_mean(mean)
    , m_stddev(stddev)
  {
  }

  template <typename TGenerator>
  __device__
  double operator()(TGenerator&& generator)
  {
    return curand_log_normal_double(&generator.getState(), m_mean, m_stddev);
  }

private:
  double m_mean;
  double m_stddev;
};

class poisson_distribution
{
public:
  __host__ __device__
  poisson_distribution(double lambda = 1.0)
    : m_lambda(lambda)
  {
  }

  template <typename TGenerator>
  __device__
  auto operator()(TGenerator&& generator)
  {
    return curand_poisson(&generator.getState(), m_lambda);
  }

private:
  float m_lambda;
};





class XORWOW_generator
{
public:
  __device__
  XORWOW_generator(unsigned long long seed, unsigned long long subsequence, unsigned long long offset = 0)
  {
    curand_init(seed, subsequence, offset, &m_state);
  }

  __device__
  XORWOW_generator(unsigned long long seed)
    : XORWOW_generator(seed, cuda::grid::flatten::thread_id_in_grid())
  {
  }

  __device__
  XORWOW_generator()
    : XORWOW_generator(clock())
  {
  }

  __host__ __device__
  curandState_t& getState()
  {
    return m_state;
  }

private:
  curandState_t m_state;
};

class Philox4_32_10_generator
{
public:
  __device__
  Philox4_32_10_generator(unsigned long long seed, unsigned long long subsequence, unsigned long long offset = 0)
  {
    curand_init(seed, subsequence, offset, &m_state);
  }

  __device__
  Philox4_32_10_generator(unsigned long long seed)
    : Philox4_32_10_generator(seed, cuda::grid::flatten::thread_id_in_grid())
  {
  }

  __device__
  Philox4_32_10_generator()
    : Philox4_32_10_generator(clock())
  {
  }

  __host__ __device__
  curandStatePhilox4_32_10_t& getState()
  {
    return m_state;
  }

private:
  curandStatePhilox4_32_10_t m_state;
};

} // end of ns device

} // end of ns cuda

#endif
