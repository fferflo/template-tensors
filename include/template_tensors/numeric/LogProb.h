#pragma once

namespace numeric {

namespace detail {

template <typename TType, bool TIsIntegral = std::is_integral<TType>::value>
struct LogProbZero;

template <typename TType>
struct LogProbZero<TType, true>
{
  static const TType ZERO = static_cast<TType>(-1);

  __host__ __device__
  static bool isZero(TType logprob)
  {
    return logprob == ZERO;
  }
};

template <typename TType>
struct LogProbZero<TType, false>
{
  static constexpr const TType ZERO = std::numeric_limits<TType>::infinity();

  __host__ __device__
  static bool isZero(TType logprob)
  {
    return isinf(logprob);
  }
};

} // end of ns detail

template <typename TType, typename TLogFunctor = ::math::functor::ln, typename TExpFunctor = ::math::functor::exp>
class LogProb
{
public:
  static constexpr const TType ZERO = detail::LogProbZero<TType>::ZERO;

  __host__ __device__
  static LogProb<TType, TLogFunctor, TExpFunctor> fromLogProb(TType logprob)
  {
    return LogProb<TType, TLogFunctor, TExpFunctor>(logprob, 0);
  }

  __host__ __device__
  LogProb()
    : m_logprob(0)
  {
  }

  __host__ __device__
  LogProb(const LogProb<TType, TLogFunctor, TExpFunctor>& other)
    : m_logprob(other.m_logprob)
  {
  }

  __host__ __device__
  LogProb(LogProb<TType, TLogFunctor, TExpFunctor>&& other)
    : m_logprob(util::move(other.m_logprob))
  {
  }

  template <typename TScalar, ENABLE_IF(std::is_floating_point<TScalar>::value || std::is_integral<TScalar>::value)>
  __host__ __device__
  LogProb(TScalar probability)
    : m_logprob(probability == 0 ? ZERO : -TLogFunctor()(probability))
  {
  }

  __host__ __device__
  LogProb<TType, TLogFunctor, TExpFunctor>& operator=(const LogProb<TType, TLogFunctor, TExpFunctor>& other)
  {
    this->m_logprob = other.m_logprob;
    return *this;
  }

  __host__ __device__
  LogProb<TType, TLogFunctor, TExpFunctor>& operator=(LogProb<TType, TLogFunctor, TExpFunctor>&& other)
  {
    this->m_logprob = util::move(other.m_logprob);
    return *this;
  }

  __host__ __device__
  bool isZero() const
  {
    return detail::LogProbZero<TType>::isZero(m_logprob);
  }

  template <typename T>
  __host__ __device__
  explicit operator T() const
  {
    return this->isZero() ? static_cast<T>(0) : TExpFunctor()(-((T) m_logprob));
  }

  __host__ __device__
  LogProb<TType, TLogFunctor, TExpFunctor> operator*(const LogProb<TType, TLogFunctor, TExpFunctor>& other) const
  {
    return fromLogProb((this->isZero() || other.isZero()) ? ZERO : (this->m_logprob + other.m_logprob));
  }

  __host__ __device__
  LogProb<TType, TLogFunctor, TExpFunctor>& operator*=(const LogProb<TType, TLogFunctor, TExpFunctor>& other)
  {
    if (other.isZero())
    {
      this->m_logprob = ZERO;
    }
    else if (!this->isZero())
    {
      this->m_logprob += other.m_logprob;
    }
    return *this;
  }

  __host__ __device__
  LogProb<TType, TLogFunctor, TExpFunctor> operator/(const LogProb<TType, TLogFunctor, TExpFunctor>& other) const
  {
    return fromLogProb((this->isZero() || other.isZero()) ? ZERO : (this->m_logprob - other.m_logprob));
  }

  __host__ __device__
  LogProb<TType, TLogFunctor, TExpFunctor>& operator/=(const LogProb<TType, TLogFunctor, TExpFunctor>& other)
  {
    if (other.isZero())
    { // TODO: this is not allowed
      this->m_logprob = ZERO;
    }
    else if (!this->isZero())
    {
      this->m_logprob -= other.m_logprob;
    }
    return *this;
  }

  __host__ __device__
  bool operator==(const LogProb<TType, TLogFunctor, TExpFunctor>& other) const
  {
    return this->m_logprob == other.m_logprob || (this->isZero() && other.isZero());
  }

  __host__ __device__
  bool operator!=(const LogProb<TType, TLogFunctor, TExpFunctor>& other) const
  {
    return !(*this == other);
  }

  __host__ __device__
  bool operator<(const LogProb<TType, TLogFunctor, TExpFunctor>& other) const
  {
    if (this->isZero())
    {
      return !other.isZero();
    }
    else if (other.isZero())
    {
      return false;
    }
    else
    {
      return this->m_logprob > other.m_logprob;
    }
  }

  __host__ __device__
  bool operator<=(const LogProb<TType, TLogFunctor, TExpFunctor>& other) const
  {
    if (this->isZero())
    {
      return true;
    }
    else if (other.isZero())
    {
      return this->isZero();
    }
    else
    {
      return this->m_logprob >= other.m_logprob;
    }
  }

  __host__ __device__
  bool operator>(const LogProb<TType, TLogFunctor, TExpFunctor>& other) const
  {
    return !(*this <= other);
  }

  __host__ __device__
  bool operator>=(const LogProb<TType, TLogFunctor, TExpFunctor>& other) const
  {
    return !(*this < other);
  }

  template <typename TType2, typename TLogFunctor2, typename TExpFunctor2>
  friend std::ostream& operator<<(std::ostream& stream, const LogProb<TType2, TLogFunctor2, TExpFunctor2>& p);

private:
  TType m_logprob;

  __host__ __device__
  LogProb(TType logprob, int dummy)
    : m_logprob(logprob)
  {
  }
};

template <typename TType, typename TLogFunctor = ::math::functor::ln, typename TExpFunctor = ::math::functor::exp>
std::ostream& operator<<(std::ostream& stream, const LogProb<TType, TLogFunctor, TExpFunctor>& p)
{
  return stream << "LogProb(p=" << static_cast<double>(p) << ", -log(p)=" << p.m_logprob << ")";
}

} // end of ns numeric