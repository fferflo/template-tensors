#pragma once

namespace numeric {

template <typename TInteger, size_t TShifts>
class FixedFloatingPoint
{
public:
  __host__ __device__
  FixedFloatingPoint()
    : m_int(0)
  {
  }

  __host__ __device__
  FixedFloatingPoint(const FixedFloatingPoint<TInteger, TShifts>& other)
    : m_int(other.m_int)
  {
  }

  template <size_t TShifts2, ENABLE_IF(TShifts != TShifts2)>
  __host__ __device__
  FixedFloatingPoint(const FixedFloatingPoint<TInteger, TShifts2>& other)
    : m_int(math::rshift(other.m_int, TShifts2 - TShifts))
  {
  }

  __host__ __device__
  FixedFloatingPoint(FixedFloatingPoint<TInteger, TShifts>&& other)
    : m_int(std::move(other.m_int))
  {
  }

  template <typename TScalar, ENABLE_IF(std::is_integral<TScalar>::value)>
  __host__ __device__
  FixedFloatingPoint(TScalar in)
    : m_int(in << TShifts)
  {
  }

  template <typename TScalar, bool TDummy = true, ENABLE_IF(std::is_floating_point<TScalar>::value)>
  __host__ __device__
  FixedFloatingPoint(TScalar in)
    : m_int(static_cast<TInteger>(in * (1 << TShifts)))
  {
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts>& operator=(const FixedFloatingPoint<TInteger, TShifts>& other)
  {
    this->m_int = other.m_int;
    return *this;
  }

  template <size_t TShifts2, ENABLE_IF(TShifts != TShifts2)>
  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts>& operator=(const FixedFloatingPoint<TInteger, TShifts2>& other)
  {
    this->m_int = math::rshift(other.m_int, TShifts2 - TShifts);
    return *this;
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts>& operator=(FixedFloatingPoint<TInteger, TShifts>&& other)
  {
    this->m_int = std::move(other.m_int);
    return *this;
  }

  template <typename T>
  __host__ __device__
  explicit operator T() const
  {
    return static_cast<T>(m_int) / (1 << TShifts);
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts> operator+(const FixedFloatingPoint<TInteger, TShifts>& other)
  {
    return FixedFloatingPoint<TInteger, TShifts>(this->m_int + other.m_int, 0);
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts> operator+=(const FixedFloatingPoint<TInteger, TShifts>& other)
  {
    this->m_int += other.m_int;
    return *this;
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts> operator-(const FixedFloatingPoint<TInteger, TShifts>& other)
  {
    return FixedFloatingPoint<TInteger, TShifts>(this->m_int - other.m_int, 0);
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts> operator-=(const FixedFloatingPoint<TInteger, TShifts>& other)
  {
    this->m_int -= other.m_int;
    return *this;
  }

  template <size_t TShifts2>
  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts + TShifts2> operator*(const FixedFloatingPoint<TInteger, TShifts2>& other)
  {
    return FixedFloatingPoint<TInteger, TShifts + TShifts2>(this->m_int * other.m_int, 0);
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts + TShifts> operator*(const FixedFloatingPoint<TInteger, TShifts>& other)
  {
    return FixedFloatingPoint<TInteger, TShifts + TShifts>(this->m_int * other.m_int, 0);
  }

  template <size_t TShifts2>
  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts>& operator*=(const FixedFloatingPoint<TInteger, TShifts2>& other)
  {
    this->m_int = math::rshift(this->m_int * other.m_int, TShifts2);
    return *this;
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts>& operator*=(const FixedFloatingPoint<TInteger, TShifts>& other)
  {
    this->m_int = math::rshift(this->m_int * other.m_int, TShifts);
    return *this;
  }

  template <size_t TShifts2>
  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts + TShifts2> operator/(const FixedFloatingPoint<TInteger, TShifts2>& other)
  {
    return FixedFloatingPoint<TInteger, TShifts - TShifts2>(this->m_int / other.m_int, 0);
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts + TShifts> operator/(const FixedFloatingPoint<TInteger, TShifts>& other)
  {
    return FixedFloatingPoint<TInteger, TShifts - TShifts>(this->m_int / other.m_int, 0);
  }

  template <size_t TShifts2>
  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts>& operator/=(const FixedFloatingPoint<TInteger, TShifts2>& other)
  {
    this->m_int = math::lshift(this->m_int, TShifts2) / other.m_int;
    return *this;
  }

  __host__ __device__
  FixedFloatingPoint<TInteger, TShifts>& operator/=(const FixedFloatingPoint<TInteger, TShifts>& other)
  {
    this->m_int = math::lshift(this->m_int, TShifts) / other.m_int;
    return *this;
  }

  __host__ __device__
  bool operator==(const FixedFloatingPoint<TInteger, TShifts>& other) const
  {
    return this->m_int == other.m_int;
  }

  __host__ __device__
  bool operator!=(const FixedFloatingPoint<TInteger, TShifts>& other) const
  {
    return this->m_int != other.m_int;
  }

  __host__ __device__
  bool operator<(const FixedFloatingPoint<TInteger, TShifts>& other) const
  {
    return this->m_int < other.m_int;
  }

  __host__ __device__
  bool operator>(const FixedFloatingPoint<TInteger, TShifts>& other) const
  {
    return this->m_int > other.m_int;
  }

  __host__ __device__
  bool operator<=(const FixedFloatingPoint<TInteger, TShifts>& other) const
  {
    return this->m_int <= other.m_int;
  }

  __host__ __device__
  bool operator>=(const FixedFloatingPoint<TInteger, TShifts>& other) const
  {
    return this->m_int >= other.m_int;
  }

  template <typename TInteger2, size_t TShifts2>
  friend std::ostream& operator<<(std::ostream& stream, const FixedFloatingPoint<TInteger2, TShifts2>& p);

  template <typename TInteger2, size_t TShifts2>
  friend class FixedFloatingPoint;

private:
  TInteger m_int;

  __host__ __device__
  FixedFloatingPoint(TInteger i, int dummy)
    : m_int(i)
  {
  }
};

template <typename TInteger, size_t TShifts>
std::ostream& operator<<(std::ostream& stream, const FixedFloatingPoint<TInteger, TShifts>& f)
{
  return stream << "FixedFloatingPoint(v=" << static_cast<double>(f) << ", v * 2^" << TShifts << "=" << f.m_int << ")";
}

} // end of ns numeric