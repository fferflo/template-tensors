#pragma once

namespace atomic {

template <typename TData, typename TAtomicOps>
class Variable
{
public:
  __host__ __device__
  Variable()
    : m_data()
    , m_atomic_ops()
  {
  }

  template <typename TData2, ENABLE_IF(std::is_constructible<TData, TData2&&>::value)>
  __host__ __device__
  Variable(TData2&& data)
    : m_data(util::forward<TData2>(data))
    , m_atomic_ops()
  {
  }

  template <typename TDummy = void>
  __host__ __device__
  TData load()
  {
    TData out;
    load(out);
    return out;
  }

  template <typename TDummy = void>
  __host__ __device__
  void load(TData& out)
  {
    m_atomic_ops.load(m_data, out);
  }

  template <typename TDummy = void>
  __host__ __device__
  void operator=(const TData& in)
  {
    m_atomic_ops.store(m_data, in);
  }

  template <typename TDummy = void>
  __host__ __device__
  void operator+=(const TData& value)
  {
    m_atomic_ops.add(m_data, value);
  }

  template <typename TDummy = void>
  __host__ __device__
  void operator*=(const TData& value)
  {
    m_atomic_ops.multiply(m_data, value);
  }

  template <typename TDummy = void>
  __host__ __device__
  void operator-=(const TData& value)
  {
    m_atomic_ops.subtract(m_data, value);
  }

  template <typename TDummy = void>
  __host__ __device__
  void operator++()
  {
    m_atomic_ops.inc(m_data);
  }

  template <typename TDummy = void>
  __host__ __device__
  void operator--()
  {
    m_atomic_ops.dec(m_data);
  }

  template <typename TDummy = void>
  __host__ __device__
  void cas(const TData& compare, const TData& val, bool& swapped)
  {
    m_atomic_ops.cas(m_data, compare, val, swapped);
  }

private:
  TData m_data;
  TAtomicOps m_atomic_ops;
};

} // end of ns atomic
