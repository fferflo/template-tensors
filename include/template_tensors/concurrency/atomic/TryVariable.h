#pragma once

namespace atomic {

template <typename TData, typename TAtomicOps>
class TryVariable
{
public:
  __host__ __device__
  TryVariable()
    : m_data()
    , m_atomic_ops()
  {
  }

  __host__ __device__
  TryVariable(TData data)
    : m_data(data)
    , m_atomic_ops()
  {
  }

  template <typename TDummy = void>
  __host__ __device__
  bool load(TData& out)
  {
    return m_atomic_ops.load(m_data, out);
  }

  template <typename TDummy = void>
  __host__ __device__
  bool operator=(const TData& in)
  {
    return m_atomic_ops.store(m_data, in);
  }

  template <typename TDummy = void>
  __host__ __device__
  bool operator+=(const TData& value)
  {
    return m_atomic_ops.add(m_data, value);
  }

  template <typename TDummy = void>
  __host__ __device__
  bool operator-=(const TData& value)
  {
    return m_atomic_ops.subtract(m_data, value);
  }

  template <typename TDummy = void>
  __host__ __device__
  bool operator++()
  {
    return m_atomic_ops.inc(m_data);
  }

  template <typename TDummy = void>
  __host__ __device__
  bool operator--()
  {
    return m_atomic_ops.dec(m_data);
  }

  template <typename TDummy = void>
  __host__ __device__
  bool cas(const TData& compare, const TData& val, bool& swapped)
  {
    return m_atomic_ops.cas(m_data, compare, val, swapped);
  }

private:
  TData m_data;
  TAtomicOps m_atomic_ops;
};

} // end of ns atomic
