#pragma once

namespace atomic {

namespace op {

namespace detail {

template <bool TIsAtomic>
struct LockLoad;

template <>
struct LockLoad<true>
{
  template <typename TData, typename TOut, typename TMutex>
  __host__ __device__
  static void load(const TData& data, TOut& out, TMutex& mutex)
  {
    out = data;
  }
};

template <>
struct LockLoad<false>
{
  template <typename TData, typename TOut, typename TMutex>
  __host__ __device__
  static void load(const TData& data, TOut& out, TMutex& mutex)
  {
    mutex::UniqueLock<TMutex> lock(mutex);
    out = data;
  }
};

template <bool TCheckBeforeLocking>
struct LockCas;

template <>
struct LockCas<true>
{
  template <typename TData, typename TCompare, typename TVal, typename TMutex>
  __host__ __device__
  static void cas(TData& data, const TCompare& compare, const TVal& val, bool& swapped, TMutex& mutex)
  {
    if (!template_tensors::eq(data, compare))
    {
      swapped = false;
    }
    else
    {
      mutex::UniqueLock<TMutex> lock(mutex);
      swapped = template_tensors::eq(data, compare);
      if (swapped)
      {
        data = val;
      }
    }
  }
};

template <>
struct LockCas<false>
{
  template <typename TData, typename TCompare, typename TVal, typename TMutex>
  __host__ __device__
  static void cas(TData& data, const TCompare& compare, const TVal& val, bool& swapped, TMutex& mutex)
  {
    mutex::UniqueLock<TMutex> lock(mutex);
    swapped = template_tensors::eq(data, compare);
    if (swapped)
    {
      data = val;
    }
  }
};

template <typename TMutex, bool TAvoidLockIfPossible>
class LockBase
{
public:
  template <typename TData, typename TOut>
  __host__ __device__
  void load(const TData& data, TOut& out)
  {
    detail::LockLoad<is_atomic<TData>::value && TAvoidLockIfPossible>::load(data, out, m_mutex);
  }

  template <typename TData, typename TValue>
  __host__ __device__
  void assign(TData& data, const TValue& value)
  {
    mutex::UniqueLock<TMutex> lock(m_mutex);
    data = value;
  }

  template <typename TData, typename TValue>
  __host__ __device__
  void add(TData& data, const TValue& value)
  {
    mutex::UniqueLock<TMutex> lock(m_mutex);
    data += value;
  }

  template <typename TData, typename TValue>
  __host__ __device__
  void multiply(TData& data, const TValue& value)
  {
    mutex::UniqueLock<TMutex> lock(m_mutex);
    data *= value;
  }

  template <typename TData, typename TValue>
  __host__ __device__
  void subtract(TData& data, const TValue& value)
  {
    mutex::UniqueLock<TMutex> lock(m_mutex);
    data -= value;
  }

  template <typename TData, typename TMax>
  __host__ __device__
  void inc(TData& data, const TMax& max)
  {
    mutex::UniqueLock<TMutex> lock(m_mutex);
    data = (data >= max) ? 0 : (data + 1);
  }

  template <typename TData, typename TMax>
  __host__ __device__
  void dec(TData& data, const TMax& max)
  {
    mutex::UniqueLock<TMutex> lock(m_mutex);
    data = (data == 0 || data > max) ? max : (data - 1);
  }

  template <typename TData, typename TCompare, typename TVal>
  __host__ __device__
  void cas(TData& data, const TCompare& compare, const TVal& val, bool& swapped)
  {
    detail::LockCas<is_atomic<TData>::value && TAvoidLockIfPossible>::cas(data, compare, val, swapped, m_mutex);
  }

private:
  TMutex m_mutex;
};

} // end of ns detail

template <typename TMutex, bool TAvoidLockIfPossible = true,
  bool TEmptyCopyAndMove = !std::is_copy_constructible<TMutex>::value ||!std::is_move_constructible<TMutex>::value>
class Lock;

template <typename TMutex, bool TAvoidLockIfPossible>
class Lock<TMutex, TAvoidLockIfPossible, false> : public detail::LockBase<TMutex, TAvoidLockIfPossible>
{
};

template <typename TMutex, bool TAvoidLockIfPossible>
class Lock<TMutex, TAvoidLockIfPossible, true> : public detail::LockBase<TMutex, TAvoidLockIfPossible>
{
public:
  __host__ __device__
  Lock()
  {
  }

  __host__ __device__
  Lock(const Lock<TMutex, TAvoidLockIfPossible>& other)
  {
  }

  __host__ __device__
  Lock(Lock<TMutex, TAvoidLockIfPossible>&& other)
  {
  }

  __host__ __device__
  Lock<TMutex, TAvoidLockIfPossible> operator=(const Lock<TMutex, TAvoidLockIfPossible>& other)
  {
    return *this;
  }

  __host__ __device__
  Lock<TMutex, TAvoidLockIfPossible> operator=(Lock<TMutex, TAvoidLockIfPossible>&& other)
  {
    return *this;
  }
};

template <mem::MemoryType TMemoryType, bool TIsOnHost = TT_IS_ON_HOST>
using default_wait_for = typename std::conditional<
    mem::isOnDevice<TMemoryType, TIsOnHost>(),
    cuda::AtomicOps<>,
    Lock<std::mutex>
  >::type;

} // end of ns op

} // end of ns atomic
