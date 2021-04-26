#pragma once

namespace atomic {

namespace op {

namespace detail {

template <bool TIsAtomic>
struct TryLockLoad;

template <>
struct TryLockLoad<true>
{
  template <typename TData, typename TOut, typename TMutex>
  __host__ __device__
  static bool load(const TData& data, TOut& out, TMutex& mutex)
  {
    out = data;
    return true;
  }
};

template <>
struct TryLockLoad<false>
{
  template <typename TData, typename TOut, typename TMutex>
  __host__ __device__
  static bool load(const TData& data, TOut& out, TMutex& mutex)
  {
    if (auto lock = mutex::UniqueTryLock<TMutex>(mutex))
    {
      out = data;
      return true;
    }
    else
    {
      return false;
    }
  }
};

template <bool TCheckBeforeLocking>
struct TryLockCas;

template <>
struct TryLockCas<true>
{
  template <typename TData, typename TCompare, typename TVal, typename TMutex>
  __host__ __device__
  static bool cas(TData& data, const TCompare& compare, const TVal& val, bool& swapped, TMutex& mutex)
  {
    if (!template_tensors::eq(data, compare))
    {
      swapped = false;
      return true;
    }
    else if (auto lock = mutex::UniqueTryLock<TMutex>(mutex))
    {
      swapped = template_tensors::eq(data, compare);
      if (swapped)
      {
        data = val;
      }
      return true;
    }
    else
    {
      return false;
    }
  }
};

template <>
struct TryLockCas<false>
{
    template <typename TData, typename TCompare, typename TVal, typename TMutex>
  __host__ __device__
  static bool cas(TData& data, const TCompare& compare, const TVal& val, bool& swapped, TMutex& mutex)
  {
    if (auto lock = mutex::UniqueTryLock<TMutex>(mutex))
    {
      swapped = template_tensors::eq(data, compare);
      if (swapped)
      {
        data = val;
      }
      return true;
    }
    else
    {
      return false;
    }
  }
};

template <typename TMutex, bool TAvoidLockIfPossible>
class TryLockBase
{
public:
  __host__ __device__
  TryLockBase()
    : m_mutex()
  {
  }

  template <typename TData, typename TOut>
  __host__ __device__
  bool load(const TData& data, TOut& out)
  {
    return detail::TryLockLoad<is_atomic<TData>::value && TAvoidLockIfPossible>::load(data, out, m_mutex);
  }

  template <typename TData, typename TValue>
  __host__ __device__
  bool assign(TData& data, const TValue& value)
  {
    if (auto lock = mutex::UniqueTryLock<TMutex>(m_mutex))
    {
      data = value;
      return true;
    }
    else
    {
      return false;
    }
  }

  template <typename TData, typename TValue>
  __host__ __device__
  bool add(TData& data, const TValue& value)
  {
    if (auto lock = mutex::UniqueTryLock<TMutex>(m_mutex))
    {
      data += value;
      return true;
    }
    else
    {
      return false;
    }
  }

  template <typename TData, typename TValue>
  __host__ __device__
  bool multiply(TData& data, const TValue& value)
  {
    if (auto lock = mutex::UniqueTryLock<TMutex>(m_mutex))
    {
      data *= value;
      return true;
    }
    else
    {
      return false;
    }
  }

  template <typename TData, typename TValue>
  __host__ __device__
  bool subtract(TData& data, const TValue& value)
  {
    if (auto lock = mutex::UniqueTryLock<TMutex>(m_mutex))
    {
      data -= value;
      return true;
    }
    else
    {
      return false;
    }
  }

  template <typename TData, typename TMax>
  __host__ __device__
  bool inc(TData& data, const TMax& max)
  {
    if (auto lock = mutex::UniqueTryLock<TMutex>(m_mutex))
    {
      data = (data >= max) ? 0 : (data + 1);
      return true;
    }
    else
    {
      return false;
    }
  }

  template <typename TData, typename TMax>
  __host__ __device__
  bool dec(TData& data, const TMax& max)
  {
    if (auto lock = mutex::UniqueTryLock<TMutex>(m_mutex))
    {
      data = (data == 0 || data > max) ? max : (data - 1);
      return true;
    }
    else
    {
      return false;
    }
  }

  template <typename TData, typename TCompare, typename TVal>
  __host__ __device__
  bool cas(TData& data, const TCompare& compare, const TVal& val, bool& swapped)
  {
    return detail::TryLockCas<is_atomic<TData>::value && TAvoidLockIfPossible>::cas(data, compare, val, swapped, m_mutex);
  }

private:
  TMutex m_mutex;
};

} // end of ns detail

template <typename TMutex, bool TAvoidLockIfPossible = true,
  bool TEmptyCopyAndMove = !std::is_copy_constructible<TMutex>::value ||!std::is_move_constructible<TMutex>::value>
class TryLock;

template <typename TMutex, bool TAvoidLockIfPossible>
class TryLock<TMutex, TAvoidLockIfPossible, false> : public detail::TryLockBase<TMutex, TAvoidLockIfPossible>
{
};

template <typename TMutex, bool TAvoidLockIfPossible>
class TryLock<TMutex, TAvoidLockIfPossible, true> : public detail::TryLockBase<TMutex, TAvoidLockIfPossible>
{
public:
  __host__ __device__
  TryLock()
  {
  }

  __host__ __device__
  TryLock(const TryLock<TMutex, TAvoidLockIfPossible>& other)
  {
  }

  __host__ __device__
  TryLock(TryLock<TMutex, TAvoidLockIfPossible>&& other)
  {
  }

  __host__ __device__
  TryLock<TMutex, TAvoidLockIfPossible> operator=(const TryLock<TMutex, TAvoidLockIfPossible>& other)
  {
    return *this;
  }

  __host__ __device__
  TryLock<TMutex, TAvoidLockIfPossible> operator=(TryLock<TMutex, TAvoidLockIfPossible>&& other)
  {
    return *this;
  }
};

template <mem::MemoryType TMemoryType, bool TIsOnHost = TT_IS_ON_HOST>
using default_try_for = typename std::conditional<
    mem::isOnDevice<TMemoryType, TIsOnHost>(),
    TryLock<cuda::Mutex>,
    TryLock<std::mutex>
  >::type;

} // end of ns op

} // end of ns atomic
