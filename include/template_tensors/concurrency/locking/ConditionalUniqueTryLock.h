#pragma once

namespace mutex {

enum ConditionalLockState
{
  LOCK_FAILED,
  CONDITION_FAILED,
  LOCKED
};

template <typename TMutex>
class ConditionalUniqueTryLock
{
public:
  template <typename TCondition>
  __host__ __device__
  ConditionalUniqueTryLock(TMutex& mutex, TCondition cond)
    : ConditionalUniqueTryLock(&mutex, cond)
  {
  }

  HD_WARNING_DISABLE
  template <typename TCondition>
  __host__ __device__
  ConditionalUniqueTryLock(TMutex* mutex, TCondition cond)
    : m_mutex(mutex)
  {
    if (!cond())
    {
      m_result = CONDITION_FAILED;
    }
    else if (!m_mutex->try_lock())
    {
      m_result = LOCK_FAILED;
    }
    else if (!cond())
    {
      m_mutex->unlock();
      m_result = CONDITION_FAILED;
    }
    else
    {
      m_result = LOCKED;
    }
  }

  __host__ __device__
  ConditionalUniqueTryLock(const ConditionalUniqueTryLock&) = delete;

  __host__ __device__
  ConditionalUniqueTryLock(ConditionalUniqueTryLock&& other)
    : m_mutex(other.m_mutex)
    , m_result(other.m_result)
  {
    other.m_mutex = nullptr;
    other.m_result = LOCK_FAILED;
  }

  __host__ __device__
  ConditionalUniqueTryLock& operator=(const ConditionalUniqueTryLock&) = delete;

  HD_WARNING_DISABLE
  __host__ __device__
  ConditionalUniqueTryLock& operator=(ConditionalUniqueTryLock&& other)
  {
    if (m_result == LOCKED)
    {
      m_mutex->unlock();
    }
    m_mutex = other.m_mutex;
    m_result = other.m_result;
    other.m_mutex = nullptr;
    other.m_result = LOCK_FAILED;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~ConditionalUniqueTryLock()
  {
    if (m_result == LOCKED)
    {
      m_mutex->unlock();
    }
  }

  __host__ __device__
  ConditionalLockState getState() const
  {
    return m_result;
  }

  __host__ __device__
  bool isLocked() const
  {
    return m_result == LOCKED;
  }

  __host__ __device__
  operator bool() const
  {
    return isLocked();
  }

private:
  TMutex* m_mutex;
  ConditionalLockState m_result;
};

} // end of ns mutex
