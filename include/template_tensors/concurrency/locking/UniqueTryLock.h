#pragma once

namespace mutex {

template <typename TMutex>
class UniqueTryLock
{
public:
  HD_WARNING_DISABLE
  __host__ __device__
  UniqueTryLock(TMutex* mutex)
    : m_mutex(mutex)
  {
    m_locked = m_mutex->try_lock();
  }

  __host__ __device__
  UniqueTryLock(TMutex& mutex)
    : UniqueTryLock(&mutex)
  {
  }

  __host__ __device__
  UniqueTryLock(const UniqueTryLock&) = delete;

  __host__ __device__
  UniqueTryLock(UniqueTryLock&& other)
    : m_mutex(other.m_mutex)
    , m_locked(other.m_locked)
  {
    other.m_mutex = nullptr;
    other.m_locked = false;
  }

  __host__ __device__
  UniqueTryLock& operator=(const UniqueTryLock&) = delete;

  HD_WARNING_DISABLE
  __host__ __device__
  UniqueTryLock& operator=(UniqueTryLock&& other)
  {
    if (m_locked)
    {
      m_mutex->unlock();
    }
    m_mutex = other.m_mutex;
    m_locked = other.m_locked;
    other.m_mutex = nullptr;
    other.m_locked = false;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~UniqueTryLock()
  {
    if (m_locked)
    {
      m_mutex->unlock();
    }
  }

  __host__ __device__
  bool isLocked() const
  {
    return m_locked;
  }

  __host__ __device__
  operator bool() const
  {
    return isLocked();
  }

private:
  TMutex* m_mutex;
  bool m_locked;
};

} // end of ns mutex
