#pragma once

namespace mutex {

template <typename TMutex>
class UniqueLock
{
public:
  HD_WARNING_DISABLE
  __host__ __device__
  UniqueLock(TMutex* mutex)
    : m_mutex(mutex)
  {
    m_mutex->lock();
  }

  __host__ __device__
  UniqueLock(TMutex& mutex)
    : UniqueLock(&mutex)
  {
  }

  __host__ __device__
  UniqueLock(const UniqueLock&) = delete;

  __host__ __device__
  UniqueLock(UniqueLock&& other)
    : m_mutex(other.m_mutex)
  {
    other.m_mutex = nullptr;
  }

  __host__ __device__
  UniqueLock& operator=(const UniqueLock&) = delete;

  HD_WARNING_DISABLE
  __host__ __device__
  UniqueLock& operator=(UniqueLock&& other)
  {
    if (m_mutex != nullptr)
    {
      m_mutex->unlock();
    }
    m_mutex = other.m_mutex;
    other.m_mutex = nullptr;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~UniqueLock()
  {
    if (m_mutex != nullptr)
    {
      m_mutex->unlock();
    }
  }

private:
  TMutex* m_mutex;
};

} // end of ns mutex
