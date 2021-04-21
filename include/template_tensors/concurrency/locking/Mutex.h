#pragma once

namespace mutex {

class Void
{
public:
  __host__ __device__
  bool isLocked()
  {
    return true;
  }

  __host__ __device__
  void lock()
  {
  }

  __host__ __device__
  bool try_lock()
  {
    return true;
  }

  __host__ __device__
  void unlock()
  {
  }
};

} // end of ns mutex
