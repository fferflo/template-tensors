#pragma once

#include "Cuda.h"

#include <template_tensors/util/Assert.h>

namespace cuda {

class Mutex
{
public:
  __device__ __host__
  Mutex()
    : m_mutex(0)
  {
  }

  __device__ __host__
  bool isLocked()
  {
    return m_mutex != 0;
  }

#ifdef __CUDACC__
  __device__
  bool try_lock()
  {
    int old = atomicCAS(&m_mutex, 0, 1);
    ASSERT((old == 0 && m_mutex == 1) || old == 1, "Invalid mutex value %i", old);
    return old == 0;
  }

  __device__
  void unlock()
  {
    int old = atomicExch(&m_mutex, 0);
    ASSERT(old == 1, "Invalid mutex value %i", old);
  }
#endif

private:
  int m_mutex;
};

} // end of ns cuda
