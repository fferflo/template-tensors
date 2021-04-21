#pragma once

#include <template_tensors/util/Memory.h>
#include <template_tensors/cuda/Cuda.h>
#include "Helper.h"

namespace for_each {

struct Sequential
{
  template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
  TVALUE(for_each::Availability, availability_v, (mem::isOnLocal<TMemoryType, TIsOnHost>() || TMemoryType == mem::UNKNOWN) ? for_each::YES : for_each::NO)

  template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
  TVALUE(bool, is_parallel_v, false)

  HD_WARNING_DISABLE
  template <size_t TNum = for_each::DYN, mem::MemoryType TMemoryType = mem::UNKNOWN, bool TMustBeAvailable = true, typename TIteratorBegin, typename TIteratorEnd, typename TFunctor>
  __host__ __device__
  static bool for_each(TIteratorBegin begin, TIteratorEnd end, TFunctor&& func)
  {
    while (begin != end)
    {
      func(*begin);
      ++begin;
    }
    return true;
  }
};

} // end of ns for_each
