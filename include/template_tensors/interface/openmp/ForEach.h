#pragma once
#ifdef OPENMP_INCLUDED

#include <template_tensors/util/Memory.h>
#include <template_tensors/cuda/Cuda.h>
#include <template_tensors/for_each/Helper.h>
#include <omp.h>

namespace openmp {

struct ForEach
{
  template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
  TVALUE(for_each::Availability, availability_v, (TIsOnHost && (mem::isOnHost<TMemoryType, TIsOnHost>() || TMemoryType == mem::UNKNOWN)) ? for_each::YES : for_each::NO)

  template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
  TVALUE(bool, is_parallel_v, true)

  template <size_t TNum = for_each::DYN, mem::MemoryType TMemoryType = mem::UNKNOWN, bool TMustBeAvailable = true, typename TIteratorBegin, typename TIteratorEnd, typename TFunctor>
  __host__
  static bool for_each(TIteratorBegin begin, TIteratorEnd end, TFunctor func)
  {
    #pragma omp parallel for
    for (TIteratorBegin it = begin; it < end; ++it)
    {
      func(*it);
    }
    return true;
  }
};

} // end of ns openmp

#endif