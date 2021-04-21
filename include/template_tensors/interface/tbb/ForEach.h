#pragma once
#ifdef TBB_INCLUDED

#include <template_tensors/util/Memory.h>
#include <template_tensors/cuda/Cuda.h>
#include <template_tensors/for_each/Helper.h>

#include <tbb/tbb.h>

namespace tbb {

struct ForEach
{
  template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
  TVALUE(for_each::Availability, availability_v, (TIsOnHost && (mem::isOnHost<TMemoryType, TIsOnHost>() || TMemoryType == mem::UNKNOWN)) ? for_each::YES : for_each::NO)

  template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
  TVALUE(bool, is_parallel_v, true)

  template <size_t TNum = for_each::DYN, mem::MemoryType TMemoryType = mem::UNKNOWN, bool TMustBeAvailable = true, typename TIterator, typename TFunctor>
  __host__
  static bool for_each(TIterator begin, TIterator end, TFunctor func)
  {
    tbb::parallel_for_each(begin, end, func);
    /* tbb::parallel_for(tbb::blocked_range<TIterator>(begin, end), [&](const tbb::blocked_range<TIterator>& range){
      for (auto it = range.begin(); it != range.end(); it++)
      {
        func(*it);
      }
    });*/
    return true;
  }
};

} // end of ns tbb

#endif