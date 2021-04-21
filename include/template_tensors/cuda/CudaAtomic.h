#pragma once

#include "Cuda.h"

namespace cuda {

namespace detail {

template <bool TLoadAsSimpleRead>
struct AtomicLoad;

template <>
struct AtomicLoad<true>
{
  template <typename TData1, typename TData2>
  __device__
  static bool load(TData1& data, TData2& out)
  {
    out = data;
    return true;
  }
};

template <>
struct AtomicLoad<false>
{
  template <typename TData1, typename TData2>
  __device__
  static bool load(TData1& data, TData2& out)
  {
    out = atomicOr(&data, static_cast<TData1>(0));
    return true;
  }
};

} // end of ns detail

template <bool TLoadAsSimpleRead = true>
class AtomicOps
{
public:
#ifdef __CUDACC__
  template <typename TData1, typename TData2>
  __device__
  bool load(TData1& data, TData2& out)
  {
    return detail::AtomicLoad<TLoadAsSimpleRead>::load(data, out);
  }

  template <typename TData1, typename TData2>
  __device__
  bool store(TData1& data, const TData2& in)
  {
    atomicExch(&data, in);
    return true;
  }

  template <typename TData1, typename TData2>
  __device__
  bool add(TData1& data, const TData2& value)
  {
    atomicAdd(&data, value);
    return true;
  }

  template <typename TData1, typename TData2>
  __device__
  bool subtract(TData1& data, const TData2& value)
  {
    atomicSub(&data, value);
    return true;
  }

  template <typename TData>
  __device__
  bool inc(TData& data, const TData& max)
  {
    atomicInc(&data, max);
    return true;
  }

  template <typename TData>
  __device__
  bool dec(TData& data, const TData& max)
  {
    atomicDec(&data, max);
    return true;
  }

  template <typename TData1, typename TData2, typename TData3>
  __device__
  bool cas(TData1& data, const TData2& compare, const TData3& val, bool& swapped)
  {
    swapped = atomicCas(&data, compare, val) == compare;
    return true;
  }
#endif
};

template <typename TArg>
struct HasAtomicCas
{
  template <typename T, typename TDummy = decltype(atomicCas(std::declval<T*>(), std::declval<T>(), std::declval<T>()))>
  TMP_IF(T)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

} // end of ns cuda
