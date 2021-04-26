#pragma once

#include <template_tensors/util/Memory.h>
#include <template_tensors/cuda/Cuda.h>

namespace for_each {

namespace detail {

struct ErrorForEach
{
  template <size_t TNum = for_each::DYN, mem::MemoryType TMemoryType = mem::UNKNOWN, bool TMustBeAvailable = true, typename TIteratorBegin, typename TIteratorEnd, typename TFunctor>
  __host__ __device__
  static bool for_each(TIteratorBegin begin, TIteratorEnd end, TFunctor&& func)
  {
    ASSERT_(false,
    HD_NAME " code: Invalid for_each operation");
    // TODO: better printing
    return false;
  }
};



template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType, typename TForEachSeq>
struct ForEachDeciderHD;

template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType, typename TForEach1, typename... TForEachs>
struct ForEachDeciderHD<TIsOnHost, TNum, TMemoryType, tmp::ts::Sequence<TForEach1, TForEachs...>>
{
  using Next = ForEachDeciderHD<TIsOnHost, TNum, TMemoryType, tmp::ts::Sequence<TForEachs...>>;
  static const for_each::Availability this_availability = TForEach1::template availability_v<TIsOnHost, TNum, TMemoryType>::value;

  static const for_each::Availability availability = this_availability == YES ? YES :
                                                     this_availability == MAYBE ? (Next::availability == YES ? YES : MAYBE) :
                                                     /*this_availability == NO => */ Next::availability;

  using ForEachSeq = typename std::conditional<this_availability == MAYBE, tmp::ts::concat_t<tmp::ts::Sequence<TForEach1>, typename Next::ForEachSeq>,
                     typename std::conditional<this_availability == YES, tmp::ts::Sequence<TForEach1>,
                     typename Next::ForEachSeq
                     >::type
                     >::type;
};

template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
struct ForEachDeciderHD<TIsOnHost, TNum, TMemoryType, tmp::ts::Sequence<>>
{
  static const for_each::Availability availability = NO;

  using ForEachSeq = tmp::ts::Sequence<ErrorForEach>;
};



template <size_t TNum, mem::MemoryType TMemoryType, typename TForEachSeq>
struct ForEachDecider
{
  using HostForEachSeq = typename ForEachDeciderHD<true, TNum, TMemoryType, TForEachSeq>::ForEachSeq;
  using DeviceForEachSeq = typename ForEachDeciderHD<false, TNum, TMemoryType, TForEachSeq>::ForEachSeq;
};

template <typename TForEachSeq>
struct ExecuteAutoForEach;

template <typename TForEach1, typename TForEach2, typename... TForEachs>
struct ExecuteAutoForEach<tmp::ts::Sequence<TForEach1, TForEach2, TForEachs...>>
{
  HD_WARNING_DISABLE
  template <size_t TNum, mem::MemoryType TMemoryType, typename TIteratorBegin, typename TIteratorEnd, typename TFunctor>
  __host__ __device__
  static bool run(TIteratorBegin begin, TIteratorEnd end, TFunctor&& functor)
  {
    if (TForEach1::template for_each<TNum, TMemoryType, false>(begin, end, util::forward<TFunctor>(functor)))
    {
      return true;
    }
    else
    {
      return ExecuteAutoForEach<tmp::ts::Sequence<TForEach2, TForEachs...>>::template run<TNum, TMemoryType>(begin, end, util::forward<TFunctor>(functor));
    }
  }
};

template <typename TForEach>
struct ExecuteAutoForEach<tmp::ts::Sequence<TForEach>>
{
  HD_WARNING_DISABLE
  template <size_t TNum, mem::MemoryType TMemoryType, typename TIteratorBegin, typename TIteratorEnd, typename TFunctor>
  __host__ __device__
  static bool run(TIteratorBegin begin, TIteratorEnd end, TFunctor&& functor)
  {
    return TForEach::template for_each<TNum, TMemoryType, false>(begin, end, util::forward<TFunctor>(functor));
  }
};

template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType, typename TForEachSeq>
struct AnyIsParallel;

template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType, typename TForEach1, typename... TForEachs>
struct AnyIsParallel<TIsOnHost, TNum, TMemoryType, tmp::ts::Sequence<TForEach1, TForEachs...>>
{
  static const bool value = TForEach1::template is_parallel_v<TIsOnHost, TNum, TMemoryType>::value
    || AnyIsParallel<TIsOnHost, TNum, TMemoryType, tmp::ts::Sequence<TForEachs...>>::value;
};

template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
struct AnyIsParallel<TIsOnHost, TNum, TMemoryType, tmp::ts::Sequence<>>
{
  static const bool value = false;
};

} // end of ns detail

template <typename... TForEachs>
struct AutoForEach
{
  using ForEachSeq = typename std::conditional<sizeof...(TForEachs) != 0,
    tmp::ts::Sequence<TForEachs...>,
    tmp::ts::Sequence<
      for_each::Sequential>
  >::type;

  template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
  TVALUE(bool, availability_v, detail::ForEachDeciderHD<TIsOnHost, TNum, TMemoryType, ForEachSeq>::availability)

  template <bool TIsOnHost, size_t TNum, mem::MemoryType TMemoryType>
  TVALUE(bool, is_parallel_v,
      TIsOnHost
    ? detail::AnyIsParallel<TIsOnHost, TNum, TMemoryType, typename detail::ForEachDecider<TNum, TMemoryType, ForEachSeq>::HostForEachSeq>::value
    : detail::AnyIsParallel<TIsOnHost, TNum, TMemoryType, typename detail::ForEachDecider<TNum, TMemoryType, ForEachSeq>::DeviceForEachSeq>::value
  )

  HD_WARNING_DISABLE
  template <size_t TNum = for_each::DYN, mem::MemoryType TMemoryType = mem::UNKNOWN, bool TMustBeAvailable = true, typename TIteratorBegin, typename TIteratorEnd, typename TFunctor>
  __host__ __device__
  static bool for_each(TIteratorBegin begin, TIteratorEnd end, TFunctor&& functor)
  {
    using HostForEachSeq = typename detail::ForEachDecider<TNum, TMemoryType, ForEachSeq>::HostForEachSeq;
    using DeviceForEachSeq = typename detail::ForEachDecider<TNum, TMemoryType, ForEachSeq>::DeviceForEachSeq;
    static_assert(
         !std::is_same<HostForEachSeq, tmp::ts::Sequence<detail::ErrorForEach>>::value
      || !std::is_same<DeviceForEachSeq, tmp::ts::Sequence<detail::ErrorForEach>>::value,
    HD_NAME " code: Invalid for_each operation");

    // Pass both host and device versions through compiler, otherwise kernels will not be compiled properly
    INSTANTIATE_HOST(ESC(detail::ExecuteAutoForEach<HostForEachSeq>::template run<TNum, TMemoryType>), INSTANTIATE_ARG(TIteratorBegin), INSTANTIATE_ARG(TIteratorEnd), INSTANTIATE_ARG(TFunctor&&));
    INSTANTIATE_DEVICE(ESC(detail::ExecuteAutoForEach<DeviceForEachSeq>::template run<TNum, TMemoryType>), INSTANTIATE_ARG(TIteratorBegin), INSTANTIATE_ARG(TIteratorEnd), INSTANTIATE_ARG(TFunctor&&));

#if TT_IS_ON_HOST
    return TT_FOR_EACH_CHECK_RESULT(detail::ExecuteAutoForEach<HostForEachSeq>::template run<TNum, TMemoryType>(begin, end, util::forward<TFunctor>(functor)));
#else
    return TT_FOR_EACH_CHECK_RESULT(detail::ExecuteAutoForEach<DeviceForEachSeq>::template run<TNum, TMemoryType>(begin, end, util::forward<TFunctor>(functor)));
#endif
    return false;
  }
};

} // end of ns for_each
