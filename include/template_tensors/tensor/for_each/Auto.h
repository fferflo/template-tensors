namespace template_tensors {

namespace op {

namespace detail {

struct ErrorForEach
{
  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_for_each_available_v, false)

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_parallel_v, false)

  template <size_t TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes>
  __host__ __device__
  static void for_each(TFunctor func, TTensorTypes&&... tensors)
  {
    ASSERT_(false, HD_NAME " code: Invalid for_each operation");
    // TODO: better printing
  }

  TT_FOR_EACH_MAP_AND_COPY(__host__ __device__)
};

struct ErrorMapper
{
  template <bool TIsOnHost, typename TTensorDest, typename... TTensorSrcs>
  TVALUE(bool, is_map_available_v, false)

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_parallel_v, false)

  template <typename TOperation, typename TTensorDest, typename... TTensorSrcs>
  __host__ __device__
  static void map(TOperation, TTensorDest&&, TTensorSrcs&&...)
  {
    ASSERT_(false, HD_NAME " code: Invalid map operation to %u",
      (uint32_t) mem::memorytype_v<TTensorDest>::value);
    // TODO: better printing
  }
};

struct ErrorCopier
{
  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v, false)

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_parallel_v, false)

  template <typename TTensorDest, typename TTensorSrc>
  __host__ __device__
  static void copy(TTensorDest&&, TTensorSrc&&)
  {
    ASSERT_(false, HD_NAME " code: Invalid copy operation from %u to %u",
      (uint32_t) mem::memorytype_v<TTensorSrc>::value, (uint32_t) mem::memorytype_v<TTensorDest>::value);
    // TODO: better printing
  }
};



template <bool TIsOnHost, typename TForEach, typename... TTensorTypes>
struct IsCopyAvailableHelper
{
  static const bool value = false;
};

template <bool TIsOnHost, typename TForEach, typename TTensorDest, typename TTensorSrc>
struct IsCopyAvailableHelper<TIsOnHost, TForEach, TTensorDest, TTensorSrc>
{
  static const bool value = TForEach::template is_copy_available_v<TIsOnHost, TTensorDest, TTensorSrc>::value;
};



template <bool TIsOnHost, typename TForEachSeq, typename TTensorTypeSeq>
struct ForEachDeciderHDSeq;

template <bool TIsOnHost, typename... TTensorTypes, typename TForEach1, typename... TForEachs>
struct ForEachDeciderHDSeq<TIsOnHost, tmp::ts::Sequence<TForEach1, TForEachs...>, tmp::ts::Sequence<TTensorTypes...>>
{
  static_assert(sizeof...(TTensorTypes) > 0, "No tensors given to AutoForEach");

  using Next = ForEachDeciderHDSeq<TIsOnHost, tmp::ts::Sequence<TForEachs...>, tmp::ts::Sequence<TTensorTypes...>>;
  static const bool this_is_for_each_available = TForEach1::template is_for_each_available_v<TIsOnHost, TTensorTypes...>::value;
  static const bool this_is_map_available = TForEach1::template is_map_available_v<TIsOnHost, TTensorTypes...>::value;
  static const bool this_is_copy_available = IsCopyAvailableHelper<TIsOnHost, TForEach1, TTensorTypes...>::value;

  static const bool is_for_each_available = this_is_for_each_available || Next::is_for_each_available;
  static const bool is_map_available = this_is_map_available || Next::is_map_available;
  static const bool is_copy_available = this_is_copy_available || Next::is_copy_available;

  using ForEach = typename std::conditional<this_is_for_each_available, TForEach1, typename Next::ForEach>::type;
  using Mapper = typename std::conditional<this_is_map_available, TForEach1, typename Next::Mapper>::type;
  using Copier = typename std::conditional<this_is_copy_available, TForEach1, typename Next::Copier>::type;
};

template <bool TIsOnHost, typename... TTensorTypes>
struct ForEachDeciderHDSeq<TIsOnHost, tmp::ts::Sequence<>, tmp::ts::Sequence<TTensorTypes...>>
{
  static_assert(sizeof...(TTensorTypes) > 0, "No tensors given to AutoForEach");

  static const bool is_for_each_available = false;
  static const bool is_map_available = false;
  static const bool is_copy_available = false;

  using ForEach = ErrorForEach;
  using Mapper = ErrorMapper;
  using Copier = ErrorCopier;
};

template <bool TIsOnHost, typename TForEachSeq, typename... TTensorTypes>
using ForEachDeciderHD = ForEachDeciderHDSeq<TIsOnHost, TForEachSeq, tmp::ts::Sequence<TTensorTypes...>>;



template <typename TForEachSeq, typename... TTensorTypes>
struct ForEachDecider
{
  using HostForEach = typename ForEachDeciderHD<true, TForEachSeq, TTensorTypes...>::ForEach;
  using DeviceForEach = typename ForEachDeciderHD<false, TForEachSeq, TTensorTypes...>::ForEach;

  using HostMapper = typename ForEachDeciderHD<true, TForEachSeq, TTensorTypes...>::Mapper;
  using DeviceMapper = typename ForEachDeciderHD<false, TForEachSeq, TTensorTypes...>::Mapper;

  using HostCopier = typename ForEachDeciderHD<true, TForEachSeq, TTensorTypes...>::Copier;
  using DeviceCopier = typename ForEachDeciderHD<false, TForEachSeq, TTensorTypes...>::Copier;
};

template <typename TForEachSeq, typename TTensorDest, typename TTensorSrc>
using CopyDecider = ForEachDecider<TForEachSeq, TTensorDest, TTensorSrc>;

} // end of ns detail



template <typename... TForEachs>
struct AutoForEach
{
  using ForEachSeq = typename std::conditional<sizeof...(TForEachs) != 0,
    tmp::ts::Sequence<TForEachs...>,
    tmp::ts::Sequence<LocalForEach, LocalArrayForEach<>, DeviceArrayForEach<>, DeviceForEach<>>
  >::type;

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_for_each_available_v, detail::ForEachDeciderHD<TIsOnHost, ForEachSeq, TTensorTypes...>::is_for_each_available)

  template <bool TIsOnHost, typename TTensorDest, typename... TTensorSrcs>
  TVALUE(bool, is_map_available_v, detail::ForEachDeciderHD<TIsOnHost, ForEachSeq, TTensorDest, TTensorSrcs...>::is_map_available)

  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v, detail::ForEachDeciderHD<TIsOnHost, ForEachSeq, TTensorDest, TTensorSrc>::is_copy_available)

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_parallel_v,
      TIsOnHost
    ? detail::ForEachDecider<ForEachSeq, TTensorTypes...>::HostForEach::template is_parallel_v<TIsOnHost, TTensorTypes...>::value
    : detail::ForEachDecider<ForEachSeq, TTensorTypes...>::DeviceForEach::template is_parallel_v<TIsOnHost, TTensorTypes...>::value
  )

  HD_WARNING_DISABLE
  template <size_t TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes>
  __host__ __device__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors)
  {
    static_assert(sizeof...(tensors) > 0, "No tensors given");
    static_assert(are_compatible_dimseqs_v<dimseq_t<TTensorTypes>...>::value, "Incompatible static dimensions");
    static_assert(
         detail::ForEachDeciderHD<true, ForEachSeq, TTensorTypes...>::is_for_each_available
      || detail::ForEachDeciderHD<false, ForEachSeq, TTensorTypes...>::is_for_each_available,
    HD_NAME " code: Invalid for_each operation");

    using HostForEach = typename detail::ForEachDecider<ForEachSeq, TTensorTypes...>::HostForEach;
    using DeviceForEach = typename detail::ForEachDecider<ForEachSeq, TTensorTypes...>::DeviceForEach;

    // Pass both host and device versions through compiler, otherwise kernel will not be compiled properly
    INSTANTIATE_HOST(HostForEach::template for_each<TCoordsRank>, INSTANTIATE_ARG(TFunctor), INSTANTIATE_ARG(TTensorTypes&&)...);
    INSTANTIATE_DEVICE(DeviceForEach::template for_each<TCoordsRank>, INSTANTIATE_ARG(TFunctor), INSTANTIATE_ARG(TTensorTypes&&)...);

#if TT_IS_ON_HOST
    HostForEach dummy; // Doesn't compile without this hack for some reason
    decltype(dummy)::template for_each<TCoordsRank>(util::forward<TFunctor>(func), util::forward<TTensorTypes>(tensors)...);
#else
    DeviceForEach dummy; // Doesn't compile without this hack for some reason
    decltype(dummy)::template for_each<TCoordsRank>(util::forward<TFunctor>(func), util::forward<TTensorTypes>(tensors)...);
#endif
  }

  HD_WARNING_DISABLE
  template <typename TOperation, typename TTensorDest, typename... TTensorSrcs>
  __host__ __device__
  static void map(TOperation&& op, TTensorDest&& dest, TTensorSrcs&&... srcs)
  {
    static_assert(are_compatible_dimseqs_v<dimseq_t<TTensorDest>, dimseq_t<TTensorSrcs>...>::value, "Incompatible static dimensions");
    static_assert(
          detail::ForEachDeciderHD<true, ForEachSeq, TTensorDest, TTensorSrcs...>::is_map_available
      || detail::ForEachDeciderHD<false, ForEachSeq, TTensorDest, TTensorSrcs...>::is_map_available,
    HD_NAME " code: Invalid map operation");

    using HostMapper = typename detail::ForEachDecider<ForEachSeq, TTensorDest, TTensorSrcs...>::HostMapper;
    using DeviceMapper = typename detail::ForEachDecider<ForEachSeq, TTensorDest, TTensorSrcs...>::DeviceMapper;

    // Pass both host and device versions through compiler, otherwise kernel will not be compiled properly
    INSTANTIATE_HOST(HostMapper::map, INSTANTIATE_ARG(TOperation), INSTANTIATE_ARG(TTensorDest&&), INSTANTIATE_ARG(TTensorSrcs&&)...);
    INSTANTIATE_DEVICE(DeviceMapper::map, INSTANTIATE_ARG(TOperation), INSTANTIATE_ARG(TTensorDest&&), INSTANTIATE_ARG(TTensorSrcs&&)...);

#if TT_IS_ON_HOST
    HostMapper dummy; // Doesn't compile without this hack for some reason
    decltype(dummy)::map(util::forward<TOperation>(op), util::forward<TTensorDest>(dest), util::forward<TTensorSrcs>(srcs)...);
#else
    DeviceMapper dummy; // Doesn't compile without this hack for some reason
    decltype(dummy)::map(util::forward<TOperation>(op), util::forward<TTensorDest>(dest), util::forward<TTensorSrcs>(srcs)...);
#endif
  }

  HD_WARNING_DISABLE
  template <typename TTensorDest, typename TTensorSrc>
  __host__ __device__
  static void copy(TTensorDest&& dest, TTensorSrc&& src)
  {
    static_assert(are_compatible_dimseqs_v<dimseq_t<TTensorDest>, dimseq_t<TTensorSrc>>::value, "Incompatible static dimensions");
    static_assert(
         detail::ForEachDeciderHD<true, ForEachSeq, TTensorDest, TTensorSrc>::is_copy_available
      || detail::ForEachDeciderHD<false, ForEachSeq, TTensorDest, TTensorSrc>::is_copy_available,
    HD_NAME " code: Invalid copy operation");

    using HostCopier = typename detail::CopyDecider<ForEachSeq, TTensorDest, TTensorSrc>::HostCopier;
    using DeviceCopier = typename detail::CopyDecider<ForEachSeq, TTensorDest, TTensorSrc>::DeviceCopier;

    // Pass both host and device versions through compiler, otherwise kernel will not be compiled properly
    INSTANTIATE_HOST(HostCopier::copy, INSTANTIATE_ARG(TTensorDest&&), INSTANTIATE_ARG(TTensorSrc&&));
    INSTANTIATE_DEVICE(DeviceCopier::copy, INSTANTIATE_ARG(TTensorDest&&), INSTANTIATE_ARG(TTensorSrc&&));

#if TT_IS_ON_HOST
    HostCopier dummy; // Doesn't compile without this hack for some reason
    decltype(dummy)::copy(util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src));
#else
    DeviceCopier dummy; // Doesn't compile without this hack for some reason
    decltype(dummy)::copy(util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src));
#endif
  }
};

} // end of ns op

template <size_t TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes, ENABLE_IF(tmp::ts::all_apply_v<template_tensors::is_tensor_v, tmp::ts::Sequence<TTensorTypes...>>::value)>
__host__ __device__
void for_each(TFunctor&& func, TTensorTypes&&... tensors)
{
  op::AutoForEach<>::for_each<TCoordsRank>(util::forward<TFunctor>(func), util::forward<TTensorTypes>(tensors)...);
}

template <typename TOperation, typename TTensorDest, typename... TTensorSrcs>
__host__ __device__
void map(TOperation&& op, TTensorDest&& dest, TTensorSrcs&&... srcs)
{
  op::AutoForEach<>::map(util::forward<TOperation>(op), util::forward<TTensorDest>(dest), util::forward<TTensorSrcs>(srcs)...);
}





namespace detail {

struct OpAssign
{
  template <typename TOp, typename TTensorTypeDest, typename TTensorTypeSrc, ENABLE_IF(is_tensor_v<TTensorTypeSrc>::value)>
  __host__ __device__
  static void op(TOp op, TTensorTypeDest&& dest, TTensorTypeSrc&& src)
  {
    template_tensors::for_each(op, util::forward<TTensorTypeDest>(dest), util::forward<TTensorTypeSrc>(src));
  }

  template <typename TOp, typename TTensorTypeDest, typename TScalarTypeSrc, ENABLE_IF(!is_tensor_v<TScalarTypeSrc>::value)>
  __host__ __device__
  static void op(TOp op, TTensorTypeDest&& dest, const TScalarTypeSrc& src)
  {
    template_tensors::for_each(op, util::forward<TTensorTypeDest>(dest),
      template_tensors::broadcast<template_tensors::dimseq_t<TTensorTypeDest>>(template_tensors::SingletonT<typename std::decay<TScalarTypeSrc>::type>(src), dest.dims()));
  }
};

} // end of ns detail
// TODO: differentiate between scalar src and tensor src
#define ELWISE_OP_ASSIGN(NAME, OPERATION) \
  template <typename TTensorTypeDest, typename TSrcType, ENABLE_IF(is_tensor_v<TTensorTypeDest>::value)> \
  __host__ __device__ \
  TTensorTypeDest&& NAME (TTensorTypeDest&& dest, TSrcType&& src) \
  { \
    template_tensors::detail::OpAssign::op(OPERATION(), util::forward<TTensorTypeDest>(dest), util::forward<TSrcType>(src)); \
    return util::forward<TTensorTypeDest>(dest); \
  }

ELWISE_OP_ASSIGN(operator+=, math::functor::addassign);
ELWISE_OP_ASSIGN(operator-=, math::functor::subtractassign);
ELWISE_OP_ASSIGN(operator*=, math::functor::multiplyassign);
ELWISE_OP_ASSIGN(operator/=, math::functor::divideassign);

} // end of ns tensor
