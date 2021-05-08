namespace template_tensors {

namespace op {

namespace detail {

template <bool TIsOnHost, typename TCopierSeq, typename TTensorDest, typename TTensorSrc>
struct CopierDeciderHD;

template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc, typename TCopier1, typename... TCopiers>
struct CopierDeciderHD<TIsOnHost, metal::list<TCopier1, TCopiers...>, TTensorDest, TTensorSrc>
{
  using Next = CopierDeciderHD<TIsOnHost, metal::list<TCopiers...>, TTensorDest, TTensorSrc>;
  static const bool this_is_valid = TCopier1::template is_copy_available_v<TIsOnHost, TTensorDest, TTensorSrc>::value;

  static const bool is_valid = this_is_valid || Next::is_valid;
  using Copier = typename std::conditional<this_is_valid, TCopier1, typename Next::Copier>::type;
};

template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
struct CopierDeciderHD<TIsOnHost, metal::list<>, TTensorDest, TTensorSrc>
{
  static const bool is_valid = false;
  using Copier = ErrorCopier;
};

template <typename TCopierSeq, mem::MemoryType TDestMemoryType, mem::MemoryType TSrcMemoryType, typename TElementTypeDest, typename TElementTypeSrc, typename TTensorDest, typename TTensorSrc>
struct CopierDecider
{
  static_assert(std::is_assignable<TElementTypeDest&, TElementTypeSrc>::value, "Element types must be assignable");
  static_assert(CopierDeciderHD<true, TCopierSeq, TTensorDest, TTensorSrc>::is_valid || CopierDeciderHD<false, TCopierSeq, TTensorDest, TTensorSrc>::is_valid,
#if TT_IS_ON_HOST
    "Host"
#else
    "Device"
#endif
    " code: Invalid copy operation");

  using HostCopier = typename CopierDeciderHD<true, TCopierSeq, TTensorDest, TTensorSrc>::Copier;
  using DeviceCopier = typename CopierDeciderHD<false, TCopierSeq, TTensorDest, TTensorSrc>::Copier;
};

} // end of ns detail


template <typename... TCopiers>
struct AutoCopier
{
  using CopierSeq = typename std::conditional<sizeof...(TCopiers) != 0,
    metal::list<TCopiers...>,
    metal::list<StorageArrayCopier, AutoForEach<>, AssignToTempMemCopyToDestCopier, MemCopyToTempAssignToDestCopier, AssignToTempMemCopyToTempAssignToDestCopier<>>
  >::type;

  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v, detail::CopierDeciderHD<TIsOnHost, CopierSeq, TTensorDest, TTensorSrc>::is_valid)

  template <typename TTensorDest, typename TTensorSrc>
  __host__ __device__
  static void copy(TTensorDest&& dest, TTensorSrc&& src)
  {
    static_assert(is_tensor_v<TTensorDest>::value && is_tensor_v<TTensorSrc>::value, "Can only copy tensors");
    static_assert(are_compatible_dimseqs_v<TTensorDest, TTensorSrc>::value, "Incompatible static dimensions");
    ASSERT(areSameDimensions(dest.dims(), src.dims()), "Incompatible runtime dimensions");

    using HostCopier = typename detail::CopierDecider<CopierSeq, mem::memorytype_v<TTensorDest>::value, mem::memorytype_v<TTensorSrc>::value, decltype(dest()), decltype(src()), TTensorDest, TTensorSrc>::HostCopier;
    using DeviceCopier = typename detail::CopierDecider<CopierSeq, mem::memorytype_v<TTensorDest>::value, mem::memorytype_v<TTensorSrc>::value, decltype(dest()), decltype(src()), TTensorDest, TTensorSrc>::DeviceCopier;

    // Pass both host and device versions through compiler, otherwise kernel will not be compiled properly
    INSTANTIATE_HOST(HostCopier::copy, INSTANTIATE_ARG(TTensorDest&&), INSTANTIATE_ARG(TTensorSrc&&));
    INSTANTIATE_DEVICE(DeviceCopier::copy, INSTANTIATE_ARG(TTensorDest&&), INSTANTIATE_ARG(TTensorSrc&&));

#if TT_IS_ON_HOST
    HostCopier::copy(util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src));
#else
    DeviceCopier::copy(util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src));
#endif
  }
};

} // end of ns op

template <typename TTensorDest, typename TTensorSrc>
__host__ __device__
void copy(TTensorDest&& dest, TTensorSrc&& src)
{
  op::AutoCopier<>::copy(util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src));
}

} // end of ns template_tensors
