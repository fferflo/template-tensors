namespace template_tensors {

namespace op {

template <typename TIndexStrategy = ColMajor>
struct AssignToTempMemCopyToTempAssignToDestCopier
{
  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v,
    !(!TIsOnHost && (mem::isOnHost<mem::memorytype_v<TTensorDest>::value, TIsOnHost>() || mem::isOnDevice<mem::memorytype_v<TTensorSrc>::value, TIsOnHost>()))
  )

  template <typename TTensorDest, typename TTensorSrc>
  __host__ __device__
  static void copy(TTensorDest&& dest, TTensorSrc&& src)
  {
    static_assert(is_tensor_v<TTensorDest>::value && is_tensor_v<TTensorSrc>::value, "Can only copy tensors");
    static_assert(are_compatible_dimseqs_v<TTensorDest, TTensorSrc>::value, "Incompatible static dimensions");
    ASSERT(areSameDimensions(dest.dims(), src.dims()), "Incompatible runtime dimensions");

    using IntermediateElementType = typename std::conditional<sizeof(decay_elementtype_t<TTensorDest>) < sizeof(decay_elementtype_t<TTensorSrc>),
      decay_elementtype_t<TTensorDest>,
      decay_elementtype_t<TTensorSrc>
    >::type;

    LocalOrAllocTensorT<
      IntermediateElementType,
      mem::alloc::default_for<mem::memorytype_v<TTensorSrc>::value>,
      TIndexStrategy, dimseq_t<TTensorSrc>> temp1(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, src.dims());

    LocalOrAllocTensorT<
      IntermediateElementType,
      mem::alloc::default_for<mem::memorytype_v<TTensorDest>::value>,
      TIndexStrategy, dimseq_t<TTensorDest>> temp2(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, dest.dims());

    temp1 = src;
    StorageArrayCopier::copy(temp2, temp1);
    dest = temp2;
  }
};

} // end of ns op

} // end of ns template_tensors