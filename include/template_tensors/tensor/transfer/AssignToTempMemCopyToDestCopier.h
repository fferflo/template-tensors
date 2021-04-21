namespace template_tensors {

namespace op {

struct AssignToTempMemCopyToDestCopier
{
  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v,
       !(!TIsOnHost && (mem::isOnHost<mem::memorytype_v<TTensorDest>::value, TIsOnHost>() || mem::isOnDevice<mem::memorytype_v<TTensorSrc>::value, TIsOnHost>()))
    && is_indexed_pointer_tensor_v<TTensorDest>::value
  )

  template <typename TTensorDest, typename TTensorSrc>
  __host__ __device__
  static void copy(TTensorDest&& dest, TTensorSrc&& src)
  {
    static_assert(is_tensor_v<TTensorDest>::value && is_tensor_v<TTensorSrc>::value, "Can only copy tensors");
    static_assert(are_compatible_dimseqs_v<TTensorDest, TTensorSrc>::value, "Incompatible static dimensions");
    ASSERT(areSameDimensions(dest.dims(), src.dims()), "Incompatible runtime dimensions");

    LocalOrAllocTensorT<
      decay_elementtype_t<TTensorDest>,
      mem::alloc::default_for<mem::memorytype_v<TTensorSrc>::value>,
      indexstrategy_t<TTensorDest>, dimseq_t<TTensorSrc>> temp(TENSOR_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, dest.getIndexStrategy(), dest.dims());

    temp = src;
    StorageArrayCopier::copy(dest, temp);
  }
};

} // end of ns op

} // end of ns tensor
