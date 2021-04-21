namespace template_tensors {

namespace op {

struct StorageArrayCopier
{
  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v,
       have_same_indexstrategy_v<TTensorDest, TTensorSrc>::value
    && !(!TIsOnHost && (mem::isOnHost<mem::memorytype_v<TTensorDest>::value, TIsOnHost>() || mem::isOnDevice<mem::memorytype_v<TTensorSrc>::value, TIsOnHost>()))
    && std::is_same<decay_elementtype_t<TTensorDest>, decay_elementtype_t<TTensorSrc>>::value
  )

  template <typename TTensorDest, typename TTensorSrc>
  __host__ __device__
  static void copy(TTensorDest&& dest, TTensorSrc&& src)
  {
    static_assert(are_compatible_dimseqs_v<TTensorDest, TTensorSrc>::value, "Incompatible static dimensions");
    static_assert(std::is_same<indexstrategy_t<TTensorDest>, indexstrategy_t<TTensorSrc>>::value, "Storages must have the same indexing strategy");
    static_assert(std::is_same<decay_elementtype_t<TTensorDest>, decay_elementtype_t<TTensorSrc>>::value, "Must have same element types");
    ASSERT(areSameDimensions(dest.dims(), src.dims()), "Incompatible runtime dimensions");
    ASSERT(dest.getIndexStrategy() == src.getIndexStrategy(), "Different index strategies given");

    const size_t STATIC_SIZE_DEST = indexed_size_v<TTensorDest>::value;
    const size_t STATIC_SIZE_SRC = indexed_size_v<TTensorSrc>::value;
    const size_t NUM =
        STATIC_SIZE_DEST != mem::DYN ? STATIC_SIZE_DEST
      : STATIC_SIZE_SRC != mem::DYN ? STATIC_SIZE_SRC
      : mem::DYN;

    mem::copy<mem::memorytype_v<TTensorDest>::value, mem::memorytype_v<TTensorSrc>::value, NUM>
      (dest.data(), src.data(), dest.size());
  }
};

} // end of ns op

} // end of ns tensor
