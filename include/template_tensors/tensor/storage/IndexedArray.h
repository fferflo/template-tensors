namespace template_tensors {

template <typename TDimSeq>
class StoreDimensions;

#define ThisType IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>
#define SuperType IndexedPointerTensor< \
                    ThisType, \
                    TElementType, \
                    TIndexStrategy, \
                    mem::memorytype_v<TArrayType>::value, \
                    TDimSeq \
                  >

template <typename TArrayType, typename TElementType, typename TIndexStrategy, typename TDimSeq>
class
#ifdef _MSC_VER
__declspec(empty_bases)
#endif
IndexedArrayTensor : public SuperType,
                     public IsIndexedArrayTensor,
                     public StoreDimensions<TDimSeq>
{
private:
  TArrayType m_array;

public:
  // TODO: static_assert(TIndexStrategy().getSize(TDimSeq()) == TArrayType::SIZE, "Invalid storage size"); or storage size or index strategy is dynamic
  static_assert(std::is_same<TElementType, ::array::elementtype_t<TArrayType>>::value, "Invalid storage type");

  using IndexStrategy = TIndexStrategy;

  template <typename... TDimArgTypes, typename TIndexStrategy2, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value && std::is_constructible<TIndexStrategy, TIndexStrategy2&&>::value)>
  __host__ __device__
  IndexedArrayTensor(ExplicitConstructWithDynDims, TArrayType array, TIndexStrategy2&& index_strategy, TDimArgTypes&&... dim_args)
    : SuperType(std::forward<TIndexStrategy2>(index_strategy), std::forward<TDimArgTypes>(dim_args)...)
    , StoreDimensions<TDimSeq>(std::forward<TDimArgTypes>(dim_args)...)
    , m_array(std::move(array))
  {
    // TODO: ASSERT(index_strategy.getSize(std::forward<TDimArgTypes>(dim_args)...) == array.size()); if array has a size
  }

  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  IndexedArrayTensor(ExplicitConstructWithDynDims, TArrayType array, TDimArgTypes&&... dim_args)
    : IndexedArrayTensor(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS,
        std::move(array), TIndexStrategy(), std::forward<TDimArgTypes>(dim_args)...)
  {
  }

  template <typename... TDimArgTypes, typename TIndexStrategy2, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value && std::is_constructible<TIndexStrategy, TIndexStrategy2&&>::value)>
  __host__ __device__
  IndexedArrayTensor(ExplicitConstructWithDynDims, TIndexStrategy2&& index_strategy, TDimArgTypes&&... dim_args)
    : IndexedArrayTensor(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS,
        TArrayType(TT_ARRAY_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, index_strategy.getSize(std::forward<TDimArgTypes>(dim_args)...)),
        std::forward<TIndexStrategy2>(index_strategy), std::forward<TDimArgTypes>(dim_args)...)
  {
  }

  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  IndexedArrayTensor(ExplicitConstructWithDynDims, TDimArgTypes&&... dim_args)
    : IndexedArrayTensor(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS,
        TIndexStrategy(), std::forward<TDimArgTypes>(dim_args)...)
  {
  }

  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value && TArrayType::HAS_DYN_SIZE_CONSTRUCTOR
    && !is_static_v<TDimSeq>::value), bool TDummy = true>
  __host__ __device__
  explicit IndexedArrayTensor(TDimArgTypes&&... dim_args)
    : IndexedArrayTensor(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, std::forward<TDimArgTypes>(dim_args)...)
  {
  }

  template <typename... TDimArgTypes, typename TIndexStrategy2, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value && TArrayType::HAS_DYN_SIZE_CONSTRUCTOR
    && !is_static_v<TDimSeq>::value && std::is_constructible<TIndexStrategy, TIndexStrategy2&&>::value), bool TDummy = true>
  __host__ __device__
  explicit IndexedArrayTensor(TIndexStrategy2&& index_strategy, TDimArgTypes&&... dim_args)
    : IndexedArrayTensor(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, std::forward<TIndexStrategy2>(index_strategy), std::forward<TDimArgTypes>(dim_args)...)
  {
  }

  template <typename... TArrayArgs, ENABLE_IF(std::is_constructible<TArrayType, TArrayArgs...>::value
    && is_static_v<TDimSeq>::value)>
  __host__ __device__
  IndexedArrayTensor(TArrayArgs&&... storage_args)
    : IndexedArrayTensor(TT_EXPLICIT_CONSTRUCT_WITH_STORAGE_ARGS, std::forward<TArrayArgs>(storage_args)...)
  {
  }

  template <typename TTensorType2, ENABLE_IF(is_tensor_v<TTensorType2>::value && are_compatible_dimseqs_v<TDimSeq, dimseq_t<TTensorType2>>::value
    && is_static_v<TDimSeq>::value)>
  __host__ __device__
  IndexedArrayTensor(TTensorType2&& other)
    : IndexedArrayTensor(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, other.dims())
  {
    *this = std::forward<TTensorType2>(other);
  }

  template <typename TOtherArrayType, typename... TDimArgTypes,
    ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value && ::array::is_array_type_v<TOtherArrayType>::value)>
  __host__ __device__
  IndexedArrayTensor(TOtherArrayType&& storage, TDimArgTypes&&... dim_args)
    : IndexedArrayTensor(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS,
        TArrayType(std::forward<TOtherArrayType>(storage)), std::forward<TDimArgTypes>(dim_args)...)
  {
  }

  template <typename... TArrayArgs,
    ENABLE_IF(std::is_constructible<TArrayType, TArrayArgs...>::value && is_static_v<dimseq_t<SuperType>>::value)>
  __host__ __device__
  IndexedArrayTensor(ExplicitConstructWithStorageArgs, TArrayArgs&&... storage_args)
    : IndexedArrayTensor(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS,
      TArrayType(std::forward<TArrayArgs>(storage_args)...), dimseq_t<SuperType>())
  {
  }

  __host__ __device__
  IndexedArrayTensor(const IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>& other)
    : SuperType(static_cast<const SuperType&>(other))
    , StoreDimensions<TDimSeq>(static_cast<const StoreDimensions<TDimSeq>&>(other))
    , m_array(other.m_array)
  {
  }

  __host__ __device__
  IndexedArrayTensor(IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>&& other)
    : SuperType(static_cast<SuperType&&>(other))
    , StoreDimensions<TDimSeq>(static_cast<StoreDimensions<TDimSeq>&&>(other))
    , m_array(std::move(other.m_array))
  {
  }

  __host__ __device__
  IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>&
    operator=(IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>&& other)
  { // This is needed, because the templated version of operator= does not represent a proper assignment operator for this specific class
    static_cast<SuperType&>(*this) = static_cast<SuperType&&>(other);
    static_cast<StoreDimensions<TDimSeq>&>(*this) = static_cast<StoreDimensions<TDimSeq>&&>(other);
    m_array = std::move(other.m_array);
    return *this;
  }

  __host__ __device__
  IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>&
    operator=(const IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>& other)
  { // This is needed, because the templated version of operator= does not represent a proper assignment operator for this specific class
    static_cast<SuperType&>(*this) = static_cast<const SuperType&>(other);
    static_cast<StoreDimensions<TDimSeq>&>(*this) = static_cast<const StoreDimensions<TDimSeq>&>(other);
    m_array = other.m_array;
    return *this;
  }

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  __host__ __device__
  TArrayType& getArray()
  {
    return m_array;
  }

  __host__ __device__
  const TArrayType& getArray() const
  {
    return m_array;
  }

  __host__ __device__
  volatile TArrayType& getArray() volatile
  {
    return m_array;
  }

  __host__ __device__
  const volatile TArrayType& getArray() const volatile
  {
    return m_array;
  }

  template <LAZY_TYPE(TArrayTypeLazy, TArrayType), typename TResultType = decltype(std::declval<TArrayTypeLazy>().data())>
  __host__ __device__
  TResultType data()
  {
    return m_array.data();
  }

  template <LAZY_TYPE(TArrayTypeLazy, TArrayType), typename TResultType = decltype(std::declval<const TArrayTypeLazy>().data())>
  __host__ __device__
  TResultType data() const
  {
    return m_array.data();
  }

  template <LAZY_TYPE(TArrayTypeLazy, TArrayType), typename TResultType = decltype(std::declval<volatile TArrayTypeLazy>().data())>
  __host__ __device__
  TResultType data() volatile
  {
    return m_array.data();
  }

  template <LAZY_TYPE(TArrayTypeLazy, TArrayType), typename TResultType = decltype(std::declval<const volatile TArrayTypeLazy>().data())>
  __host__ __device__
  TResultType data() const volatile
  {
    return m_array.data();
  }

  __host__ __device__
  auto begin()
  RETURN_AUTO(getArray().begin())

  __host__ __device__
  auto begin() const
  RETURN_AUTO(getArray().begin())
  // TODO: specialized iterators should only be here for dense bijective index strategies
  __host__ __device__
  auto end()
  RETURN_AUTO(getArray().end())

  __host__ __device__
  auto end() const
  RETURN_AUTO(getArray().end())

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }
  // TODO: transform array here
  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }

#ifdef __CUDACC__
  template <typename TTensorType, ENABLE_IF(mem::memorytype_v<TTensorType>::value == mem::LOCAL)>
  __host__
  static auto toKernel(TTensorType&& tensor)
  RETURN_AUTO(typename std::decay<TTensorType>::type(std::forward<TTensorType>(tensor)))

  template <typename TTensorType, ENABLE_IF(mem::memorytype_v<TTensorType>::value != mem::LOCAL)>
  __host__
  static auto toKernel(TTensorType&& tensor)
  RETURN_AUTO(SuperType::toKernel(std::forward<TTensorType>(tensor)))
#endif
};
#undef SuperType
#undef ThisType

#ifdef CEREAL_INCLUDED
template <typename TArchive, typename TArrayType, typename TElementType, typename TIndexStrategy, typename TDimSeq, ENABLE_IF(is_static_v<TDimSeq>::value)>
void save(TArchive& archive, const IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>& m)
{
  archive(m.getArray());
  archive(m.getIndexStrategy());
}

template <typename TArchive, typename TArrayType, typename TElementType, typename TIndexStrategy, typename TDimSeq, ENABLE_IF(!is_static_v<TDimSeq>::value), bool TDummy = false>
void save(TArchive& archive, const IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>& m)
{
  template_tensors::VectorXs<non_trivial_dimensions_num_v<TDimSeq>::value> dims = m.dims();
  archive(dims);
  archive(m.getArray());
  archive(m.getIndexStrategy());
}

template <typename TArchive, typename TArrayType, typename TElementType, typename TIndexStrategy, typename TDimSeq, ENABLE_IF(is_static_v<TDimSeq>::value)>
void load(TArchive& archive, IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>& m)
{
  archive(m.getArray());
  archive(m.getIndexStrategy());
}

template <typename TArchive, typename TArrayType, typename TElementType, typename TIndexStrategy, typename TDimSeq, ENABLE_IF(!is_static_v<TDimSeq>::value), bool TDummy = false>
void load(TArchive& archive, IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>& m)
{
  template_tensors::VectorXs<non_trivial_dimensions_num_v<TDimSeq>::value> dims;
  archive(dims);
  TArrayType array;
  archive(array);
  TIndexStrategy index_strategy;
  archive(index_strategy);
  m = IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>(ExplicitConstructWithDynDims(), std::move(array), std::move(index_strategy), std::move(dims));
}
#endif

} // end of ns template_tensors

namespace atomic {

template <typename TArrayType, typename TElementType, typename TIndexStrategy, typename TDimSeq>
struct is_atomic<template_tensors::IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>>
{
  static const bool value = TIndexStrategy::IS_STATIC && is_atomic<TArrayType>::value && template_tensors::is_static_v<TDimSeq>::value;
};

} // end of ns atomic

template <typename TArrayType, typename TElementType, typename TIndexStrategy, typename TDimSeq>
TT_PROCLAIM_TRIVIALLY_RELOCATABLE((template_tensors::IndexedArrayTensor<TArrayType, TElementType, TIndexStrategy, TDimSeq>),
  mem::is_trivially_relocatable_v<TIndexStrategy>::value && mem::is_trivially_relocatable_v<TElementType>::value && mem::is_trivially_relocatable_v<TArrayType>::value);
