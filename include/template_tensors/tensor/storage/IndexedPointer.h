namespace template_tensors {

namespace detail {

template <bool TIsStatic>
struct IndexedStorageElementAccess;

template <>
struct IndexedStorageElementAccess<true>
{
  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    util::decay_if<std::is_rvalue_reference<TThisType&&>::value>(
      ptr::toRawPointer(self.data())
        [self.getIndexStrategy().toIndex(dimseq_t<TThisType>(), std::forward<TCoordArgTypes>(coords)...)]
    )
  )
};

template <>
struct IndexedStorageElementAccess<false>
{
  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    util::decay_if<std::is_rvalue_reference<TThisType&&>::value>(
      ptr::toRawPointer(self.data())
        [self.getIndexStrategy().toIndex(self.template dims<non_trivial_dimensions_num_v<TThisType>::value>(), std::forward<TCoordArgTypes>(coords)...)]
    )
  )
};

} // end of ns detail

#define SuperType TensorBase< \
    TThisType, \
    TMemoryType, \
    TDimSeq \
  >

template <typename TThisType, typename TElementType, typename TIndexStrategy, mem::MemoryType TMemoryType, typename TDimSeq>
class
#ifdef _MSC_VER
__declspec(empty_bases)
#endif
IndexedPointerTensor : public SuperType,
                       private TIndexStrategy,
                       public IsIndexedPointerTensor
{
public:
  template <typename... TDimArgTypes,
    ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  IndexedPointerTensor(TDimArgTypes&&... dim_args)
    : SuperType(std::forward<TDimArgTypes>(dim_args)...)
    , TIndexStrategy()
  {
  }

  template <typename... TDimArgTypes, typename TIndexStrategy2,
    ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value && !are_dim_args_v<TIndexStrategy>::value && std::is_constructible<TIndexStrategy, TIndexStrategy2&&>::value)>
  __host__ __device__
  IndexedPointerTensor(TIndexStrategy2&& index_strategy, TDimArgTypes&&... dim_args)
    : SuperType(std::forward<TDimArgTypes>(dim_args)...)
    , TIndexStrategy(std::forward<TIndexStrategy2>(index_strategy))
  {
  }

  __host__ __device__
  IndexedPointerTensor(const IndexedPointerTensor<TThisType, TElementType, TIndexStrategy, TMemoryType, TDimSeq>& other)
    : SuperType(static_cast<const SuperType&>(other))
    , TIndexStrategy(static_cast<const TIndexStrategy&>(other))
  {
  }

  __host__ __device__
  IndexedPointerTensor(IndexedPointerTensor<TThisType, TElementType, TIndexStrategy, TMemoryType, TDimSeq>&& other)
    : SuperType(static_cast<SuperType&&>(other))
    , TIndexStrategy(static_cast<TIndexStrategy&&>(other))
  {
  }

  template <typename... TDimArgTypes, typename TIndexStrategy2, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value && std::is_constructible<TIndexStrategy, TIndexStrategy2&&>::value)>
  __host__ __device__
  IndexedPointerTensor(ExplicitConstructWithDynDims, TIndexStrategy2&& index_strategy, TDimArgTypes&&... dim_args)
    : SuperType(std::forward<TDimArgTypes>(dim_args)...)
    , TIndexStrategy(std::forward<TIndexStrategy2>(index_strategy))
  {
  }

  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  IndexedPointerTensor(ExplicitConstructWithDynDims, TDimArgTypes&&... dim_args)
    : SuperType(std::forward<TDimArgTypes>(dim_args)...)
    , TIndexStrategy()
  {
    // TODO: ASSERT(index_strategy.getSize(std::forward<TDimArgTypes>(dim_args)...) == array.size()); if array has a size
  }

  __host__ __device__
  IndexedPointerTensor<TThisType, TElementType, TIndexStrategy, TMemoryType, TDimSeq>&
    operator=(IndexedPointerTensor<TThisType, TElementType, TIndexStrategy, TMemoryType, TDimSeq>&& other)
  { // This is needed, because the templated version of operator= does not represent a proper assignment operator for this specific class
    static_cast<SuperType&>(*this) = static_cast<SuperType&&>(other);
    static_cast<TIndexStrategy&>(*this) = static_cast<TIndexStrategy&&>(other);
    return *this;
  }

  __host__ __device__
  IndexedPointerTensor<TThisType, TElementType, TIndexStrategy, TMemoryType, TDimSeq>&
    operator=(const IndexedPointerTensor<TThisType, TElementType, TIndexStrategy, TMemoryType, TDimSeq>& other)
  { // This is needed, because the templated version of operator= does not represent a proper assignment operator for this specific class
    static_cast<SuperType&>(*this) = static_cast<const SuperType&>(other);
    static_cast<TIndexStrategy&>(*this) = static_cast<const TIndexStrategy&>(other);
    return *this;
  }

  TT_TENSOR_SUBCLASS_ASSIGN(TThisType)

  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(static_cast<util::copy_qualifiers_t<TThisType, TThisType2&&>>(self).data())
  FORWARD_ALL_QUALIFIERS(data, data2)

  TT_TENSOR_SUBCLASS_FORWARD_ELEMENT_ACCESS(detail::IndexedStorageElementAccess<is_static_v<TDimSeq>::value>::getElement)

  __host__ __device__
  TIndexStrategy& getIndexStrategy()
  {
    return *this;
  }

  __host__ __device__
  const TIndexStrategy& getIndexStrategy() const
  {
    return *this;
  }

  __host__ __device__
  volatile TIndexStrategy& getIndexStrategy() volatile
  {
    return *this;
  }

  __host__ __device__
  const volatile TIndexStrategy& getIndexStrategy() const volatile
  {
    return *this;
  }

  __host__ __device__
  size_t size() const
  {
    return this->getIndexStrategy().getSize(this->dims());
  }

  __host__ __device__
  size_t size() const volatile
  {
    return this->getIndexStrategy().getSize(this->dims());
  }

  template <typename TTensorType,
    ENABLE_IF(mem::memorytype_v<TTensorType>::value == mem::DEVICE),
    typename TResultTensor = LocalOrAllocTensorT<decay_elementtype_t<TTensorType&&>, mem::alloc::host_heap, indexstrategy_t<TTensorType&&>, dimseq_t<TTensorType&&>>
  >
  __host__
  static TResultTensor toHost(TTensorType&& tensor)
  {
    TResultTensor result(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, tensor.getIndexStrategy(), tensor.dims());
    result = std::forward<TTensorType>(tensor);
    return result;
  }

  template <typename TTensorType, ENABLE_IF(mem::memorytype_v<TTensorType>::value != mem::DEVICE)>
  __host__
  static auto toHost(TTensorType&& tensor)
  RETURN_AUTO(
    SuperType::toHost(std::forward<TTensorType>(tensor))
  )

#ifdef __CUDACC__
  template <typename TTensorType,
    ENABLE_IF(mem::memorytype_v<TTensorType>::value == mem::HOST || mem::memorytype_v<TTensorType>::value == mem::LOCAL),
    typename TResultTensor = AllocTensorTEx<decay_elementtype_t<TTensorType&&>, mem::alloc::device, indexstrategy_t<TTensorType&&>, dimseq_t<TTensorType&&>>
  >
  __host__
  static TResultTensor toDevice(TTensorType&& tensor)
  { // TODO: here and toHost -> copy with same memory layout and index strategy instead
    TResultTensor result(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, tensor.getIndexStrategy(), tensor.dims());
    result = std::forward<TTensorType>(tensor);
    return result;
  }

  template <typename TTensorType, ENABLE_IF(mem::memorytype_v<TTensorType>::value == mem::DEVICE)>
  __host__
  static auto toDevice(TTensorType&& tensor)
  RETURN_AUTO(
    SuperType::toDevice(std::forward<TTensorType>(tensor))
  )

  template <typename TTensorType, ENABLE_IF(mem::memorytype_v<TTensorType>::value == mem::LOCAL || mem::memorytype_v<TTensorType>::value == mem::HOST)>
  __host__
  static void toKernel(TTensorType&& tensor)
  {
    static_assert(mem::memorytype_v<TTensorType>::value != mem::HOST, "Host allocated memory cannot be passed to kernel");
    static_assert(mem::memorytype_v<TTensorType>::value != mem::LOCAL, "Pointer to local memory cannot be passed to kernel");
  }

  template <typename TTensorType,
    ENABLE_IF(mem::memorytype_v<TTensorType>::value == mem::DEVICE)>
  __host__
  static auto toKernel(TTensorType&& tensor)
  RETURN_AUTO(
    template_tensors::ref(std::forward<TTensorType>(tensor))
  )
#endif
};
#undef SuperType

} // end of ns template_tensors
