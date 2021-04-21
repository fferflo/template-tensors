namespace template_tensors {

#define ThisType DummyTensor<TElementType, TMemoryType, TDimSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        TMemoryType, \
                                        TDimSeq \
                              >
template <typename TElementType, mem::MemoryType TMemoryType, typename TDimSeq>
class DummyTensor : public SuperType
{
public:
  __host__ __device__
  DummyTensor()
    : SuperType()
  {
    ASSERT_(false, "Dummy tensor functions should never be called");
  }

  TENSOR_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static TElementType getElement(TThisType&& self, TCoordArgTypes&&... coords)
  {
    ASSERT_(false, "Dummy tensor functions should never be called");
    return INSTANTIATE_ARG(TElementType);
  }
  TENSOR_FORWARD_ELEMENT_ACCESS(getElement)

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    ASSERT_(false, "Dummy tensor functions should never be called");
    return 1;
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    ASSERT_(false, "Dummy tensor functions should never be called");
    return 1;
  }

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  DummyTensor<TElementType, TMemoryType, TDimSeq> map(TTransform transform)
  {
    ASSERT_(false, "Dummy tensor functions should never be called");
    return *this;
  }

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  DummyTensor<TElementType, TMemoryType, TDimSeq> map(TTransform transform) const
  {
    ASSERT_(false, "Dummy tensor functions should never be called");
    return *this;
  }
};
#undef SuperType
#undef ThisType

template <typename TElementType, mem::MemoryType TMemoryType, typename TDimSeq>
auto dummy()
RETURN_AUTO(DummyTensor<TElementType, TMemoryType, TDimSeq>())

template <typename TTensorType,
  mem::MemoryType TMemoryType = mem::memorytype_v<TTensorType>::value,
  typename TDimSeq = template_tensors::dimseq_t<TTensorType>,
  typename TElementType = template_tensors::decay_elementtype_t<TTensorType>>
auto dummy_like()
RETURN_AUTO(DummyTensor<TElementType, TMemoryType, TDimSeq>())

} // end of ns tensor
