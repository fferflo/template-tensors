#ifdef DLPACK_INCLUDED

#include <dlpack/dlpack.h>
#include <boost/make_unique.hpp>

namespace template_tensors {

#define ThisType FromDlPack<TElementType, TRank, TMemoryType>
#define SuperType IndexedPointerTensor< \
                                        ThisType, \
                                        TElementType, \
                                        template_tensors::Stride<TRank>, \
                                        TMemoryType, \
                                        template_tensors::dyn_dimseq_t<TRank> \
                              >
template <typename TElementType, metal::int_ TRank, mem::MemoryType TMemoryType>
class FromDlPack : public SuperType
{
public:
  __host__
  FromDlPack(SafeDLManagedTensor&& data)
    : SuperType(
        template_tensors::Stride<TRank>(template_tensors::ref<template_tensors::ColMajor, mem::HOST, TRank>(data->dl_tensor.strides)),
        template_tensors::ref<template_tensors::ColMajor, mem::HOST, TRank>(data->dl_tensor.shape)
      )
    , m_data(std::move(data))
  {
  }

  __host__
  FromDlPack(const FromDlPack<TElementType, TRank, TMemoryType>& other) = delete;

  __host__
  FromDlPack(FromDlPack<TElementType, TRank, TMemoryType>&& other)
    : SuperType(static_cast<SuperType&&>(other))
    , m_data(std::move(other.m_data))
  {
  }

  __host__
  FromDlPack<TElementType, TRank, TMemoryType>& operator=(const FromDlPack<TElementType, TRank, TMemoryType>& other) = delete;

  __host__
  FromDlPack<TElementType, TRank, TMemoryType>& operator=(FromDlPack<TElementType, TRank, TMemoryType>&& other)
  {
    static_cast<SuperType&>(*this) = static_cast<SuperType&&>(other);
    m_data = std::move(other.m_data);
  }

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(reinterpret_cast<TElementType*>(reinterpret_cast<uint8_t*>(self.m_data->dl_tensor.data) + self.m_data->dl_tensor.byte_offset))
  FORWARD_ALL_QUALIFIERS(data, data2)

  template <metal::int_ TIndex>
  __host__
  dim_t getDynDim() const
  {
    return TIndex < TRank ? m_data->dl_tensor.shape[TIndex] : 1;
  }

  __host__
  dim_t getDynDim(size_t index) const
  {
    return index < TRank ? m_data->dl_tensor.shape[index] : 1;
  }

private:
  SafeDLManagedTensor m_data;
};
#undef SuperType
#undef ThisType

template <typename TElementType, metal::int_ TRank, mem::MemoryType TMemoryType>
__host__
FromDlPack<TElementType, TRank, TMemoryType> fromDlPack(SafeDLManagedTensor&& dl)
{
  if (dl->dl_tensor.ndim != TRank)
  {
    throw template_tensors::InvalidDlPackShapeException(dl->dl_tensor.ndim, TRank);
  }
  if (template_tensors::dlpack_elementtype<TElementType>::getCode() != dl->dl_tensor.dtype.code
   || template_tensors::dlpack_elementtype<TElementType>::getBits() != dl->dl_tensor.dtype.bits
   || template_tensors::dlpack_elementtype<TElementType>::getLanes() != dl->dl_tensor.dtype.lanes)
  {
    throw template_tensors::InvalidDlPackElementTypeException(
      template_tensors::dlpack_elementtype<TElementType>::getCode(), dlpack_elementtype<TElementType>::getBits(), template_tensors::dlpack_elementtype<TElementType>::getLanes(),
      static_cast<DLDataTypeCode>(dl->dl_tensor.dtype.code), dl->dl_tensor.dtype.bits, dl->dl_tensor.dtype.lanes
    );
  }
  if (template_tensors::dlpack_devicetype<TMemoryType>::value != dl->dl_tensor.device.device_type)
  {
    throw InvalidDlPackMemoryType(dl->dl_tensor.device.device_type, template_tensors::dlpack_devicetype<TMemoryType>::value);
  }

  return FromDlPack<TElementType, TRank, TMemoryType>(std::move(dl));
}

namespace detail {

template <typename TTensorPtr>
struct DLManagedTensorContext
{
  static void deleter(DLManagedTensor* dl)
  {
    DLManagedTensorContext<TTensorPtr>* manager = reinterpret_cast<DLManagedTensorContext<TTensorPtr>*>(dl->manager_ctx);
    delete manager;
  }

  template <typename TTensorPtrIn>
  DLManagedTensorContext(TTensorPtrIn&& ptr)
    : ptr(std::forward<TTensorPtrIn>(ptr))
  {
    static const metal::int_ RANK = non_trivial_dimensions_num_v<decltype(*ptr)>::value;
    shape.resize(RANK);
    strides.resize(RANK);
    template_tensors::fromStdVector(shape) = this->ptr->template dims<RANK>();
    template_tensors::fromStdVector(strides) = this->ptr->getIndexStrategy().template toStride<RANK>(this->ptr->dims());
  }

  ~DLManagedTensorContext()
  {
  }

  TTensorPtr ptr;
  DLManagedTensor dl;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

} // end of ns detail

template <typename TTensorPtr,
  typename TDummy = decltype(indexstrategy_t<decltype(*std::declval<TTensorPtr>())>().toStride())>
SafeDLManagedTensor toDlPack(TTensorPtr&& tensor)
{
  using ManagerContext = detail::DLManagedTensorContext<typename std::decay<TTensorPtr>::type>;
  auto* manager_ctx = new ManagerContext(std::forward<TTensorPtr>(tensor));

  using TensorType = decltype(*tensor);
  using ElementType = template_tensors::decay_elementtype_t<TensorType>;
  static const mem::MemoryType MEMORY_TYPE = mem::memorytype_v<TensorType>::value;

  manager_ctx->dl.deleter = &ManagerContext::deleter;
  manager_ctx->dl.manager_ctx = manager_ctx;
  manager_ctx->dl.dl_tensor.data = manager_ctx->ptr->data();
  manager_ctx->dl.dl_tensor.device.device_type = template_tensors::dlpack_devicetype<MEMORY_TYPE>::value;
  manager_ctx->dl.dl_tensor.device.device_id = 0; // TODO: get different device ids (compare https://github.com/tensorflow/tensorflow/blob/22e07fb204386768e5bcbea563641ea11f96ceb8/tensorflow/c/eager/dlpack.cc#L110)
  manager_ctx->dl.dl_tensor.ndim = manager_ctx->shape.size();
  manager_ctx->dl.dl_tensor.dtype.code = template_tensors::dlpack_elementtype<ElementType>::getCode();
  manager_ctx->dl.dl_tensor.dtype.bits = template_tensors::dlpack_elementtype<ElementType>::getBits();
  manager_ctx->dl.dl_tensor.dtype.lanes = template_tensors::dlpack_elementtype<ElementType>::getLanes();
  manager_ctx->dl.dl_tensor.shape = manager_ctx->shape.data();
  manager_ctx->dl.dl_tensor.strides = manager_ctx->strides.data();
  manager_ctx->dl.dl_tensor.byte_offset = 0;

  return SafeDLManagedTensor(&manager_ctx->dl);
}

template <typename TTensorType,
  typename TDummy = decltype(indexstrategy_t<TTensorType>().toStride()), typename TDummy2 = void>
SafeDLManagedTensor toDlPack(TTensorType&& tensor)
{
  return toDlPack(::boost::make_unique<typename std::decay<TTensorType>::type>(std::forward<TTensorType>(tensor)));
}

} // end of ns template_tensors

#endif
