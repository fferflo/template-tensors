#ifdef DLPACK_INCLUDED

#include <dlpack/dlpack.h>
#include <boost/make_unique.hpp>

namespace template_tensors {

class InvalidDlPackShapeException : public std::exception
{
public:
  InvalidDlPackShapeException(size_t got, size_t expected)
    : m_message(std::string("Invalid DlPack shape. Got rank ") + util::to_string(got) + " expected rank " + util::to_string(expected))
  {
  }

  template <typename TVector1, typename TVector2>
  InvalidDlPackShapeException(TVector1&& got, TVector2&& expected)
    : m_message(std::string("Invalid DlPack shape. Got dimensions ") + util::to_string(got) + " expected dimensions " + util::to_string(expected))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class InvalidDlPackElementTypeException : public std::exception
{
public:
  InvalidDlPackElementTypeException(
      DLDataTypeCode expected_code, uint8_t expected_bits, uint16_t expected_lanes,
      DLDataTypeCode got_code, uint8_t got_bits, uint16_t got_lanes)
    : m_message(
      std::string("Invalid DlPack elementtype. Got {code=") + util::to_string(got_code) + ", bits=" + util::to_string(got_bits) + ", lanes=" + util::to_string(got_lanes) + "}"
      " Expected {code=" + util::to_string(expected_code) + ", bits=" + util::to_string(expected_bits) + ", lanes=" + util::to_string(expected_lanes) + "}"
    )
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class InvalidDlPackMemoryType : public std::exception
{
public:
  InvalidDlPackMemoryType(DLDeviceType got, DLDeviceType expected)
    : m_message(std::string("Invalid DlPack memory type. Got type ") + util::to_string(got) + " expected type " + util::to_string(expected))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

template <typename TElementType>
struct dlpack_elementtype;

#define DLPACK_ELEMENTTYPE(TYPENAME, CODE, BITS, LANES) \
  template <> \
  struct dlpack_elementtype<TYPENAME> \
  { \
    static DLDataTypeCode getCode() \
    { \
      return CODE; \
    } \
    static uint8_t getBits() \
    { \
      return BITS; \
    } \
    static uint16_t getLanes() \
    { \
      return LANES; \
    } \
  }

DLPACK_ELEMENTTYPE(float, kDLFloat, 32, 1);
DLPACK_ELEMENTTYPE(double, kDLFloat, 64, 1);
DLPACK_ELEMENTTYPE(uint8_t, kDLUInt, 8, 1);
DLPACK_ELEMENTTYPE(uint16_t, kDLUInt, 16, 1);
DLPACK_ELEMENTTYPE(uint32_t, kDLUInt, 32, 1);
DLPACK_ELEMENTTYPE(uint64_t, kDLUInt, 64, 1);
DLPACK_ELEMENTTYPE(int8_t, kDLInt, 8, 1);
DLPACK_ELEMENTTYPE(int16_t, kDLInt, 16, 1);
DLPACK_ELEMENTTYPE(int32_t, kDLInt, 32, 1);
DLPACK_ELEMENTTYPE(int64_t, kDLInt, 64, 1);

#undef DLPACK_ELEMENTTYPE

template <mem::MemoryType TMemoryType>
struct dlpack_devicetype;

template <>
struct dlpack_devicetype<mem::HOST>
{
  static const DLDeviceType value = kDLCPU;
};

template <>
struct dlpack_devicetype<mem::DEVICE>
{
  static const DLDeviceType value = kDLCUDA;
};

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

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

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
    throw InvalidDlPackShapeException(dl->dl_tensor.ndim, TRank);
  }
  if (dlpack_elementtype<TElementType>::getCode() != dl->dl_tensor.dtype.code
   || dlpack_elementtype<TElementType>::getBits() != dl->dl_tensor.dtype.bits
   || dlpack_elementtype<TElementType>::getLanes() != dl->dl_tensor.dtype.lanes)
  {
    throw InvalidDlPackElementTypeException(
      dlpack_elementtype<TElementType>::getCode(), dlpack_elementtype<TElementType>::getBits(), dlpack_elementtype<TElementType>::getLanes(),
      static_cast<DLDataTypeCode>(dl->dl_tensor.dtype.code), dl->dl_tensor.dtype.bits, dl->dl_tensor.dtype.lanes
    );
  }
  if (dlpack_devicetype<TMemoryType>::value != dl->dl_tensor.device.device_type)
  {
    throw InvalidDlPackMemoryType(dl->dl_tensor.device.device_type, dlpack_devicetype<TMemoryType>::value);
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
  manager_ctx->dl.dl_tensor.device.device_type = dlpack_devicetype<MEMORY_TYPE>::value;
  manager_ctx->dl.dl_tensor.device.device_id = 0; // TODO: get different device ids (compare https://github.com/tensorflow/tensorflow/blob/22e07fb204386768e5bcbea563641ea11f96ceb8/tensorflow/c/eager/dlpack.cc#L110)
  manager_ctx->dl.dl_tensor.ndim = manager_ctx->shape.size();
  manager_ctx->dl.dl_tensor.dtype.code = dlpack_elementtype<ElementType>::getCode();
  manager_ctx->dl.dl_tensor.dtype.bits = dlpack_elementtype<ElementType>::getBits();
  manager_ctx->dl.dl_tensor.dtype.lanes = dlpack_elementtype<ElementType>::getLanes();
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
