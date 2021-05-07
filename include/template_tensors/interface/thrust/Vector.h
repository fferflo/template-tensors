#ifdef __CUDACC__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace thrust {

namespace detail {

template <typename TVector>
struct ThrustMemoryType;

template <typename TElementType, typename TAllocator>
struct ThrustMemoryType<thrust::device_vector<TElementType, TAllocator>>
{
  static const mem::MemoryType value = mem::DEVICE;
};

template <typename TElementType, typename TAllocator>
struct ThrustMemoryType<thrust::host_vector<TElementType, TAllocator>>
{
  static const mem::MemoryType value = mem::HOST;
};

template <mem::MemoryType TMemoryType, typename TElementType>
struct VectorByMemoryType
{
  static_assert(std::is_same<TElementType, void>::value, "No thrust vector defined for this memory type");
};

template <typename TElementType>
struct VectorByMemoryType<mem::HOST, TElementType>
{
  using type = thrust::host_vector<TElementType>;
};

template <typename TElementType>
struct VectorByMemoryType<mem::DEVICE, TElementType>
{
  using type = thrust::device_vector<TElementType>;
};

template <typename TElementType>
struct VectorByMemoryType<mem::LOCAL, TElementType>
{
  using type =
#if TT_IS_ON_HOST
    thrust::host_vector<TElementType>
#else
    thrust::device_vector<TElementType>
#endif
  ;
};

} // end of ns detail

template <mem::MemoryType TMemoryType, typename TElementType>
using vector_for = typename detail::VectorByMemoryType<TMemoryType, TElementType>::type;

template <typename TThrustVector>
TVALUE(mem::MemoryType, memorytype_v, detail::ThrustMemoryType<typename std::decay<TThrustVector>::type>::value)

} // end of ns thrust

namespace template_tensors {

#define ThisType FromThrustVector<TThrustVector>
#define SuperType IndexedPointerTensor< \
                                        ThisType, \
                                        decltype(ptr::toRawPointer(std::declval<TThrustVector>().data())[0]), \
                                        template_tensors::ColMajor, \
                                        thrust::memorytype_v<TThrustVector>::value, \
                                        template_tensors::DimSeq<template_tensors::DYN> \
                              >

template <typename TThrustVector>
class FromThrustVector : public SuperType
{
private:
  TThrustVector m_vector;

public:
  using ThrustVector = TThrustVector;
  using ThrustElementType = typename std::decay<TThrustVector>::type::value_type;

  template <bool TDummy = true>
  __host__
  FromThrustVector()
    : SuperType(0)
    , m_vector()
  {
  }

  __host__
  FromThrustVector(TThrustVector vector)
    : SuperType(vector.size())
    , m_vector(vector)
  {
  }

  __host__
  FromThrustVector(size_t num)
    : SuperType(num)
    , m_vector(num)
  {
  }

  __host__
  FromThrustVector(const FromThrustVector<TThrustVector>& other)
    : SuperType(static_cast<const SuperType&>(other))
    , m_vector(other.m_vector)
  {
  }

  __host__
  FromThrustVector(FromThrustVector<TThrustVector>&& other)
    : SuperType(static_cast<SuperType&&>(other))
    , m_vector(static_cast<TThrustVector&&>(other.m_vector))
  {
  }

  __host__
  ~FromThrustVector()
  {
  }

  __host__
  FromThrustVector<TThrustVector>& operator=(const FromThrustVector<TThrustVector>& other)
  {
    static_cast<SuperType&>(*this) = static_cast<const SuperType&>(other);
    m_vector = other.m_vector;
    return *this;
  }

  __host__
  FromThrustVector<TThrustVector>& operator=(FromThrustVector<TThrustVector>&& other)
  {
    static_cast<SuperType&>(*this) = static_cast<SuperType&&>(other);
    m_vector = static_cast<TThrustVector&&>(other.m_vector);
    return *this;
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(ptr::toRawPointer(self.m_vector.data()))
  FORWARD_ALL_QUALIFIERS(data, data2)

  template <metal::int_ TIndex>
  __host__
  dim_t getDynDim() const
  {
    return TIndex == 0 ? m_vector.size() : 1;
  }

  __host__
  dim_t getDynDim(size_t index) const
  {
    return index == 0 ? m_vector.size() : 1;
  }

  __host__
  TThrustVector& getVector()
  {
    return m_vector;
  }

  __host__
  const TThrustVector& getVector() const
  {
    return m_vector;
  }

  __host__
  auto begin()
  RETURN_AUTO(m_vector.begin())

  __host__
  auto begin() const
  RETURN_AUTO(m_vector.begin())

  __host__
  auto end()
  RETURN_AUTO(m_vector.end())

  __host__
  auto end() const
  RETURN_AUTO(m_vector.end())

  template <typename TTensorType, LAZY_TYPE(TElementType, ThrustElementType),
    ENABLE_IF(thrust::memorytype_v<TThrustVector>::value == mem::DEVICE)>
  __host__
  static auto toHost(TTensorType&& tensor)
  RETURN_AUTO(
    FromThrustVector<thrust::host_vector<TElementType>>(thrust::host_vector<TElementType>(tensor.getVector()))
  )

  template <typename TTensorType, ENABLE_IF(thrust::memorytype_v<TThrustVector>::value == mem::HOST)>
  __host__
  static auto toHost(TTensorType&& tensor)
  RETURN_AUTO(
    SuperType::toHost(util::forward<TTensorType>(tensor))
  )

  template <typename TTensorType, LAZY_TYPE(TElementType, ThrustElementType),
    ENABLE_IF(thrust::memorytype_v<TThrustVector>::value == mem::HOST)>
  __host__
  static auto toDevice(TTensorType&& tensor)
  RETURN_AUTO(
    FromThrustVector<thrust::device_vector<TElementType>>(thrust::device_vector<TElementType>(tensor.getVector()))
  )

  template <typename TTensorType, ENABLE_IF(thrust::memorytype_v<TThrustVector>::value == mem::DEVICE)>
  __host__
  static auto toDevice(TTensorType&& tensor)
  RETURN_AUTO(
    SuperType::toDevice(util::forward<TTensorType>(tensor))
  )
};
#undef SuperType
#undef ThisType





template <typename TThrustVector>
__host__
auto fromThrust(TThrustVector&& vector)
RETURN_AUTO(FromThrustVector<util::store_member_t<TThrustVector>>(util::forward<TThrustVector>(vector)))

template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType, ENABLE_IF(mem::isOnHost<mem::memorytype_v<TTensorType>::value>()),
  typename TElementType = TT_WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__
thrust::host_vector<TElementType> toThrust(TTensorType&& tensor)
{
  static_assert(math::lte(non_trivial_dimensions_num_v<TTensorType>::value, 1UL), "Must be vector");

  thrust::host_vector<TElementType> result(tensor.rows());
  fromThrust(result) = tensor;
  return result;
}

template <typename TElementTypeIn = util::EmptyDefaultType, typename TTensorType, ENABLE_IF(mem::isOnDevice<mem::memorytype_v<TTensorType>::value>()),
  typename TElementType = TT_WITH_DEFAULT_TYPE(TElementTypeIn, decay_elementtype_t<TTensorType>)>
__host__
thrust::device_vector<TElementType> toThrust(TTensorType&& tensor)
{
  static_assert(math::lte(non_trivial_dimensions_num_v<TTensorType>::value, 1UL), "Must be vector");

  thrust::device_vector<TElementType> result(tensor.rows());
  fromThrust(result) = tensor;
  return result;
}

} // end of ns tensor

#endif
