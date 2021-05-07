#include <vector>

namespace std {

namespace detail {

template <typename TVector>
struct StdVectorMemoryType
{
  static_assert(std::is_same<TVector, void>::value, "Not a std::vector");
};

template <typename TElementType, typename TAllocator>
struct StdVectorMemoryType<std::vector<TElementType, TAllocator>>
{
  static const mem::MemoryType value = mem::HOST;
};

template <mem::MemoryType TMemoryType, typename TElementType>
struct StdVectorByMemoryType
{
  static_assert(std::is_same<TElementType, void>::value, "No std::vector defined for this memory type");
};

template <typename TElementType>
struct StdVectorByMemoryType<mem::HOST, TElementType>
{
  using type = std::vector<TElementType>;
};

template <typename TElementType>
struct StdVectorByMemoryType<mem::LOCAL, TElementType>
{
  using type = std::vector<TElementType>;
};

} // end of ns detail

template <mem::MemoryType TMemoryType, typename TElementType>
using vector_for = typename detail::StdVectorByMemoryType<TMemoryType, TElementType>::type;

template <typename TStdVector>
TVALUE(mem::MemoryType, memorytype_v, detail::StdVectorMemoryType<typename std::decay<TStdVector>::type>::value)

} // end of ns std

namespace template_tensors {

#define ThisType FromStdVector<TStdVector>
#define SuperType IndexedPointerTensor< \
                                        ThisType, \
                                        decltype(std::declval<TStdVector>()[0]), \
                                        template_tensors::ColMajor, \
                                        mem::HOST, \
                                        template_tensors::DimSeq<template_tensors::DYN> \
                              >
template <typename TStdVector>
class FromStdVector : public SuperType
{
private:
  TStdVector m_vector;

public:
  template <bool TDummy = true>
  __host__
  FromStdVector()
    : SuperType(0)
    , m_vector()
  {
  }

  __host__
  FromStdVector(TStdVector vector)
    : SuperType(vector.size())
    , m_vector(vector)
  {
  }

  __host__
  FromStdVector(size_t num)
    : SuperType(num)
    , m_vector(num)
  {
  }

  __host__
  FromStdVector(const FromStdVector<TStdVector>& other)
    : SuperType(other.dims())
    , m_vector(other.m_vector)
  {
  }

  __host__
  FromStdVector(FromStdVector<TStdVector>&& other)
    : SuperType(other.dims())
    , m_vector(static_cast<TStdVector&&>(other.m_vector))
  {
  }

  __host__
  FromStdVector<TStdVector>& operator=(const FromStdVector<TStdVector>& other)
  {
    m_vector = other.m_vector;
    return *this;
  }

  __host__
  FromStdVector<TStdVector>& operator=(FromStdVector<TStdVector>&& other)
  {
    m_vector = static_cast<TStdVector&&>(other.m_vector);
    return *this;
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(self.m_vector.data())
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

  template <typename TTransform>
  __host__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }

  __host__
  TStdVector& getVector()
  {
    return m_vector;
  }

  __host__
  const TStdVector& getVector() const
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
};
#undef SuperType
#undef ThisType

template <typename TStdVector>
__host__
auto fromStdVector(TStdVector&& vector)
RETURN_AUTO(FromStdVector<util::store_member_t<TStdVector>>(util::forward<TStdVector>(vector)))

template <typename TAllocatorIn = util::EmptyDefaultType, typename TVectorType,
  typename TAllocator = TT_WITH_DEFAULT_TYPE(TAllocatorIn, std::allocator<decay_elementtype_t<TVectorType>>), ENABLE_IF(is_vector_v<TVectorType>::value)>
__host__
std::vector<decay_elementtype_t<TVectorType>, TAllocator> toStdVector(TVectorType&& vector)
{
  static_assert(TT_IS_ON_DEVICE || mem::isOnHost<mem::memorytype_v<TVectorType>::value>(), "Cannot convert device vector to std::vector");
  std::vector<decay_elementtype_t<TVectorType>, TAllocator> result(vector.rows());
  template_tensors::copy(fromStdVector(result), vector);
  return result;
}

} // end of ns tensor
