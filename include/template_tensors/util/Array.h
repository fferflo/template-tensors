#pragma once

#include <cstring>
#include <type_traits>
#include <metal.hpp>

#include <template_tensors/cuda/Cuda.h>
#include <template_tensors/util/Memory.h>
#include <template_tensors/util/Util.h>
#include <template_tensors/util/Constexpr.h>
#include <template_tensors/util/Assert.h>
#include <template_tensors/util/Ptr.h>

namespace array {

#if TT_CUDA_FUNCTIONS_AVAILABLE
#define TT_ARRAY_SUBCLASS_ITERATORS \
  using Iterator = typename std::conditional<mem::isOnHost<ThisType>(), \
    TElementType*, \
    thrust::device_ptr<TElementType> \
  >::type; \
  using ConstIterator = typename std::conditional<mem::isOnHost<ThisType>(), \
    const TElementType*, \
    thrust::device_ptr<const TElementType> \
  >::type;
#else
#define TT_ARRAY_SUBCLASS_ITERATORS \
  using Iterator = TElementType*; \
  using ConstIterator = const TElementType*;
#endif

#define TT_ARRAY_SUBCLASS_ITERATORS_BEGIN \
  __host__ __device__ \
  Iterator begin() \
  { \
    return Iterator(data()); \
  } \
  __host__ __device__ \
  ConstIterator begin() const \
  { \
    return ConstIterator(data()); \
  } \
  __host__ __device__ \
  ConstIterator cbegin() const \
  { \
    return ConstIterator(data()); \
  }

#define TT_ARRAY_SUBCLASS_ITERATORS_END \
  __host__ __device__ \
  Iterator end() \
  { \
    return Iterator(data() + size()); \
  } \
  __host__ __device__ \
  ConstIterator end() const \
  { \
    return ConstIterator(data() + size()); \
  } \
  __host__ __device__ \
  ConstIterator cend() const \
  { \
    return ConstIterator(data() + size()); \
  }

#define TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE \
  util::constexpr_if<mem::isOnHost<mem::memorytype_v<ThisType>::value>() && TT_IS_ON_DEVICE>([]__host__ __device__(){ \
      ASSERT_(false, "Cannot access host elements from device"); \
    }); \
  util::constexpr_if<mem::isOnDevice<mem::memorytype_v<ThisType>::value>() && TT_IS_ON_HOST>([]__host__ __device__(){ \
      ASSERT_(false, "Cannot access device elements from host"); \
    });

static const metal::int_ DYN = static_cast<metal::int_>(-1);

struct ExplicitConstructWithDynDims
{
};
#define TT_ARRAY_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS (::array::ExplicitConstructWithDynDims())

class IsArray
{
};

template <typename TArrayType>
TVALUE(bool, is_array_type_v, std::is_base_of<IsArray, typename std::decay<TArrayType>::type>::value)
template <typename TArrayType>
TVALUE(metal::int_, size_v, std::decay<TArrayType>::type::SIZE)
template <typename TArrayType>
using elementtype_t = typename std::decay<TArrayType>::type::ElementType;



namespace detail {

template <metal::int_ TIndex>
struct LocalArrayFillHelper
{
  template <typename T>
  __host__ __device__
  static constexpr const T& get(const T& fill)
  {
    return fill;
  }
};

} // end of ns detail

#define ThisType LocalArray<TElementType, TSize>
template <typename TElementType, metal::int_ TSize>
class LocalArray : public IsArray
                 , public mem::HasMemoryType<ThisType, mem::LOCAL>
{
public:
  static_assert(TSize != DYN, "LocalArray cannot have dynamic size");
  static const bool HAS_DYN_SIZE_CONSTRUCTOR = false;
  static const metal::int_ SIZE = TSize;
  using ElementType = TElementType;

  TT_ARRAY_SUBCLASS_ITERATORS

  __host__ __device__
  constexpr LocalArray()
    : m_data()
  {
  }

  template <typename TValue0, typename... TValues, ENABLE_IF(sizeof...(TValues) + 1 == TSize && std::is_convertible<TValue0, TElementType>::value)>
  __host__ __device__
  constexpr LocalArray(TValue0 arg0, TValues... args)
    : m_data{static_cast<TElementType>(arg0), static_cast<TElementType>(args)...}
  {
  }

  template <typename TValueFill, ENABLE_IF(TSize != 1 && TSize < MAX_COMPILE_RECURSION_DEPTH && std::is_convertible<TValueFill, TElementType>::value)>
  __host__ __device__
  constexpr LocalArray(TValueFill fill)
    : LocalArray(metal::iota<metal::number<0>, metal::number<TSize>>(), static_cast<TElementType>(fill))
  {
  }

  template <typename TValueFill, bool TDummy = true, ENABLE_IF(TSize != 1 && TSize >= MAX_COMPILE_RECURSION_DEPTH && std::is_convertible<TValueFill, TElementType>::value)>
  __host__ __device__
  LocalArray(TValueFill fill)
    : m_data()
  {
    for (auto i = 0; i < TSize; i++)
    {
      m_data[i] = fill;
    }
  }

  __host__ __device__
  LocalArray(ExplicitConstructWithDynDims, size_t size)
    : m_data()
  {
    ASSERT(size == TSize, "Incompatible dynamic and static size");
  }

  __host__ __device__
  TElementType& operator[](size_t index)
  {
    ASSERT(TSize != 0 && index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    return m_data[index];
  }

  __host__ __device__
  const TElementType& operator[](size_t index) const
  {
    ASSERT(TSize != 0 && index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    return m_data[index];
  }

  __host__ __device__
  volatile TElementType& operator[](size_t index) volatile
  {
    ASSERT(TSize != 0 && index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    return m_data[index];
  }

  __host__ __device__
  const volatile TElementType& operator[](size_t index) const volatile
  {
    ASSERT(TSize != 0 && index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    return m_data[index];
  }

  __host__ __device__
  constexpr size_t size() const
  {
    return TSize;
  }

  __host__ __device__
  TElementType* data()
  {
    return m_data;
  }

  __host__ __device__
  constexpr const TElementType* data() const
  {
    return m_data;
  }

  __host__ __device__
  volatile TElementType* data() volatile
  {
    return m_data;
  }

  __host__ __device__
  const volatile TElementType* data() const volatile
  {
    return m_data;
  }

  TT_ARRAY_SUBCLASS_ITERATORS_BEGIN
  TT_ARRAY_SUBCLASS_ITERATORS_END

private:
  TElementType m_data[TSize];

  template <metal::int_... TIndices, typename TValueFill>
  __host__ __device__
  constexpr LocalArray(metal::numbers<TIndices...>, TValueFill fill)
    : LocalArray(detail::LocalArrayFillHelper<TIndices>::get(fill)...)
  {
  }
};
#undef ThisType

#define ThisType LocalArray<TElementType, 0>
template <typename TElementType>
class LocalArray<TElementType, 0> : public IsArray
                                  , public mem::HasMemoryType<ThisType, mem::LOCAL>
{
public:
  static const bool HAS_DYN_SIZE_CONSTRUCTOR = false;
  static const metal::int_ SIZE = 0;
  using ElementType = TElementType;

  TT_ARRAY_SUBCLASS_ITERATORS

  __host__ __device__
  constexpr LocalArray()
    : m_dummy(true)
  {
  }

  __host__ __device__
  LocalArray(ExplicitConstructWithDynDims, size_t size)
    : m_dummy(true)
  {
    ASSERT(size == 0, "Incompatible dynamic and static size");
  }

  __host__ __device__
  TElementType& operator[](size_t index)
  {
    ASSERT(false, "Index %llu out of bounds", (long long unsigned int) index);
    return data()[0];
  }

  __host__ __device__
  const TElementType& operator[](size_t index) const
  {
    ASSERT(false, "Index %llu out of bounds", (long long unsigned int) index);
    return data()[0];
  }

  __host__ __device__
  volatile TElementType& operator[](size_t index) volatile
  {
    ASSERT(false, "Index %llu out of bounds", (long long unsigned int) index);
    return data()[0];
  }

  __host__ __device__
  const volatile TElementType& operator[](size_t index) const volatile
  {
    ASSERT(false, "Index %llu out of bounds", (long long unsigned int) index);
    return data()[0];
  }

  __host__ __device__
  constexpr size_t size() const
  {
    return 0;
  }

  __host__ __device__
  TElementType* data()
  {
    return reinterpret_cast<TElementType*>(this);
  }

  __host__ __device__
  constexpr const TElementType* data() const
  {
    return reinterpret_cast<const TElementType*>(this);
  }

  __host__ __device__
  volatile TElementType* data() volatile
  {
    return reinterpret_cast<TElementType*>(this);
  }

  __host__ __device__
  const volatile TElementType* data() const volatile
  {
    return reinterpret_cast<TElementType*>(this);
  }

  TT_ARRAY_SUBCLASS_ITERATORS_BEGIN
  TT_ARRAY_SUBCLASS_ITERATORS_END

private:
  bool m_dummy;
};
#undef ThisType

static_assert(sizeof(LocalArray<float, 3>) == 3 * sizeof(float), "Invalid size");

} // end of ns array
template <typename TElementType, metal::int_ TSize>
TT_PROCLAIM_TRIVIALLY_RELOCATABLE((::array::LocalArray<TElementType, TSize>), mem::is_trivially_relocatable_v<TElementType>::value);
namespace array {

#ifdef CEREAL_INCLUDED
template <typename TArchive, typename TElementType, metal::int_ TSize>
void save(TArchive& archive, const LocalArray<TElementType, TSize>& m)
{
  for (const TElementType& el : m)
  {
    archive(el);
  }
}

template <typename TArchive, typename TElementType, metal::int_ TSize>
void load(TArchive& archive, LocalArray<TElementType, TSize>& m)
{
  for (TElementType& el : m)
  {
    archive(el);
  }
}
#endif



#define ThisType DynamicAllocArray<TElementType, TAllocator>
template <typename TElementType, typename TAllocator>
class DynamicAllocArray : public IsArray
                        , public mem::HasMemoryType<ThisType, TAllocator::MEMORY_TYPE>
{
public:
  static const bool HAS_DYN_SIZE_CONSTRUCTOR = true;
  static const metal::int_ SIZE = DYN;
  using ElementType = TElementType;

  TT_ARRAY_SUBCLASS_ITERATORS

  __host__ __device__
  DynamicAllocArray()
    : m_data(nullptr)
    , m_size(0)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  DynamicAllocArray(size_t size)
    : m_data(TAllocator::template allocate<TElementType>(size))
    , m_size(size)
  {
  }

  __host__ __device__
  DynamicAllocArray(ExplicitConstructWithDynDims, size_t size)
    : DynamicAllocArray(size)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  DynamicAllocArray(TElementType* data, size_t size)
    : m_data(TAllocator::template allocate<TElementType>(size))
    , m_size(size)
  {
    mem::copy<mem::isOnHost<mem::memorytype_v<ThisType>::value>(), mem::isOnHost<mem::memorytype_v<ThisType>::value>()>(m_data, data, size);
  }

  HD_WARNING_DISABLE
  __host__ __device__
  DynamicAllocArray(const DynamicAllocArray<TElementType, TAllocator>& other)
    : m_data(TAllocator::template allocate<TElementType>(other.m_size))
    , m_size(other.m_size)
  {
    mem::copy<mem::memorytype_v<ThisType>::value, mem::memorytype_v<ThisType>::value>(m_data, other.m_data, m_size);
  }

  __host__ __device__
  DynamicAllocArray(DynamicAllocArray<TElementType, TAllocator>&& other)
    : m_data(other.m_data)
    , m_size(other.m_size)
  {
    other.m_data = nullptr;
    other.m_size = 0;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  DynamicAllocArray<TElementType, TAllocator>& operator=(const DynamicAllocArray<TElementType, TAllocator>& other)
  {
    TAllocator::free(m_data);
    m_data = TAllocator::template allocate<TElementType>(other.m_size);
    m_size = other.m_size;
    mem::copy<mem::memorytype_v<ThisType>::value, mem::memorytype_v<ThisType>::value>(m_data, other.m_data, m_size);

    return *this;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  DynamicAllocArray<TElementType, TAllocator>& operator=(DynamicAllocArray<TElementType, TAllocator>&& other)
  {
    TAllocator::free(m_data);
    m_data = other.m_data;
    m_size = other.m_size;

    other.m_data = nullptr;
    other.m_size = 0;

    return *this;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~DynamicAllocArray()
  {
    TAllocator::free(m_data);
  }

  template <bool TDummy = true>
  __host__ __device__
  TElementType& operator[](size_t index)
  {
    ASSERT(index < m_size, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) m_size);
    ASSERT(m_data != nullptr, "Array data pointer is null");
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  template <bool TDummy = true>
  __host__ __device__
  const TElementType& operator[](size_t index) const
  {
    ASSERT(index < m_size, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) m_size);
    ASSERT(m_data != nullptr, "Array data pointer is null");
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  template <bool TDummy = true>
  __host__ __device__
  volatile TElementType& operator[](size_t index) volatile
  {
    ASSERT(index < m_size, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) m_size);
    ASSERT(m_data != nullptr, "Array data pointer is null");
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  template <bool TDummy = true>
  __host__ __device__
  const volatile TElementType& operator[](size_t index) const volatile
  {
    ASSERT(index < m_size, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) m_size);
    ASSERT(m_data != nullptr, "Array data pointer is null");
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  __host__ __device__
  size_t size() const
  {
    return m_size;
  }

  __host__ __device__
  TElementType* data()
  {
    return m_data;
  }

  __host__ __device__
  const TElementType* data() const
  {
    return m_data;
  }

  __host__ __device__
  volatile TElementType* data() volatile
  {
    return m_data;
  }

  __host__ __device__
  const volatile TElementType* data() const volatile
  {
    return m_data;
  }

  TT_ARRAY_SUBCLASS_ITERATORS_BEGIN
  TT_ARRAY_SUBCLASS_ITERATORS_END

private:
  TElementType* m_data;
  size_t m_size;
};
#undef ThisType

#ifdef CEREAL_INCLUDED
template <typename TArchive, typename TElementType, typename TAllocator>
void save(TArchive& archive, const DynamicAllocArray<TElementType, TAllocator>& m)
{
  archive(m.size());
  for (const TElementType& el : m)
  {
    archive(el);
  }
}

template <typename TArchive, typename TElementType, typename TAllocator>
void load(TArchive& archive, DynamicAllocArray<TElementType, TAllocator>& m)
{
  size_t size;
  archive(size);
  m = DynamicAllocArray<TElementType, TAllocator>(size);

  for (TElementType& el : m)
  {
    archive(el);
  }
}
#endif

template <typename TElementType, typename TAllocator>
using AllocArray = DynamicAllocArray<TElementType, TAllocator>;



#define ThisType StaticAllocArray<TElementType, TAllocator, TSize>
template <typename TElementType, typename TAllocator, metal::int_ TSize>
class StaticAllocArray : public IsArray
                       , public mem::HasMemoryType<ThisType, mem::memorytype_v<TAllocator>::value>
{
public:
  static_assert(TSize != DYN, "StaticAllocArray cannot have dynamic size");
  static const bool HAS_DYN_SIZE_CONSTRUCTOR = false;
  static const metal::int_ SIZE = TSize;
  using ElementType = TElementType;

  TT_ARRAY_SUBCLASS_ITERATORS

  __host__ __device__
  StaticAllocArray()
    : m_data(TAllocator::template allocate<TElementType>(TSize))
  {
  }

  __host__ __device__
  StaticAllocArray(ExplicitConstructWithDynDims, size_t size)
    : StaticAllocArray()
  {
    ASSERT(size == TSize, "Incompatible dynamic and static size");
  }

  __host__ __device__
  StaticAllocArray(const StaticAllocArray<TElementType, TAllocator, TSize>& other)
    : m_data(TAllocator::template allocate<TElementType>(TSize))
  {
    mem::copy<mem::memorytype_v<ThisType>::value, mem::memorytype_v<ThisType>::value, SIZE>(m_data, other.m_data, TSize);
  }

  __host__ __device__
  StaticAllocArray(StaticAllocArray<TElementType, TAllocator, TSize>&& other)
    : m_data(other.m_data)
  {
    other.m_data = nullptr;
  }

  __host__ __device__
  StaticAllocArray<TElementType, TAllocator, TSize>& operator=(const StaticAllocArray<TElementType, TAllocator, TSize>& other)
  {
    TAllocator::free(m_data);
    m_data = TAllocator::template allocate<TElementType>(TSize);
    mem::copy<mem::memorytype_v<ThisType>::value, mem::memorytype_v<ThisType>::value, SIZE>(m_data, other.m_data, TSize);

    return *this;
  }

  __host__ __device__
  StaticAllocArray<TElementType, TAllocator, TSize>& operator=(StaticAllocArray<TElementType, TAllocator, TSize>&& other)
  {
    TAllocator::free(m_data);
    m_data = other.m_data;

    other.m_data = nullptr;

    return *this;
  }

  __host__ __device__
  ~StaticAllocArray()
  {
    TAllocator::free(m_data);
  }

  template <bool TDummy = true>
  __host__ __device__
  TElementType& operator[](size_t index)
  {
    ASSERT(index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    ASSERT(m_data != nullptr, "Array data pointer is null");
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  template <bool TDummy = true>
  __host__ __device__
  const TElementType& operator[](size_t index) const
  {
    ASSERT(index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    ASSERT(m_data != nullptr, "Array data pointer is null");
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  template <bool TDummy = true>
  __host__ __device__
  volatile TElementType& operator[](size_t index) volatile
  {
    ASSERT(index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    ASSERT(m_data != nullptr, "Array data pointer is null");
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  template <bool TDummy = true>
  __host__ __device__
  const volatile TElementType& operator[](size_t index) const volatile
  {
    ASSERT(index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    ASSERT(m_data != nullptr, "Array data pointer is null");
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  __host__ __device__
  constexpr size_t size() const
  {
    return TSize;
  }

  __host__ __device__
  TElementType* data()
  {
    return m_data;
  }

  __host__ __device__
  const TElementType* data() const
  {
    return m_data;
  }

  __host__ __device__
  volatile TElementType* data() volatile
  {
    return m_data;
  }

  __host__ __device__
  const volatile TElementType* data() const volatile
  {
    return m_data;
  }

  TT_ARRAY_SUBCLASS_ITERATORS_BEGIN
  TT_ARRAY_SUBCLASS_ITERATORS_END

private:
  TElementType* m_data;
};
#undef ThisType





// TODO: this is a dangerous class and should be removed/ only be used with managed pointers?
#define ThisType ReferenceArray<TElementType, TMemoryType, TSize, TPointerType>
template <typename TElementType, mem::MemoryType TMemoryType, metal::int_ TSize, typename TPointerType = TElementType*>
class ReferenceArray : public IsArray
                     , public mem::HasMemoryType<ThisType, TMemoryType>
{
public:
  static const bool HAS_DYN_SIZE_CONSTRUCTOR = false;
  static const metal::int_ SIZE = TSize;
  using ElementType = TElementType;

  TT_ARRAY_SUBCLASS_ITERATORS

  __host__ __device__
  ReferenceArray()
    : m_data(nullptr)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ReferenceArray(TPointerType data)
    : m_data(data)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ReferenceArray(TPointerType data, size_t size)
    : m_data(data)
  {
    ASSERT(size == TSize, "Incompatible runtime dimensions");
  }

  HD_WARNING_DISABLE
  template <typename TOtherArrayType, ENABLE_IF(is_array_type_v<TOtherArrayType>::value)>
  __host__ __device__
  ReferenceArray(TOtherArrayType&& other)
    : m_data(other.data())
  {
    static_assert(SIZE == DYN || size_v<TOtherArrayType>::value == DYN || SIZE == size_v<TOtherArrayType>::value, "Sizes must be equal");
    static_assert(std::is_same<elementtype_t<TOtherArrayType>, ElementType>::value
      || std::is_same<const elementtype_t<TOtherArrayType>, ElementType>::value, "Must have same element type");
    static_assert(TMemoryType == mem::memorytype_v<TOtherArrayType>::value, "Must have same memory type");

    ASSERT(other.size() == TSize, "Incompatible runtime dimensions");
  }

  HD_WARNING_DISABLE
  template <typename TOtherArrayType, ENABLE_IF(is_array_type_v<TOtherArrayType>::value)>
  __host__ __device__
  ReferenceArray<TElementType, TMemoryType, TSize, TPointerType>& operator=(TOtherArrayType&& other)
  {
    static_assert(SIZE == DYN || size_v<TOtherArrayType>::value == DYN || SIZE == size_v<TOtherArrayType>::value, "Sizes must be equal");
    static_assert(std::is_same<elementtype_t<TOtherArrayType>, ElementType>::value
      || std::is_same<const elementtype_t<TOtherArrayType>, ElementType>::value, "Must have same element type");
    static_assert(TMemoryType == mem::memorytype_v<TOtherArrayType>::value, "Must have same memory type");
    m_data = other.data();

    return *this;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~ReferenceArray()
  {
  }

  HD_WARNING_DISABLE
  template <bool TDummy = true>
  __host__ __device__
  TElementType& operator[](size_t index)
  {
    ASSERT(m_data, "Array data pointer is null");
    ASSERT(index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return ptr::toRawPointer(m_data)[index];
  }

  HD_WARNING_DISABLE
  template <bool TDummy = true>
  __host__ __device__
  const TElementType& operator[](size_t index) const
  {
    ASSERT(m_data, "Array data pointer is null");
    ASSERT(index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return ptr::toRawPointer(m_data)[index];
  }

  HD_WARNING_DISABLE
  template <bool TDummy = true>
  __host__ __device__
  volatile TElementType& operator[](size_t index) volatile
  {
    ASSERT(m_data, "Array data pointer is null");
    ASSERT(index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  HD_WARNING_DISABLE
  template <bool TDummy = true>
  __host__ __device__
  const volatile TElementType& operator[](size_t index) const volatile
  {
    ASSERT(m_data, "Array data pointer is null");
    ASSERT(index < TSize, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) TSize);
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return ptr::toRawPointer(m_data)[index];
  }

  __host__ __device__
  constexpr size_t size() const
  {
    return TSize;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  TPointerType data()
  {
    return m_data;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  const TPointerType data() const
  {
    return m_data;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  volatile TPointerType data() volatile
  {
    return m_data;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  const volatile TPointerType data() const volatile
  {
    return m_data;
  }

  TT_ARRAY_SUBCLASS_ITERATORS_BEGIN
  TT_ARRAY_SUBCLASS_ITERATORS_END

private:
  TPointerType m_data;
};
#undef ThisType

#define ThisType ReferenceArray<TElementType, TMemoryType, ::array::DYN, TPointerType>
template <typename TElementType, mem::MemoryType TMemoryType, typename TPointerType>
class ReferenceArray<TElementType, TMemoryType, ::array::DYN, TPointerType> : public IsArray
                                                                          , public mem::HasMemoryType<ThisType, TMemoryType>
{
public:
  static const bool HAS_DYN_SIZE_CONSTRUCTOR = false;
  static const metal::int_ SIZE = ::array::DYN;
  using ElementType = TElementType;

  TT_ARRAY_SUBCLASS_ITERATORS

  __host__ __device__
  ReferenceArray()
    : m_data(nullptr)
    , m_size(0)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ReferenceArray(TPointerType data, size_t size)
    : m_data(std::move(data))
    , m_size(size)
  {
  }

  HD_WARNING_DISABLE
  template <typename TOtherArrayType, ENABLE_IF(is_array_type_v<TOtherArrayType>::value)>
  __host__ __device__
  ReferenceArray(TOtherArrayType&& other)
    : m_data(other.data())
    , m_size(other.size())
  {
    static_assert(SIZE == DYN || size_v<TOtherArrayType>::value == DYN || SIZE == size_v<TOtherArrayType>::value, "Sizes must be equal");
    static_assert(std::is_same<elementtype_t<TOtherArrayType>, ElementType>::value
      || std::is_same<const elementtype_t<TOtherArrayType>, ElementType>::value, "Must have same element type");
    static_assert(TMemoryType == mem::memorytype_v<TOtherArrayType>::value, "Must have same memory type");
  }

  HD_WARNING_DISABLE
  template <typename TOtherArrayType, ENABLE_IF(is_array_type_v<TOtherArrayType>::value)>
  __host__ __device__
  ReferenceArray<TElementType, TMemoryType, ::array::DYN, TPointerType>& operator=(TOtherArrayType&& other)
  {
    static_assert(SIZE == DYN || size_v<TOtherArrayType>::value == DYN || SIZE == size_v<TOtherArrayType>::value, "Sizes must be equal");
    static_assert(std::is_same<elementtype_t<TOtherArrayType>, ElementType>::value
      || std::is_same<const elementtype_t<TOtherArrayType>, ElementType>::value, "Must have same element type");
    static_assert(TMemoryType == mem::memorytype_v<TOtherArrayType>::value, "Must have same memory type");
    m_data = other.data();
    m_size = other.size();

    return *this;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~ReferenceArray()
  {
  }

  HD_WARNING_DISABLE
  template <bool TDummy = true>
  __host__ __device__
  TElementType& operator[](size_t index)
  {
    ASSERT(m_data, "Array data pointer is null");
    ASSERT(index < m_size, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) m_size);
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return ptr::toRawPointer(m_data)[index];
  }

  HD_WARNING_DISABLE
  template <bool TDummy = true>
  __host__ __device__
  const TElementType& operator[](size_t index) const
  {
    ASSERT(m_data, "Array data pointer is null");
    ASSERT(index < m_size, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) m_size);
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return ptr::toRawPointer(m_data)[index];
  }

  HD_WARNING_DISABLE
  template <bool TDummy = true>
  __host__ __device__
  volatile TElementType& operator[](size_t index) volatile
  {
    ASSERT(m_data, "Array data pointer is null");
    ASSERT(index < m_size, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) m_size);
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return m_data[index];
  }

  HD_WARNING_DISABLE
  template <bool TDummy = true>
  __host__ __device__
  const volatile TElementType& operator[](size_t index) const volatile
  {
    ASSERT(m_data, "Array data pointer is null");
    ASSERT(index < m_size, "Index %llu out of range %llu", (long long unsigned int) index, (long long unsigned int) m_size);
    TT_ARRAY_SUBCLASS_ASSERT_ELEMENT_ACCESS_MEMORY_TYPE
    return ptr::toRawPointer(m_data)[index];
  }

  __host__ __device__
  size_t size() const
  {
    return m_size;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  TPointerType data()
  {
    return m_data;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  const TPointerType data() const
  {
    return m_data;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  volatile TPointerType data() volatile
  {
    return m_data;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  const volatile TPointerType data() const volatile
  {
    return m_data;
  }

  TT_ARRAY_SUBCLASS_ITERATORS_BEGIN
  TT_ARRAY_SUBCLASS_ITERATORS_END

private:
  TPointerType m_data;
  size_t m_size;
};
#undef ThisType

template <typename TArray, ENABLE_IF(::array::is_array_type_v<TArray&&>::value)>
__host__ __device__
auto ref(TArray&& array)
RETURN_AUTO(ReferenceArray<ptr::value_t<decltype(std::declval<TArray&&>().data())>, mem::memorytype_v<TArray&&>::value, size_v<TArray&&>::value>(std::forward<TArray>(array)))

template <mem::MemoryType TMemoryType, metal::int_ TSize, typename TPointerType, ENABLE_IF(!::array::is_array_type_v<TPointerType&&>::value)>
__host__ __device__
auto ref(TPointerType&& ptr)
RETURN_AUTO(ReferenceArray<ptr::value_t<TPointerType&&>, TMemoryType, TSize, typename std::decay<TPointerType>::type>(std::forward<TPointerType>(ptr)))

template <mem::MemoryType TMemoryType, metal::int_ TSize = ::array::DYN, typename TPointerType, ENABLE_IF(!::array::is_array_type_v<TPointerType&&>::value)>
__host__ __device__
auto ref(TPointerType&& ptr, size_t size)
RETURN_AUTO(ReferenceArray<ptr::value_t<TPointerType&&>, TMemoryType, TSize, typename std::decay<TPointerType>::type>(std::forward<TPointerType>(ptr), size))



template <typename TArray1, typename TArray2, ENABLE_IF(::array::is_array_type_v<TArray1&&>::value && ::array::is_array_type_v<TArray2&&>::value)>
bool eq(TArray1&& array1, TArray2&& array2)
{
  if (array1.size() != array2.size())
  {
    return false;
  }
  // TODO: this should use template_tensors::eq
  for (size_t i = 0; i < array1.size(); i++)
  {
    if (array1[i] != array2[i])
    {
      return false;
    }
  }
  return true;
}

} // end of ns array

namespace atomic {

template <typename T>
struct is_atomic;

template <typename TElementType, metal::int_ TSize>
struct is_atomic<::array::LocalArray<TElementType, TSize>>
{
  static const bool value = is_atomic<TElementType>::value && TSize == 1;
};

} // end of ns atomic
