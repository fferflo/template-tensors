#include <string>

namespace template_tensors {

#define ThisType FromStdString<TStdString>
#define SuperType IndexedPointerTensor< \
                                        ThisType, \
                                        decltype(std::declval<TStdString>()[0]), \
                                        template_tensors::RowMajor, \
                                        mem::HOST, \
                                        template_tensors::DimSeq<template_tensors::DYN> \
                              >
template <typename TStdString>
class FromStdString : public SuperType
{
public:
  __host__
  FromStdString(TStdString string)
    : SuperType(string.length())
    , m_string(string)
  {
  }

  __host__
  FromStdString(const FromStdString<TStdString>& other)
    : SuperType(other.dims())
    , m_string(other.m_string)
  {
  }

  __host__
  FromStdString(FromStdString<TStdString>&& other)
    : SuperType(other.dims())
    , m_string(static_cast<TStdString&&>(other.m_string))
  {
  }

  __host__
  FromStdString<TStdString>& operator=(const FromStdString<TStdString>& other)
  {
    m_string = other.m_string;
    return *this;
  }

  __host__
  FromStdString<TStdString>& operator=(FromStdString<TStdString>&& other)
  {
    m_string = static_cast<TStdString&&>(other.m_string);
    return *this;
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(self.data())
  FORWARD_ALL_QUALIFIERS(data, data2)

  template <metal::int_ TIndex>
  __host__
  dim_t getDynDim() const
  {
    return TIndex == 0 ? m_string.length() : 1;
  }

  __host__
  dim_t getDynDim(size_t index) const
  {
    return index == 0 ? m_string.length() : 1;
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

private:
  TStdString m_string;
};
#undef SuperType
#undef ThisType

template <typename TStdString>
__host__
auto fromStdString(TStdString&& string)
RETURN_AUTO(FromStdString<util::store_member_t<TStdString>>(util::forward<TStdString>(string)))

template <typename TAllocatorIn = util::EmptyDefaultType, typename TVectorType,
  typename TAllocator = TT_WITH_DEFAULT_TYPE(TAllocatorIn, std::allocator<decay_elementtype_t<TVectorType>>), ENABLE_IF(is_vector_v<TVectorType>::value)>
__host__
std::basic_string<decay_elementtype_t<TVectorType>, TAllocator> toStdstring(TVectorType&& string)
{
  static_assert(TT_IS_ON_DEVICE || mem::isOnHost<mem::memorytype_v<TVectorType>::value>(), "Cannot convert device string to std::basic_string");
  std::basic_string<decay_elementtype_t<TVectorType>, TAllocator> result(string.rows());
  template_tensors::copy(fromStdString(result), string);
  return result;
}

} // end of ns template_tensors
