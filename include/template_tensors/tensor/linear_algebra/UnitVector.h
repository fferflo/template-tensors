namespace template_tensors {

#define ThisType UnitVector<TElementType, TRows, TDirection>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        template_tensors::DimSeq<TRows> \
                              >
template <typename TElementType, size_t TRows = template_tensors::DYN, size_t TDirection = template_tensors::DYN>
class UnitVector : public SuperType, public StoreDimensions<template_tensors::DimSeq<TRows>>
{
public:
  static_assert(TDirection < TRows, "Direction out of range");

  template <typename... TDimArgTypes>
  __host__ __device__
  UnitVector(TDimArgTypes&&... dims)
    : SuperType(util::forward<TDimArgTypes>(dims)...)
    , StoreDimensions<template_tensors::DimSeq<TRows>>(util::forward<TDimArgTypes>(dims)...)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, size_t row)
  RETURN_AUTO(
    row == TDirection ? 1 : 0
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SIZE_T_N(getElement, 1)

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }
};
#undef SuperType
#undef ThisType

#define ThisType UnitVector<TElementType, TRows, DYN>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        template_tensors::DimSeq<TRows> \
                              >

template <typename TElementType, size_t TRows>
class UnitVector<TElementType, TRows, DYN> : public SuperType, public StoreDimensions<template_tensors::DimSeq<TRows>>
{
public:
  template <typename... TDimArgTypes>
  __host__ __device__
  UnitVector(size_t direction, TDimArgTypes&&... dims)
    : SuperType(util::forward<TDimArgTypes>(dims)...)
    , StoreDimensions<template_tensors::DimSeq<TRows>>(util::forward<TDimArgTypes>(dims)...)
    , m_direction(direction)
  {
    ASSERT(direction < TRows, "Direction out of range");
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, size_t row)
  RETURN_AUTO(
    row == self.m_direction ? 1 : 0
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SIZE_T_N(getElement, 1)

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }

private:
  size_t m_direction;
};
#undef SuperType
#undef ThisType

#define ThisType UnitVector<TElementType, DYN, TDirection>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        template_tensors::DimSeq<DYN> \
                              >

template <typename TElementType, size_t TDirection>
class UnitVector<TElementType, DYN, TDirection> : public SuperType, public StoreDimensions<template_tensors::DimSeq<DYN>>
{
public:
  template <typename... TDimArgTypes>
  __host__ __device__
  UnitVector(TDimArgTypes&&... dims)
    : SuperType(util::forward<TDimArgTypes>(dims)...)
    , StoreDimensions<template_tensors::DimSeq<DYN>>(util::forward<TDimArgTypes>(dims)...)
  {
    ASSERT(TDirection < this->rows(), "Direction out of range");
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, size_t row)
  RETURN_AUTO(
    row == TDirection ? 1 : 0
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SIZE_T_N(getElement, 1)

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }
};
#undef SuperType
#undef ThisType

#define ThisType UnitVector<TElementType, DYN, DYN>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        template_tensors::DimSeq<DYN> \
                              >

template <typename TElementType>
class UnitVector<TElementType, DYN, DYN> : public SuperType, public StoreDimensions<template_tensors::DimSeq<DYN>>
{
public:
  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  UnitVector(size_t direction, TDimArgTypes&&... dims)
    : SuperType(util::forward<TDimArgTypes>(dims)...)
    , StoreDimensions<template_tensors::DimSeq<DYN>>(util::forward<TDimArgTypes>(dims)...)
    , m_direction(direction)
  {
    ASSERT(direction < this->rows(), "Direction out of range");
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, size_t row)
  RETURN_AUTO(
    row == self.m_direction ? 1 : 0
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SIZE_T_N(getElement, 1)

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }

private:
  size_t m_direction;
};
#undef SuperType
#undef ThisType





#define ThisType UnitVectors<TElementType, TRows>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        template_tensors::DimSeq<TRows> \
                              >

template <typename TElementType, size_t TRows>
class UnitVectors : public SuperType, public StoreDimensions<template_tensors::DimSeq<TRows>>
{
public:
  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  UnitVectors(TDimArgTypes&&... dims)
    : SuperType(util::forward<TDimArgTypes>(dims)...)
    , StoreDimensions<template_tensors::DimSeq<TRows>>(util::forward<TDimArgTypes>(dims)...)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, size_t row)
  RETURN_AUTO(
    UnitVector<TElementType, TRows>(row, self.rows())
  )
 TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SIZE_T_N(getElement, 1)

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }
};
#undef SuperType
#undef ThisType

} // end of ns tensor
