#pragma once

namespace field {

namespace wrap {

namespace detail {

template <typename TDims>
class repeatf
{
private:
  TDims dims;

  template <typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(template_tensors::elwise(math::functor::positive_fmod(), std::forward<TCoordVector>(coords), std::forward<TThisType>(self).dims))

public:
  __host__ __device__
  repeatf(TDims dims)
    : dims(dims)
  {
  }

  FORWARD_ALL_QUALIFIERS(operator(), get)
};

template <typename TDims, typename TIntType = int32_t>
class repeati
{
private:
  TDims dims;

  template <typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(template_tensors::elwise(math::functor::positive_mod(), template_tensors::static_cast_to<TIntType>(std::forward<TCoordVector>(coords)), template_tensors::static_cast_to<TIntType>(std::forward<TThisType>(self).dims)))

public:
  __host__ __device__
  repeati(TDims dims)
    : dims(dims)
  {
  }

  FORWARD_ALL_QUALIFIERS(operator(), get)
};

} // end of ns detail

template <typename TField, typename TDims>
__host__ __device__
auto repeat(TField&& field, TDims&& dims)
RETURN_AUTO(
  field::transform(std::forward<TField>(field), detail::repeatf<TDims>(std::forward<TDims>(dims)))
)

template <typename TIntType = int32_t, typename TField, typename TDims>
__host__ __device__
auto repeat_discrete(TField&& field, TDims&& dims)
RETURN_AUTO(
  field::transform(std::forward<TField>(field), detail::repeati<TDims, TIntType>(std::forward<TDims>(dims)))
)

namespace detail {

template <metal::int_ TRank, typename TIntType = int32_t, typename TTensorType, typename TDims>
__host__ __device__
auto repeat(TTensorType&& tensor, TDims&& dims)
RETURN_AUTO(
  field::wrap::repeat_discrete<TIntType>(field::fromSupplier<TRank>(std::forward<TTensorType>(tensor)), std::forward<TDims>(dims))
)

} // end of ns detail

template <metal::int_ TRank, typename TIntType = int32_t, typename TTensorType, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
__host__ __device__
auto repeat(TTensorType&& tensor)
RETURN_AUTO(
  detail::repeat<TRank, TIntType>(std::forward<TTensorType>(tensor), template_tensors::eval(std::forward<TTensorType>(tensor).template dims<TRank>()))
)



template <typename TField, typename TSize>
__host__ __device__
auto clamp(TField&& field, TSize&& size)
RETURN_AUTO(
  field::transform(std::forward<TField>(field), template_tensors::functor::clamp(template_tensors::VectorXT<template_tensors::decay_elementtype_t<TSize>, template_tensors::rows_v<TSize>::value>(0), std::forward<TSize>(size)))
)

namespace detail {

template <metal::int_ TRank, typename TTensorType, typename TDims>
__host__ __device__
auto clamp(TTensorType&& tensor, TDims&& dims)
RETURN_AUTO(
  field::wrap::clamp(field::fromSupplier<TRank>(std::forward<TTensorType>(tensor)), std::forward<TDims>(dims))
)

} // end of ns detail

template <metal::int_ TRank, typename TTensorType, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
__host__ __device__
auto clamp(TTensorType&& tensor)
RETURN_AUTO(
  detail::clamp<TRank>(std::forward<TTensorType>(tensor), template_tensors::eval(std::forward<TTensorType>(tensor).template dims<TRank>() - 1))
)



template <typename TField, typename TSize, typename TConstant>
class Constant
{
private:
  TField m_field;
  TSize m_size;
  TConstant m_constant;

  template <typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get(TThisType&& self, TCoordVector&& coords) -> typename std::decay<decltype(m_field(coords))>::type
  {
    if (template_tensors::all(0 <= coords && coords < self.m_size))
    {
      return self.m_field(std::forward<TCoordVector>(coords));
    }
    else
    {
      return self.m_constant;
    }
  }

public:
  static const metal::int_ RANK = std::decay<TField>::type::RANK;

  __host__ __device__
  Constant(TField field, TSize size, TConstant constant)
    : m_field(field)
    , m_size(size)
    , m_constant(constant)
  {
  }

  FORWARD_ALL_QUALIFIERS(operator(), get)
};

template <typename TField, typename TSize, typename TConstant>
__host__ __device__
auto constant(TField&& field, TSize&& size, TConstant&& constant)
RETURN_AUTO(
  Constant<TField, TSize, typename std::decay<TConstant&&>::type>
    (std::forward<TField>(field), std::forward<TSize>(size), std::forward<TConstant>(constant))
)

namespace detail {

template <metal::int_ TRank, typename TTensorType, typename TDims, typename TConstant>
__host__ __device__
auto constant_helper(TTensorType&& tensor, TDims&& dims, TConstant&& constant)
RETURN_AUTO(
  field::wrap::constant(field::fromSupplier<TRank>(std::forward<TTensorType>(tensor)), std::forward<TDims>(dims), std::forward<TConstant>(constant))
)

} // end of ns detail

template <metal::int_ TRank, typename TTensorType, typename TConstant, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
__host__ __device__
auto constant(TTensorType&& tensor, TConstant&& constant)
RETURN_AUTO(
  detail::constant_helper<TRank>(std::forward<TTensorType>(tensor), template_tensors::eval(tensor.template dims<TRank>()), std::forward<TConstant>(constant))
)

} // end of ns wrap

} // end of ns field
