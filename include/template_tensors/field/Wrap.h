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
  RETURN_AUTO(template_tensors::elwise(math::functor::positive_fmod(), util::forward<TCoordVector>(coords), util::forward<TThisType>(self).dims))

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
  RETURN_AUTO(template_tensors::elwise(math::functor::positive_mod(), template_tensors::static_cast_to<TIntType>(util::forward<TCoordVector>(coords)), template_tensors::static_cast_to<TIntType>(util::forward<TThisType>(self).dims)))

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
  field::transform(util::forward<TField>(field), detail::repeatf<util::store_member_t<TDims&&>>(util::forward<TDims>(dims)))
)

template <typename TIntType = int32_t, typename TField, typename TDims>
__host__ __device__
auto repeat_discrete(TField&& field, TDims&& dims)
RETURN_AUTO(
  field::transform(util::forward<TField>(field), detail::repeati<util::store_member_t<TDims&&>, TIntType>(util::forward<TDims>(dims)))
)

namespace detail {

template <size_t TRank, typename TIntType = int32_t, typename TTensorType, typename TDims>
__host__ __device__
auto repeat(TTensorType&& tensor, TDims&& dims)
RETURN_AUTO(
  field::wrap::repeat_discrete<TIntType>(field::fromSupplier<TRank>(util::forward<TTensorType>(tensor)), util::forward<TDims>(dims))
)

} // end of ns detail

template <size_t TRank, typename TIntType = int32_t, typename TTensorType, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
__host__ __device__
auto repeat(TTensorType&& tensor)
RETURN_AUTO(
  detail::repeat<TRank, TIntType>(util::forward<TTensorType>(tensor), template_tensors::eval(util::forward<TTensorType>(tensor).template dims<TRank>()))
)



template <typename TField, typename TSize>
__host__ __device__
auto clamp(TField&& field, TSize&& size)
RETURN_AUTO(
  field::transform(util::forward<TField>(field), template_tensors::functor::clamp(template_tensors::VectorXT<template_tensors::decay_elementtype_t<TSize>, template_tensors::rows_v<TSize>::value>(0), util::forward<TSize>(size)))
)

namespace detail {

template <size_t TRank, typename TTensorType, typename TDims>
__host__ __device__
auto clamp(TTensorType&& tensor, TDims&& dims)
RETURN_AUTO(
  field::wrap::clamp(field::fromSupplier<TRank>(util::forward<TTensorType>(tensor)), util::forward<TDims>(dims))
)

} // end of ns detail

template <size_t TRank, typename TTensorType, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
__host__ __device__
auto clamp(TTensorType&& tensor)
RETURN_AUTO(
  detail::clamp<TRank>(util::forward<TTensorType>(tensor), template_tensors::eval(util::forward<TTensorType>(tensor).template dims<TRank>() - 1))
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
      return self.m_field(util::forward<TCoordVector>(coords));
    }
    else
    {
      return self.m_constant;
    }
  }

public:
  static const size_t RANK = std::decay<TField>::type::RANK;

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
  Constant<util::store_member_t<TField&&>, util::store_member_t<TSize&&>, typename std::decay<TConstant&&>::type>
    (util::forward<TField>(field), util::forward<TSize>(size), util::forward<TConstant>(constant))
)

namespace detail {

template <size_t TRank, typename TTensorType, typename TDims, typename TConstant>
__host__ __device__
auto constant_helper(TTensorType&& tensor, TDims&& dims, TConstant&& constant)
RETURN_AUTO(
  field::wrap::constant(field::fromSupplier<TRank>(util::forward<TTensorType>(tensor)), util::forward<TDims>(dims), util::forward<TConstant>(constant))
)

} // end of ns detail

template <size_t TRank, typename TTensorType, typename TConstant, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
__host__ __device__
auto constant(TTensorType&& tensor, TConstant&& constant)
RETURN_AUTO(
  detail::constant_helper<TRank>(util::forward<TTensorType>(tensor), template_tensors::eval(tensor.template dims<TRank>()), util::forward<TConstant>(constant))
)

} // end of ns wrap

} // end of ns field
