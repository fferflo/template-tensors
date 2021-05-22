#pragma once

#include <jtuple/tuple.hpp>
#include <jtuple/tuple_utility.hpp>
#include <metal.hpp>

namespace template_tensors {

namespace detail {

template <typename TElementType, metal::int_ TRows, metal::int_ N>
struct ToTupleHelper
{
  template <typename TVectorType, typename... TValues>
  __host__ __device__
  static auto toTuple(TVectorType&& vector, TValues&&... values)
  RETURN_AUTO(
    ToTupleHelper<TElementType, TRows, N + 1>::toTuple(util::forward<TVectorType>(vector), util::forward<TValues>(values)..., vector(N))
  )
};

template <typename TElementType, metal::int_ TRows>
struct ToTupleHelper<TElementType, TRows, TRows>
{
  template <typename TVectorType, typename... TValues>
  __host__ __device__
  static auto toTuple(TVectorType&& vector, TValues&&... values)
  RETURN_AUTO(
    metal::apply<metal::lambda<jtuple::tuple>, metal::repeat<TElementType, metal::number<TRows>>>(util::forward<TValues>(values)...)
  )
};

} // end of ns detail

template <typename TVectorType, metal::int_ TRows = rows_v<TVectorType>::value,
  typename TElementType = decay_elementtype_t<TVectorType>, ENABLE_IF(template_tensors::is_vector_v<TVectorType>::value)>
__host__ __device__
auto toTuple(TVectorType&& vector)
RETURN_AUTO(
  detail::ToTupleHelper<TElementType, TRows, 0>::toTuple(util::forward<TVectorType>(vector))
)

template <typename TElementType2 = util::EmptyDefaultType, typename TTuple,
  typename TElementType = TT_WITH_DEFAULT_TYPE(TElementType2, typename std::tuple_element<0, TTuple>::type)>
__host__ __device__
auto fromTuple(TTuple&& tuple)
RETURN_AUTO(jtuple::tuple_apply(
  util::functor::construct<template_tensors::VectorXT<TElementType, std::tuple_size<TTuple>::value>>(),
  util::forward<TTuple>(tuple)
))

} // end of ns template_tensors
