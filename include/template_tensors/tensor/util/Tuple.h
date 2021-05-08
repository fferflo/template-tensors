namespace template_tensors {

namespace detail {

template <typename TElementType, metal::int_ TRows, metal::int_ N>
struct ToTupleHelper
{
  template <typename TVectorType, typename... TValues>
  __host__ __device__
  static ::tuple::TupleEx<metal::repeat<TElementType, metal::number<TRows>>> toTuple(TVectorType&& vector, TValues&&... values)
  {
    return ToTupleHelper<TElementType, TRows, N + 1>::toTuple(util::forward<TVectorType>(vector), util::forward<TValues>(values)..., vector(N));
  }
};

template <typename TElementType, metal::int_ TRows>
struct ToTupleHelper<TElementType, TRows, TRows>
{
  template <typename TVectorType, typename... TValues>
  __host__ __device__
  static ::tuple::TupleEx<metal::repeat<TElementType, metal::number<TRows>>> toTuple(TVectorType&& vector, TValues&&... values)
  {
    return ::tuple::TupleEx<metal::repeat<TElementType, metal::number<TRows>>>(util::forward<TValues>(values)...);
  }
};

} // end of ns detail

template <typename TVectorType, metal::int_ TRows = rows_v<TVectorType>::value,
  typename TElementType = decay_elementtype_t<TVectorType>, ENABLE_IF(template_tensors::is_vector_v<TVectorType>::value)>
__host__ __device__
::tuple::TupleEx<metal::repeat<TElementType, metal::number<TRows>>> toTuple(TVectorType&& vector)
{
  return detail::ToTupleHelper<TElementType, TRows, 0>::toTuple(util::forward<TVectorType>(vector));
}

template <typename TElementType2 = util::EmptyDefaultType, typename TTuple,
  typename TElementType = TT_WITH_DEFAULT_TYPE(TElementType2, metal::front<::tuple::types_t<TTuple>>)>
__host__ __device__
auto fromTuple(TTuple&& tuple)
RETURN_AUTO(::tuple::for_all(
  util::functor::construct<template_tensors::VectorXT<TElementType, ::tuple::size_v<TTuple>::value>>(),
  util::forward<TTuple>(tuple)
))

} // end of ns template_tensors
