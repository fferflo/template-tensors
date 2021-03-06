#pragma once

#include <metal.hpp>

namespace field {

template <typename TField>
class StaticDifferentiatedField
{
public:
  static const metal::int_ RANK = std::decay<TField>::type::RANK;

private:
  TField m_field;

  template <metal::int_ TRow, typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get_row(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(
     (std::forward<TThisType>(self).m_field(coords + template_tensors::UnitVector<size_t, RANK, TRow>())
    - std::forward<TThisType>(self).m_field(coords - template_tensors::UnitVector<size_t, RANK, TRow>()))
    / 2
  )

  template <metal::int_... TRows, typename TThisType, typename TCoordVector,
    typename TElementType = decltype(get_row<0>(std::declval<TThisType&&>(), std::declval<TCoordVector&&>()))>
  __host__ __device__
  static template_tensors::VectorXT<TElementType, RANK> get2(TThisType&& self, TCoordVector&& coords, metal::numbers<TRows...>)
  {
    return template_tensors::VectorXT<TElementType, RANK>(
      get_row<TRows>(std::forward<TThisType>(self), std::forward<TCoordVector>(coords))...
    );
  }

  template <typename TThisType, typename TCoordVector,
    typename TElementType = decltype(std::forward<TThisType>(std::declval<TThisType&&>()).m_field(std::declval<template_tensors::VectorXs<RANK>>()))>
  __host__ __device__
  static auto get1(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(
    get2(std::forward<TThisType>(self), std::forward<TCoordVector>(coords), metal::iota<metal::number<0>, metal::number<RANK>>())
  )

public:
  __host__ __device__
  StaticDifferentiatedField(TField field)
    : m_field(field)
  {
  }

  FORWARD_ALL_QUALIFIERS(operator(), get1)
};

template <typename TField>
__host__ __device__
auto differentiate(TField&& field)
RETURN_AUTO(StaticDifferentiatedField<TField>(std::forward<TField>(field)))

} // end of ns field
