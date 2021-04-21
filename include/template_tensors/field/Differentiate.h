#pragma once

namespace field {

template <typename TField>
class StaticDifferentiatedField
{
public:
  static const size_t RANK = std::decay<TField>::type::RANK;

private:
  TField m_field;

  template <size_t TRow, typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get_row(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(
     (util::forward<TThisType>(self).m_field(coords + template_tensors::UnitVector<size_t, RANK, TRow>())
    - util::forward<TThisType>(self).m_field(coords - template_tensors::UnitVector<size_t, RANK, TRow>()))
    / 2
  )

  template <size_t... TRows, typename TThisType, typename TCoordVector,
    typename TElementType = decltype(get_row<0>(std::declval<TThisType&&>(), std::declval<TCoordVector&&>()))>
  __host__ __device__
  static template_tensors::VectorXT<TElementType, RANK> get2(TThisType&& self, TCoordVector&& coords, tmp::vs::IndexSequence<TRows...>)
  {
    return template_tensors::VectorXT<TElementType, RANK>(
      get_row<TRows>(util::forward<TThisType>(self), util::forward<TCoordVector>(coords))...
    );
  }

  template <typename TThisType, typename TCoordVector,
    typename TElementType = decltype(util::forward<TThisType>(std::declval<TThisType&&>()).m_field(std::declval<template_tensors::VectorXs<RANK>>()))>
  __host__ __device__
  static auto get1(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(
    get2(util::forward<TThisType>(self), util::forward<TCoordVector>(coords), tmp::vs::ascending_numbers_t<RANK>())
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
RETURN_AUTO(StaticDifferentiatedField<util::store_member_t<TField&&>>(util::forward<TField>(field)))

} // end of ns field
