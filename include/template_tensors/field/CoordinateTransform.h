#pragma once

namespace field {

template <typename TField, typename TCoordFunctor>
class CoordinateTransformedField
{
private:
  TField m_field;
  TCoordFunctor m_coord_functor;

  template <typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(std::forward<TThisType>(self).m_field(std::forward<TThisType>(self).m_coord_functor(std::forward<TCoordVector>(coords))))

public:
  static const metal::int_ RANK = std::decay<TField>::type::RANK;

  __host__ __device__
  CoordinateTransformedField(TField field, TCoordFunctor coord_functor)
    : m_field(field)
    , m_coord_functor(coord_functor)
  {
  }

  FORWARD_ALL_QUALIFIERS(operator(), get)
};

template <typename TField, typename TCoordFunctor>
__host__ __device__
auto transform(TField&& field, TCoordFunctor&& coord_functor)
RETURN_AUTO(CoordinateTransformedField<util::store_member_t<TField&&>, util::store_member_t<TCoordFunctor&&>>(std::forward<TField>(field), std::forward<TCoordFunctor>(coord_functor)))

} // end of ns field
