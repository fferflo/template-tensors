#pragma once

namespace field {

template <typename TField, typename TMapper>
class MapField
{
public:
  static const metal::int_ RANK = std::decay<TField>::type::RANK;

private:
  TField m_field;
  TMapper m_mapper;

  template <typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(
    util::forward<TThisType>(self).m_mapper(util::forward<TThisType>(self).m_field(util::forward<TCoordVector>(coords)))
  )

public:
  __host__ __device__
  MapField(TField field, TMapper mapper)
    : m_field(field)
    , m_mapper(mapper)
  {
  }

  FORWARD_ALL_QUALIFIERS(operator(), get)
};

template <typename TField, typename TMapper>
__host__ __device__
auto map(TField&& field, TMapper&& mapper)
RETURN_AUTO(MapField<util::store_member_t<TField&&>, util::store_member_t<TMapper&&>>(util::forward<TField>(field), util::forward<TMapper>(mapper)))

} // end of ns field
