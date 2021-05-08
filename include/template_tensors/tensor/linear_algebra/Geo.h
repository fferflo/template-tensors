#pragma once
// TODO: where to put this file
namespace template_tensors {

namespace geo {

template <typename TScalar>
TVALUE(TScalar, earth_radius_v, 6371 * 1000)

// LatLon: (latitude [-pi/2, pi/2]), longitude [-pi, pi]
namespace latlon {

template <typename TVectorType1, typename TVectorType2, typename TScalar = typename std::common_type<
  decay_elementtype_t<TVectorType1>,
  decay_elementtype_t<TVectorType2>
>::type>
TScalar distance(TVectorType1&& latlon1, TVectorType2&& latlon2, TScalar radius = earth_radius_v<TScalar>::value)
{
  TScalar diff_lat = latlon2(0) - latlon1(0);
  TScalar diff_lon = latlon2(1) - latlon1(1);
  TScalar a =
      math::sin(diff_lat / 2) * math::sin(diff_lat / 2)
    + math::cos(latlon1(0)) * math::cos(latlon2(0)) * math::sin(diff_lon / 2) * math::sin(diff_lon / 2);
  TScalar c = 2 * math::atan2(math::sqrt(a), math::sqrt(1 - a));
  return radius * c;
}

template <typename TVectorType, typename TScalar = decay_elementtype_t<TVectorType>>
__host__ __device__
template_tensors::VectorXT<TScalar, 2> toMercator(TVectorType&& latlon)
{
  return template_tensors::VectorXT<TScalar, 2>(
    latlon(1),
    math::ln(math::tan(math::consts<TScalar>::PI / 4 + latlon(0) / 2))
  );
}

} // end of ns latlon

// Mercator: (x-axis=equator [-pi, pi], y-axis=prime-meridian [-pi, pi])
namespace mercator {

template <typename TVectorType, typename TScalar = decay_elementtype_t<TVectorType>>
__host__ __device__
template_tensors::VectorXT<TScalar, 2> toLatLon(TVectorType&& mercator)
{
  return template_tensors::VectorXT<TScalar, 2>(
    2 * (math::atan(math::exp(mercator(1))) - math::consts<TScalar>::PI / 4),
    mercator(0)
  );
}

} // end of ns mercator

} // end of ns geo

} // end of ns template_tensors
