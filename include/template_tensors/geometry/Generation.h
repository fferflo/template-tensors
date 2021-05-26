#include <vector>

namespace template_tensors {

namespace geometry {

template <typename TScalar>
class SpherePoint
{
public:
  using Scalar = TScalar;

  __host__ __device__
  SpherePoint(template_tensors::VectorXT<TScalar, 3> position, template_tensors::VectorXT<TScalar, 3> normal)
    : m_position(position)
    , m_normal(normal)
  {
  }

  __host__ __device__
  SpherePoint()
    : m_position((TScalar) math::consts<TScalar>::NaN)
    , m_normal((TScalar) math::consts<TScalar>::NaN)
  {
  }

  __host__ __device__
  const template_tensors::VectorXT<TScalar, 3>& getPosition() const
  {
    return m_position;
  }

  __host__ __device__
  const template_tensors::VectorXT<TScalar, 3>& getNormal() const
  {
    return m_normal;
  }

private:
  template_tensors::VectorXT<TScalar, 3> m_position;
  template_tensors::VectorXT<TScalar, 3> m_normal;
};

template <typename TScalar>
class SpherePointConstructor
{
public:
  __host__ __device__
  SpherePointConstructor(size_t resolution, TScalar radius, template_tensors::VectorXT<TScalar, 3> center)
    : m_resolution(resolution)
    , m_radius(radius)
    , m_center(center)
  {
  }

  __host__ __device__
  SpherePoint<TScalar> operator()(size_t index) const
  {
    size_t lat_num = m_resolution;
    size_t lng_num = 2 * m_resolution;

    for (size_t i = 0; i < lat_num; i++)
    {
      float lat = math::consts<float>::PI * ((float) i) / (lat_num - 1);
      size_t cur_lng_num = lng_num * math::cos(lat - math::consts<float>::PI / 2);
      cur_lng_num = math::max((size_t) 1, cur_lng_num);
      if (index < cur_lng_num)
      {
        float lng = cur_lng_num == 1 ? 0 : (2 * math::consts<float>::PI * ((float) index) / (cur_lng_num - 1));

        template_tensors::VectorXT<TScalar, 3> normal = template_tensors::normalize(template_tensors::VectorXT<TScalar, 3>(
            math::sin(lat) * math::cos(lng),
            math::sin(lat) * math::sin(lng),
            math::cos(lat)
          ));

        return SpherePoint<TScalar>(m_center + m_radius * normal, normal);
      }
      else
      {
        index -= cur_lng_num;
      }
    }
    ASSERT_(false, "This should never happen");
    return SpherePoint<TScalar>();
  }

private:
  size_t m_resolution;
  TScalar m_radius;
  template_tensors::VectorXT<TScalar, 3> m_center;
};

template <typename TScalar>
struct SpherePointToSurfaceSplat
{
  __host__ __device__
  SpherePointToSurfaceSplat(TScalar splat_radius)
    : splat_radius(splat_radius)
  {
  }

  template <typename TPoint>
  __host__ __device__
  geometry::render::SurfaceSplat<TScalar> operator()(const TPoint& point)
  {
    return geometry::render::SurfaceSplat<TScalar>(point.getPosition(), point.getNormal(), splat_radius);
  }

  TScalar splat_radius;
};

namespace detail {

__host__ __device__
inline size_t getSpherePoints(size_t resolution)
{
  size_t lat_num = resolution;
  size_t lng_num = 2 * resolution;

  size_t num = 0;
  for (size_t i = 0; i < lat_num; i++)
  {
    float lat = math::consts<float>::PI * ((float) i) / (lat_num - 1);
    size_t cur_lng_num = lng_num * math::cos(lat - math::consts<float>::PI / 2);
    cur_lng_num = math::max((size_t) 1, cur_lng_num);
    num += cur_lng_num;
  }
  return num;
}

} // end of ns detail

template <typename TScalar>
__host__ __device__
auto generateSphere(size_t resolution, TScalar radius, template_tensors::VectorXT<TScalar, 3> center)
RETURN_AUTO(iterable::transform(
  iterable::count<size_t>(0, detail::getSpherePoints(resolution)), // TODO: dont have to store number of sphere points, just detect whether end is reached while iterating
  SpherePointConstructor<TScalar>(resolution, radius, center)
))

template <typename TScalar>
__host__ __device__
auto generateSphereOfSurfaceSplats(size_t resolution, TScalar radius, template_tensors::VectorXT<TScalar, 3> center)
RETURN_AUTO(iterable::transform(generateSphere(resolution, radius, center),
  SpherePointToSurfaceSplat<TScalar>(0.6f * 2 * math::consts<TScalar>::PI * radius / (resolution - 1))))

template <typename TScalar>
std::vector<render::Triangle<TScalar>> generateBoxOfTriangles(template_tensors::VectorXT<TScalar, 3> corner, template_tensors::VectorXT<TScalar, 3> size)
{
  std::vector<render::Triangle<TScalar>> result(12);

  auto vertex = [corner, size]__host__ __device__(size_t x, size_t y, size_t z){
    return corner + size * Vector3s(x, y, z);
  };

  // Z = 0
  result[0]  = render::Triangle<TScalar>(vertex(0, 0, 0), vertex(1, 0, 0), vertex(0, 1, 0));
  result[1]  = render::Triangle<TScalar>(vertex(1, 1, 0), vertex(1, 0, 0), vertex(0, 1, 0));

  // Z = 1
  result[2]  = render::Triangle<TScalar>(vertex(0, 0, 1), vertex(1, 0, 1), vertex(0, 1, 1));
  result[3]  = render::Triangle<TScalar>(vertex(1, 1, 1), vertex(1, 0, 1), vertex(0, 1, 1));

  // Y = 0
  result[4]  = render::Triangle<TScalar>(vertex(0, 0, 0), vertex(1, 0, 0), vertex(0, 0, 1));
  result[5]  = render::Triangle<TScalar>(vertex(1, 0, 1), vertex(1, 0, 0), vertex(0, 0, 1));

  // Y = 1
  result[6]  = render::Triangle<TScalar>(vertex(0, 1, 0), vertex(1, 1, 0), vertex(0, 1, 1));
  result[7]  = render::Triangle<TScalar>(vertex(1, 1, 1), vertex(1, 1, 0), vertex(0, 1, 1));

  // X = 0
  result[8]  = render::Triangle<TScalar>(vertex(0, 0, 0), vertex(0, 1, 0), vertex(0, 0, 1));
  result[9]  = render::Triangle<TScalar>(vertex(0, 1, 1), vertex(0, 1, 0), vertex(0, 0, 1));

  // X = 1
  result[10] = render::Triangle<TScalar>(vertex(1, 0, 0), vertex(1, 1, 0), vertex(1, 0, 1));
  result[11] = render::Triangle<TScalar>(vertex(1, 1, 1), vertex(1, 1, 0), vertex(1, 0, 1));

  return result;
}


// TODO: docs: discrete pixel n is described by floating point coordinates [n, n + 1.0]
namespace discrete {

// Includes all pixels for line coefficient t in [0, 1)
template <typename TScalar, size_t TRank, typename TInteger = int32_t>
class Line
{
private:
  template_tensors::VectorXT<TScalar, TRank> m_start;
  template_tensors::VectorXT<TScalar, TRank> m_direction;
  size_t m_num;

public:
  __host__ __device__
  Line(template_tensors::VectorXT<TScalar, TRank> start, template_tensors::VectorXT<TScalar, TRank> end)
    : m_start(start)
  {
    template_tensors::VectorXT<TScalar, TRank> direction = end - start;
    m_direction = direction / template_tensors::max_el(template_tensors::abs(direction));
    TScalar max_el = template_tensors::max_el(template_tensors::abs(template_tensors::ceil(end) - template_tensors::floor(start)));
    m_num = static_cast<size_t>(math::ceil(max_el));
  }

  template <typename TThisType>
  __host__ __device__
  static template_tensors::VectorXT<TInteger, TRank> get(TThisType&& self, size_t index)
  {
    return template_tensors::static_cast_to<TInteger>(template_tensors::floor(self.m_start + index * self.m_direction));
  }
  FORWARD_ALL_QUALIFIERS(operator(), get)

  template <typename TThisType>
  __host__ __device__
  static auto begin1(TThisType&& self)
  RETURN_AUTO(
    ::iterator::transform(::iterator::counting_iterator<size_t>(0), typename std::decay<TThisType>::type(self))
  )
  FORWARD_LVALUE_QUALIFIERS(begin, begin1)

  template <typename TThisType>
  __host__ __device__
  static auto end1(TThisType&& self)
  RETURN_AUTO(
    ::iterator::transform(::iterator::counting_iterator<size_t>(self.m_num), typename std::decay<TThisType>::type(self))
  )
  FORWARD_LVALUE_QUALIFIERS(end, end1)
};

template <typename TInteger = int32_t, typename TVectorType1, typename TVectorType2>
__host__ __device__
auto line(TVectorType1&& start, TVectorType2&& end)
RETURN_AUTO(
  Line<decay_elementtype_t<TVectorType1&&>, rows_v<TVectorType1>::value, TInteger>(
    std::forward<TVectorType1>(start), std::forward<TVectorType2>(end)
  )
)



namespace detail {

template <typename TVector>
struct add
{
  TVector vector;

  template <typename TVector2, ENABLE_IF(std::is_constructible<TVector, TVector2&&>::value)>
  __host__ __device__
  add(TVector2&& v)
    : vector(std::forward<TVector2>(v))
  {
  }

  template <typename TThisType, typename TVector2>
  __host__ __device__
  static auto get(TThisType&& self, TVector2&& vector)
  RETURN_AUTO(template_tensors::eval(self.vector + std::forward<TVector2>(vector))) // thrust::transform_iterator makes weird reference conversions, so have to call template_tensors::eval here
  FORWARD_ALL_QUALIFIERS(operator(), get)
};

} // end of ns detail

template <typename TIterable, typename TVectorType>
__host__ __device__
auto offset(TIterable&& object, TVectorType&& offset)
RETURN_AUTO(
  iterable::transform(std::forward<TIterable>(object), detail::add<util::store_member_t<TVectorType&&>>(std::forward<TVectorType>(offset)))
)



template <size_t TRank, typename TIndexStrategy>
class OriginBox
{
private:
  template_tensors::VectorXT<size_t, TRank> m_size;
  TIndexStrategy m_index_strategy;

public:
  template <typename TVectorType, typename TIndexStrategy2>
  __host__ __device__
  OriginBox(TVectorType&& size, TIndexStrategy2&& index_strategy)
    : m_size(std::forward<TVectorType>(size))
    , m_index_strategy(std::forward<TIndexStrategy2>(index_strategy))
  {
  }

  template <typename TThisType>
  __host__ __device__
  static template_tensors::VectorXT<size_t, TRank> get(TThisType&& self, size_t index)
  {
    return self.m_index_strategy.template fromIndex<TRank>(index, self.m_size);
  }
  FORWARD_ALL_QUALIFIERS(operator(), get)

  template <typename TThisType>
  __host__ __device__
  static auto begin1(TThisType&& self)
  RETURN_AUTO(
    ::iterator::transform(::iterator::count<size_t>(0), typename std::decay<TThisType>::type(self))
  )
  FORWARD_LVALUE_QUALIFIERS(begin, begin1)

  template <typename TThisType>
  __host__ __device__
  static auto end1(TThisType&& self)
  RETURN_AUTO(
    ::iterator::transform(::iterator::count<size_t>(self.m_index_strategy.getSize(self.m_size)), typename std::decay<TThisType>::type(self))
  )
  FORWARD_LVALUE_QUALIFIERS(end, end1)
};

template <typename TIndexStrategy = template_tensors::ColMajor, typename TVectorType>
__host__ __device__
auto origin_box(TVectorType&& size, TIndexStrategy&& index_strategy = TIndexStrategy())
RETURN_AUTO(
  OriginBox<rows_v<TVectorType>::value, util::store_member_t<TIndexStrategy&&>>(
    std::forward<TVectorType>(size), std::forward<TIndexStrategy>(index_strategy)
  )
)

template <typename TIndexStrategy = template_tensors::ColMajor, typename TInteger = int32_t, typename TVectorType1, typename TVectorType2>
__host__ __device__
auto box(TVectorType1&& start, TVectorType2&& end, TIndexStrategy&& index_strategy = TIndexStrategy())
RETURN_AUTO(
  discrete::offset(
    discrete::origin_box(
      template_tensors::static_cast_to<TInteger>(
        template_tensors::ceil(std::forward<TVectorType2&&>(end)) - template_tensors::floor(static_cast<util::store_member_t<TVectorType1&&>>(start))
      ),
      std::forward<TIndexStrategy>(index_strategy)
    ),
    template_tensors::static_cast_to<TInteger>(
      template_tensors::floor(static_cast<util::store_member_t<TVectorType1&&>>(start))
    )
  )
)



namespace detail {

template <typename TScalar, typename TVectorType, typename TDistanceFunctor>
struct sphere_helper
{
  TScalar radius;
  TVectorType center;
  TDistanceFunctor distance;

  template <typename TVectorType2, typename TDistanceFunctor2>
  __host__ __device__
  sphere_helper(TScalar radius, TVectorType2&& center, TDistanceFunctor2&& distance)
    : radius(radius)
    , center(std::forward<TVectorType2>(center))
    , distance(std::forward<TDistanceFunctor2>(distance))
  {
  }

  template <typename TThisType, typename TVector>
  __host__ __device__
  static bool get(TThisType&& self, TVector&& vector)
  {
    return self.distance(std::forward<TVector>(vector), self.center) <= self.radius;
  }
  FORWARD_ALL_QUALIFIERS(operator(), get)
};

} // end of ns detail

template <typename TIndexStrategy = template_tensors::ColMajor, typename TVectorType, typename TScalar, typename TDistanceFunctor = template_tensors::functor::distance>
__host__ __device__
auto sphere(TVectorType&& center, TScalar radius, TDistanceFunctor&& distance = TDistanceFunctor(), TIndexStrategy&& index_strategy = TIndexStrategy())
RETURN_AUTO(
  filter(
    box(
      TVectorType(center) - radius,
      TVectorType(center) + radius,
      std::forward<TIndexStrategy>(index_strategy)
    ),
    detail::sphere_helper<TScalar, decltype(template_tensors::eval(center - static_cast<TScalar>(0.5))), util::store_member_t<TDistanceFunctor&&>>(radius, template_tensors::eval(center - static_cast<TScalar>(0.5)), std::forward<TDistanceFunctor>(distance))
  )
)

} // end of ns discrete

} // end of ns geometry

} // end of ns template_tensors
