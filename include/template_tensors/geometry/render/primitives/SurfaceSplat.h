namespace template_tensors {

namespace geometry {

namespace render {

template <typename TScalar>
class SurfaceSplat
{
public:
  using Scalar = TScalar;

  struct Intersection
  {
    TScalar z;
    TScalar t;
    template_tensors::VectorXT<TScalar, 3> intersection_c;
    template_tensors::VectorXT<TScalar, 3> normal_c;
  };

  struct Data
  {
    template_tensors::VectorXT<TScalar, 3> center_c;
    template_tensors::VectorXT<TScalar, 3> normal_c;

    template_tensors::VectorXT<TScalar, 3> axis1_c;
    template_tensors::VectorXT<TScalar, 3> axis2_c;

    template_tensors::Vector2i lower_bound;
    template_tensors::Vector2i upper_bound;

    TScalar d;
    TScalar r4;

    __host__ __device__
    template_tensors::VectorXT<TScalar, 3> corner(size_t i) const
    {
      template_tensors::Vector2i corner_index;
      switch (i)
      {
        case 0: corner_index = template_tensors::Vector2i(-1, -1); break;
        case 1: corner_index = template_tensors::Vector2i( 1, -1); break;
        case 2: corner_index = template_tensors::Vector2i(-1,  1); break;
        case 3: corner_index = template_tensors::Vector2i( 1,  1); break;
        default: ASSERT_(false, "Invalid corner"); corner_index = template_tensors::Vector2i(0, 0); break;
      }

      return center_c + corner_index(0) * axis1_c + corner_index(1) * axis2_c;
    }
  };

private:
  template_tensors::VectorXT<TScalar, 3> m_position;
  template_tensors::VectorXT<TScalar, 3> m_normal;
  TScalar m_radius;

public:

  __host__ __device__
  SurfaceSplat(template_tensors::VectorXT<TScalar, 3> position, template_tensors::VectorXT<TScalar, 3> normal, TScalar radius)
    : m_position(position)
    , m_normal(normal)
    , m_radius(radius)
  {
  }

  __host__ __device__
  SurfaceSplat()
    : m_radius(-1)
  {
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, 3> getPosition() const
  {
    return m_position;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, 3> getNormal() const
  {
    return m_normal;
  }

  __host__ __device__
  TScalar getRadius() const
  {
    return m_radius;
  }

  __host__ __device__
  bool isClear() const
  {
    return m_radius < 0;
  }

#if defined(__CUDACC__)
  template <typename TProjection, typename THandler>
  __device__
  bool intersect(template_tensors::Vector2s pos, const Data& data, THandler&& handler, const transform::Rigid<TScalar, 3>& camera_pose, TProjection&& projection) const
  {
    Intersection intersection;

    // Calculate intersection of pixel with the splat's plane in camera space
    intersection.intersection_c = projection.unproject(pos); // Initialized with z = 1
    intersection.z = data.d / template_tensors::dot(intersection.intersection_c, data.normal_c);
    intersection.intersection_c *= intersection.z;

    // Check if point is on splat
    intersection.t = math::squared(template_tensors::dot(intersection.intersection_c - data.center_c, data.axis1_c))
                   + math::squared(template_tensors::dot(intersection.intersection_c - data.center_c, data.axis2_c));
    if (intersection.t <= data.r4)
    {
      if (!handler(pos, intersection))
      {
        return false;
      }
    }

    return true;
  }

  template <typename TProjection>
  __device__
  bool precompute(
    Data& data,
    template_tensors::Vector2s viewport_size,
    const transform::Rigid<TScalar, 3>& camera_pose,
    TProjection&& projection) const
  {
    if (this->isClear())
    {
      return false;
    }

    // 1. Project center to screenspace and check if it is in render bounds
    data.center_c = camera_pose.transformPoint(m_position);
    if (data.center_c(2) < 0)
    {
      return false;
    }

    template_tensors::Vector2i center_s = projection.project(data.center_c);
    if (template_tensors::any(center_s < 0 || center_s >= viewport_size)) // TODO: renders only splats whose centers are on screen
    {
      return false;
    }

    // 2. Calculate surface splat properties in global space
    template_tensors::VectorXT<TScalar, 3> axis1_g = m_radius * template_tensors::normalize(template_tensors::eval(template_tensors::cross(template_tensors::eval(m_position - camera_pose.getTranslation()), m_normal)));
    template_tensors::VectorXT<TScalar, 3> axis2_g = m_radius * template_tensors::normalize(template_tensors::eval(template_tensors::cross(axis1_g, m_normal)));

    // 3. Calculate surface splat properties in camera space
    data.normal_c = camera_pose.transformDirection(m_normal);
    if (template_tensors::all(axis1_g == 0))
    {
      data.axis1_c = template_tensors::VectorXT<TScalar, 3>(m_radius, 0, 0);
      data.axis2_c = template_tensors::VectorXT<TScalar, 3>(0, m_radius, 0);
    }
    else
    {
      data.axis1_c = camera_pose.transformDirection(axis1_g);
      data.axis2_c = camera_pose.transformDirection(axis2_g);
    }

    // 4. Calculate surface splat bounds in screen space
    data.lower_bound = template_tensors::static_cast_to<int32_t>(viewport_size - 1);
    data.upper_bound = template_tensors::Vector2i(0, 0);
    for (size_t c = 0; c < 4; c++)
    {
      template_tensors::Vector2i corner = projection.project(data.corner(c));
      data.lower_bound = template_tensors::min(data.lower_bound, corner);
      data.upper_bound = template_tensors::max(data.upper_bound, corner);
    }
    data.lower_bound = template_tensors::max(data.lower_bound, (int32_t) 1) - 1;
    data.upper_bound = template_tensors::min(data.upper_bound, template_tensors::static_cast_to<int32_t>(viewport_size - 1) - 1) + 1;

    data.d = template_tensors::dot(data.center_c, data.normal_c);
    data.r4 = math::squared(math::squared(m_radius));

    return true;
  }

  template <size_t TThreadsPerPrimitive, typename THandler, typename TProjection, typename TIndexStrategy = ColMajor>
  __device__
  void rasterize(
    THandler&& handler,
    template_tensors::Vector2s viewport_size,
    const transform::Rigid<TScalar, 3>& camera_pose,
    TProjection&& projection,
    TIndexStrategy&& index_strategy = TIndexStrategy()) const
  {
    Data data;
    if (!precompute(data, viewport_size, camera_pose, projection))
    {
      return;
    }

    const size_t offset = cuda::grid::thread_id_in_grid<1>()() % TThreadsPerPrimitive;
    const size_t step = TThreadsPerPrimitive;
    const template_tensors::Vector2s screen_aabb_dims = (data.upper_bound + 1) - data.lower_bound;
    const size_t num = index_strategy.getSize(screen_aabb_dims);

    for (size_t index = offset; index < num; index += step)
    {
      template_tensors::Vector2i pos = data.lower_bound + index_strategy.fromIndex(index, screen_aabb_dims);
      if (!intersect(pos, data, std::forward<THandler>(handler), camera_pose, projection))
      {
        index -= step;
      }
    }
  }
#endif
};

} // end of ns render

} // end of ns geometry

} // end of ns template_tensors
