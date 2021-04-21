namespace template_tensors {

namespace geometry {

namespace render {

template <typename TScalar>
class ScreenSplat
{
public:
  using Scalar = TScalar;

  struct Intersection
  {
    TScalar z;

    TScalar t;
  };

  struct Data
  {
    TScalar z;
    TScalar r2;

    template_tensors::Vector2i center_s;

    template_tensors::Vector2i lower_bound;
    template_tensors::Vector2i upper_bound;
  };

private:
  template_tensors::VectorXT<TScalar, 3> m_position;

public:
  __host__ __device__
  ScreenSplat(template_tensors::VectorXT<TScalar, 3> position)
    : m_position(position)
  {
  }

  __host__ __device__
  ScreenSplat()
    : m_position(math::consts<TScalar>::NaN)
  {
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, 3> getPosition() const
  {
    return m_position;
  }

  __host__ __device__
  bool isClear() const
  {
    return math::isnan(m_position(0));
  }

#if defined(__CUDACC__)
  template <typename TProjection, typename THandler>
  __device__
  bool intersect(template_tensors::Vector2s pos, const Data& data, THandler&& handler, const transform::Rigid<TScalar, 3>& camera_pose, TProjection&& projection) const
  {
    Intersection intersection;

    intersection.t = template_tensors::length_squared(template_tensors::static_cast_to<int32_t>(pos) - data.center_s);
    if (intersection.t <= data.r2)
    {
      intersection.z = data.z;
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
    TProjection&& projection,
    TScalar radius) const
  {
    if (this->isClear())
    {
      return false;
    }

    // 1. Project center to screenspace and check if it is in render bounds
    VectorXT<TScalar, 3> center_c = camera_pose.transformPoint(m_position);
    if (center_c(2) < 0)
    {
      return false;
    }
    data.z = center_c(2);

    data.center_s = projection.project(center_c);
    if (template_tensors::any(data.center_s < 0 || data.center_s >= viewport_size)) // TODO: renders only splats whose centers are on screen
    {
      return false;
    }

    // 2. Calculate screen splat bounds in screen space
    radius = radius / -center_c(2);
    data.r2 = radius * radius;
    data.lower_bound = template_tensors::static_cast_to<int32_t>(viewport_size - 1);
    data.upper_bound = template_tensors::Vector2i(0, 0);
    data.lower_bound = template_tensors::min(data.lower_bound, data.center_s - (int32_t) radius - 1);
    data.upper_bound = template_tensors::max(data.upper_bound, data.center_s + (int32_t) radius + 1);
    data.lower_bound = template_tensors::max(data.lower_bound, template_tensors::Vector2i(0, 0));
    data.upper_bound = template_tensors::min(data.upper_bound, template_tensors::static_cast_to<int32_t>(viewport_size - 1));

    return true;
  }

  template <size_t TThreadsPerPrimitive, typename THandler, typename TProjection, typename TIndexStrategy = ColMajor>
  __device__
  void rasterize(
    THandler&& handler,
    template_tensors::Vector2s viewport_size,
    const transform::Rigid<TScalar, 3>& camera_pose,
    TProjection&& projection,
    TScalar radius,
    TIndexStrategy&& index_strategy = TIndexStrategy()) const
  {
    Data data;
    if (!precompute(data, viewport_size, camera_pose, projection, radius))
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
      if (!intersect(pos, data, util::forward<THandler>(handler), camera_pose, util::forward<TProjection>(projection)))
      {
        index -= step;
      }
    }
  }
#endif
};

} // end of ns render

} // end of ns geometry

} // end of ns tensor
