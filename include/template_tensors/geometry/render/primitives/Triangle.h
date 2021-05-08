namespace template_tensors {

namespace geometry {

namespace render {

template <typename TScalar, typename TThisType = void>
class Triangle
{
public:
  using Scalar = TScalar;

  struct Intersection
  {
    TScalar z;
    template_tensors::VectorXT<TScalar, 3> intersection_c;
    template_tensors::VectorXT<TScalar, 3> barycentric_coords;
    template_tensors::VectorXT<TScalar, 3> normal_c;
  };

  struct Data
  {
    template_tensors::Vector2i lower_bound;
    template_tensors::Vector2i upper_bound;

    template_tensors::VectorXT<TScalar, 3> vertices_c[3];
    template_tensors::VectorXT<TScalar, 3> edges_c[3];
    template_tensors::VectorXT<TScalar, 3> normal_c;

    TScalar denom;
    TScalar d;
  };

  __host__ __device__
  template_tensors::VectorXT<TScalar, 3> getVertex(size_t i) const
  {
    return static_cast<const TThisType*>(this)->getVertex(i);
  }

  __host__ __device__
  bool isClear() const
  {
    return static_cast<const TThisType*>(this)->isClear();
  }

#if defined(__CUDACC__)
  template <typename TProjection, typename THandler>
  __device__
  bool intersect(template_tensors::Vector2s pos, const Data& data, THandler&& handler, const transform::Rigid<TScalar, 3>& camera_pose, TProjection&& projection) const
  {
    Intersection intersection;

    template_tensors::VectorXT<TScalar, 3> ray_direction = projection.unproject(pos);
    ray_direction = template_tensors::normalize(ray_direction); // TODO: test if necessary

    TScalar a = template_tensors::dot(data.normal_c, ray_direction);
    if (a == 0) // TODO: abs smaller than epsilon
    {
      return true; // Ray and triangle are parallel
    }
    TScalar t = data.d / a;
    if (t < 0)
    {
      return true; // Triangle behind camera
    }
    intersection.intersection_c = ray_direction * t;
    intersection.z = intersection.intersection_c(2);

    // Calculate barycentric coordinates
    for (size_t i = 0; i < 3; i++)
    {
      intersection.barycentric_coords((i + 2) % 3) = template_tensors::dot(data.normal_c, template_tensors::cross(data.edges_c[i], intersection.intersection_c - data.vertices_c[i]));
    }

    if (template_tensors::all(intersection.barycentric_coords >= 0))
    {
      intersection.barycentric_coords /= data.denom;
      intersection.normal_c = template_tensors::normalize(data.normal_c);
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
    if (isClear())
    {
      return false;
    }

    for (size_t i = 0; i < 3; i++)
    {
      data.vertices_c[i] = camera_pose.transformPoint(this->getVertex(i));
    }

    // Triangle behind camera
    if (data.vertices_c[0](2) < 0 && data.vertices_c[1](2) < 0 && data.vertices_c[2](2) < 0)
    {
      return false;
    }

    for (size_t i = 0; i < 3; i++)
    {
      data.edges_c[i] = data.vertices_c[(i + 1) % 3] - data.vertices_c[i];
    }

    data.normal_c = template_tensors::cross(data.edges_c[0], -data.edges_c[2]);
    data.denom = template_tensors::dot(data.normal_c, data.normal_c);

    data.d = template_tensors::dot(data.normal_c, data.vertices_c[0]);

    data.lower_bound = template_tensors::static_cast_to<int32_t>(viewport_size - 1);
    data.upper_bound = template_tensors::Vector2i(0, 0);
    for (size_t i = 0; i < 3; i++)
    {
      template_tensors::Vector2i vertex_s = projection.project(data.vertices_c[i]);
      data.lower_bound = template_tensors::min(data.lower_bound, vertex_s);
      data.upper_bound = template_tensors::max(data.upper_bound, vertex_s);
    }
    data.lower_bound = template_tensors::max(data.lower_bound, (int32_t) 1) - 1;
    data.upper_bound = template_tensors::min(data.upper_bound, template_tensors::static_cast_to<int32_t>(viewport_size - 1) - 1) + 1;

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
      if (!intersect(pos, data, util::forward<THandler>(handler), camera_pose, projection))
      {
        index -= step;
      }
    }
  }
#endif
};


// Specialization that stores vertices in member variable
template <typename TScalar>
class Triangle<TScalar, void> : public Triangle<TScalar, Triangle<TScalar, void>>
{
public:
  __host__ __device__
  Triangle(template_tensors::VectorXT<TScalar, 3> vertex0, template_tensors::VectorXT<TScalar, 3> vertex1, template_tensors::VectorXT<TScalar, 3> vertex2)
    : m_vertices{vertex0, vertex1, vertex2}
  {
  }

  __host__ __device__
  Triangle()
  {
    m_vertices[0](0) = ((TScalar) math::consts<TScalar>::NaN);
  }

  __host__ __device__
  bool isClear() const
  {
    return math::isnan(m_vertices[0](0));
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, 3> getVertex(size_t i) const
  {
    return m_vertices[i];
  }

private:
  template_tensors::VectorXT<TScalar, 3> m_vertices[3];
};

template <typename TScalar, typename TIndexType>
class VertexIndexTriangle : public Triangle<TScalar, VertexIndexTriangle<TScalar, TIndexType>>
{
public:
  __host__ __device__
  VertexIndexTriangle(template_tensors::VectorXT<TScalar, 3>* vertices, template_tensors::VectorXT<TIndexType, 3> indices)
    : m_vertices(vertices)
    , m_indices(indices)
  {
  }

  __host__ __device__
  VertexIndexTriangle()
  {
    m_vertices = nullptr;
  }

  __host__ __device__
  bool isClear() const
  {
    return m_vertices == nullptr;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, 3> getVertex(size_t i) const
  {
    return m_vertices[m_indices(i)];
  }
  // TODO: pass m_vertices pointer here when rendering, so that it isnt stored in every triangle
  __host__ __device__
  template_tensors::VectorXT<TIndexType, 3> getIndices() const
  {
    return m_indices;
  }

private:
  template_tensors::VectorXT<TScalar, 3>* m_vertices;
  template_tensors::VectorXT<TIndexType, 3> m_indices;
};

} // end of ns render

} // end of ns geometry

} // end of ns template_tensors

template <typename TScalar, typename TIndexType>
TT_PROCLAIM_TRIVIALLY_RELOCATABLE((template_tensors::geometry::render::VertexIndexTriangle<TScalar, TIndexType>),
  mem::is_trivially_relocatable_v<TScalar>::value && mem::is_trivially_relocatable_v<TIndexType>::value);
