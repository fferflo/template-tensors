namespace template_tensors {

namespace geometry {

namespace render {

struct NopShader
{
  template <typename TPixel, typename TPrimitive, typename TIntersect>
  __host__ __device__
  void operator()(TPixel& pixel, TPrimitive& primitive, TIntersect& intersect) const
  {
  }
};

template <typename TScalar>
class AmbientDirectionalGrayShader
{
public:
  __host__ __device__
  AmbientDirectionalGrayShader(
        TScalar ambient_intensity,
        TScalar directional_intensity,
        template_tensors::VectorXT<TScalar, 3> direction_g)
    : m_ambient_intensity(ambient_intensity)
    , m_directional_intensity(directional_intensity)
    , m_direction_g(template_tensors::normalize(direction_g))
  {
  }

  template <typename TPixel, typename TPrimitive, typename TIntersect>
  __host__ __device__
  void operator()(TPixel& pixel, TPrimitive& primitive, TIntersect& intersect) const
  {
    TScalar intensity = m_ambient_intensity + m_directional_intensity * math::abs(template_tensors::dot(m_direction_g, primitive.getNormal()));
    intensity = math::clamp(intensity, (TScalar) 0, (TScalar) 1);
    pixel.gray = intensity;
  }

private:
  TScalar m_ambient_intensity;
  TScalar m_directional_intensity;
  template_tensors::VectorXT<TScalar, 3> m_direction_g;
};

template <typename TScalar>
class AmbientDirectionalGrayShaderFromIntersect
{
public:
  __host__ __device__
  AmbientDirectionalGrayShaderFromIntersect(
        TScalar ambient_intensity,
        TScalar directional_intensity,
        template_tensors::VectorXT<TScalar, 3> direction_c) // TODO: this should be direction_g?
    : m_ambient_intensity(ambient_intensity)
    , m_directional_intensity(directional_intensity)
    , m_direction_c(template_tensors::normalize(direction_c))
  {
  }

  template <typename TPixel, typename TPrimitive, typename TIntersect>
  __host__ __device__
  void operator()(TPixel& pixel, TPrimitive& primitive, TIntersect& intersect) const
  {
    TScalar intensity = m_ambient_intensity + m_directional_intensity * math::abs(template_tensors::dot(m_direction_c, intersect.normal_c));
    intensity = math::clamp(intensity, (TScalar) 0, (TScalar) 1);
    pixel.gray = intensity;
  }

private:
  TScalar m_ambient_intensity;
  TScalar m_directional_intensity;
  template_tensors::VectorXT<TScalar, 3> m_direction_c;
};

struct CopyNormalShader
{
  template <typename TPixel, typename TPrimitive, typename TIntersect>
  __host__ __device__
  void operator()(TPixel& pixel, TPrimitive& primitive, TIntersect& intersect) const
  {
    pixel.normal = primitive.getNormal();
  }
};

} // end of ns render

} // end of ns geometry

} // end of ns tensor