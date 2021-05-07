#ifdef OROCOS_KDL_INCLUDED

#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>

namespace template_tensors {

using OrocosKdlScalar = typename std::decay<decltype(std::declval<KDL::Frame>()(0, 0))>::type;

template <typename TScalar = OrocosKdlScalar>
__host__
MatrixXXT<TScalar, 3, 3> fromKdl(const KDL::Rotation& kdl)
{
  MatrixXXT<TScalar, 3, 3> result;
  for (dim_t r = 0; r < 3; r++)
  {
    for (dim_t c = 0; c < 3; c++)
    {
      result(r, c) = kdl(r, c);
    }
  }
  return result;
}

template <typename TScalar = OrocosKdlScalar>
__host__
VectorXT<TScalar, 3> fromKdl(const KDL::Vector& kdl)
{
  VectorXT<TScalar, 3> result;
  for (dim_t r = 0; r < 3; r++)
  {
    result(r) = kdl[r];
  }
  return result;
}

template <typename TScalar = OrocosKdlScalar>
__host__
geometry::transform::Rigid<TScalar, 3> fromKdl(const KDL::Frame& kdl)
{
  return geometry::transform::Rigid<TScalar, 3>(fromKdl(kdl.M), fromKdl(kdl.p));
}

template <typename TTensor>
__host__
KDL::Rotation toKdlRot(TTensor&& rotation)
{
  ASSERT(areSameDimensions(rotation.dims(), Vector2s(3, 3)), "Invalid dimensions");
  KDL::Rotation result;
  for (dim_t r = 0; r < 3; r++)
  {
    for (dim_t c = 0; c < 3; c++)
    {
      result(r, c) = rotation(r, c);
    }
  }
  return result;
}

template <typename TTensor>
__host__
KDL::Vector toKdlVec(TTensor&& vector)
{
  ASSERT(areSameDimensions(vector.dims(), Vector1s(3)), "Invalid dimensions");
  KDL::Vector result;
  for (dim_t r = 0; r < 3; r++)
  {
    result(r) = vector(r);
  }
  return result;
}

template <typename TScalar>
__host__
KDL::Frame toKdlFrame(const geometry::transform::Rigid<TScalar, 3>& transform)
{
  return KDL::Frame(toKdlRot(transform.getRotation()), toKdlVec(transform.getTranslation()));
}

} // end of ns tensor

#endif
