#ifdef TF2_INCLUDED
#include <tf2/LinearMath/Transform.h>
#endif
// TODO: why is this here
#ifdef TF_INCLUDED
#include <tf/transform_datatypes.h>
#endif

#ifdef G2O_INCLUDED
#include <g2o/types/slam3d/se3quat.h>
#endif

namespace template_tensors {

namespace geometry {

namespace transform {

template <typename TScalar, size_t TRank>
class Rigid
{
private:
  template_tensors::MatrixXXT<TScalar, TRank, TRank> m_rotation;
  template_tensors::VectorXT<TScalar, TRank> m_translation;

public:
  __host__ __device__
  Rigid()
    : m_rotation(template_tensors::IdentityMatrix<TScalar, TRank>())
    , m_translation(0)
  {
  }

  __host__ __device__
  Rigid(const template_tensors::MatrixXXT<TScalar, TRank + 1, TRank + 1>& transformation_matrix)
    : m_rotation(template_tensors::head<TRank, TRank>(transformation_matrix))
    , m_translation(template_tensors::head<TRank>(template_tensors::col<TRank>(transformation_matrix)))
  {
    // TODO: checks that all other elements of matrix are 0, etc
  }

  __host__ __device__
  Rigid(const template_tensors::MatrixXXT<TScalar, TRank, TRank>& rotation, const template_tensors::VectorXT<TScalar, TRank>& translation)
    : m_rotation(rotation)
    , m_translation(translation)
  {
  }

  template <typename TScalar2>
  Rigid(const Rigid<TScalar2, TRank>& other)
    : m_rotation(other.m_rotation)
    , m_translation(other.m_translation)
  {
  }

  template <typename TScalar2>
  Rigid<TScalar, TRank>& operator=(const Rigid<TScalar2, TRank>& other)
  {
    this->m_rotation = other.m_rotation;
    this->m_translation = other.m_translation;
    return *this;
  }

#ifdef OPENCV_INCLUDED
  __host__
  Rigid(const cv::Mat& transformation_matrix)
    : Rigid(template_tensors::fromCv<TScalar>(transformation_matrix))
  {
  }

  __host__
  operator cv::Mat() const
  {
    return template_tensors::toCv(this->matrix());
  }
#endif

#ifdef G2O_INCLUDED
  __host__
  Rigid(const g2o::SE3Quat& se3)
    : Rigid(template_tensors::fromEigen(se3.rotation().toRotationMatrix()), template_tensors::fromEigen(se3.translation()))
  {
  }

  __host__
  operator g2o::SE3Quat() const
  {
    return g2o::SE3Quat(template_tensors::toEigen(template_tensors::static_cast_to<double>(getRotation())), template_tensors::toEigen(template_tensors::static_cast_to<double>(getTranslation())));
  }
#endif

  template <typename TPoint>
  __host__ __device__
  auto transformPoint(TPoint&& point) const
  RETURN_AUTO(matmul(m_rotation, util::forward<TPoint>(point)) + m_translation)

  template <typename TPoint>
  __host__ __device__
  auto operator()(TPoint&& point) const
  RETURN_AUTO(transformPoint(util::forward<TPoint>(point)))

  template <typename TPoint>
  __host__ __device__
  auto transformPointInverted(TPoint&& point) const
  RETURN_AUTO(matmul(template_tensors::transpose<2>(m_rotation), template_tensors::eval(util::forward<TPoint>(point) - m_translation)))

  template <typename TDirection>
  __host__ __device__
  auto transformDirection(TDirection&& direction) const
  RETURN_AUTO(matmul(m_rotation, util::forward<TDirection>(direction)))

  template <typename TDirection>
  __host__ __device__
  auto transformDirectionInverted(TDirection&& direction) const
  RETURN_AUTO(matmul(template_tensors::transpose<2>(m_rotation), util::forward<TDirection>(direction)))

  __host__ __device__
  Rigid<TScalar, TRank> inverse() const
  {
    Rigid<TScalar, TRank> result;
    result.getRotation() = template_tensors::transpose<2>(m_rotation);
    result.getTranslation() = matmul(result.getRotation(), -m_translation);
    return result;
  }

  __host__ __device__
  Rigid<TScalar, TRank>& operator*=(const Rigid<TScalar, TRank>& right)
  {
    m_translation = (*this)(right.getTranslation());
    m_rotation = template_tensors::eval(matmul(m_rotation, right.getRotation()));
    return *this;
  }

  __host__ __device__
  template_tensors::MatrixXXT<TScalar, TRank, TRank>& getRotation()
  {
    return m_rotation;
  }

  __host__ __device__
  const template_tensors::MatrixXXT<TScalar, TRank, TRank>& getRotation() const
  {
    return m_rotation;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank>& getTranslation()
  {
    return m_translation;
  }

  __host__ __device__
  const template_tensors::VectorXT<TScalar, TRank>& getTranslation() const
  {
    return m_translation;
  }

  __host__ __device__
  auto matrix() const
  RETURN_AUTO(
    template_tensors::concat<0>(template_tensors::concat<1>(m_rotation, m_translation), template_tensors::transpose<2>(template_tensors::concat<0>(template_tensors::VectorXT<TScalar, TRank>(0), template_tensors::VectorXT<TScalar, 1>(1))))
  )

  template <typename TScalar2, size_t TRank2>
  friend class Rigid;
};



template <typename TScalar, size_t TRank>
__host__ __device__
Rigid<TScalar, TRank> operator*(const Rigid<TScalar, TRank>& left, const Rigid<TScalar, TRank>& right)
{
  return Rigid<TScalar, TRank>(matmul(left.getRotation(), right.getRotation()), left(right.getTranslation()));
}

template <typename TScalar, size_t TRank>
__host__ __device__
Rigid<TScalar, TRank> operator/(const Rigid<TScalar, TRank>& left, const Rigid<TScalar, TRank>& right)
{
  return left * right.inverse();
}



template <typename TScalarIn = util::EmptyDefaultType, typename TVec1, typename TVec2, typename TVec3,
  typename TScalar = TT_WITH_DEFAULT_TYPE(TScalarIn, typename std::common_type<decay_elementtype_t<TVec1>, decay_elementtype_t<TVec2>, decay_elementtype_t<TVec3>>::type)>
__host__ __device__
Rigid<TScalar, 3> lookAt(TVec1&& eye, TVec2&& at, TVec3&& up)
{
  Rigid<TScalar, 3> result;
  result.getTranslation() = eye;

  template_tensors::VectorXT<TScalar, 3> zaxis = template_tensors::normalize(at - eye);
  template_tensors::VectorXT<TScalar, 3> xaxis = template_tensors::normalize(template_tensors::eval(template_tensors::cross(template_tensors::normalize(up), zaxis)));
  template_tensors::VectorXT<TScalar, 3> yaxis = template_tensors::cross(zaxis, xaxis);
  result.getRotation() = template_tensors::concat<1>(xaxis, yaxis, zaxis);

  return result.inverse();
}

template <typename TScalar, size_t TRank>
__host__
std::ostream& operator<<(std::ostream& stream, const Rigid<TScalar, TRank>& transform)
{
  stream << "Rigid(t=" << transform.getTranslation() << " R=" << transform.getRotation() << ")";
  return stream;
}

} // end of ns transform

} // end of ns geometry

template <typename TScalar, size_t TRank, typename TElwiseEqualsOp = math::functor::eq_real<TScalar>>
__host__ __device__
bool eq(const geometry::transform::Rigid<TScalar, TRank>& left, const geometry::transform::Rigid<TScalar, TRank>& right, TElwiseEqualsOp elwise_equals_op)
{
  return template_tensors::eq(left.getRotation(), right.getRotation(), elwise_equals_op) && template_tensors::eq(left.getTranslation(), right.getTranslation(), elwise_equals_op);
}

template <typename TScalar, size_t TRank, typename TElwiseEqualsOp = math::functor::eq_real<TScalar>>
__host__ __device__
bool neq(const geometry::transform::Rigid<TScalar, TRank>& left, const geometry::transform::Rigid<TScalar, TRank>& right, TElwiseEqualsOp elwise_equals_op)
{
  return !eq(left, right, elwise_equals_op);
}

} // end of ns tensor
