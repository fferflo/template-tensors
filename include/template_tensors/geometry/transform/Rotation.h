namespace template_tensors {

namespace geometry {

namespace transform {

template <typename TScalar, size_t TRank>
class Rotation
{
public:
  __host__ __device__
  Rotation()
    : m_rotation(template_tensors::IdentityMatrix<TScalar, TRank>())
  {
  }

  __host__ __device__
  Rotation(const template_tensors::MatrixXXT<TScalar, TRank, TRank>& rotation)
    : m_rotation(rotation)
  {
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> operator()(template_tensors::VectorXT<TScalar, TRank> point) const
  {
    return transformPoint(point);
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformPoint(template_tensors::VectorXT<TScalar, TRank> point) const
  {
    return m_rotation * point;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformPointInverted(template_tensors::VectorXT<TScalar, TRank> point) const
  {
    return template_tensors::transpose<2>(m_rotation) * point;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformDirection(template_tensors::VectorXT<TScalar, TRank> direction) const
  {
    return m_rotation * direction;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformDirectionInverted(template_tensors::VectorXT<TScalar, TRank> direction) const
  {
    return template_tensors::transpose<2>(m_rotation) * direction;
  }

  __host__ __device__
  Rotation<TScalar, TRank> inverse() const
  {
    Rotation<TScalar, TRank> result;
    result.getRotation() = template_tensors::transpose<2>(m_rotation);
    return result;
  }

  __host__ __device__
  Rotation<TScalar, TRank>& operator*=(const Rotation<TScalar, TRank>& right)
  {
    m_rotation = template_tensors::eval(m_rotation * right.getRotation());
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
  template_tensors::MatrixXXT<TScalar, TRank + 1, TRank + 1> matrix() const
  {
    return homogenizeRotation(m_rotation);
  }

private:
  template_tensors::MatrixXXT<TScalar, TRank, TRank> m_rotation;
};

} // end of ns transform

} // end of ns geometry

} // end of ns template_tensors
