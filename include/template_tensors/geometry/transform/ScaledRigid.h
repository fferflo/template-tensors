

namespace template_tensors {

namespace geometry {

namespace transform {

template <typename TScalar, size_t TRank>
class ScaledRigid
{
public:
  __host__ __device__
  ScaledRigid()
    : m_rotation(template_tensors::IdentityMatrix<TScalar, TRank>())
    , m_translation(0)
    , m_scale(1)
  {
  }

  __host__ __device__
  ScaledRigid(const template_tensors::MatrixXXT<TScalar, TRank, TRank>& rotation, const template_tensors::VectorXT<TScalar, TRank>& translation, TScalar scale)
    : m_rotation(rotation)
    , m_translation(translation)
    , m_scale(scale)
  {
  }

  template <typename TScalar2>
  ScaledRigid(const ScaledRigid<TScalar2, TRank>& other)
    : m_rotation(other.m_rotation)
    , m_translation(other.m_translation)
    , m_scale(other.m_scale)
  {
  }

  template <typename TScalar2>
  ScaledRigid<TScalar, TRank>& operator=(const ScaledRigid<TScalar2, TRank>& other)
  {
    this->m_rotation = other.m_rotation;
    this->m_translation = other.m_translation;
    this->m_scale = other.m_scale;
    return *this;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> operator()(template_tensors::VectorXT<TScalar, TRank> point) const
  {
    return transformPoint(point);
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformPoint(template_tensors::VectorXT<TScalar, TRank> point) const
  {
    return m_scale * matmul(m_rotation, point) + m_translation;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformPointInverted(template_tensors::VectorXT<TScalar, TRank> point) const
  {
    return (1 / m_scale) * matmul(template_tensors::transpose<2>(m_rotation), template_tensors::eval(point - m_translation));
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformDirection(template_tensors::VectorXT<TScalar, TRank> direction) const
  {
    return m_scale * matmul(m_rotation, direction);
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformDirectionInverted(template_tensors::VectorXT<TScalar, TRank> direction) const
  {
    return (1 / m_scale) * matmul(template_tensors::transpose<2>(m_rotation), direction);
  }

  __host__ __device__
  ScaledRigid<TScalar, TRank> inverse() const
  {
    ScaledRigid<TScalar, TRank> result;
    result.getRotation() = template_tensors::transpose<2>(m_rotation);
    result.getTranslation() = (1 / m_scale) * matmul(result.getRotation(), -m_translation);
    result.getScale() = 1 / m_scale;
    return result;
  }

  __host__ __device__
  ScaledRigid<TScalar, TRank>& operator*=(const ScaledRigid<TScalar, TRank>& right)
  {
    m_translation = (*this)(right.getTranslation());
    m_rotation = template_tensors::eval(matmul(m_rotation, right.getRotation()));
    m_scale = m_scale * right.getScale();
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
  TScalar& getScale()
  {
    return m_scale;
  }

  __host__ __device__
  const TScalar& getScale() const
  {
    return m_scale;
  }

  __host__ __device__
  template_tensors::MatrixXXT<TScalar, TRank + 1, TRank + 1> matrix() const
  {
    return template_tensors::concat<0>(template_tensors::concat<1>(m_scale * m_rotation, m_translation), template_tensors::transpose<2>(template_tensors::concat<0>(template_tensors::VectorXT<TScalar, TRank>(0), template_tensors::VectorXT<TScalar, 1>(1))));
  }

  template <typename TScalar2, size_t TRank2>
  friend class ScaledRigid;

private:
  template_tensors::MatrixXXT<TScalar, TRank, TRank> m_rotation;
  template_tensors::VectorXT<TScalar, TRank> m_translation;
  TScalar m_scale;
};



template <typename TScalar, size_t TRank>
__host__ __device__
ScaledRigid<TScalar, TRank> operator*(const ScaledRigid<TScalar, TRank>& left, const ScaledRigid<TScalar, TRank>& right)
{
  return ScaledRigid<TScalar, TRank>(matmul(left.getRotation(), right.getRotation()), left(right.getTranslation()), left.getScale() * right.getScale());
}

template <typename TScalar, size_t TRank>
__host__ __device__
ScaledRigid<TScalar, TRank> operator/(const ScaledRigid<TScalar, TRank>& left, const ScaledRigid<TScalar, TRank>& right)
{
  return left * right.inverse();
}



template <typename TScalar, size_t TRank>
__host__
std::ostream& operator<<(std::ostream& stream, const ScaledRigid<TScalar, TRank>& transform)
{
  stream << "ScaledRigid(t=" << transform.getTranslation() << " R=" << transform.getRotation() << " s=" << transform.getScale() << ")";
  return stream;
}

} // end of ns transform

} // end of ns geometry

template <typename TScalar, size_t TRank, typename TElwiseEqualsOp = math::functor::eq_real<TScalar>>
__host__ __device__
bool eq(const geometry::transform::ScaledRigid<TScalar, TRank>& left, const geometry::transform::ScaledRigid<TScalar, TRank>& right, TElwiseEqualsOp elwise_equals_op)
{
  return template_tensors::eq(left.getRotation(), right.getRotation(), elwise_equals_op) && template_tensors::eq(left.getTranslation(), right.getTranslation(), elwise_equals_op) && elwise_equals_op(left.getScale(), right.getScale());
}

template <typename TScalar, size_t TRank, typename TElwiseEqualsOp = math::functor::eq_real<TScalar>>
__host__ __device__
bool neq(const geometry::transform::ScaledRigid<TScalar, TRank>& left, const geometry::transform::ScaledRigid<TScalar, TRank>& right, TElwiseEqualsOp elwise_equals_op)
{
  return !eq(left, right, elwise_equals_op);
}

} // end of ns template_tensors
