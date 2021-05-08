namespace template_tensors {

namespace geometry {

namespace transform {

template <typename TScalar, size_t TRank>
class Translation
{
public:
  __host__ __device__
  Translation()
    : m_translation(0)
  {
  }

  __host__ __device__
  Translation(const template_tensors::VectorXT<TScalar, TRank>& translation)
    : m_translation(translation)
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
    return point + m_translation;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformPointInverted(template_tensors::VectorXT<TScalar, TRank> point) const
  {
    return point - m_translation;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformDirection(template_tensors::VectorXT<TScalar, TRank> direction) const
  {
    return direction;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> transformDirectionInverted(template_tensors::VectorXT<TScalar, TRank> direction) const
  {
    return direction;
  }

  __host__ __device__
  Translation<TScalar, TRank> inverse() const
  {
    Translation<TScalar, TRank> result;
    result.getTranslation() = -m_translation;
    return result;
  }

  __host__ __device__
  Translation<TScalar, TRank>& operator*=(const Translation<TScalar, TRank>& right)
  {
    m_translation = (*this)(right.getTranslation());
    return *this;
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
  template_tensors::MatrixXXT<TScalar, TRank + 1, TRank + 1> matrix() const
  {
    return homogenizeTranslation(m_translation);
  }

private:
  template_tensors::VectorXT<TScalar, TRank> m_translation;
};

} // end of ns transform

} // end of ns geometry

} // end of ns template_tensors
