namespace template_tensors {

namespace geometry {

namespace projection {

template <typename TScalar, size_t TRank>
class Orthographic
{
public:
  __host__ __device__
  Orthographic(template_tensors::VectorXT<TScalar, TRank - 1> lower, template_tensors::VectorXT<TScalar, TRank - 1> upper)
    : m_lower(lower)
    , m_upper(upper)
    , m_1_over_upper_minus_lower(static_cast<TScalar>(1) / (upper - lower))
  {
    ASSERT(template_tensors::all(m_lower < m_upper), "Lower bound must be strictly less than upper bound");
  }

  __host__ __device__
  Orthographic()
    : Orthographic(template_tensors::VectorXT<TScalar, TRank>(-1), template_tensors::VectorXT<TScalar, TRank>(1))
  {
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank> unproject(template_tensors::VectorXT<TScalar, TRank - 1> point) const
  {
    return template_tensors::concat<0>((point + (m_upper + m_lower) * m_1_over_upper_minus_lower) * static_cast<TScalar>(0.5) * (m_upper - m_lower), VectorXT<TScalar, 1>(1));
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank - 1> project(template_tensors::VectorXT<TScalar, TRank> point) const
  {
    return template_tensors::head<TRank - 1>(point) * 2 * m_1_over_upper_minus_lower - (m_upper + m_lower) * m_1_over_upper_minus_lower;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank - 1> getLowerBound() const
  {
    return m_lower;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, TRank - 1> getUpperBound() const
  {
    return m_upper;
  }

  __host__ __device__
  template_tensors::MatrixXXT<TScalar, TRank + 1, TRank + 1> matrix(TScalar near, TScalar far)
  {
    template_tensors::MatrixXXT<TScalar, TRank + 1, TRank + 1> result(0);

    for (size_t i = 0; i < TRank - 1; i++)
    {
      result(i, i) = 2 * m_1_over_upper_minus_lower(i);
      result(i, TRank) = -(m_upper(i) + m_lower(i)) * m_1_over_upper_minus_lower(i);
    }
    result(TRank - 1, TRank - 1) = 2 / (far - near);
    result(TRank - 1, TRank) = -(far + near) / (far - near);

    result(TRank, TRank) = 1.0f;
    return result;
  }

private:
  template_tensors::VectorXT<TScalar, TRank - 1> m_lower;
  template_tensors::VectorXT<TScalar, TRank - 1> m_upper;
  template_tensors::VectorXT<TScalar, TRank - 1> m_1_over_upper_minus_lower;
};

template <typename TScalar, size_t TRank>
__host__
std::ostream& operator<<(std::ostream& stream, const Orthographic<TScalar, TRank>& p)
{
  stream << "Orthographic(lower=" << p.getLowerBound() << " upper=" << p.getUpperBound() << ")";
  return stream;
}

} // end of ns projection

} // end of ns geometry

} // end of ns template_tensors
