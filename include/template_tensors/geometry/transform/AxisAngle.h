#pragma once

namespace template_tensors {

namespace geometry {

namespace transform {

template <typename TScalar>
class Rodrigues;

template <typename TScalar>
class AxisAngle
{
public:
  __host__ __device__
  AxisAngle()
    : m_axis(0)
    , m_angle(0)
  {
    m_axis(0) = 1;
  }

  __host__ __device__
  AxisAngle(template_tensors::VectorXT<TScalar, 3> axis, TScalar angle)
    : m_axis(axis)
    , m_angle(angle)
  {
    // TODO: assert axis is normalized
  }

  __host__ __device__
  AxisAngle(Rodrigues<TScalar> rodrigues)
  {
    m_angle = template_tensors::l2_norm(rodrigues.getVector());
    m_axis = rodrigues.getVector() / m_angle;
  }

  __host__ __device__
  template_tensors::VectorXT<TScalar, 3> getAxis() const
  {
    return m_axis;
  }

  __host__ __device__
  TScalar getAngle() const
  {
    return m_angle;
  }

  __host__ __device__
  template_tensors::MatrixXXT<TScalar, 3, 3> matrix3() const
  {
    TScalar cos = math::cos(m_angle);
    TScalar sin = math::sin(m_angle);

    template_tensors::MatrixXXT<TScalar, 3, 3> result;
    result(0, 0) = cos + m_axis(0) * m_axis(0) * (1 - cos);
    result(1, 1) = cos + m_axis(1) * m_axis(1) * (1 - cos);
    result(2, 2) = cos + m_axis(2) * m_axis(2) * (1 - cos);

    result(1, 0) = m_axis(1) * m_axis(0) * (1 - cos) + m_axis(2) * sin;
    result(0, 1) = m_axis(1) * m_axis(0) * (1 - cos) - m_axis(2) * sin;

    result(2, 0) = m_axis(2) * m_axis(0) * (1 - cos) - m_axis(1) * sin;
    result(0, 2) = m_axis(2) * m_axis(0) * (1 - cos) + m_axis(1) * sin;

    result(1, 2) = m_axis(1) * m_axis(2) * (1 - cos) - m_axis(0) * sin;
    result(2, 1) = m_axis(1) * m_axis(2) * (1 - cos) + m_axis(0) * sin;

    return result;
  }

private:
  template_tensors::VectorXT<TScalar, 3> m_axis;
  TScalar m_angle;
};

template <typename TScalar>
class Rodrigues
{
public:
  __host__ __device__
  Rodrigues()
    : m_vector(0)
  {
  }

  __host__ __device__
  Rodrigues(template_tensors::VectorXT<TScalar, 3> vector)
    : m_vector(vector)
  {
  }

  const template_tensors::VectorXT<TScalar, 3>& getVector() const
  {
    return m_vector;
  }

  __host__ __device__
  template_tensors::MatrixXXT<TScalar, 3, 3> matrix3() const
  {
    return AxisAngle<TScalar>(*this).matrix3();
  }

private:
  template_tensors::VectorXT<TScalar, 3> m_vector;
};

} // end of ns transform

} // end of ns geometry

} // end of ns tensor
