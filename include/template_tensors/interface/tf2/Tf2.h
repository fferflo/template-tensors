#ifdef TF2_INCLUDED

#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace template_tensors {

template_tensors::VectorXT<tf2Scalar, 3> fromTf2Vec(tf2::Vector3 vec)
{
  return template_tensors::VectorXT<tf2Scalar, 3>(vec.m_floats[0], vec.m_floats[1], vec.m_floats[2]);
}

tf2::Vector3 toTf2Vec(template_tensors::VectorXT<tf2Scalar, 3> vec)
{
  return tf2::Vector3(vec(0), vec(1), vec(2));
}

template_tensors::MatrixXXT<tf2Scalar, 3, 3, ColMajor> fromTf2Mat(tf2::Matrix3x3 mat)
{
  return template_tensors::MatrixXXT<tf2Scalar, 3, 3, ColMajor>(
      mat.getRow(0).m_floats[0], mat.getRow(1).m_floats[0], mat.getRow(2).m_floats[0],
      mat.getRow(0).m_floats[1], mat.getRow(1).m_floats[1], mat.getRow(2).m_floats[1],
      mat.getRow(0).m_floats[2], mat.getRow(1).m_floats[2], mat.getRow(2).m_floats[2]
    );
}

tf2::Matrix3x3 toTf2Mat(template_tensors::MatrixXXT<tf2Scalar, 3, 3, ColMajor> mat)
{
  tf2::Matrix3x3 result;
  result.setValue(
      mat(0, 0), mat(0, 1), mat(0, 2),
      mat(1, 0), mat(1, 1), mat(1, 2),
      mat(2, 0), mat(2, 1), mat(2, 2)
    );
  return result;
}

namespace geometry {

template <typename TScalar = tf2Scalar>
__host__
transform::Rigid<TScalar, 3> fromTf2(const tf2::Transform& t)
{
  return transform::Rigid<TScalar, 3>(template_tensors::fromTf2Mat(t.getBasis()), template_tensors::fromTf2Vec(t.getOrigin()));
}

template <typename TScalar>
__host__
tf2::Transform toTf2(const transform::Rigid<TScalar, 3>& t)
{
  return tf2::Transform(template_tensors::toTf2Mat(t.getRotation()), template_tensors::toTf2Vec(t.getTranslation()));
}

} // end of ns geometry

} // end of ns template_tensors

#endif