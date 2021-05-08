#ifdef TF_INCLUDED

#include <tf/LinearMath/Vector3.h>
#include <tf/LinearMath/Matrix3x3.h>

namespace template_tensors {

template_tensors::VectorXT<tfScalar, 3> fromTfVec(tf::Vector3 vec)
{
  return template_tensors::VectorXT<tfScalar, 3>(vec.m_floats[0], vec.m_floats[1], vec.m_floats[2]);
}

tf::Vector3 toTfVec(template_tensors::VectorXT<tfScalar, 3> vec)
{
  return tf::Vector3(vec(0), vec(1), vec(2));
}

template_tensors::MatrixXXT<tfScalar, 3, 3, ColMajor> fromTfMat(tf::Matrix3x3 mat)
{
  return template_tensors::MatrixXXT<tfScalar, 3, 3, ColMajor>(
      mat.getRow(0).m_floats[0], mat.getRow(1).m_floats[0], mat.getRow(2).m_floats[0],
      mat.getRow(0).m_floats[1], mat.getRow(1).m_floats[1], mat.getRow(2).m_floats[1],
      mat.getRow(0).m_floats[2], mat.getRow(1).m_floats[2], mat.getRow(2).m_floats[2]
    );
}

tf::Matrix3x3 toTfMat(template_tensors::MatrixXXT<tfScalar, 3, 3, ColMajor> mat)
{
  tf::Matrix3x3 result;
  result.setValue(
      mat(0, 0), mat(0, 1), mat(0, 2),
      mat(1, 0), mat(1, 1), mat(1, 2),
      mat(2, 0), mat(2, 1), mat(2, 2)
    );
  return result;
}

namespace geometry {

template <typename TScalar = tfScalar>
__host__
transform::Rigid<TScalar, 3> fromTf(const tf::Transform& t)
{
  return transform::Rigid<TScalar, 3>(template_tensors::fromTfMat(t.getBasis()), template_tensors::fromTfVec(t.getOrigin()));
}

template <typename TScalar>
__host__
tf::Transform toTf(const transform::Rigid<TScalar, 3>& t)
{
  return tf::Transform(template_tensors::toTfMat(t.getRotation()), template_tensors::toTfVec(t.getTranslation()));
}

} // end of ns geometry

} // end of ns template_tensors

#endif