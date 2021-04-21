namespace template_tensors {

namespace quaternion {

// TODO: Add quaternion representations wxyz and xyzw

template <typename T>
using Quaternion = template_tensors::VectorXT<T, 4>; // wxyz

template <typename T>
__host__ __device__
template_tensors::MatrixXXT<T, 3, 3> toMatrix(Quaternion<T> quaternion)
{
  template_tensors::MatrixXXT<T, 3, 3> result;

  result(0, 1) = static_cast<T>(2) * (quaternion(1) * quaternion(2) - quaternion(0) * quaternion(3));
  result(0, 2) = static_cast<T>(2) * (quaternion(1) * quaternion(3) + quaternion(0) * quaternion(2));
  result(1, 0) = static_cast<T>(2) * (quaternion(1) * quaternion(2) + quaternion(0) * quaternion(3));
  result(1, 2) = static_cast<T>(2) * (quaternion(2) * quaternion(3) - quaternion(0) * quaternion(1));
  result(2, 0) = static_cast<T>(2) * (quaternion(1) * quaternion(3) - quaternion(0) * quaternion(2));
  result(2, 1) = static_cast<T>(2) * (quaternion(2) * quaternion(3) + quaternion(0) * quaternion(1));

  result(0, 0) = static_cast<T>(1) - static_cast<T>(2) * (quaternion(2) * quaternion(2) + quaternion(3) * quaternion(3));
  result(1, 1) = static_cast<T>(1) - static_cast<T>(2) * (quaternion(1) * quaternion(1) + quaternion(3) * quaternion(3));
  result(2, 2) = static_cast<T>(1) - static_cast<T>(2) * (quaternion(1) * quaternion(1) + quaternion(2) * quaternion(2));

  return result;
}

template <typename TMatrix, typename TElementType = decay_elementtype_t<TMatrix>>
__host__ __device__
Quaternion<TElementType> fromMatrix(TMatrix&& matrix)
{
  Quaternion<TElementType> result;
  result(0) = math::sqrt(static_cast<TElementType>(1) + matrix(0, 0) + matrix(1, 1) + matrix(2, 2)) / static_cast<TElementType>(2);
  TElementType w4 = 4 * result(0);
  result(1) = (matrix(2, 1) - matrix(1, 2)) / w4;
  result(2) = (matrix(0, 2) - matrix(2, 0)) / w4;
  result(3) = (matrix(1, 0) - matrix(0, 1)) / w4;
  return result;
}

template <typename T1, typename T2, typename T = decltype(std::declval<T1>() * std::declval<T2>())>
__host__ __device__
Quaternion<T> fromAxisAngle(template_tensors::VectorXT<T1, 3> axis, T2 angle)
{
  T sin = math::sin(angle / static_cast<T>(2));
  T cos = math::cos(angle / static_cast<T>(2));

  return Quaternion<T>(cos, axis(0) * sin, axis(1) * sin, axis(2) * sin);
}

template <typename T>
__host__ __device__
Quaternion<T> conjugate(Quaternion<T> quaternion)
{
  return Quaternion<T>(quaternion(0), -quaternion(1), -quaternion(2), -quaternion(3));
}

template <typename T>
__host__ __device__
Quaternion<T> inverse(Quaternion<T> quaternion)
{
  return conjugate(quaternion) / template_tensors::length_squared(quaternion);
}

} // end of ns quaternion

} // end of ns tensor