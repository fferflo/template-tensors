namespace template_tensors {

// TODO: dimension assertion in this file

/*!
 * \brief Returns whether the given matrix is quadratic
 * \ingroup MiscTensorOps
 *
 * @param m the matrix
 * @return true if the given matrix is quadratic, false otherwise
 */
template <typename TMatrixType>
__host__ __device__
bool isQuadratic(const TMatrixType& m)
{
  return m.template dim<0>() == m.template dim<1>();
}

/*!
 * \brief Returns whether the given matrix is symmetric
 * \ingroup MiscTensorOps
 *
 * @param m the matrix
 * @return true if the given matrix is symmetric, false otherwise
 */
template <typename TMatrixType>
__host__ __device__
bool isSymmetric(const TMatrixType& m)
{
  if (!isQuadratic(m))
  {
    return false;
  }

  for (dim_t r = 0; r < m.template dim<0>(); r++)
  {
    for (dim_t c = r + 1; c < m.template dim<1>(); c++)
    {
      if (m(r, c) != m(c, r))
      {
        return false;
      }
    }
  }
  return true;
}



#define OPERATION_T(NAME, OPERATION) \
  template <typename TTensorType> \
  __host__ __device__ \
  auto NAME(TTensorType&& t) \
  RETURN_AUTO(OPERATION); \
  namespace functor { \
    struct NAME \
    { \
      template <typename TTensorType> \
      __host__ __device__ \
      auto operator()(TTensorType&& t) const \
      RETURN_AUTO(OPERATION) \
    }; \
  }

#define OPERATION_TT(NAME, OPERATION) \
  template <typename TTensorType1, typename TTensorType2> \
  __host__ __device__ \
  auto NAME(TTensorType1&& t1, TTensorType2&& t2) \
  RETURN_AUTO(OPERATION); \
  namespace functor { \
    struct NAME \
    { \
      template <typename TTensorType1, typename TTensorType2> \
      __host__ __device__ \
      auto operator()(TTensorType1&& t1, TTensorType2&& t2) const \
      RETURN_AUTO(OPERATION) \
    }; \
  }

#define T (util::forward<TTensorType>(t))
#define T1 (util::forward<TTensorType1>(t1))
#define T2 (util::forward<TTensorType2>(t2))

/*!
 * \brief Returns the dot product of the given tensors
 * \ingroup MiscTensorOps
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return the reduced sum of the element-wise product of t1 and t2
 */
OPERATION_TT(dot, template_tensors::sum((T1) * (T2)))
/*!
 * \brief Returns the squared euclidean length of the given tensor
 * \ingroup MiscTensorOps
 *
 * @param t the input tensor
 * @return dot(t, t)
 */
OPERATION_T(length_squared, template_tensors::dot(T, T))
/*!
 * \brief Returns the euclidean length of the given tensor
 * \ingroup MiscTensorOps
 *
 * @param t the input tensor
 * @return sqrt(dot(t, t))
 */
OPERATION_T(length, math::sqrt(template_tensors::length_squared(T)))
/*!
 * \brief Returns the euclidean distance of the given tensors
 * \ingroup MiscTensorOps
 *
 * @param t1 the left input tensor
 * @param t2 the right input tensor
 * @return length(t2 - t1)
 */
OPERATION_TT(distance, template_tensors::length(T2 - T1))

// TODO: replace length, length_squared, distance with norm functions
// OPERATION_T(l0_norm, template_tensors::count(template_tensors::abs(T) > 1e-6)) TODO: implement
OPERATION_T(l1_norm, template_tensors::sum(template_tensors::abs(T)))
OPERATION_T(l2_norm, math::sqrt(template_tensors::dot(T, T)))
OPERATION_T(linf_norm, template_tensors::max_el(template_tensors::abs(T)))

template <typename TNorm = template_tensors::functor::l2_norm, typename TTensorType>
__host__ __device__
auto normalize(TTensorType&& t, TNorm norm = TNorm())
RETURN_AUTO(T * (static_cast<decay_elementtype_t<TTensorType>>(1) / norm(T)));
namespace functor {
template <typename TNorm = template_tensors::functor::l2_norm>
struct normalize
{
  TNorm norm;

  __host__ __device__
  normalize(TNorm norm = TNorm())
    : norm(norm)
  {
  }

  template <typename TTensorType>
  __host__ __device__
  auto operator()(TTensorType&& t) const
  RETURN_AUTO(template_tensors::normalize(T, norm))
};
} // end of ns functor

OPERATION_TT(cos_angle, template_tensors::dot(template_tensors::normalize(T1), template_tensors::normalize(T2)))
OPERATION_TT(angle, math::acos(template_tensors::cos_angle(T1, T2)))
OPERATION_TT(acute_angle, math::acos(math::abs(template_tensors::dot(template_tensors::normalize(T1), template_tensors::normalize(T2)))))

// TODO: assert dimensions, or add implementation for n dimensions
template <typename TVector1, typename TVector2>
__host__ __device__
auto directed_angle(TVector1&& a, TVector2&& b)
RETURN_AUTO(math::atan2(a(0) * b(1) - a(1) * b(0), a(0) * b(0) + a(1) * b(1)))

#undef OPERATION_T
#undef OPERATION_TT

#undef T
#undef T1
#undef T2

template <typename TVectorType>
__host__ __device__
auto toPolar(const TVectorType& v)
RETURN_AUTO(VectorXT<decay_elementtype_t<TVectorType>, 2>(length(v), math::atan2(v(1), v(0))))

namespace functor {
struct toPolar
{
  template <typename TTensorType>
  __host__ __device__
  auto operator()(TTensorType&& t) const
  RETURN_AUTO(template_tensors::toPolar(util::forward<TTensorType>(t)))
};
}

template <typename TVectorType>
__host__ __device__
auto toCartesian(const TVectorType& v)
RETURN_AUTO(VectorXT<decay_elementtype_t<TVectorType>, 2>(v(0) * math::cos(v(1)), v(0) * math::sin(v(1))))

namespace functor {
struct toCartesian
{
  template <typename TTensorType>
  __host__ __device__
  auto operator()(TTensorType&& t) const
  RETURN_AUTO(template_tensors::toCartesian(util::forward<TTensorType>(t)))
};
}

} // end of ns tensor
