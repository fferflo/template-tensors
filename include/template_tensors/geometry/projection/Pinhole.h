namespace template_tensors {

namespace geometry {

namespace projection {

template <typename TIntrinsicsMatrix>
class PinholeK;

template <typename TFocalLength, typename TPrincipalPoint>
class PinholeFC
{
public:
  static const size_t RANK = template_tensors::rows_v<TPrincipalPoint>::value + 1;
  using Scalar = template_tensors::decay_elementtype_t<TPrincipalPoint>;

  __host__ __device__
  PinholeFC(TFocalLength f, TPrincipalPoint c)
    : m_f(f)
    , m_1_over_f(1 / f)
    , m_c(c)
  {
  }

  __host__ __device__
  PinholeFC()
    : PinholeFC(1, 0)
  {
  }

  template <typename TFocalLength2, typename TPrincipalPoint2>
  __host__ __device__
  PinholeFC(const PinholeFC<TFocalLength2, TPrincipalPoint2>& other)
    : m_f(other.m_f)
    , m_1_over_f(other.m_1_over_f)
    , m_c(other.m_c)
  {
  }

  template <typename TFocalLength2, typename TPrincipalPoint2>
  __host__ __device__
  PinholeFC<TFocalLength, TPrincipalPoint>& operator=(const PinholeFC<TFocalLength2, TPrincipalPoint2>& other)
  {
    m_f = other.m_f;
    m_1_over_f = other.m_1_over_f;
    m_c = other.m_c;
    return *this;
  }

  __host__ __device__
  template_tensors::VectorXT<Scalar, RANK> unproject(template_tensors::VectorXT<Scalar, RANK - 1> point) const
  {
    return template_tensors::concat<0>((point - m_c) * m_1_over_f, VectorXT<Scalar, 1>(1));
  }

  __host__ __device__
  template_tensors::VectorXT<Scalar, RANK - 1> project(template_tensors::VectorXT<Scalar, RANK> point) const
  {
    return template_tensors::head<RANK - 1>(point) * m_f / point(RANK - 1) + m_c;
  }

  __host__ __device__
  const TFocalLength& getFocalLength() const
  {
    return m_f;
  }

  __host__ __device__
  const TPrincipalPoint& getPrincipalPoint() const
  {
    return m_c;
  }

  template <typename TFocalLength2, typename TPrincipalPoint2>
  friend class PinholeFC;
  template <typename TIntrinsicsMatrix>
  friend class PinholeK;

  template <typename TElementType = Scalar>
  template_tensors::MatrixXXT<Scalar, RANK, RANK> asMatrix() const
  {
    return PinholeK<template_tensors::MatrixXXT<Scalar, RANK, RANK>>(*this).asMatrix();
  }

  template <typename TElementType>
  operator template_tensors::MatrixXXT<TElementType, RANK, RANK>() const
  {
    return asMatrix<TElementType>();
  }

#ifdef OPENCV_INCLUDED
  operator ::cv::Mat() const
  {
    return template_tensors::toCv(this->asMatrix());
  }
#endif

private:
  TFocalLength m_f;
  TFocalLength m_1_over_f;
  TPrincipalPoint m_c;
};

template <typename TFocalLength, typename TPrincipalPoint>
__host__
std::ostream& operator<<(std::ostream& stream, const PinholeFC<TFocalLength, TPrincipalPoint>& p)
{
  stream << "Pinhole(f=" << p.getFocalLength() << " c=" << p.getPrincipalPoint() << ")";
  return stream;
}

template <typename TIntrinsicsMatrix>
class PinholeK
{
public:
  static const size_t RANK = template_tensors::rows_v<TIntrinsicsMatrix>::value;
  using Scalar = template_tensors::decay_elementtype_t<TIntrinsicsMatrix>;

  __host__ __device__
  PinholeK(TIntrinsicsMatrix k)
    : m_k(k)
    , m_k_inv(template_tensors::inverse(k))
  {
  }

  __host__ __device__
  PinholeK()
    : PinholeK(template_tensors::eye<Scalar, RANK>())
  {
  }

  template <typename TIntrinsicsMatrix2>
  __host__ __device__
  PinholeK(const PinholeK<TIntrinsicsMatrix2>& other)
    : m_k(other.m_k)
    , m_k_inv(other.m_k_inv)
  {
  }

  template <typename TFocalLength, typename TPrincipalPoint>
  __host__ __device__
  PinholeK(const PinholeFC<TFocalLength, TPrincipalPoint>& other)
  {
    *this = other;
  }

  template <typename TIntrinsicsMatrix2>
  __host__ __device__
  PinholeK<TIntrinsicsMatrix>& operator=(const PinholeK<TIntrinsicsMatrix2>& other)
  {
    m_k = other.m_k;
    m_k_inv = other.m_k_inv;
    return *this;
  }

  template <typename TFocalLength, typename TPrincipalPoint>
  __host__ __device__
  PinholeK<TIntrinsicsMatrix>& operator=(const PinholeFC<TFocalLength, TPrincipalPoint>& other)
  {
    template_tensors::VectorXT<Scalar, RANK - 1> f(other.m_f);
    template_tensors::VectorXT<Scalar, RANK - 1> one_over_f(other.m_1_over_f);
    template_tensors::VectorXT<Scalar, RANK - 1> c(other.m_c);

    m_k = 0;
    for (size_t i = 0; i < RANK - 1; i++)
    {
      m_k(i, i) = f(i);
      m_k(i, RANK - 1) = c(i);
    }
    m_k(RANK - 1, RANK - 1) = 1;

    m_k_inv = 0;
    for (size_t i = 0; i < RANK - 1; i++)
    {
      m_k_inv(i, i) = one_over_f(i);
      m_k_inv(i, RANK - 1) = -c(i) * one_over_f(i);
    }
    m_k_inv(RANK - 1, RANK - 1) = 1;

    return *this;
  }

  __host__ __device__
  template_tensors::VectorXT<Scalar, RANK> unproject(template_tensors::VectorXT<Scalar, RANK - 1> point) const
  {
    return template_tensors::matmul(m_k_inv, template_tensors::homogenize(point));
  }

  __host__ __device__
  template_tensors::VectorXT<Scalar, RANK - 1> project(template_tensors::VectorXT<Scalar, RANK> point) const
  {
    return template_tensors::dehomogenize(template_tensors::matmul(m_k, point));
  }

  __host__ __device__
  const TIntrinsicsMatrix& getMatrix() const
  {
    return m_k;
  }

  template <typename TElementType = Scalar>
  template_tensors::MatrixXXT<TElementType, RANK, RANK> asMatrix() const
  {
    return getMatrix();
  }

  template <typename TElementType>
  operator template_tensors::MatrixXXT<TElementType, RANK, RANK>() const
  {
    return asMatrix<TElementType>();
  }

#ifdef OPENCV_INCLUDED
  operator ::cv::Mat() const
  {
    return template_tensors::toCv(this->asMatrix());
  }
#endif

  template <typename TIntrinsicsMatrix2>
  friend class PinholeK;

private:
  TIntrinsicsMatrix m_k;
  TIntrinsicsMatrix m_k_inv;
};

template <typename TIntrinsicsMatrix>
__host__
std::ostream& operator<<(std::ostream& stream, const PinholeK<TIntrinsicsMatrix>& p)
{
  stream << "Pinhole(K=" << p.getMatrix() << ")";
  return stream;
}



// principal point = (0, 0)
template <typename TScalar>
__host__ __device__
PinholeFC<VectorXT<TScalar, 2>, VectorXT<TScalar, 2>> fromSymmetricFov(TScalar aspect_ratio_x_over_y, TScalar field_of_view_y)
{
  TScalar fx = static_cast<TScalar>(1) / (aspect_ratio_x_over_y * math::tan(field_of_view_y / 2));
  TScalar fy = static_cast<TScalar>(1) / math::tan(field_of_view_y / 2);
  return PinholeFC<VectorXT<TScalar, 2>, VectorXT<TScalar, 2>>(VectorXT<TScalar, 2>(fx, fy), VectorXT<TScalar, 2>(0));
}
// TODO: refactor these two methods
// principal point = resolution / 2
template <typename TScalar>
__host__ __device__
PinholeFC<VectorXT<TScalar, 2>, VectorXT<TScalar, 2>> fromSymmetricFov(template_tensors::Vector2s resolution, TScalar field_of_view_y)
{
  TScalar cx = resolution(0) / 2.0f;
  TScalar cy = resolution(1) / 2.0f;
  TScalar fx = cx / (((TScalar) resolution(0)) / resolution(1) * math::tan(field_of_view_y / 2));
  TScalar fy = cy / math::tan(field_of_view_y / 2);
  return PinholeFC<VectorXT<TScalar, 2>, VectorXT<TScalar, 2>>(VectorXT<TScalar, 2>(fx, fy), VectorXT<TScalar, 2>(cx, cy));
}
/*
TODO:
  __host__ __device__
  template_tensors::MatrixXXT<Scalar, RANK + 1, RANK + 1> matrix(Scalar near, Scalar far)
  {
    template_tensors::MatrixXXT<Scalar, RANK + 1, RANK + 1> result(0);
    for (size_t i = 0; i < RANK - 1; i++)
    {
      result(i, i) = m_f(i);
      result(i, RANK - 1) = -m_c(i);
    }
    result(RANK - 1, RANK - 1) = -(far + near) / (far - near);
    result(RANK - 1, RANK) = -2 * far * near / (far - near);
    result(RANK, RANK - 1) = -1;
    return result;
  }*/

} // end of ns projection

} // end of ns geometry

} // end of ns tensor
