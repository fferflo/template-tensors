namespace template_tensors {

namespace op {

template <metal::int_ TRank>
class ClosedFormInverse;
// TODO: memory types
template <>
class ClosedFormInverse<2>
{
public:
  template <typename TMatrixTypeDest, typename TMatrixTypeSrc>
  __host__ __device__
  bool operator()(TMatrixTypeDest&& dest, TMatrixTypeSrc&& src)
  {
    TT_MATRIX_INVERSE_CHECK_DIMS

    auto determinant = src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0); // TODO: closed form determinant
    if (determinant == 0)
    { // TODO: add is_zero check as template arg
      return false;
    }
    auto determinant_inv = 1 / determinant;
    dest(0, 0) = determinant_inv * src(1, 1);
    dest(1, 1) = determinant_inv * src(0, 0);
    dest(0, 1) = -determinant_inv * src(0, 1);
    dest(1, 0) = -determinant_inv * src(1, 0);
    return true;
  }
};

template <>
class ClosedFormInverse<3>
{
public:
  template <typename TMatrixTypeDest, typename TMatrixTypeSrc>
  __host__ __device__
  bool operator()(TMatrixTypeDest&& dest, TMatrixTypeSrc&& src)
  {
    TT_MATRIX_INVERSE_CHECK_DIMS

    auto determinant =
        src(0, 0) * (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1))
      - src(0, 1) * (src(1, 0) * src(2, 2) - src(1, 2) * src(2, 0))
      + src(0, 2) * (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0)); // TODO: closed form determinant
    if (determinant == 0)
    { // TODO: add is_zero check as template arg
      return false;
    }
    auto determinant_inv = 1 / determinant;

    dest(0, 0) = determinant_inv * (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1));
    dest(1, 1) = determinant_inv * (src(0, 0) * src(2, 2) - src(0, 2) * src(2, 0));
    dest(2, 2) = determinant_inv * (src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0));

    dest(0, 1) = -determinant_inv * (src(0, 1) * src(2, 2) - src(0, 2) * src(2, 1));
    dest(1, 0) = -determinant_inv * (src(1, 0) * src(2, 2) - src(1, 2) * src(2, 0));
    dest(1, 2) = -determinant_inv * (src(0, 0) * src(1, 2) - src(0, 2) * src(1, 0));
    dest(2, 1) = -determinant_inv * (src(0, 0) * src(2, 1) - src(0, 1) * src(2, 0));

    dest(0, 2) = determinant_inv * (src(0, 1) * src(1, 2) - src(0, 2) * src(1, 1));
    dest(2, 0) = determinant_inv * (src(1, 0) * src(2, 1) - src(2, 0) * src(1, 1));
    return true;
  }
};

} // end of ns op

} // end of ns tensor
