namespace template_tensors {

namespace op {

template <typename TScalar>
class GaussInverse
{
public:
  __host__ __device__
  GaussInverse(TScalar epsilon)
    : m_epsilon(epsilon)
  {
  }

  template <typename TMatrixTypeDest, typename TMatrixTypeSrc>
  __host__ __device__
  bool operator()(TMatrixTypeDest&& dest, TMatrixTypeSrc&& src)
  {
    TT_MATRIX_INVERSE_CHECK_DIMS

    return GaussSolver<TScalar>(m_epsilon)
      (std::forward<TMatrixTypeDest>(dest), std::forward<TMatrixTypeSrc>(src), template_tensors::IdentityMatrix<TScalar, RANK>(dest.rows()));
  }

private:
  TScalar m_epsilon;
};

} // end of ns op

} // end of ns template_tensors
