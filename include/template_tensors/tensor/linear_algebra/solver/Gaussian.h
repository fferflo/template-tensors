namespace template_tensors {

namespace op {

template <typename TScalar>
class GaussSolver
{
public:
  __host__ __device__
  GaussSolver(TScalar epsilon)
    : m_epsilon(epsilon)
  {
  }

  template <typename TMatrixTypeX, typename TMatrixTypeAb>
  __host__ __device__
  bool operator()(TMatrixTypeX&& x, TMatrixTypeAb&& Ab) const
  {
    TT_SOLVER_CHECK_X_AB_DIMS

    LocalOrAllocTensorT<TScalar, mem::alloc::heap, ColMajor, DimSeq<RANK, COLS_TOTAL>>
      Ab_copy(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, Ab.dims());
    Ab_copy = util::forward<TMatrixTypeAb>(Ab);

    gaussian_elimination(Ab_copy, m_epsilon); // TODO: try index strategies for this
    back_substitution(Ab_copy, x.cols(), m_epsilon); // and this?
    return find_unique_solution(util::forward<TMatrixTypeX>(x), Ab_copy, m_epsilon);
  }

  TT_SOLVER_FORWARD_X_A_B(__host__ __device__)

private:
  TScalar m_epsilon;
};

} // end of ns op

} // end of ns tensor
