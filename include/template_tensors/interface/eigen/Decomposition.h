#ifdef EIGEN_INCLUDED

#include <Eigen/Dense>

namespace template_tensors {

template <typename TScalar>
struct EigenColPivHouseholderQRSolver
{
  template <typename TMatrixTypeX, typename TMatrixTypeA, typename TMatrixTypeB>
  __host__
  bool operator()(TMatrixTypeX&& x, TMatrixTypeA&& A, TMatrixTypeB&& b) const
  {
    TT_SOLVER_CHECK_X_A_B_DIMS

    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> A_eigen = toEigen(template_tensors::static_cast_to<TScalar>(A));
    auto decomp = A_eigen.colPivHouseholderQr();
    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> b_eigen = toEigen(template_tensors::static_cast_to<TScalar>(b));
    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> x_eigen = decomp.solve(b_eigen);
    if (static_cast<size_t>(decomp.rank()) == A.rows())
    {
      x = fromEigen(x_eigen);
      return true;
    }
    else
    {
      return false;
    }
  }
  // TODO: this solves only with unique solutions. what about a solver for solution spaces? same with gaussian solver
  TT_SOLVER_FORWARD_X_AB(__host__)
};

template <typename TScalar>
struct EigenColPivHouseholderQRInverse
{
  HD_WARNING_DISABLE
  template <typename TMatrixTypeDest, typename TMatrixTypeSrc>
  __host__
  bool operator()(TMatrixTypeDest&& dest, TMatrixTypeSrc&& src)
  {
    TT_MATRIX_INVERSE_CHECK_DIMS

    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> src_eigen = toEigen(template_tensors::static_cast_to<TScalar>(src));
    auto decomp = src_eigen.colPivHouseholderQr();
    if (decomp.isInvertible())
    {
      Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> dest_eigen = decomp.inverse();
      dest = fromEigen(dest_eigen);
      return true;
    }
    else
    {
      return false;
    }
  }
};

} // end of ns template_tensors

#endif
