namespace template_tensors {

#define TENSOR_SOLVER_CHECK_X_AB_DIMS \
  static const size_t RANK = rows_v<TMatrixTypeX>::value != DYN ? rows_v<TMatrixTypeX>::value \
                           : rows_v<TMatrixTypeAb>::value != DYN ? rows_v<TMatrixTypeAb>::value \
                           : (cols_v<TMatrixTypeAb>::value && cols_v<TMatrixTypeX>::value != DYN) != DYN ? cols_v<TMatrixTypeAb>::value - cols_v<TMatrixTypeX>::value \
                           : DYN; \
  static const size_t COLS_TOTAL = \
                             (RANK != DYN && cols_v<TMatrixTypeX>::value != DYN) ? (RANK + cols_v<TMatrixTypeX>::value) \
                           : cols_v<TMatrixTypeAb>::value != DYN ? cols_v<TMatrixTypeAb>::value \
                           : DYN; \
 \
  static_assert(RANK == DYN || rows_v<TMatrixTypeAb>::value == DYN || RANK == rows_v<TMatrixTypeAb>::value, "Incompatible dimensions"); \
  static_assert(RANK == DYN || rows_v<TMatrixTypeX>::value == DYN || RANK == rows_v<TMatrixTypeX>::value, "Incompatible dimensions"); \
  static_assert(COLS_TOTAL == DYN || cols_v<TMatrixTypeAb>::value == DYN || COLS_TOTAL == cols_v<TMatrixTypeAb>::value, "Incompatible dimensions"); \
  ASSERT(Ab.rows() + x.cols() == Ab.cols(), "Incompatible dimensions"); \
  ASSERT(x.rows() == Ab.rows(), "Incompatible dimensions");

#define TENSOR_SOLVER_CHECK_X_A_B_DIMS \
  static const size_t RANK = \
                      rows_v<TMatrixTypeX>::value != DYN ? rows_v<TMatrixTypeX>::value \
                    : rows_v<TMatrixTypeA>::value != DYN ? rows_v<TMatrixTypeA>::value \
                    : cols_v<TMatrixTypeA>::value != DYN ? cols_v<TMatrixTypeA>::value \
                    : rows_v<TMatrixTypeB>::value != DYN ? rows_v<TMatrixTypeB>::value \
                    : DYN; \
  static const size_t COLS_RIGHT \
                    = cols_v<TMatrixTypeX>::value != DYN ? cols_v<TMatrixTypeX>::value \
                    : cols_v<TMatrixTypeB>::value != DYN ? cols_v<TMatrixTypeB>::value \
                    : DYN; \
 \
  static_assert(RANK == DYN || rows_v<TMatrixTypeA>::value == DYN || RANK == rows_v<TMatrixTypeA>::value, "Incompatible dimensions"); \
  static_assert(RANK == DYN || rows_v<TMatrixTypeX>::value == DYN || RANK == rows_v<TMatrixTypeX>::value, "Incompatible dimensions"); \
  static_assert(RANK == DYN || rows_v<TMatrixTypeB>::value == DYN || RANK == rows_v<TMatrixTypeB>::value, "Incompatible dimensions"); \
  static_assert(RANK == DYN || cols_v<TMatrixTypeA>::value == DYN || RANK == rows_v<TMatrixTypeA>::value, "Incompatible dimensions"); \
  static_assert(COLS_RIGHT == DYN || cols_v<TMatrixTypeX>::value == DYN || COLS_RIGHT == cols_v<TMatrixTypeX>::value, "Incompatible dimensions"); \
  static_assert(COLS_RIGHT == DYN || cols_v<TMatrixTypeB>::value == DYN || COLS_RIGHT == cols_v<TMatrixTypeB>::value, "Incompatible dimensions"); \
  ASSERT(x.rows() == A.rows(), "Incompatible dimensions"); \
  ASSERT(x.rows() == b.rows(), "Incompatible dimensions"); \
  ASSERT(x.cols() == b.cols(), "Incompatible dimensions"); \
  ASSERT(A.rows() == A.cols(), "Incompatible dimensions");

#define TENSOR_SOLVER_FORWARD_X_A_B(...) \
  template <typename TMatrixTypeX, typename TMatrixTypeA, typename TMatrixTypeB> \
  __VA_ARGS__ \
  bool operator()(TMatrixTypeX&& x, TMatrixTypeA&& A, TMatrixTypeB&& b) const \
  { \
    TENSOR_SOLVER_CHECK_X_A_B_DIMS \
 \
    return (*this)(util::forward<TMatrixTypeX>(x), \
      template_tensors::concat<1>(util::forward<TMatrixTypeA>(A), util::forward<TMatrixTypeB>(b))); \
  }

#define TENSOR_SOLVER_FORWARD_X_AB(...) \
  template <typename TMatrixTypeX, typename TMatrixTypeAb> \
  __VA_ARGS__ \
  bool operator()(TMatrixTypeX&& x, TMatrixTypeAb&& Ab) const \
  { \
    TENSOR_SOLVER_CHECK_X_AB_DIMS \
 \
    return (*this)( \
      util::forward<TMatrixTypeX>(x), \
      template_tensors::head(util::forward<TMatrixTypeAb>(Ab), Ab.rows(), Ab.rows()), \
      template_tensors::tail(util::forward<TMatrixTypeAb>(Ab), Ab.rows(), x.cols()) \
    ); \
  }
// TODO: template_tensors::head and template_tensors::tail don't consider static dims
} // end of ns tensor
