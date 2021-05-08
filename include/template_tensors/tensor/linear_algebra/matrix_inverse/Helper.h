namespace template_tensors {

namespace op {

#define TT_MATRIX_INVERSE_CHECK_DIMS \
  static const metal::int_ RANK = combine_dims_v< \
                              rows_v<TMatrixTypeDest>::value, \
                              cols_v<TMatrixTypeDest>::value, \
                              rows_v<TMatrixTypeSrc>::value, \
                              cols_v<TMatrixTypeSrc>::value \
                          >::value; \
  static_assert(RANK == DYN || RANK != DYN, "Incompatible dimensions"); \
  ASSERT(dest.rows() == dest.cols(), "Incompatible dimensions"); \
  ASSERT(src.rows() == src.cols(), "Incompatible dimensions"); \
  ASSERT(dest.rows() == src.rows(), "Incompatible dimensions");

} // end of ns op

} // end of ns template_tensors
