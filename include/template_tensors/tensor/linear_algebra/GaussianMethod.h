namespace template_tensors {

namespace op {

/*!
 * \defgroup LinearSystem Systems of Linear Equations
 * \ingroup TensorOperations
 * @{
 */

/*!
 * \brief Runs a gaussian elimination with partial pivoting on the given matrix, resulting in an upper triangular matrix.
 *
 * The matrix is modified inplace.
 *
 * @param matrix the the system of linear equations
 * @param epsilon a small floating point value used to determine whether a value is approximately zero
 */
template <typename TMatrixType>
__host__ __device__
void gaussian_elimination(TMatrixType&& matrix, decay_elementtype_t<TMatrixType> epsilon)
{
  using ElementType = decay_elementtype_t<TMatrixType>;

  size_t pivotRow = 0;
  for (size_t pivotCol = 0; pivotCol < matrix.template dim<1>(); pivotCol++)
  {
    // Find non-zero element in column
    ElementType largestAbsValue = 0;
    size_t r = 0;
    for (size_t r2 = pivotRow; r2 < matrix.template dim<0>(); r2++)
    {
      if (math::abs(matrix(r2, pivotCol)) > largestAbsValue)
      {
        largestAbsValue = math::abs(matrix(r2, pivotCol));
        r = r2;
      }
    }
    if (largestAbsValue > epsilon)
    {
      // Move row with non-zero element up
      for (size_t c = pivotCol; c < matrix.template dim<1>(); c++)
      {
        util::swap(matrix(r, c), matrix(pivotRow, c));
      }

      // Normalize row -> first element is 1.0
      ElementType divisor = matrix(pivotRow, pivotCol);
      matrix(pivotRow, pivotCol) = 1;
      for (size_t c = pivotCol + 1; c < matrix.template dim<1>(); c++)
      {
        matrix(pivotRow, c) /= divisor;
      }

      // Subtract row from lower rows -> column elements below pivotRow are 0.0
      for (r = pivotRow + 1; r < matrix.template dim<0>(); r++)
      {
        ElementType multiplier = matrix(r, pivotCol);
        matrix(r, pivotCol) = 0;
        for (size_t c = pivotCol + 1; c < matrix.template dim<1>(); c++)
        {
          matrix(r, c) -= multiplier * matrix(pivotRow, c);
        }
      }

      pivotRow++;
    }
  }
}

/*!
 * \brief Runs a back substitution on the given upper triangular matrix, resulting in a matrix in reduced row echelon form.
 *
 * The matrix is modified inplace.
 *
 * @param matrix the the system of linear equations
 * @param rightHandSideColumns the number of columns on the right side of the equation
 * @param epsilon a small floating point value used to determine whether a value is approximately zero
 */
template <typename TMatrixType>
__host__ __device__
void back_substitution(TMatrixType&& matrix, size_t rightHandSideColumns, decay_elementtype_t<TMatrixType> epsilon)
{
  using ElementType = decay_elementtype_t<TMatrixType>;

  size_t pivotCol = matrix.template dim<1>() - rightHandSideColumns;
  for (size_t pivotRow = matrix.template dim<0>() - 1; pivotRow != static_cast<size_t>(-1); pivotRow--)
  {
    // Find first non-zero element in row
    size_t newPivotCol;
    for (newPivotCol = 0; newPivotCol < pivotCol && math::abs(matrix(pivotRow, newPivotCol)) <= epsilon; newPivotCol++)
    {
    }
    if (newPivotCol < pivotCol)
    {
      // Subtract row from higher rows -> column elements above pivotRow are 0.0
      for (size_t r = pivotRow - 1; r != static_cast<size_t>(-1); r--)
      {
        ElementType multiplier = matrix(r, newPivotCol);

        // Left hand side
        matrix(r, newPivotCol) = 0;
        for (size_t c = newPivotCol + 1; c < pivotCol; c++)
        {
          matrix(r, c) -= multiplier * matrix(pivotRow, c);
        }

        // Right hand side
        for (size_t c = matrix.template dim<1>() - rightHandSideColumns; c < matrix.template dim<1>(); c++)
        {
          matrix(r, c) -= multiplier * matrix(pivotRow, c);
        }
      }
      pivotCol = newPivotCol;
    }
  }
}

/*!
 * \brief Finds the unique solution of the given system of linear equations in reduced row echelon form.
 *
 * @param solution[out] the unique solution
 * @param matrix the system of linear equations
 * @param epsilon a small floating point value used to determine whether a value is approximately zero
 * @return true if the given system of linear equations has a unique solution, false otherwise
 */
template <typename TMatrixTypeX, typename TMatrixTypeAb>
__host__ __device__
bool find_unique_solution(TMatrixTypeX&& x, const TMatrixTypeAb& Ab, decay_elementtype_t<TMatrixTypeAb> epsilon)
{
  const size_t RANK = rows_v<TMatrixTypeX>::value != DYN ? rows_v<TMatrixTypeX>::value
                    : rows_v<TMatrixTypeAb>::value != DYN ? rows_v<TMatrixTypeAb>::value
                    : (cols_v<TMatrixTypeAb>::value && cols_v<TMatrixTypeX>::value != DYN) != DYN ? cols_v<TMatrixTypeAb>::value - cols_v<TMatrixTypeX>::value
                    : DYN;
  const size_t COLS = (RANK != DYN && cols_v<TMatrixTypeX>::value != DYN) ? (RANK + cols_v<TMatrixTypeX>::value)
                    : cols_v<TMatrixTypeAb>::value != DYN ? cols_v<TMatrixTypeAb>::value
                    : DYN;

  static_assert(RANK == DYN || rows_v<TMatrixTypeAb>::value == DYN || RANK == rows_v<TMatrixTypeAb>::value, "Incompatible dimensions");
  static_assert(RANK == DYN || rows_v<TMatrixTypeX>::value == DYN || RANK == rows_v<TMatrixTypeX>::value, "Incompatible dimensions");
  static_assert(COLS == DYN || cols_v<TMatrixTypeAb>::value == DYN || COLS == cols_v<TMatrixTypeAb>::value, "Incompatible dimensions");
  ASSERT(Ab.rows() + x.cols() == Ab.cols(), "Incompatible dimensions");
  ASSERT(x.rows() == Ab.rows(), "Incompatible dimensions");

  for (size_t r = Ab.rows() - 1; r != static_cast<size_t>(-1); r--)
  {
    // Find non-zero element in row on left side of system
    size_t c;
    for (c = 0; c < Ab.rows() && math::abs(Ab(r, c)) <= epsilon; c++)
    {
    }
    if (c == Ab.rows())
    {
      // Equation in row: 0 = 1 or 0 = 0
      return false;
    }
  }

  x = template_tensors::offset(Ab, 0, Ab.cols() - x.cols());
  return true;
}

/*!
 * @}
 */

} // end of ns op

} // end of ns tensor
