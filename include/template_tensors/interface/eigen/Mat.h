#ifdef EIGEN_INCLUDED

#include <Eigen/Dense>

namespace template_tensors {

constexpr size_t fromEigenDim(int eigen_dim)
{
  return eigen_dim == Eigen::Dynamic ? template_tensors::DYN : eigen_dim;
}

constexpr int toEigenDim(size_t dim)
{
  return dim == template_tensors::DYN ? Eigen::Dynamic : static_cast<int>(dim);
}

template <int TOption>
struct EigenOptionToIndexStrategy;

template <>
struct EigenOptionToIndexStrategy<Eigen::ColMajor>
{
  using type = template_tensors::ColMajor;
};

template <>
struct EigenOptionToIndexStrategy<Eigen::RowMajor>
{
  using type = template_tensors::RowMajor;
};

template <typename TIndexStrategy>
struct EigenIndexStrategyToOption;

template <>
struct EigenIndexStrategyToOption<template_tensors::ColMajor>
{
  static const int value = Eigen::ColMajor;
};

template <>
struct EigenIndexStrategyToOption<template_tensors::RowMajor>
{
  static const int value = Eigen::RowMajor;
};


#define ThisType FromEigenMatrix<TEigenMatrix>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::HOST, \
                                        template_tensors::DimSeq<fromEigenDim(Eigen::internal::traits<typename std::decay<TEigenMatrix>::type>::RowsAtCompileTime), \
                                                       fromEigenDim(Eigen::internal::traits<typename std::decay<TEigenMatrix>::type>::ColsAtCompileTime)> \
                              >
template <typename TEigenMatrix>
class FromEigenMatrix : public SuperType
{
public:
  static_assert(!std::is_rvalue_reference<TEigenMatrix>::value, "Cannot store rvalue reference");

  __host__ __device__
  FromEigenMatrix(TEigenMatrix eigen_matrix)
    : SuperType(eigen_matrix.rows(), eigen_matrix.cols())
    , m_eigen_matrix(eigen_matrix)
  {
  }

  TENSOR_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TSizeTCoords>
  __host__
  static auto getElement(TThisType&& self, size_t rows, size_t cols)
  RETURN_AUTO(
    self.m_eigen_matrix(rows, cols)
  )
  TENSOR_FORWARD_ELEMENT_ACCESS_SIZE_T_N(getElement, 2) // TODO: overhead for eigen vectors

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    return TIndex == 0 ? m_eigen_matrix.rows() : TIndex == 1 ? m_eigen_matrix.cols() : 1;
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    switch (index)
    {
      case 0: return m_eigen_matrix.rows();
      case 1: return m_eigen_matrix.cols();
      default: return 1;
    }
  }

private:
  TEigenMatrix m_eigen_matrix;
};
#undef SuperType
#undef ThisType

namespace detail {

template <typename TEigenDerived>
__host__
auto fromEigenDerived(TEigenDerived&& eigen_matrix)
RETURN_AUTO(
  FromEigenMatrix<util::store_member_t<TEigenDerived&&>>(util::forward<TEigenDerived>(eigen_matrix))
)

} // end of ns detail

template <typename TEigenBase>
__host__
auto fromEigen(TEigenBase&& eigen_matrix)
RETURN_AUTO(
  detail::fromEigenDerived(util::move_if<std::is_rvalue_reference<TEigenBase&&>::value>(eigen_matrix.derived()))
)



template <typename TArg>
struct EigenSupportsTensor
{
  template <typename TTensorType2, ENABLE_IF(
       is_indexed_pointer_tensor_v<TTensorType2>::value
    && !std::is_rvalue_reference<TTensorType2&&>::value
    && !std::is_const<typename std::remove_reference<TTensorType2>::type>::value
  )>
  TMP_IF(TTensorType2&&)
  TMP_RETURN_VALUE(
       std::is_same<indexstrategy_t<TTensorType2>, template_tensors::RowMajor>::value
    || std::is_same<indexstrategy_t<TTensorType2>, template_tensors::ColMajor>::value
  )

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

template <typename TMatrixType,
  typename TElementType = typename std::remove_reference<decltype(std::declval<TMatrixType>()())>::type,
  ENABLE_IF(EigenSupportsTensor<TMatrixType&&>::value)>
__host__
auto toEigen(TMatrixType&& matrix)
RETURN_AUTO(
  Eigen::Map<Eigen::Matrix<
    TElementType,
    toEigenDim(rows_v<TMatrixType>::value),
    toEigenDim(cols_v<TMatrixType>::value),
    EigenIndexStrategyToOption<indexstrategy_t<TMatrixType>>::value
  >>(
    &matrix(), matrix.rows(), matrix.cols()
))

template <typename TMatrixType,
  typename TElementType = decay_elementtype_t<TMatrixType>,
  typename TEigenMatrix = Eigen::Matrix<
    TElementType,
    toEigenDim(rows_v<TMatrixType>::value),
    toEigenDim(cols_v<TMatrixType>::value),
    EigenIndexStrategyToOption<template_tensors::ColMajor>::value
  >,
  ENABLE_IF(!EigenSupportsTensor<TMatrixType>::value)>
__host__
TEigenMatrix toEigen(TMatrixType&& matrix)
{
  TEigenMatrix eigen(matrix.rows(), matrix.cols());
  fromEigen(eigen) = util::forward<TMatrixType>(matrix);
  return eigen;
}

} // end of ns tensor

#endif
