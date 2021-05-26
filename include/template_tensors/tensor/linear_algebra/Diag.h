namespace template_tensors {

#define ThisType DiagTensor<TMatrixType>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TMatrixType>::value, \
                                        template_tensors::DimSeq<rows_v<TMatrixType>::value> \
                              >

template <typename TMatrixType>
class DiagTensor : public SuperType
{
public:
  static_assert(is_matrix_v<TMatrixType>::value, "TMatrixType must be matrices");

  __host__ __device__
  DiagTensor(TMatrixType matrix)
    : SuperType(matrix.template dim<0>())
    , m_matrix(matrix)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, dim_t row)
  RETURN_AUTO(
    self.m_matrix(row, row)
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 1)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return TIndex == 0 ? m_matrix.rows() : 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index == 0 ? m_matrix.rows() : 1;
  }

private:
  TMatrixType m_matrix;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(DiagTensor<decltype(transform(m_matrix))>
    (transform(m_matrix))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(DiagTensor<decltype(transform(m_matrix))>
    (transform(m_matrix))
  )
};
#undef SuperType
#undef ThisType



template <typename TMatrixType>
__host__ __device__
auto diag(TMatrixType&& matrix)
RETURN_AUTO(DiagTensor<TMatrixType>
  (std::forward<TMatrixType>(matrix))
)

template <typename TMatrixType>
__host__ __device__
auto trace(TMatrixType&& matrix)
RETURN_AUTO(template_tensors::sum(template_tensors::diag(std::forward<TMatrixType>(matrix))))

} // end of ns template_tensors
