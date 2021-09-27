namespace template_tensors {

#define ThisType HomogenizedRotation<TMatrixType>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TMatrixType>::value, \
                                        template_tensors::DimSeq< \
                                          detail::HomogenizedRowsHelper<rows_v<TMatrixType>::value>::value, \
                                          detail::HomogenizedRowsHelper<cols_v<TMatrixType>::value>::value \
                                        > \
                              >

template <typename TMatrixType>
class HomogenizedRotation : public SuperType
{
public:
  static_assert(is_matrix_v<TMatrixType>::value, "TMatrixType must be a matrix");

  static const metal::int_ ROWS = rows_v<SuperType>::value;
  static const metal::int_ COLS = cols_v<SuperType>::value;

  static_assert(ROWS == COLS, "Rotation matrix must be square");
  static_assert(ROWS == DYN || COLS == DYN || ROWS == COLS, "Must be square matrix");

  __host__ __device__
  HomogenizedRotation(TMatrixType input)
    : SuperType(input.template dim<0>() + 1, input.template dim<1>() + 1)
    , m_input(input)
  {
  }

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static decay_elementtype_t<TMatrixType> getElement(TThisType&& self, dim_t row, dim_t col)
  {
    if  (row == self.template dim<0>() - 1
      && col == self.template dim<1>() - 1)
    {
      return static_cast<decay_elementtype_t<TMatrixType>>(1);
    }
    else if (row < self.template dim<0>() - 1
          && col < self.template dim<1>() - 1)
    {
      return self.m_input(row, col);
    }
    else
    {
      return static_cast<decay_elementtype_t<TMatrixType>>(0);
    }
  }
  TT_TENSOR_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 2)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return TIndex < 2 ? m_input.template dim<TIndex>() + 1 : 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index < 2 ? m_input.dim(index) + 1 : 1;
  }

private:
  TMatrixType m_input;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(HomogenizedRotation<decltype(transform(m_input))>
    (transform(m_input))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(HomogenizedRotation<decltype(transform(m_input))>
    (transform(m_input))
  )
};
#undef SuperType
#undef ThisType

template <typename TMatrixType>
__host__ __device__
auto homogenizeRotation(TMatrixType&& rotation)
RETURN_AUTO(
  HomogenizedRotation<TMatrixType>(std::forward<TMatrixType>(rotation))
)

} // end of ns template_tensors
