namespace template_tensors {

#define ThisType IdentityMatrix<TElementType, TRowsCols>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        template_tensors::DimSeq<TRowsCols, TRowsCols> \
                              >
template <typename TElementType, size_t TRowsCols>
class IdentityMatrix : public SuperType
{
public:
  __host__ __device__
  IdentityMatrix()
    : SuperType(TRowsCols, TRowsCols)
  {
  }

  __host__ __device__
  IdentityMatrix(size_t rows_cols)
    : SuperType(rows_cols, rows_cols)
  {
    ASSERT(TRowsCols == DYN || TRowsCols == rows_cols, "Static and dynamic dimensions must be equal");
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, size_t row, size_t col)
  RETURN_AUTO(
    row == col ? 1 : 0
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SIZE_T_N(getElement, 2)

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }
};
#undef SuperType
#undef ThisType

// TODO: combine these two classes, constructor can be overloaded with SFINAE

#define ThisType IdentityMatrix<TElementType, DYN>
#define SuperType TensorBase< \
                              ThisType, \
                              mem::LOCAL, \
                              template_tensors::DimSeq<DYN, DYN> \
                            >

template <typename TElementType>
class IdentityMatrix<TElementType, DYN> : public SuperType
{
public:
  __host__ __device__
  IdentityMatrix(size_t rows_cols)
    : SuperType(rows_cols, rows_cols)
    , m_rows_cols(rows_cols)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, size_t row, size_t col)
  RETURN_AUTO(
    row == col ? 1 : 0
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_SIZE_T_N(getElement, 2)

  template <size_t TIndex>
  __host__ __device__
  size_t getDynDim() const
  {
    return TIndex < 2UL ? m_rows_cols : 1;
  }

  __host__ __device__
  size_t getDynDim(size_t index) const
  {
    return index < 2UL ? m_rows_cols : 1;
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }

private:
  size_t m_rows_cols;
};
#undef SuperType
#undef ThisType
// TODO: replace all IdentityMatrix usages with template_tensors::eye
template <typename TElementType, size_t TRank = template_tensors::DYN, typename... TDimArgTypes>
__host__ __device__
auto eye(TDimArgTypes&&... dim_args)
RETURN_AUTO(IdentityMatrix<TElementType, TRank>(util::forward<TDimArgTypes>(dim_args)...))

} // end of ns tensor