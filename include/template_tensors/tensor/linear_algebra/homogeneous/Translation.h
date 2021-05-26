namespace template_tensors {

#define ThisType HomogenizedTranslation<TVectorType>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TVectorType>::value, \
                                        template_tensors::DimSeq< \
                                          detail::HomogenizedRowsHelper<rows_v<TVectorType>::value>::value, \
                                          detail::HomogenizedRowsHelper<rows_v<TVectorType>::value>::value \
                                        > \
                              >

template <typename TVectorType>
class HomogenizedTranslation : public SuperType
{
public:
  static_assert(is_vector_v<TVectorType>::value, "TVectorType must be a vector");

  static const metal::int_ COLS = cols_v<SuperType>::value;

  __host__ __device__
  HomogenizedTranslation(TVectorType input)
    : SuperType(input.template dim<0>() + 1, input.template dim<0>() + 1)
    , m_input(input)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static decay_elementtype_t<TVectorType> getElement(TThisType&& self, dim_t row, dim_t col)
  {
    if (row == col)
    {
      return static_cast<decay_elementtype_t<TVectorType>>(1);
    }
    else if (col == COLS - 1)
    {
      return self.m_input(row);
    }
    else
    {
      return static_cast<decay_elementtype_t<TVectorType>>(0);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 2)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return TIndex < 2 ? m_input.template dim<0>() + 1 : 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index < 2 ? m_input.template dim<0>() + 1 : 1;
  }

private:
  TVectorType m_input;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(HomogenizedTranslation<decltype(transform(m_input))>
    (transform(m_input))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(HomogenizedTranslation<decltype(transform(m_input))>
    (transform(m_input))
  )
};
#undef SuperType
#undef ThisType

template <typename TVectorType>
__host__ __device__
auto homogenizeTranslation(TVectorType&& translation)
RETURN_AUTO(
  HomogenizedTranslation<util::store_member_t<TVectorType&&>>(std::forward<TVectorType>(translation))
)

} // end of ns template_tensors
