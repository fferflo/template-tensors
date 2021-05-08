namespace template_tensors {

namespace detail {

template <metal::int_ TRows>
struct HomogenizedRowsHelper
{
  static const metal::int_ value = TRows + 1;
};

template <>
struct HomogenizedRowsHelper<DYN>
{
  static const metal::int_ value = DYN;
};

template <metal::int_ TRows>
struct DehomogenizedRowsHelper
{
  static const metal::int_ value = TRows - 1;
};
// TODO: constexpr application of lambda? or with all args plus result and checking first n-1 values for DYN?, grep DYN
template <>
struct DehomogenizedRowsHelper<DYN>
{
  static const metal::int_ value = DYN;
};

} // end of ns detail

#define ThisType HomogenizedVector<TVectorType>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TVectorType>::value, \
                                        template_tensors::DimSeq<detail::HomogenizedRowsHelper<rows_v<TVectorType>::value>::value> \
                              >

template <typename TVectorType>
class HomogenizedVector : public SuperType
{
public:
  static_assert(is_vector_v<TVectorType>::value, "TVectorType must be a vector");

  __host__ __device__
  HomogenizedVector(TVectorType input)
    : SuperType(input.template dim<0>() + 1)
    , m_input(input)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static decay_elementtype_t<TVectorType> getElement(TThisType&& self, dim_t row)
  {
    if (row < self.template dim<0>() - 1)
    {
      return self.m_input(row);
    }
    else
    {
      return static_cast<decay_elementtype_t<TVectorType>>(1);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 1)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return TIndex == 0 ? m_input.template dim<0>() + 1 : 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index == 0 ? m_input.template dim<0>() + 1 : 1;
  }

private:
  TVectorType m_input;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(HomogenizedVector<decltype(transform(m_input))>
    (transform(m_input))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(HomogenizedVector<decltype(transform(m_input))>
    (transform(m_input))
  )
};
#undef SuperType
#undef ThisType



#define ThisType DehomogenizedVector<TVectorType>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TVectorType>::value, \
                                        template_tensors::DimSeq<detail::DehomogenizedRowsHelper<rows_v<TVectorType>::value>::value> \
                              >

template <typename TVectorType>
class DehomogenizedVector : public SuperType
{
public:
  static_assert(metal::front<dimseq_t<TVectorType>>::value >= 1, "Cannot dehomogenize a vector with 0 rows!");
  static_assert(is_vector_v<TVectorType>::value, "TVectorType must be a vector");

  __host__ __device__
  DehomogenizedVector(TVectorType input)
    : SuperType(input.template dim<0>() - 1)
    , m_input(input)
  {
    ASSERT(input.template dim<0>() >= 1, "Cannot dehomogenize a vector with 0 rows!");
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, dim_t row)
  RETURN_AUTO(
    self.m_input(row) / self.m_input(self.template dim<0>())
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 1)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return TIndex == 0 ? m_input.template dim<0>() - 1 : 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index == 0 ? m_input.template dim<0>() - 1 : 1;
  }

private:
  TVectorType m_input;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(DehomogenizedVector<decltype(transform(m_input))>
    (transform(m_input))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(DehomogenizedVector<decltype(transform(m_input))>
    (transform(m_input))
  )
};
#undef SuperType
#undef ThisType

template <typename TVectorType>
__host__ __device__
auto homogenize(TVectorType&& vector)
RETURN_AUTO(
  HomogenizedVector<util::store_member_t<TVectorType&&>>(util::forward<TVectorType>(vector))
)

template <typename TVectorType>
__host__ __device__
auto dehomogenize(TVectorType&& vector)
RETURN_AUTO(
  DehomogenizedVector<util::store_member_t<TVectorType&&>>(util::forward<TVectorType>(vector))
)

} // end of ns template_tensors
