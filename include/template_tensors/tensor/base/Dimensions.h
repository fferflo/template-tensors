namespace template_tensors {

namespace detail {

template <typename TDimSeq>
struct multiply_all_but_first;

template <metal::int_ TFirst, metal::int_... TDims>
struct multiply_all_but_first<template_tensors::DimSeq<TFirst, TDims...>>
{
  static const metal::int_ value = math::multiply(TDims..., 1, 1);
};

template <>
struct multiply_all_but_first<template_tensors::DimSeq<>>
{
  static const metal::int_ value = 1;
};

template <typename TThisType, typename TDimSeq, bool TIsStatic = template_tensors::is_static_v<TDimSeq>::value>
struct UseStaticDimensionsIfPossible;

template <typename TThisType, typename TDimSeq>
struct UseStaticDimensionsIfPossible<TThisType, TDimSeq, true>
{
  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = template_tensors::non_trivial_dimensions_num_v<TDimSeq>::value;

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t dim() const
  {
    return nth_dimension_v<TIndex, TDimSeq>::value;
  }

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t dim() const volatile
  {
    return nth_dimension_v<TIndex, TDimSeq>::value;
  }

  __host__ __device__
  dim_t dim(size_t index) const
  {
    return getNthDimension(index, TDimSeq());
  }

  __host__ __device__
  dim_t dim(size_t index) const volatile
  {
    return getNthDimension(index, TDimSeq());
  }
};

template <typename TThisType, typename TDimSeq>
struct UseStaticDimensionsIfPossible<TThisType, TDimSeq, false>
{
  HD_WARNING_DISABLE
  template <metal::int_ TIndex>
  __host__ __device__
  dim_t dim() const
  {
    return static_cast<const TThisType*>(this)->template getDynDim<TIndex>();
  }

  HD_WARNING_DISABLE
  template <metal::int_ TIndex>
  __host__ __device__
  dim_t dim() const volatile
  {
    return static_cast<const volatile TThisType*>(this)->template getDynDim<TIndex>();
  }

  HD_WARNING_DISABLE
  __host__ __device__
  dim_t dim(size_t index) const
  {
    return static_cast<const TThisType*>(this)->getDynDim(index);
  }

  HD_WARNING_DISABLE
  __host__ __device__
  dim_t dim(size_t index) const volatile
  {
    return static_cast<const volatile TThisType*>(this)->getDynDim(index);
  }
};

template <typename TTensor, typename TDimSeq, metal::int_ TRank>
class DimensionVector;

} // end of ns detail

template <typename TThisType, typename TDimSeq>
struct HasDimensions : public detail::UseStaticDimensionsIfPossible<TThisType, TDimSeq>
{
  static_assert(detail::multiply_all_but_first<TDimSeq>::value != 0, "Only first dimension can be zero");

  static const metal::int_ NON_TRIVIAL_DIMENSIONS_NUM = template_tensors::non_trivial_dimensions_num_v<TDimSeq>::value;

  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  explicit HasDimensions(TDimArgTypes&&... dim_args)
  {
    ASSERT(areCompatibleDimensions<TDimSeq>(std::forward<TDimArgTypes>(dim_args)...), "Dynamic dimensions do not match static dimensions");
  }

  __host__ __device__
  dim_t rows() const
  {
    return this->template dim<0>();
  }

  __host__ __device__
  dim_t rows() const volatile
  {
    return this->template dim<0>();
  }

  __host__ __device__
  dim_t cols() const
  {
    return this->template dim<1>();
  }

  __host__ __device__
  dim_t cols() const volatile
  {
    return this->template dim<1>();
  }

  template <metal::int_ TRank = NON_TRIVIAL_DIMENSIONS_NUM>
  __host__ __device__
  auto dims()
  RETURN_AUTO(detail::DimensionVector<HasDimensions<TThisType, TDimSeq>&, TDimSeq, TRank>(*this))

  template <metal::int_ TRank = NON_TRIVIAL_DIMENSIONS_NUM>
  __host__ __device__
  auto dims() const
  RETURN_AUTO(detail::DimensionVector<const HasDimensions<TThisType, TDimSeq>&, TDimSeq, TRank>(*this))

  template <metal::int_ TRank = NON_TRIVIAL_DIMENSIONS_NUM>
  __host__ __device__
  auto dims() volatile
  RETURN_AUTO(detail::DimensionVector<volatile HasDimensions<TThisType, TDimSeq>&, TDimSeq, TRank>(*this))

  template <metal::int_ TRank = NON_TRIVIAL_DIMENSIONS_NUM>
  __host__ __device__
  auto dims() const volatile
  RETURN_AUTO(detail::DimensionVector<const volatile HasDimensions<TThisType, TDimSeq>&, TDimSeq, TRank>(*this))
};

namespace detail {

#define ThisType DimensionVector<TTensor, TDimSeq, TRank>
#define SuperType TensorBase<ThisType, mem::LOCAL, template_tensors::DimSeq<TRank>>
// TODO: should this only store refs? or param_forward? also map(transform) function
template <typename TTensor, typename TDimSeq, metal::int_ TRank>
class DimensionVector : public SuperType
{
public:
  static_assert(template_tensors::non_trivial_dimensions_num_v<TDimSeq>::value <= TRank, "Rank is too small");

  __host__ __device__
  DimensionVector(TTensor tensor)
    : SuperType(TRank)
    , m_tensor(tensor)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto getElement(TThisType&& self, dim_t row)
  RETURN_AUTO(
    self.m_tensor.dim(row)
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 1)

  template <typename TTensorType>
  __host__
  static auto toKernel(TTensorType&& tensor)
  RETURN_AUTO(VectorXs<TRank>(std::forward<TTensorType>(tensor)))

private:
  TTensor m_tensor;
};
#undef SuperType
#undef ThisType

} // end of ns detail

} // end of ns template_tensors
