namespace template_tensors {

namespace detail {

template <metal::int_ TTensorDim, metal::int_ TKernelDim>
struct StaticConvDim
{
  static const metal::int_ value = (TTensorDim == DYN || TKernelDim == DYN) ? DYN : (TTensorDim - TKernelDim + 1);
};

template <typename TInputType, typename TKernelType, typename TIndexSequence>
struct StaticConvDims;

template <typename TInputType, typename TKernelType, metal::int_... TIndices>
struct StaticConvDims<TInputType, TKernelType, metal::numbers<TIndices...>>
{
  using type = DimSeq<StaticConvDim<
    nth_dimension_v<TIndices, TInputType>::value,
    nth_dimension_v<TIndices, TKernelType>::value
  >::value...>;
};

template <typename TIndexSequence>
struct DynamicConvDims;

template <metal::int_... TIndices>
struct DynamicConvDims<metal::numbers<TIndices...>>
{
  template <typename TInputType, typename TKernelType>
  __host__ __device__
  static VectorXs<sizeof...(TIndices)> get(TInputType&& input, TKernelType&& kernel)
  {
    return VectorXs<sizeof...(TIndices)>((input.template dim<TIndices>() - kernel.template dim<TIndices>() + 1)...);
  }
};

} // end of ns detail

#define ThisType ConvolutionTensor<TInputType, TKernelType>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::combine<mem::memorytype_v<TInputType>::value, mem::memorytype_v<TKernelType>::value>(), \
                                        typename detail::StaticConvDims<TInputType, TKernelType, metal::iota< \
                                          metal::number<0>, metal::number<math::min(non_trivial_dimensions_num_v<TInputType>::value, non_trivial_dimensions_num_v<TKernelType>::value)> \
                                        >>::type \
                              >

template <typename TInputType, typename TKernelType>
class ConvolutionTensor : public SuperType
{
public:
  __host__ __device__
  ConvolutionTensor(TInputType input, TKernelType kernel)
    : SuperType(detail::DynamicConvDims<metal::iota<metal::number<0>, metal::number<non_trivial_dimensions_num_v<SuperType>::value>>>::get(input, kernel))
    , m_input(input)
    , m_kernel(kernel)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    dot(
      self.m_kernel,
      headEx<dimseq_t<TKernelType>>(
        offset(self.m_input, util::forward<TCoordArgTypes>(coords)...),
        self.m_kernel.dims()
      )
    )
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return m_input.template dim<TIndex>() - m_kernel.template dim<TIndex>() + 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return m_input.dim(index) - m_kernel.dim(index) + 1;
  }

private:
  TInputType m_input;
  TKernelType m_kernel;

public:
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(ConvolutionTensor<decltype(transform(m_input)), decltype(transform(m_kernel))>
    (transform(m_input), transform(m_kernel))
  )

  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(ConvolutionTensor<decltype(transform(m_input)), decltype(transform(m_kernel))>
    (transform(m_input), transform(m_kernel))
  )
};
#undef SuperType
#undef ThisType

template <typename TInputType, typename TKernelType>
__host__ __device__
auto conv(TInputType&& input, TKernelType&& kernel)
RETURN_AUTO(ConvolutionTensor<util::store_member_t<TInputType&&>, util::store_member_t<TKernelType&&>>
  (util::forward<TInputType>(input), util::forward<TKernelType>(kernel))
);

} // end of ns tensor
