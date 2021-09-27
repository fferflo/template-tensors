namespace template_tensors {

namespace detail {

template <bool TWrap>
struct ApplyWrapperHelper
{
  template <typename TOperation, metal::int_... TIndices, typename... TCoordArgTypes>
  __host__ __device__
  static auto apply(TOperation&& op, metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(std::forward<TOperation>(op)(getNthCoordinate<TIndices>(std::forward<TCoordArgTypes>(coords)...)...))
};

template <>
struct ApplyWrapperHelper<true>
{
  template <typename TOperation, metal::int_... TIndices, typename... TCoordArgTypes>
  __host__ __device__
  static auto apply(TOperation&& op, metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(std::forward<TOperation>(op)(VectorXs<sizeof...(TIndices)>(getNthCoordinate<TIndices>(std::forward<TCoordArgTypes>(coords)...)...)))
};

template <metal::int_ TSupplierDims, typename TOperation, typename... TCoordArgTypes>
__host__ __device__
auto apply_supplier(TOperation&& op, TCoordArgTypes&&... coords)
RETURN_AUTO(ApplyWrapperHelper<tmp::takes_arguments_v<TOperation, VectorXs<TSupplierDims>>::value>::apply(
  std::forward<TOperation>(op),
  metal::iota<metal::number<0>, metal::number<TSupplierDims>>(),
  std::forward<TCoordArgTypes>(coords)...
))

} // end of ns detail

#define ThisType ElementSupplierTensor<TOperation, TSupplierDims, TDimSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        TDimSeq \
                              >
template <typename TOperation, metal::int_ TSupplierDims, typename TDimSeq>
class ElementSupplierTensor : public SuperType,
                              public StoreDimensions<TDimSeq>
{
public:
  static_assert(non_trivial_dimensions_num_v<SuperType>::value <= TSupplierDims, "Too few supplier dimensions");

  template <typename... TDimArgTypes>
  __host__ __device__
  ElementSupplierTensor(TOperation supplier, TDimArgTypes&&... dim_args)
    : SuperType(std::forward<TDimArgTypes>(dim_args)...)
    , StoreDimensions<TDimSeq>(std::forward<TDimArgTypes>(dim_args)...)
    , m_supplier(supplier)
  {
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

  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

private:
  TOperation m_supplier;

public:
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    detail::apply_supplier<TSupplierDims>(self.m_supplier, std::forward<TCoordArgTypes>(coords)...)
  )
  TT_TENSOR_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)
};
#undef SuperType
#undef ThisType


/*!
 * \brief Returns a tensor whose elements are given by a supplier
 *
 * \ingroup SpecialTensorConstants
 *
 * The supplier is a functor that takes as input n indices and returns the tensor's value at that location, where n
 * is equal to the number of dimensions given to this function, regardless of whether they are trivial (i.e. equal to one) or not.
 *
 * @param supplier the element supplier
 * @param dim_args... the run-time dimensions of the resulting tensor
 * @tparam TDims... the compile-time dimensions of the resulting tensor
 * @return the supplied tensor
 */
template <metal::int_... TDims, typename TOperation>
__host__ __device__
auto fromSupplier(TOperation&& supplier)
RETURN_AUTO(ElementSupplierTensor<TOperation, sizeof...(TDims), DimSeq<TDims...>>(std::forward<TOperation>(supplier)));

template <typename TOperation, typename... TDimArgTypes, ENABLE_IF(sizeof...(TDimArgTypes) != 0)>
__host__ __device__
auto fromSupplier(TOperation&& supplier, TDimArgTypes&&... dim_args)
RETURN_AUTO(ElementSupplierTensor<TOperation, dimension_num_v<TDimArgTypes...>::value, dyn_dimseq_t<dimension_num_v<TDimArgTypes...>::value>>(std::forward<TOperation>(supplier), std::forward<TDimArgTypes>(dim_args)...));

template <typename TDimSeq, metal::int_ TSupplierDims = non_trivial_dimensions_num_v<TDimSeq>::value, typename TOperation>
__host__ __device__
auto fromSupplier(TOperation&& supplier)
RETURN_AUTO(ElementSupplierTensor<TOperation, TSupplierDims, TDimSeq>(std::forward<TOperation>(supplier)));

template <metal::int_... TDims>
__host__ __device__
auto coordinates()
RETURN_AUTO(fromSupplier<TDims...>(math::functor::id()))

template <typename... TDimArgTypes, ENABLE_IF(sizeof...(TDimArgTypes) != 0)>
__host__ __device__
auto coordinates(TDimArgTypes&&... dim_args)
RETURN_AUTO(fromSupplier(math::functor::id(), std::forward<TDimArgTypes>(dim_args)...))

template <typename TDimSeq, metal::int_ TSupplierDims = non_trivial_dimensions_num_v<TDimSeq>::value>
__host__ __device__
auto coordinates()
RETURN_AUTO(fromSupplier<TDimSeq, TSupplierDims>(math::functor::id()))

} // end of ns template_tensors
