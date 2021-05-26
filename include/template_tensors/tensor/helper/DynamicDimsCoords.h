namespace template_tensors {

template <typename TArg>
struct is_vector_v;

template <typename TArg>
struct is_dim_or_coord_vector_v
{
  template <typename TVectorType, ENABLE_IF(template_tensors::is_vector_v<TVectorType>::value)>
  TMP_IF(TVectorType&&)
  TMP_RETURN_VALUE(std::is_assignable<dim_t&, decltype(std::declval<TVectorType>()())>::value && rows_v<TVectorType>::value != template_tensors::DYN)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

template <typename... TArgs>
using are_dim_t_v = metal::all_of<metal::list<TArgs...>, metal::bind<
  metal::trait<std::is_convertible>,
  metal::arg<1>,
  metal::always<dim_t>
>>;

static_assert(are_dim_t_v<long unsigned int>::value, "are_dim_t_v not working");

template <typename... TArgs>
struct are_dim_or_coord_args_v
{
  template <typename TArg, ENABLE_IF(is_dimseq_v<TArg>::value || is_dim_or_coord_vector_v<TArg>::value)>
  TMP_IF(const TArg&)
  TMP_RETURN_VALUE(true)

  template <typename... TSizetArgs, ENABLE_IF(are_dim_t_v<TSizetArgs...>::value)>
  TMP_IF(TSizetArgs&&...)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(sizeof...(TArgs) == 0)

  TMP_DEDUCE_VALUE(TArgs...);
};
static_assert(are_dim_or_coord_args_v<>::value, "are_dim_or_coord_args_v not working");

template <typename... TArgs>
TVALUE(bool, are_dim_args_v, are_dim_or_coord_args_v<TArgs...>::value)
template <typename... TArgs>
TVALUE(bool, are_coord_args_v, are_dim_or_coord_args_v<TArgs...>::value)



namespace detail {

template <typename... TArgs>
struct get_coord_or_dim_num_v
{
  template <typename TSeq, ENABLE_IF(is_dimseq_v<TSeq>::value || is_coordseq_v<TSeq>::value)>
  TMP_IF(TSeq&&)
  TMP_RETURN_VALUE(metal::size<TSeq>::value)

  template <typename TVectorType, ENABLE_IF(is_dim_or_coord_vector_v<TVectorType>::value)>
  TMP_IF(TVectorType&&...)
  TMP_RETURN_VALUE(rows_v<TVectorType>::value)

  template <typename... TSizetArgs, ENABLE_IF(sizeof...(TSizetArgs) != 0 && are_dim_t_v<TSizetArgs...>::value)>
  TMP_IF(TSizetArgs&&...)
  TMP_RETURN_VALUE(sizeof...(TSizetArgs))

  template <typename TDummy = void>
  TMP_IF()
  TMP_RETURN_VALUE(0)

  TMP_DEDUCE_VALUE(TArgs...);
};

} // end of ns detail

template <typename... TCoordArgTypes>
TVALUE(metal::int_, coordinate_num_v, detail::get_coord_or_dim_num_v<TCoordArgTypes...>::value);
template <typename... TDimArgTypes>
TVALUE(metal::int_, dimension_num_v, detail::get_coord_or_dim_num_v<TDimArgTypes...>::value);



namespace detail {

template <bool TInRange, metal::int_ TRank, metal::int_ TDefault>
struct NthValue;

template <metal::int_ TRank, metal::int_ TDefault>
struct NthValue<true, TRank, TDefault>
{
  template <typename... TSizetArgs, ENABLE_IF(are_dim_t_v<TSizetArgs...>::value)>
  __host__ __device__
  static constexpr dim_t get(TSizetArgs&&... values)
  {
    return util::nth<TRank>(std::forward<TSizetArgs>(values)...);
  }

  template <typename TVectorType, ENABLE_IF(is_dim_or_coord_vector_v<TVectorType>::value)>
  __host__ __device__
  static constexpr dim_t get(TVectorType&& vector)
  {
    return vector(TRank);
  }

  template <typename TDimOrCoordSeqArg, typename TDimOrCoordSeq = typename std::decay<TDimOrCoordSeqArg>::type,
    ENABLE_IF(is_dimseq_v<TDimOrCoordSeq>::value || is_coordseq_v<TDimOrCoordSeq>::value)>
  __host__ __device__
  static constexpr dim_t get(TDimOrCoordSeqArg)
  {
    return metal::at<TDimOrCoordSeq, metal::number<TRank>>::value;
  }
};

template <metal::int_ TRank, metal::int_ TDefault>
struct NthValue<false, TRank, TDefault>
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static constexpr dim_t get(TCoordArgTypes&&... coords)
  {
    return TDefault;
  }
};

template <bool TAreDims, metal::int_ TRank, metal::int_ TDefault, typename... TDimOrCoordArgTypes>
__host__ __device__
constexpr dim_t getNthValue(TDimOrCoordArgTypes&&... dim_or_coord_args)
{
  return detail::NthValue<math::lt(TRank, TAreDims ?
        dimension_num_v<TDimOrCoordArgTypes&&...>::value
      : coordinate_num_v<TDimOrCoordArgTypes&&...>::value),
    TRank, TDefault>::get(std::forward<TDimOrCoordArgTypes>(dim_or_coord_args)...);
}

} // end of ns detail

template <metal::int_ TRank, typename... TDimArgTypes>
__host__ __device__
constexpr dim_t getNthDimension(TDimArgTypes&&... dim_args)
{
  return detail::getNthValue<true, TRank, 1>(std::forward<TDimArgTypes>(dim_args)...);
}

template <metal::int_ TRank, typename... TCoordArgTypes>
__host__ __device__
constexpr dim_t getNthCoordinate(TCoordArgTypes&&... coord_args)
{
  return detail::getNthValue<false, TRank, 0>(std::forward<TCoordArgTypes>(coord_args)...);
}



namespace detail {

template <metal::int_... TIndices, typename... TDimArgTypes>
__host__ __device__
constexpr dim_t multiplyDimensionsHelper(metal::numbers<TIndices...>, TDimArgTypes&&... dims)
{
  static_assert(sizeof...(TIndices) >= dimension_num_v<TDimArgTypes&&...>::value, "Invalid number of dimensions");
  return math::multiply(template_tensors::getNthDimension<TIndices>(std::forward<TDimArgTypes>(dims)...)...);
}

} // end of ns detail

template <typename... TDimArgTypes>
__host__ __device__
constexpr dim_t multiplyDimensions(TDimArgTypes&&... dim_args)
{
  return detail::multiplyDimensionsHelper(metal::iota<metal::number<0>, metal::number<dimension_num_v<TDimArgTypes&&...>::value>>(),
    std::forward<TDimArgTypes>(dim_args)...);
}



struct ColMajor;
/*!
 * \brief The default index strategy used for storage tensors when no other is specified.
 */
using DefaultIndexStrategy = ColMajor;

template <typename TThisType, typename TElementType, typename TIndexStrategy, mem::MemoryType TMemoryType, typename TDimSeq>
class IndexedPointerTensor;
template <typename TArrayType, typename TElementType, typename TIndexStrategy, typename TDimSeq>
class IndexedArrayTensor;
template <metal::int_ TRank>
using VectorXs = IndexedArrayTensor<::array::LocalArray<size_t, TRank>, size_t, DefaultIndexStrategy, template_tensors::DimSeq<TRank>>;
// TODO: can storage/Typedefs.h be moved forward in inclusion stack to remove this?
// TODO: VectorXs should have dim_t?

namespace detail {

template <metal::int_... TIndices, typename... TDimArgTypes>
__host__ __device__
VectorXs<sizeof...(TIndices)> toDimVectorHelper(metal::numbers<TIndices...>, TDimArgTypes&&... dims)
{
  return VectorXs<sizeof...(TIndices)>(template_tensors::getNthDimension<TIndices>(std::forward<TDimArgTypes>(dims)...)...);
}
// TODO: assert rest is 0 or 1 if more dims/ coords are given than TIndices
template <metal::int_... TIndices, typename... TCoordArgTypes>
__host__ __device__
VectorXs<sizeof...(TIndices)> toCoordVectorHelper(metal::numbers<TIndices...>, TCoordArgTypes&&... coords)
{
  return VectorXs<sizeof...(TIndices)>(getNthCoordinate<TIndices>(std::forward<TCoordArgTypes>(coords)...)...);
}

} // end of ns detail

template <metal::int_ TDims = DYN, typename... TDimArgTypes>
__host__ __device__
auto toDimVector(TDimArgTypes&&... dims)
RETURN_AUTO(
  detail::toDimVectorHelper(metal::iota<metal::number<0>, metal::number<TDims == DYN ? dimension_num_v<TDimArgTypes&&...>::value : TDims>>(),
    std::forward<TDimArgTypes>(dims)...)
)

template <metal::int_ TDims = DYN, typename... TCoordArgTypes>
__host__ __device__
auto toCoordVector(TCoordArgTypes&&... coords)
RETURN_AUTO(
  detail::toCoordVectorHelper(metal::iota<metal::number<0>, metal::number<TDims == DYN ? coordinate_num_v<TCoordArgTypes&&...>::value : TDims>>(),
    std::forward<TCoordArgTypes>(coords)...)
)

template <typename... TDimArgTypes>
__host__ __device__
constexpr dim_t getNthDimension(size_t n, TDimArgTypes&&... dims)
{
  return n < dimension_num_v<TDimArgTypes&&...>::value ? toDimVector(std::forward<TDimArgTypes>(dims)...)(n) : 1;
}

template <typename... TCoordArgTypes>
__host__ __device__
constexpr dim_t getNthCoordinate(size_t n, TCoordArgTypes&&... coords)
{
  return n < coordinate_num_v<TCoordArgTypes&&...>::value ? toDimVector(std::forward<TCoordArgTypes>(coords)...)(n) : 0;
}




namespace detail {

template <metal::int_ I>
struct GetNonTrivialDimensionsNumHelper
{
  template <typename... TDimArgTypes>
  __host__ __device__
  static constexpr size_t getNonTrivialDimensionsNum(TDimArgTypes&&... dims)
  {
    return getNthDimension<I - 1>(std::forward<TDimArgTypes>(dims)...) == 0 ? I
      : GetNonTrivialDimensionsNumHelper<I - 1>::getNonTrivialDimensionsNum(std::forward<TDimArgTypes>(dims)...);
  }
};

template <>
struct GetNonTrivialDimensionsNumHelper<0>
{
  template <typename... TDimArgTypes>
  __host__ __device__
  static constexpr size_t getNonTrivialDimensionsNum(TDimArgTypes&&... dims)
  {
    return 0;
  }
};

} // end of ns detail

template <typename... TDimArgTypes>
__host__ __device__
constexpr size_t getNonTrivialDimensionsNum(TDimArgTypes&&... dims)
{
  return detail::GetNonTrivialDimensionsNumHelper<dimension_num_v<TDimArgTypes&&...>::value>::getNonTrivialDimensionsNum(std::forward<TDimArgTypes>(dims)...);
}

} // end of ns template_tensors
