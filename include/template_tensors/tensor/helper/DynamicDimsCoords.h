namespace template_tensors {

template <typename TArg>
struct is_vector_v;

template <typename TArg>
struct is_dim_or_coord_vector_v
{
  template <typename TVectorType, ENABLE_IF(template_tensors::is_vector_v<TVectorType>::value)>
  TMP_IF(TVectorType&&)
  TMP_RETURN_VALUE(std::is_assignable<size_t&, decltype(std::declval<TVectorType>()())>::value && rows_v<TVectorType>::value != template_tensors::DYN)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

template <typename... TArgs>
TVALUE(bool, are_size_t_v, tmp::ts::all_apply_v<tmp::ts::pred::is_convertible_to<size_t>::pred, tmp::ts::Sequence<TArgs...>>::value)

static_assert(are_size_t_v<long unsigned int>::value, "are_size_t_v not working");

template <typename... TArgs>
struct are_dim_or_coord_args_v
{
  template <typename TArg, ENABLE_IF(is_dimseq_v<TArg>::value || is_dim_or_coord_vector_v<TArg>::value)>
  TMP_IF(const TArg&)
  TMP_RETURN_VALUE(true)

  template <typename... TSizetArgs, ENABLE_IF(are_size_t_v<TSizetArgs...>::value)>
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
  TMP_RETURN_VALUE(tmp::vs::length_v<TSeq>::value)

  template <typename TVectorType, ENABLE_IF(is_dim_or_coord_vector_v<TVectorType>::value)>
  TMP_IF(TVectorType&&...)
  TMP_RETURN_VALUE(rows_v<TVectorType>::value)

  template <typename... TSizetArgs, ENABLE_IF(sizeof...(TSizetArgs) != 0 && are_size_t_v<TSizetArgs...>::value)>
  TMP_IF(TSizetArgs&&...)
  TMP_RETURN_VALUE(sizeof...(TSizetArgs))

  template <typename TDummy = void>
  TMP_IF()
  TMP_RETURN_VALUE(0)

  TMP_DEDUCE_VALUE(TArgs...);
};

} // end of ns detail

template <typename... TCoordArgTypes>
TVALUE(size_t, coordinate_num_v, detail::get_coord_or_dim_num_v<TCoordArgTypes...>::value);
template <typename... TDimArgTypes>
TVALUE(size_t, dimension_num_v, detail::get_coord_or_dim_num_v<TDimArgTypes...>::value);



namespace detail {

template <bool TInRange, size_t TRank, size_t TDefault>
struct NthValue;

template <size_t TRank, size_t TDefault>
struct NthValue<true, TRank, TDefault>
{
  template <typename... TSizetArgs, ENABLE_IF(are_size_t_v<TSizetArgs...>::value)>
  __host__ __device__
  static constexpr size_t get(TSizetArgs&&... values)
  {
    return util::nth<TRank>(util::forward<TSizetArgs>(values)...);
  }

  template <typename TVectorType, ENABLE_IF(is_dim_or_coord_vector_v<TVectorType>::value)>
  __host__ __device__
  static constexpr size_t get(TVectorType&& vector)
  {
    return vector(TRank);
  }

  template <typename TDimOrCoordSeqArg, typename TDimOrCoordSeq = typename std::decay<TDimOrCoordSeqArg>::type,
    ENABLE_IF(is_dimseq_v<TDimOrCoordSeq>::value || is_coordseq_v<TDimOrCoordSeq>::value)>
  __host__ __device__
  static constexpr size_t get(TDimOrCoordSeqArg)
  {
    return tmp::vs::get_v<TRank, TDimOrCoordSeq>::value;
  }
};

template <size_t TRank, size_t TDefault>
struct NthValue<false, TRank, TDefault>
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static constexpr size_t get(TCoordArgTypes&&... coords)
  {
    return TDefault;
  }
};

template <bool TAreDims, size_t TRank, size_t TDefault, typename... TDimOrCoordArgTypes>
__host__ __device__
constexpr size_t getNthValue(TDimOrCoordArgTypes&&... dim_or_coord_args)
{
  return detail::NthValue<math::lt(TRank, TAreDims ?
        dimension_num_v<TDimOrCoordArgTypes&&...>::value
      : coordinate_num_v<TDimOrCoordArgTypes&&...>::value),
    TRank, TDefault>::get(util::forward<TDimOrCoordArgTypes>(dim_or_coord_args)...);
}

} // end of ns detail

template <size_t TRank, typename... TDimArgTypes>
__host__ __device__
constexpr size_t getNthDimension(TDimArgTypes&&... dim_args)
{
  return detail::getNthValue<true, TRank, 1>(util::forward<TDimArgTypes>(dim_args)...);
}

template <size_t TRank, typename... TCoordArgTypes>
__host__ __device__
constexpr size_t getNthCoordinate(TCoordArgTypes&&... coord_args)
{
  return detail::getNthValue<false, TRank, 0>(util::forward<TCoordArgTypes>(coord_args)...);
}



namespace detail {

template <size_t... TIndices, typename... TDimArgTypes>
__host__ __device__
constexpr size_t multiplyDimensionsHelper(tmp::vs::IndexSequence<TIndices...>, TDimArgTypes&&... dims)
{
  static_assert(sizeof...(TIndices) >= dimension_num_v<TDimArgTypes&&...>::value, "Invalid number of dimensions");
  return math::multiply(template_tensors::getNthDimension<TIndices>(util::forward<TDimArgTypes>(dims)...)...);
}

} // end of ns detail

template <typename... TDimArgTypes>
__host__ __device__
constexpr size_t multiplyDimensions(TDimArgTypes&&... dim_args)
{
  return detail::multiplyDimensionsHelper(tmp::vs::ascending_numbers_t<dimension_num_v<TDimArgTypes&&...>::value>(),
    util::forward<TDimArgTypes>(dim_args)...);
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
template <size_t TRank>
using VectorXs = IndexedArrayTensor<::array::LocalArray<size_t, TRank>, size_t, DefaultIndexStrategy, template_tensors::DimSeq<TRank>>;
// TODO: can storage/Typedefs.h be moved forward in inclusion stack to remove this?

namespace detail {

template <size_t... TIndices, typename... TDimArgTypes>
__host__ __device__
VectorXs<sizeof...(TIndices)> toDimVectorHelper(tmp::vs::IndexSequence<TIndices...>, TDimArgTypes&&... dims)
{
  return VectorXs<sizeof...(TIndices)>(template_tensors::getNthDimension<TIndices>(util::forward<TDimArgTypes>(dims)...)...);
}
// TODO: assert rest is 0 or 1 if more dims/ coords are given than TIndices
template <size_t... TIndices, typename... TCoordArgTypes>
__host__ __device__
VectorXs<sizeof...(TIndices)> toCoordVectorHelper(tmp::vs::IndexSequence<TIndices...>, TCoordArgTypes&&... coords)
{
  return VectorXs<sizeof...(TIndices)>(getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...)...);
}

} // end of ns detail

template <size_t TDims = DYN, typename... TDimArgTypes>
__host__ __device__
auto toDimVector(TDimArgTypes&&... dims)
RETURN_AUTO(
  detail::toDimVectorHelper(tmp::vs::ascending_numbers_t<TDims == DYN ? dimension_num_v<TDimArgTypes&&...>::value : TDims>(),
    util::forward<TDimArgTypes>(dims)...)
)

template <size_t TDims = DYN, typename... TCoordArgTypes>
__host__ __device__
auto toCoordVector(TCoordArgTypes&&... coords)
RETURN_AUTO(
  detail::toCoordVectorHelper(tmp::vs::ascending_numbers_t<TDims == DYN ? coordinate_num_v<TCoordArgTypes&&...>::value : TDims>(),
    util::forward<TCoordArgTypes>(coords)...)
)





namespace detail {

template <size_t I>
struct GetNonTrivialDimensionsNumHelper
{
  template <typename... TDimArgTypes>
  __host__ __device__
  static constexpr size_t getNonTrivialDimensionsNum(TDimArgTypes&&... dims)
  {
    return getNthDimension<I - 1>(util::forward<TDimArgTypes>(dims)...) == 0 ? I
      : GetNonTrivialDimensionsNumHelper<I - 1>::getNonTrivialDimensionsNum(util::forward<TDimArgTypes>(dims)...);
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
  return detail::GetNonTrivialDimensionsNumHelper<dimension_num_v<TDimArgTypes&&...>::value>::getNonTrivialDimensionsNum(util::forward<TDimArgTypes>(dims)...);
}

} // end of ns tensor
