namespace template_tensors {

/*!
 * \brief A compile-time dimension value indicating that the dimension will be determined at run-time.
 */
static const size_t DYN = static_cast<size_t>(-1);

template <size_t... TDims>
using DimSeq = tmp::vs::Sequence<size_t, TDims...>;
template <size_t... TCoords>
using CoordSeq = tmp::vs::Sequence<size_t, TCoords...>;

template <typename TArg>
struct is_dimseq_v
{
  template <size_t... TDims>
  TMP_IF(const DimSeq<TDims...>&)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(typename std::decay<TArg>::type);
};
template <typename TArg>
TVALUE(bool, is_coordseq_v, is_dimseq_v<TArg>::value)



template <typename TThisType, typename TDimSeq>
class HasDimensions;

template <typename TArg>
struct has_dimensions_v
{
  template <typename TThisType, typename TDimSeq>
  TMP_IF(const HasDimensions<TThisType, TDimSeq>&)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(typename std::decay<TArg>::type);
};

namespace detail {

template <typename TDimSeqOrTensor>
struct ToDimSeq
{
#ifdef __CUDACC__
  template <typename TElement, typename TPointer, typename TDerived>
  TMP_IF(thrust::reference<TElement, TPointer, TDerived>)
  TMP_RETURN_TYPE(typename ToDimSeq<TElement>::type)
#endif

  template <typename TThisType, typename TDimSeq>
  TMP_IF(const HasDimensions<TThisType, TDimSeq>&)
  TMP_RETURN_TYPE(TDimSeq)

  template <size_t... TDims>
  TMP_IF(const DimSeq<TDims...>&)
  TMP_RETURN_TYPE(DimSeq<TDims...>)

  TMP_DEDUCE_TYPE(typename std::decay<TDimSeqOrTensor>::type);
};

} // end of ns detail

template <typename TDimSeqOrTensor, ENABLE_IF(has_dimensions_v<TDimSeqOrTensor>::value || is_dimseq_v<TDimSeqOrTensor>::value)>
using dimseq_t = typename detail::ToDimSeq<typename std::decay<TDimSeqOrTensor>::type>::type;



namespace detail {

template <typename TSeq>
struct SeqHelper;

template <size_t... TSeq>
struct SeqHelper<DimSeq<TSeq...>>
{
  static const size_t SIZE = tmp::vs::length_v<DimSeq<TSeq...>>::value;

  template <size_t TFill>
  static constexpr size_t non_trivial_num()
  {
    return non_trivial_num<TFill>(::array::LocalArray<size_t, SIZE>(TSeq...), SIZE);
  }

  template <size_t TFill>
  static constexpr size_t non_trivial_num(const ::array::LocalArray<size_t, SIZE> dims, size_t i)
  {
    return i == 0 ? 0
         : (dims.data()[i - 1] == TFill ? non_trivial_num<TFill>(dims, i - 1)
            : i);
  }

  static constexpr bool are_static()
  {
    return are_static(::array::LocalArray<size_t, SIZE>(TSeq...), 0);
  }

  static constexpr bool are_static(const ::array::LocalArray<size_t, SIZE> dims, size_t i)
  {
    return i == SIZE ? true : dims.data()[i] != DYN && are_static(dims, i + 1);
  }

  template <size_t TFill>
  static constexpr size_t nth(size_t n)
  {
    return nth<TFill>(::array::LocalArray<size_t, SIZE>(TSeq...), n);
  }

  template <size_t TFill>
  static constexpr size_t nth(const ::array::LocalArray<size_t, SIZE> dims, size_t n)
  {
    return math::gte(n, SIZE) ? TFill : dims.data()[n];
  }
};

template <typename TDimSeq>
struct MultiplyDimensions;

template <size_t... TDims>
struct MultiplyDimensions<DimSeq<TDims...>>
{
  static const size_t value = SeqHelper<DimSeq<TDims...>>::are_static() ? math::multiply(TDims...) : DYN;
};

} // end of ns detail

template <typename TDimSeqOrCoordSeqOrTensor>
TVALUE(size_t, is_static_v, detail::SeqHelper<dimseq_t<TDimSeqOrCoordSeqOrTensor>>::are_static());
template <typename TDimSeqOrCoordSeqOrTensor>
TVALUE(size_t, non_trivial_dimensions_num_v, detail::SeqHelper<dimseq_t<TDimSeqOrCoordSeqOrTensor>>::template non_trivial_num<1>());
template <size_t N, typename TDimSeqOrTensor>
TVALUE(size_t, nth_dimension_v, detail::SeqHelper<dimseq_t<TDimSeqOrTensor>>::template nth<1>(N))
template <typename TDimSeqOrTensor>
TVALUE(size_t, rows_v, nth_dimension_v<0, TDimSeqOrTensor>::value)
template <typename TDimSeqOrTensor>
TVALUE(size_t, cols_v, nth_dimension_v<1, TDimSeqOrTensor>::value)

template <typename TCoordSeq>
TVALUE(size_t, non_trivial_coordinates_num_v, detail::SeqHelper<typename std::decay<TCoordSeq>::type>::template non_trivial_num<0>());
template <size_t N, typename TCoordSeq>
TVALUE(size_t, nth_coordinate_v, detail::SeqHelper<typename std::decay<TCoordSeq>::type>::template nth<0>(N))

template <typename TArg>
TVALUE(size_t, multiply_dimensions_v, detail::MultiplyDimensions<dimseq_t<TArg>>::value)

template <size_t TRank>
using dyn_dimseq_t = tmp::vs::repeat_t<size_t, DYN, TRank>;



static_assert(non_trivial_dimensions_num_v<DimSeq<1, 2, 1>>::value == 2, "non_trivial_dimensions_num not working");
static_assert(nth_dimension_v<0, DimSeq<4, 2, 6>>::value == 4, "nth_dimension not working");
static_assert(nth_dimension_v<1, DimSeq<4, 2, 6>>::value == 2, "nth_dimension not working");
static_assert(nth_dimension_v<2, DimSeq<4, 2, 6>>::value == 6, "nth_dimension not working");
static_assert(nth_dimension_v<3, DimSeq<4, 2, 6>>::value == 1, "nth_dimension not working");



namespace detail {

template <typename TCoordSeq, typename TIndexSeq>
struct CoordSeqMakeLength;

template <typename TCoordSeq, size_t... TIndices>
struct CoordSeqMakeLength<TCoordSeq, tmp::vs::IndexSequence<TIndices...>>
{
  using type = tmp::vs::Sequence<size_t, nth_coordinate_v<TIndices, TCoordSeq>::value...>;
  static_assert(non_trivial_coordinates_num_v<TCoordSeq>::value == non_trivial_coordinates_num_v<type>::value, "Cannot cut non-trivial coordinates");
};

template <typename TDimSeq, typename TIndexSeq>
struct DimSeqMakeLength;

template <typename TDimSeq, size_t... TIndices>
struct DimSeqMakeLength<TDimSeq, tmp::vs::IndexSequence<TIndices...>>
{
  using type = tmp::vs::Sequence<size_t, nth_dimension_v<TIndices, TDimSeq>::value...>;
  static_assert(non_trivial_dimensions_num_v<TDimSeq>::value == non_trivial_dimensions_num_v<type>::value, "Cannot cut non-trivial dimensions");
};

} // end of ns detail

template <typename TDimSeqOrTensor, size_t TLength>
using dimseq_make_length_t = typename detail::DimSeqMakeLength<dimseq_t<TDimSeqOrTensor>, tmp::vs::ascending_numbers_t<TLength>>::type;
template <typename TCoordSeq, size_t TLength>
using coordseq_make_length_t = typename detail::CoordSeqMakeLength<TCoordSeq, tmp::vs::ascending_numbers_t<TLength>>::type;

} // end of ns tensor
