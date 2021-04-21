namespace template_tensors {

namespace detail {

template <size_t... TDims>
struct CombineDims;

template <size_t TDim>
struct CombineDims<TDim>
{
  static const size_t value = TDim;
};

template <size_t TDim0, size_t TDim1, size_t... TRestDims>
struct CombineDims<TDim0, TDim1, TRestDims...>
{
  static_assert(TDim0 == DYN || TDim1 == DYN || TDim0 == TDim1, "Incompatible dimensions");
  static const size_t value = TDim0 != DYN ? CombineDims<TDim0, TRestDims...>::value : CombineDims<TDim1, TRestDims...>::value;
};

template <typename... TDimSeqs>
struct CombineDimSeqs
{
  static const size_t MAX_RANK = math::max(non_trivial_dimensions_num_v<TDimSeqs>::value...);

  template <size_t TRank>
  static constexpr size_t combine_at()
  {
    return CombineDims<nth_dimension_v<TRank, TDimSeqs>::value...>::value;
  }

  template <size_t... TIndices>
  static auto deduce(tmp::vs::IndexSequence<TIndices...>)
  RETURN_AUTO(DimSeq<combine_at<TIndices>()...>())

  using type = decltype(deduce(tmp::vs::ascending_numbers_t<MAX_RANK>()));
};

template <size_t... TDims>
struct AreCompatibleDims;

template <size_t TDim>
struct AreCompatibleDims<TDim>
{
  static const bool value = true;
};

template <size_t TDim0, size_t TDim1, size_t... TRestDims>
struct AreCompatibleDims<TDim0, TDim1, TRestDims...>
{
  static const bool value = (TDim0 == DYN || TDim1 == DYN || TDim0 == TDim1)
    && AreCompatibleDims<TDim0 == DYN ? TDim1 : TDim0, TRestDims...>::value;
};

template <typename... TDimSeqs>
struct AreCompatibleDimSeqs
{
  static const size_t MAX_RANK = math::max(non_trivial_dimensions_num_v<TDimSeqs>::value...);

  template <size_t TRank>
  static constexpr bool compatible_at()
  {
    return AreCompatibleDims<nth_dimension_v<TRank, TDimSeqs>::value...>::value;
  }

  template <size_t... TIndices>
  static constexpr bool get(tmp::vs::IndexSequence<TIndices...>)
  {
    return math::landsc(compatible_at<TIndices>()...);
  }

  static const bool value = get(tmp::vs::ascending_numbers_t<MAX_RANK>());
};

} // end of ns detail





template <size_t... TDims>
TVALUE(size_t, combine_dims_v, detail::CombineDims<TDims...>::value);
template <typename... TDimSeqOrTensors>
using combine_dimseqs_t = typename detail::CombineDimSeqs<dimseq_t<TDimSeqOrTensors>...>::type;

template <size_t... TDims>
TVALUE(size_t, are_compatible_dims_v, detail::AreCompatibleDims<TDims...>::value);
template <typename... TDimSeqOrTensors>
TVALUE(size_t, are_compatible_dimseqs_v, detail::AreCompatibleDimSeqs<TDimSeqOrTensors...>::value);





static_assert(are_compatible_dimseqs_v<DimSeq<1, 2>, DimSeq<1, 2>>::value, "are_compatible_dimseqs_v not working");
static_assert(are_compatible_dimseqs_v<DimSeq<1, 2, 1>, DimSeq<1, 2>>::value, "are_compatible_dimseqs_v not working");
static_assert(!are_compatible_dimseqs_v<DimSeq<1, 3>, DimSeq<1>>::value, "are_compatible_dimseqs_v not working");
static_assert(are_compatible_dimseqs_v<DimSeq<>, DimSeq<1, 1, 1>>::value, "are_compatible_dimseqs_v not working");
static_assert(!are_compatible_dimseqs_v<DimSeq<0>, DimSeq<>>::value, "are_compatible_dimseqs_v not working");

} // end of ns tensor
