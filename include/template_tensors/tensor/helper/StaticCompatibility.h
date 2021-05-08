namespace template_tensors {

namespace detail {

template <metal::int_... TDims>
struct CombineDims;

template <metal::int_ TDim>
struct CombineDims<TDim>
{
  static const metal::int_ value = TDim;
};

template <metal::int_ TDim0, metal::int_ TDim1, metal::int_... TRestDims>
struct CombineDims<TDim0, TDim1, TRestDims...>
{
  static_assert(TDim0 == DYN || TDim1 == DYN || TDim0 == TDim1, "Incompatible dimensions");
  static const metal::int_ value = TDim0 != DYN ? CombineDims<TDim0, TRestDims...>::value : CombineDims<TDim1, TRestDims...>::value;
};

template <typename... TDimSeqs>
struct CombineDimSeqs
{
  static const metal::int_ MAX_RANK = math::max(non_trivial_dimensions_num_v<TDimSeqs>::value...);

  template <metal::int_ TRank>
  static constexpr metal::int_ combine_at()
  {
    return CombineDims<nth_dimension_v<TRank, TDimSeqs>::value...>::value;
  }

  template <metal::int_... TIndices>
  static auto deduce(metal::numbers<TIndices...>)
  RETURN_AUTO(DimSeq<combine_at<TIndices>()...>())

  using type = decltype(deduce(metal::iota<metal::number<0>, metal::number<MAX_RANK>>()));
};

template <metal::int_... TDims>
struct AreCompatibleDims;

template <metal::int_ TDim>
struct AreCompatibleDims<TDim>
{
  static const bool value = true;
};

template <metal::int_ TDim0, metal::int_ TDim1, metal::int_... TRestDims>
struct AreCompatibleDims<TDim0, TDim1, TRestDims...>
{
  static const bool value = (TDim0 == DYN || TDim1 == DYN || TDim0 == TDim1)
    && AreCompatibleDims<TDim0 == DYN ? TDim1 : TDim0, TRestDims...>::value;
};

template <typename... TDimSeqs>
struct AreCompatibleDimSeqs
{
  static const metal::int_ MAX_RANK = math::max(non_trivial_dimensions_num_v<TDimSeqs>::value...);

  template <metal::int_ TRank>
  static constexpr bool compatible_at()
  {
    return AreCompatibleDims<nth_dimension_v<TRank, TDimSeqs>::value...>::value;
  }

  template <metal::int_... TIndices>
  static constexpr bool get(metal::numbers<TIndices...>)
  {
    return math::landsc(compatible_at<TIndices>()...);
  }

  static const bool value = get(metal::iota<metal::number<0>, metal::number<MAX_RANK>>());
};

} // end of ns detail





template <metal::int_... TDims>
TVALUE(metal::int_, combine_dims_v, detail::CombineDims<TDims...>::value);
template <typename... TDimSeqOrTensors>
using combine_dimseqs_t = typename detail::CombineDimSeqs<dimseq_t<TDimSeqOrTensors>...>::type;

template <metal::int_... TDims>
TVALUE(metal::int_, are_compatible_dims_v, detail::AreCompatibleDims<TDims...>::value);
template <typename... TDimSeqOrTensors>
TVALUE(metal::int_, are_compatible_dimseqs_v, detail::AreCompatibleDimSeqs<TDimSeqOrTensors...>::value);





static_assert(are_compatible_dimseqs_v<DimSeq<1, 2>, DimSeq<1, 2>>::value, "are_compatible_dimseqs_v not working");
static_assert(are_compatible_dimseqs_v<DimSeq<1, 2, 1>, DimSeq<1, 2>>::value, "are_compatible_dimseqs_v not working");
static_assert(!are_compatible_dimseqs_v<DimSeq<1, 3>, DimSeq<1>>::value, "are_compatible_dimseqs_v not working");
static_assert(are_compatible_dimseqs_v<DimSeq<>, DimSeq<1, 1, 1>>::value, "are_compatible_dimseqs_v not working");
static_assert(!are_compatible_dimseqs_v<DimSeq<0>, DimSeq<>>::value, "are_compatible_dimseqs_v not working");

} // end of ns template_tensors
