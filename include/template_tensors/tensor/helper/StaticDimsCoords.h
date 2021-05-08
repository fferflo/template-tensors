#include <metal.hpp>

namespace template_tensors {

using dim_t = size_t;

/*!
 * \brief A compile-time dimension value indicating that the dimension will be determined at run-time.
 */
static const metal::int_ DYN = static_cast<metal::int_>(-1);

template <metal::int_... TDims>
using DimSeq = metal::numbers<TDims...>;
template <metal::int_... TCoords>
using CoordSeq = metal::numbers<TCoords...>;

template <typename TArg>
struct is_dimseq_v
{
  template <metal::int_... TDims>
  TMP_IF(const DimSeq<TDims...>&)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(typename std::decay<TArg>::type);
};
template <typename TArg>
TVALUE(bool, is_coordseq_v, is_dimseq_v<TArg>::value)

static_assert(is_dimseq_v<DimSeq<1, 2, 1>>::value, "is_dimseq_v not working");
static_assert(!is_dimseq_v<int>::value, "is_dimseq_v not working");


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

  template <metal::int_... TDims>
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

template <metal::int_... TSeq>
struct SeqHelper<DimSeq<TSeq...>>
{
  static const metal::int_ SIZE = metal::size<DimSeq<TSeq...>>::value;

  template <metal::int_ TFill>
  static constexpr metal::int_ non_trivial_num()
  {
    return non_trivial_num<TFill>(::array::LocalArray<metal::int_, SIZE>(TSeq...), SIZE);
  }

  template <metal::int_ TFill>
  static constexpr metal::int_ non_trivial_num(const ::array::LocalArray<metal::int_, SIZE> dims, metal::int_ i)
  {
    return i == 0 ? 0
         : (dims.data()[i - 1] == TFill ? non_trivial_num<TFill>(dims, i - 1)
            : i);
  }

  static constexpr bool are_static()
  {
    return are_static(::array::LocalArray<metal::int_, SIZE>(TSeq...), 0);
  }

  static constexpr bool are_static(const ::array::LocalArray<metal::int_, SIZE> dims, metal::int_ i)
  {
    return i == SIZE ? true : dims.data()[i] != DYN && are_static(dims, i + 1);
  }

  template <metal::int_ TFill>
  static constexpr metal::int_ nth(metal::int_ n)
  {
    return nth<TFill>(::array::LocalArray<metal::int_, SIZE>(TSeq...), n);
  }

  template <metal::int_ TFill>
  static constexpr metal::int_ nth(const ::array::LocalArray<metal::int_, SIZE> dims, metal::int_ n)
  {
    return math::gte(n, SIZE) ? TFill : dims.data()[n];
  }
};

template <typename TDimSeq>
struct MultiplyDimensions;

template <metal::int_... TDims>
struct MultiplyDimensions<DimSeq<TDims...>>
{
  static const metal::int_ value = SeqHelper<DimSeq<TDims...>>::are_static() ? math::multiply(TDims...) : DYN;
};

} // end of ns detail

template <typename TDimSeqOrCoordSeqOrTensor>
TVALUE(bool, is_static_v, detail::SeqHelper<dimseq_t<TDimSeqOrCoordSeqOrTensor>>::are_static());
template <typename TDimSeqOrCoordSeqOrTensor>
TVALUE(metal::int_, non_trivial_dimensions_num_v, detail::SeqHelper<dimseq_t<TDimSeqOrCoordSeqOrTensor>>::template non_trivial_num<1>());
template <metal::int_ N, typename TDimSeqOrTensor>
TVALUE(metal::int_, nth_dimension_v, detail::SeqHelper<dimseq_t<TDimSeqOrTensor>>::template nth<1>(N))
template <typename TDimSeqOrTensor>
TVALUE(metal::int_, rows_v, nth_dimension_v<0, TDimSeqOrTensor>::value)
template <typename TDimSeqOrTensor>
TVALUE(metal::int_, cols_v, nth_dimension_v<1, TDimSeqOrTensor>::value)

template <typename TCoordSeq>
TVALUE(metal::int_, non_trivial_coordinates_num_v, detail::SeqHelper<typename std::decay<TCoordSeq>::type>::template non_trivial_num<0>());
template <metal::int_ N, typename TCoordSeq>
TVALUE(metal::int_, nth_coordinate_v, detail::SeqHelper<typename std::decay<TCoordSeq>::type>::template nth<0>(N))

template <typename TArg>
TVALUE(metal::int_, multiply_dimensions_v, detail::MultiplyDimensions<dimseq_t<TArg>>::value)


namespace detail {

template <metal::int_ TValue, metal::int_ N, metal::int_... TNumbers>
struct RepeatDimseqHelper
{
  using type = typename RepeatDimseqHelper<TValue, N - 1, TValue, TNumbers...>::type;
};

template <metal::int_ TValue, metal::int_... TNumbers>
struct RepeatDimseqHelper<TValue, 0, TNumbers...>
{
  using type = metal::list<metal::number<TNumbers>...>;
};

} // end of ns detail

template <metal::int_ TValue, metal::int_ TRank>
using repeat_dimseq_t = typename detail::RepeatDimseqHelper<TValue, TRank>::type;
// The below definition should be equivalent, but sometimes causes errors when compiling CUDA code
//template <metal::int_ TRank>
//using repeat_dimseq_t = metal::repeat<metal::number<TValue>, metal::number<TRank>>;

template <metal::int_ TRank>
using dyn_dimseq_t = repeat_dimseq_t<DYN, TRank>;



static_assert(non_trivial_dimensions_num_v<DimSeq<1, 2, 1>>::value == 2, "non_trivial_dimensions_num not working");
static_assert(nth_dimension_v<0, DimSeq<4, 2, 6>>::value == 4, "nth_dimension not working");
static_assert(nth_dimension_v<1, DimSeq<4, 2, 6>>::value == 2, "nth_dimension not working");
static_assert(nth_dimension_v<2, DimSeq<4, 2, 6>>::value == 6, "nth_dimension not working");
static_assert(nth_dimension_v<3, DimSeq<4, 2, 6>>::value == 1, "nth_dimension not working");



namespace detail {

template <typename TCoordSeq, typename TIndexSeq>
struct CoordSeqMakeLength;

template <typename TCoordSeq, metal::int_... TIndices>
struct CoordSeqMakeLength<TCoordSeq, metal::numbers<TIndices...>>
{
  using type = metal::numbers<nth_coordinate_v<TIndices, TCoordSeq>::value...>;
  static_assert(non_trivial_coordinates_num_v<TCoordSeq>::value == non_trivial_coordinates_num_v<type>::value, "Cannot cut non-trivial coordinates");
};

template <typename TDimSeq, typename TIndexSeq>
struct DimSeqMakeLength;

template <typename TDimSeq, metal::int_... TIndices>
struct DimSeqMakeLength<TDimSeq, metal::numbers<TIndices...>>
{
  using type = metal::numbers<nth_dimension_v<TIndices, TDimSeq>::value...>;
  static_assert(non_trivial_dimensions_num_v<TDimSeq>::value == non_trivial_dimensions_num_v<type>::value, "Cannot cut non-trivial dimensions");
};

} // end of ns detail

template <typename TDimSeqOrTensor, metal::int_ TLength>
using dimseq_make_length_t = typename detail::DimSeqMakeLength<dimseq_t<TDimSeqOrTensor>, metal::iota<metal::number<0>, metal::number<TLength>>>::type;
template <typename TCoordSeq, metal::int_ TLength>
using coordseq_make_length_t = typename detail::CoordSeqMakeLength<TCoordSeq, metal::iota<metal::number<0>, metal::number<TLength>>>::type;

} // end of ns template_tensors
