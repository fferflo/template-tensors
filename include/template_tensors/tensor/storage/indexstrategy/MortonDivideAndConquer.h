namespace template_tensors {

namespace detail {

template <size_t TBit, size_t TBits, size_t TStep, size_t TRank>
struct MortonMaskHelper
{
  static const size_t steps = math::ilog2(TBits / TRank) + 1 + (TBits / TRank * TRank == TBits ? 0 : 1);
  static const bool bit_is_set = (TBit / (1 << (steps - TStep - 1))) % TRank == 0;
  static const size_t value =
    (((bit_is_set ? (size_t) 1UL : (size_t) 0UL) << TBit)
      | MortonMaskHelper<TBit + 1, TBits, TStep, TRank>::value)
    & (((size_t) -1) >> (sizeof(size_t) * 8 - TBits));
};

template <size_t TBits, size_t TStep, size_t TRank>
struct MortonMaskHelper<TBits, TBits, TStep, TRank>
{
  static const size_t value = 0UL;
};

static_assert(MortonMaskHelper<0, 32, 2, 2>::value == 0b00001111000011110000111100001111, "MortonMaskHelper not working");
static_assert(MortonMaskHelper<0, 32, 2, 5>::value == 0b11000000001100000000110000000011, "MortonMaskHelper not working");

template <size_t TStep, size_t TRank>
using MortonMask = MortonMaskHelper<0, sizeof(size_t) * 8, TStep, TRank>;



template <size_t I, size_t N, size_t TRank>
struct MortonSeparateHelper
{
  static const size_t shift = (TRank - 1) * (1 << (N - I - 1));
  static const size_t mask = MortonMask<I, TRank>::value;

  __host__ __device__
  static size_t separate(size_t field)
  {
    return MortonSeparateHelper<I + 1, N, TRank>::separate((field ^ (field << shift)) & mask);
  }
};

template <size_t N, size_t TRank>
struct MortonSeparateHelper<0, N, TRank>
{
  static const size_t mask = MortonMask<0, TRank>::value;

  __host__ __device__
  static size_t separate(size_t field)
  {
    ASSERT((field & mask) == field, "Coordinate out of morton range");
    return MortonSeparateHelper<1, N, TRank>::separate(field);
  }
};

template <size_t N, size_t TRank>
struct MortonSeparateHelper<N, N, TRank>
{
  __host__ __device__
  static size_t separate(size_t field)
  {
    return field;
  }
};

template <size_t TRank>
__host__ __device__
size_t mortonSeparate(size_t field)
{
  return MortonSeparateHelper<0, MortonMask<0, TRank>::steps, TRank>::separate(field);
}



template <size_t I, size_t N, size_t TRank>
struct MortonCompactHelper
{
  static const size_t shift = (TRank - 1) * (1 << (I - 1));
  static const size_t mask = MortonMask<N - I - 1, TRank>::value;

  __host__ __device__
  static size_t compact(size_t index)
  {
    return MortonCompactHelper<I + 1, N, TRank>::compact((index ^ (index >> shift)) & mask);
  }
};

template <size_t N, size_t TRank>
struct MortonCompactHelper<0, N, TRank>
{
  static const size_t mask = MortonMask<N - 1, TRank>::value;

  __host__ __device__
  static size_t compact(size_t index)
  {
    return MortonCompactHelper<1, N, TRank>::compact(index & mask);
  }
};

template <size_t N, size_t TRank>
struct MortonCompactHelper<N, N, TRank>
{
  __host__ __device__
  static size_t compact(size_t index)
  {
    return index;
  }
};

template <size_t TRank>
size_t mortonCompact(size_t index)
{
  return MortonCompactHelper<0, MortonMask<0, TRank>::steps, TRank>::compact(index);
}





template <size_t I, size_t N>
struct MortonDivideAndConquerToIndexHelper
{
  template <typename TDimArgType, typename... TCoordArgTypes>
  __host__ __device__
  static size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords)
  {
    return (mortonSeparate<N>(getNthCoordinate<I>(util::forward<TCoordArgTypes>(coords)...)) << I)
      | MortonDivideAndConquerToIndexHelper<I + 1, N>::toIndex(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...);
  }
};

template <size_t N>
struct MortonDivideAndConquerToIndexHelper<N, N>
{
  template <typename TDimArgType, typename... TCoordArgTypes>
  __host__ __device__
  static size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords)
  {
    return 0UL;
  }
};

template <size_t I, size_t N>
struct MortonDivideAndConquerFromIndexHelper
{
  template <size_t TDims, typename... TDimArgTypes>
  __host__ __device__
  static void fromIndex(VectorXs<TDims>& result, size_t index, TDimArgTypes&&... dims)
  {
    if (I < TDims)
    {
      result(I) = mortonCompact<N>(index >> I);
    }

    MortonDivideAndConquerFromIndexHelper<I + 1, N>::fromIndex(result, index, util::forward<TDimArgTypes>(dims)...);
  }
};

template <size_t N>
struct MortonDivideAndConquerFromIndexHelper<N, N>
{
  template <size_t TDims, typename... TDimArgTypes>
  __host__ __device__
  static void fromIndex(VectorXs<TDims>& result, size_t index, TDimArgTypes&&... dims)
  {
    for (size_t i = N; i < TDims; i++)
    {
      result(i) = 0;
    }
  }
};

} // end of ns detail





template <size_t TRank>
struct MortonDivideAndConquer
{
  static const size_t FIELD_BITS = sizeof(size_t) * 8 / TRank;
  static const bool IS_STATIC = false;

  template <typename TDimArgType, typename... TCoordArgTypes, ENABLE_IF(are_dim_args_v<TDimArgType&&>::value)>
  __host__ __device__
  size_t toIndex(TDimArgType&& dims, TCoordArgTypes&&... coords) const volatile
  {
    ASSERT(getNonTrivialDimensionsNum(util::forward<TDimArgType>(dims)) <= TRank, "Dimensions out of morton range");
    ASSERT(coordsAreInRange(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...), "Coordinates are out of range");
    return detail::MortonDivideAndConquerToIndexHelper<0, TRank>::toIndex(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...);
  }

  INDEXSTRATEGY_TO_INDEX_2

  template <size_t TDimsArg = DYN, typename... TDimArgTypes, size_t TDims = TDimsArg == DYN ? dimension_num_v<TDimArgTypes&&...>::value : TDimsArg>
  __host__ __device__
  VectorXs<TDims> fromIndex(size_t index, TDimArgTypes&&... dims) const volatile
  {
    ASSERT(getNonTrivialDimensionsNum(util::forward<TDimArgTypes>(dims)...) <= TRank, "Dimensions out of morton range");
    VectorXs<TDims> result;
    detail::MortonForLoopFromIndexHelper<0, TRank>::fromIndex(result, index, util::forward<TDimArgTypes>(dims)...);
    return result;
  }

  INDEXSTRATEGY_FROM_INDEX_2

  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  size_t getSize(TDimArgTypes&&... dim_args) const volatile
  {
    return toIndex(toDimVector(util::forward<TDimArgTypes>(dim_args)...), toDimVector(util::forward<TDimArgTypes>(dim_args)...) - 1) + 1;
  }
};

template <size_t TRank>
__host__ __device__
bool operator==(const volatile MortonDivideAndConquer<TRank>&, const volatile MortonDivideAndConquer<TRank>&)
{
  return true;
}

HD_WARNING_DISABLE
template <typename TStreamType, size_t TRank>
__host__ __device__
TStreamType&& operator<<(TStreamType&& stream, const MortonDivideAndConquer<TRank>& index_strategy)
{
  stream << "MortonDivideAndConquer<" << TRank << ">";
  return util::forward<TStreamType>(stream);
}

#ifdef CEREAL_INCLUDED
template <typename TArchive, size_t TRank>
void save(TArchive& archive, const MortonDivideAndConquer<TRank>& m)
{
}

template <typename TArchive, size_t TRank>
void load(TArchive& archive, MortonDivideAndConquer<TRank>& m)
{
}
#endif

template <size_t TRank>
using Morton = MortonDivideAndConquer<TRank>;

} // end of ns tensor

template <size_t TRank>
PROCLAIM_TRIVIALLY_RELOCATABLE((template_tensors::MortonDivideAndConquer<TRank>));
