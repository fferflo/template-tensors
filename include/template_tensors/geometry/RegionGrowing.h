#pragma once

namespace detail {

template <typename TOutput, typename TValues, typename TGetNeighbors, typename TAreSimilar, typename TAtomicOps>
struct RegionGrowing
{
  using Index = template_tensors::decay_elementtype_t<TOutput>;
  using Value = template_tensors::decay_elementtype_t<TValues>;

  TOutput output;
  TValues values;
  TGetNeighbors get_neighbors;
  TAreSimilar are_similar;
  TAtomicOps atomic_ops;

  template <typename TOutput2, typename TValues2, typename TGetNeighbors2, typename TAreSimilar2, typename TAtomicOps2>
  __host__ __device__
  RegionGrowing(TOutput2&& output, TValues2&& values, TGetNeighbors2&& get_neighbors, TAreSimilar2&& are_similar, TAtomicOps2&& atomic_ops)
    : output(util::forward<TOutput2>(output))
    , values(util::forward<TValues2>(values))
    , get_neighbors(util::forward<TGetNeighbors2>(get_neighbors))
    , are_similar(util::forward<TAreSimilar2>(are_similar))
    , atomic_ops(util::forward<TAtomicOps2>(atomic_ops))
  {
  }

  __host__ __device__
  RegionGrowing(const RegionGrowing<TOutput, TValues, TGetNeighbors, TAreSimilar, TAtomicOps>& other)
    : output(static_cast<TOutput>(other.output))
    , values(static_cast<TValues>(other.values))
    , get_neighbors(static_cast<TGetNeighbors>(other.get_neighbors))
    , are_similar(static_cast<TAreSimilar>(other.are_similar))
    , atomic_ops(static_cast<TAtomicOps>(other.atomic_ops))
  {
  }

  __host__ __device__
  void reset(Index idx, Index& value)
  {
    value = idx;
  }

  __host__ __device__
  void merge(Index center_idx, Value center_value)
  {
    auto neighbors = get_neighbors(center_idx);
    auto neighbor_it = neighbors.begin();
    auto neighbors_end = neighbors.end();
    while (neighbor_it != neighbors_end)
    {
      bool increment = true;
      Index neighbor_idx = *neighbor_it;
      if (template_tensors::lt(center_idx, neighbor_idx))
      {
        Value neighbor_value = values(neighbor_idx);
        if (are_similar(center_value, neighbor_value))
        {
          Index center_root = getRoot(center_idx);
          Index neighbor_root = getRoot(neighbor_idx);
          // If regions are connected: Neighbor root is higher in tree than center root or the same
          if (!template_tensors::eq(neighbor_root, center_root) && !isPredecessor(center_root, neighbor_root))
          {
            // Regions are not connected yet
            bool swapped;
            if (!atomic_ops(center_root).cas(output(center_root), center_root, neighbor_root, swapped) || !swapped)
            {
              // Either the atomic operation failed, or center_root was modified concurrently and is not a root anymore
              increment = false;
            }
          }
        }
      }
      if (increment)
      {
        ++neighbor_it;
      }
    }
  }

  __host__ __device__
  void postprocess(Index idx, Index& value)
  {
    value = getRoot(idx);
  }

  __host__ __device__
  Index getRoot(Index idx)
  {
#ifdef DEBUG
    for (size_t i = 0; i < 100000; i++)
#else
    while (true)
#endif
    {
      Index parent_idx;
      if (atomic_ops(idx).load(output(idx), parent_idx))
      {
        if (template_tensors::eq(parent_idx, idx))
        {
          return idx;
        }
        idx = parent_idx;
      }
    }
#ifdef DEBUG
    ASSERT_(false, "RegionGrowing: getRoot timeout\n");
#endif
    return idx;
  }

  __host__ __device__
  bool isPredecessor(Index child, Index parent)
  {
#ifdef DEBUG
    for (size_t i = 0; i < 100000; i++)
#else
    while (true)
#endif
    {
      Index next;
      if (atomic_ops(child).load(output(child), next))
      {
        if (template_tensors::eq(next, parent))
        {
          return true;
        }
        else if (template_tensors::eq(next, child))
        {
          return false;
        }
        child = next;
      }
    }
  #ifdef DEBUG
      ASSERT_(false, "RegionGrowing: isPredecessor timeout\n");
  #endif
    return false;
  }
};

} // end of ns detail

DECLARE_MEMBER_FUNCTOR(reset)
DECLARE_MEMBER_FUNCTOR(merge)
DECLARE_MEMBER_FUNCTOR(postprocess)

template <typename TForEach = template_tensors::op::AutoForEach<>, bool TIsOnHost = TT_IS_ON_HOST, typename TOutput, typename TValues, typename TGetNeighbors, typename TAreSimilar, typename TAtomicOps>
__host__ __device__
void regionGrow(TOutput&& output, TValues&& values, TGetNeighbors&& get_neighbors, TAreSimilar&& are_similar, TAtomicOps&& atomic_ops)
{
  INSTANTIATE_HOST(ESC(regionGrow<TForEach, true, TOutput, TValues, TGetNeighbors, TAreSimilar, TAtomicOps>),
    INSTANTIATE_ARG(TOutput&&), INSTANTIATE_ARG(TValues&&), INSTANTIATE_ARG(TGetNeighbors&&), INSTANTIATE_ARG(TAreSimilar&&), INSTANTIATE_ARG(TAtomicOps&&));
  INSTANTIATE_DEVICE(ESC(regionGrow<TForEach, false, TOutput, TValues, TGetNeighbors, TAreSimilar, TAtomicOps>),
    INSTANTIATE_ARG(TOutput&&), INSTANTIATE_ARG(TValues&&), INSTANTIATE_ARG(TGetNeighbors&&), INSTANTIATE_ARG(TAreSimilar&&), INSTANTIATE_ARG(TAtomicOps&&));

  using Index = template_tensors::decay_elementtype_t<TOutput>;
  static const size_t Rank = template_tensors::rows_v<Index>::value;

  static const mem::MemoryType MEMORY_TYPE = mem::memorytype_v<TValues>::value;
  auto region_grower = ::detail::RegionGrowing<
    decltype(mem::toFunctor<MEMORY_TYPE, TIsOnHost>(output)),
    decltype(mem::toFunctor<MEMORY_TYPE, TIsOnHost>(values)),
    decltype(mem::toFunctor<MEMORY_TYPE, TIsOnHost>(get_neighbors)),
    decltype(mem::toFunctor<MEMORY_TYPE, TIsOnHost>(are_similar)),
    decltype(mem::toFunctor<MEMORY_TYPE, TIsOnHost>(atomic_ops))
  >(
    mem::toFunctor<MEMORY_TYPE, TIsOnHost>(output),
    mem::toFunctor<MEMORY_TYPE, TIsOnHost>(values),
    mem::toFunctor<MEMORY_TYPE, TIsOnHost>(get_neighbors),
    mem::toFunctor<MEMORY_TYPE, TIsOnHost>(are_similar),
    mem::toFunctor<MEMORY_TYPE, TIsOnHost>(atomic_ops)
  );
  using RegionGrower = decltype(region_grower);

  static const bool HOST = mem::isOnHost<MEMORY_TYPE>();
  TForEach::template for_each<Rank>(member::hd::reset<HOST, RegionGrower>(region_grower), util::forward<TOutput>(output));
  TForEach::template for_each<Rank>(member::hd::merge<HOST, RegionGrower>(region_grower), util::forward<TValues>(values));
  TForEach::template for_each<Rank>(member::hd::postprocess<HOST, RegionGrower>(region_grower), util::forward<TOutput>(output));
}



template <size_t TRank, typename TForEach = template_tensors::op::AutoForEach<>, typename TIndexStrategy = template_tensors::RowMajor, bool TIsOnHost = TT_IS_ON_HOST, typename TValues, typename TGetNeighbors, typename TAreSimilar, typename TAtomicOps,
  typename TIndex = template_tensors::VectorXs<TRank>,
  typename TResultTensor = template_tensors::AllocTensorTEx<TIndex, mem::alloc::default_for<mem::memorytype_v<TValues>::value, TIsOnHost>, TIndexStrategy, template_tensors::dimseq_t<TValues&&>>
>
__host__ __device__
TResultTensor regionGrow(TValues&& values, TGetNeighbors&& get_neighbors, TAreSimilar&& are_similar, TAtomicOps&& atomic_ops)
{
  INSTANTIATE_HOST(ESC(regionGrow<TRank, TForEach, TIndexStrategy, true, TValues, TGetNeighbors, TAreSimilar, TAtomicOps>),
    INSTANTIATE_ARG(TValues&&), INSTANTIATE_ARG(TGetNeighbors&&), INSTANTIATE_ARG(TAreSimilar&&), INSTANTIATE_ARG(TAtomicOps&&));
  INSTANTIATE_DEVICE(ESC(regionGrow<TRank, TForEach, TIndexStrategy, false, TValues, TGetNeighbors, TAreSimilar, TAtomicOps>),
    INSTANTIATE_ARG(TValues&&), INSTANTIATE_ARG(TGetNeighbors&&), INSTANTIATE_ARG(TAreSimilar&&), INSTANTIATE_ARG(TAtomicOps&&));

  TResultTensor result(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, TIndexStrategy(), util::forward<TValues>(values).dims());
  regionGrow<TForEach>(result, util::forward<TValues>(values), util::forward<TGetNeighbors>(get_neighbors), util::forward<TAreSimilar>(are_similar), util::forward<TAtomicOps>(atomic_ops));
  return result;
}



namespace detail {

template <bool TIsParallel>
struct RegionGrowAtomicOps;

template <>
struct RegionGrowAtomicOps<true>
{
  template <typename TAtomicOps, typename TAllocator, typename TIndexStrategy, typename TDimSeq, typename TDims>
  __host__ __device__
  static auto get(TDims&& dims)
  RETURN_AUTO(
    template_tensors::AllocTensorTEx<TAtomicOps, TAllocator, TIndexStrategy, TDimSeq>
      (TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, TIndexStrategy(), util::forward<TDims>(dims))
  )
};

template <>
struct RegionGrowAtomicOps<false>
{
  template <typename TAtomicOps, typename TAllocator, typename TIndexStrategy, typename TDimSeq, typename TDims>
  __host__ __device__
  static auto get(TDims&& dims)
  RETURN_AUTO(
    util::functor::constant<::atomic::op::Void>(::atomic::op::Void())
  )
};

} // end of ns detail
// TODO: take index strategy as value parameter?
template <size_t TRank, typename TForEach = template_tensors::op::AutoForEach<>, typename TIndexStrategy = template_tensors::RowMajor, bool TIsOnHost = TT_IS_ON_HOST, typename TValues, typename TGetNeighbors, typename TAreSimilar,
  typename TIndex = template_tensors::VectorXs<TRank>,
  typename TResultTensor = template_tensors::AllocTensorTEx<TIndex, mem::alloc::default_for<mem::memorytype_v<TValues>::value>, TIndexStrategy, template_tensors::dimseq_t<TValues&&>>
>
__host__ __device__
TResultTensor regionGrow(TValues&& values, TGetNeighbors&& get_neighbors, TAreSimilar&& are_similar)
{
  INSTANTIATE_HOST(ESC(regionGrow<TRank, TForEach, TIndexStrategy, true, TValues, TGetNeighbors, TAreSimilar>),
    INSTANTIATE_ARG(TValues&&), INSTANTIATE_ARG(TGetNeighbors&&), INSTANTIATE_ARG(TAreSimilar&&));
  INSTANTIATE_DEVICE(ESC(regionGrow<TRank, TForEach, TIndexStrategy, false, TValues, TGetNeighbors, TAreSimilar>),
    INSTANTIATE_ARG(TValues&&), INSTANTIATE_ARG(TGetNeighbors&&), INSTANTIATE_ARG(TAreSimilar&&));

  auto atomic_ops = ::detail::RegionGrowAtomicOps<TForEach::template is_parallel_v<TIsOnHost, TValues&&>::value>::template get<
      ::atomic::op::default_try_for<mem::memorytype_v<TValues>::value, TIsOnHost>,
      mem::alloc::default_for<mem::memorytype_v<TValues>::value, TIsOnHost>,
      TIndexStrategy,
      template_tensors::dimseq_t<TValues&&>
    >(util::forward<TValues>(values).dims());
  return regionGrow<TRank, TForEach, TIndexStrategy>(util::forward<TValues>(values), util::forward<TGetNeighbors>(get_neighbors), util::forward<TAreSimilar>(are_similar), atomic_ops);
}
