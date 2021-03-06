#pragma once

namespace aggregator {

namespace detail {

template <typename TBucketAggregator, typename TAllocator, typename TIndexStrategy, typename TBucketDimSeq>
class partial_histogram : public aggregator::IsAggregator
{
private:
  template_tensors::LocalOrAllocTensorT<TBucketAggregator, TAllocator, TIndexStrategy, TBucketDimSeq> m_buckets;

public:
  template <typename... TDimArgs>
  __host__ __device__
  partial_histogram(TBucketAggregator bucket_aggregator, TIndexStrategy index_strategy, TDimArgs&&... dim_args)
    : m_buckets(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, index_strategy, std::forward<TDimArgs>(dim_args)...)
  {
    template_tensors::fill(m_buckets, bucket_aggregator);
  }

  __host__ __device__
  partial_histogram()
  {
  }

  HD_WARNING_DISABLE
  template <typename TFirst, typename... TRest>
  __host__ __device__
  void operator()(TFirst&& first, TRest&&... rest)
  {
    m_buckets(template_tensors::static_cast_to<size_t>(std::forward<TFirst>(first)))(std::forward<TRest>(rest)...);
  }

  __host__ __device__
  auto get() const
  RETURN_AUTO(template_tensors::elwise(util::functor::get(), m_buckets))

  __host__ __device__
  auto get() const volatile
  RETURN_AUTO(template_tensors::elwise(util::functor::get(), m_buckets))
};

} // end of ns detail

template <typename TAllocator = mem::alloc::heap, typename TIndexStrategy = template_tensors::ColMajor, typename TBucketAggregator,
  typename... TDimArgs, ENABLE_IF(template_tensors::are_dim_args_v<TDimArgs...>::value && !template_tensors::are_dim_args_v<TIndexStrategy, TDimArgs&&...>::value)>
__host__ __device__
auto partial_histogram(TBucketAggregator aggregator, TIndexStrategy index_strategy, TDimArgs&&... dim_args)
RETURN_AUTO(detail::partial_histogram<typename std::decay<TBucketAggregator>::type, TAllocator, TIndexStrategy, template_tensors::dyn_dimseq_t<template_tensors::dimension_num_v<TDimArgs...>::value>>(
  aggregator,
  index_strategy,
  std::forward<TDimArgs>(dim_args)...
))

template <typename TAllocator = mem::alloc::heap, typename TIndexStrategy = template_tensors::ColMajor, typename TBucketAggregator,
  typename... TDimArgs, ENABLE_IF(template_tensors::are_dim_args_v<TDimArgs&&...>::value)>
__host__ __device__
auto partial_histogram(TBucketAggregator aggregator, TDimArgs&&... dim_args)
RETURN_AUTO(aggregator::partial_histogram(
  aggregator,
  TIndexStrategy(),
  std::forward<TDimArgs>(dim_args)...
))

template <metal::int_... TDims, typename TAllocator = mem::alloc::heap, typename TIndexStrategy = template_tensors::ColMajor, typename TBucketAggregator,
  ENABLE_IF(math::gt(sizeof...(TDims), 0UL))>
__host__ __device__
auto partial_histogram(TBucketAggregator aggregator, TIndexStrategy index_strategy)
RETURN_AUTO(detail::partial_histogram<typename std::decay<TBucketAggregator>::type, TAllocator, TIndexStrategy, template_tensors::DimSeq<TDims...>>(
  aggregator,
  index_strategy
))

template <metal::int_... TDims, typename TAllocator = mem::alloc::heap, typename TIndexStrategy = template_tensors::ColMajor, typename TBucketAggregator,
  ENABLE_IF(math::gt(sizeof...(TDims), 0UL))>
__host__ __device__
auto partial_histogram(TBucketAggregator aggregator)
RETURN_AUTO(detail::partial_histogram<typename std::decay<TBucketAggregator>::type, TAllocator, TIndexStrategy, template_tensors::DimSeq<TDims...>>(
  aggregator,
  TIndexStrategy()
))

} // end of ns aggregator
