#pragma once

namespace aggregator {

namespace detail {

template <typename TElwiseAggregator, typename TAllocator, typename TIndexStrategy, typename TDimSeq>
class elwise_helper : public aggregator::IsAggregator
{
private:
  template_tensors::LocalOrAllocTensorT<TElwiseAggregator, TAllocator, TIndexStrategy, TDimSeq> m_aggregators;

public:
  template <typename... TDimArgs>
  __host__ __device__
  elwise_helper(TElwiseAggregator aggregator, TIndexStrategy index_strategy, TDimArgs&&... dim_args)
    : m_aggregators(TT_EXPLICIT_CONSTRUCT_WITH_DYN_DIMS, index_strategy, std::forward<TDimArgs>(dim_args)...)
  {
    template_tensors::fill(m_aggregators, aggregator);
  }

  __host__ __device__
  elwise_helper()
    : m_aggregators()
  {
  }

  struct AggregateElwise
  {
    template <typename TAggregator, typename... TInput>
    __host__ __device__
    void operator()(TAggregator&& aggregator, TInput&&... input) const
    {
      aggregator(std::forward<TInput>(input)...);
    }
  };

  template <typename... TInput>
  __host__ __device__
  void operator()(TInput&&... input)
  {
    template_tensors::op::LocalForEach::for_each(AggregateElwise(), m_aggregators, std::forward<TInput>(input)...);
  }

  __host__ __device__
  auto get() const
  RETURN_AUTO(template_tensors::elwise(util::functor::get(), m_aggregators))
};

} // end of ns detail

template <typename TAllocator = mem::alloc::heap, typename TIndexStrategy = template_tensors::ColMajor, typename TElwiseAggregator,
  typename... TDimArgs, ENABLE_IF(template_tensors::are_dim_args_v<TDimArgs...>::value && !template_tensors::are_dim_args_v<TIndexStrategy, TDimArgs&&...>::value)>
__host__ __device__
auto elwise(TElwiseAggregator aggregator, TIndexStrategy index_strategy, TDimArgs&&... dim_args)
RETURN_AUTO(detail::elwise_helper<typename std::decay<TElwiseAggregator>::type, TAllocator, TIndexStrategy, template_tensors::dyn_dimseq_t<template_tensors::dimension_num_v<TDimArgs...>::value>>(
  aggregator,
  index_strategy,
  std::forward<TDimArgs>(dim_args)...
))

template <typename TAllocator = mem::alloc::heap, typename TIndexStrategy = template_tensors::ColMajor, typename TElwiseAggregator,
  typename... TDimArgs, ENABLE_IF(template_tensors::are_dim_args_v<TDimArgs&&...>::value)>
__host__ __device__
auto elwise(TElwiseAggregator aggregator, TDimArgs&&... dim_args)
RETURN_AUTO(aggregator::elwise(
  aggregator,
  TIndexStrategy(),
  std::forward<TDimArgs>(dim_args)...
))

template <metal::int_... TDims, typename TAllocator = mem::alloc::heap, typename TIndexStrategy = template_tensors::ColMajor, typename TElwiseAggregator,
  ENABLE_IF(math::gt(sizeof...(TDims), 0UL))>
__host__ __device__
auto elwise(TElwiseAggregator aggregator, TIndexStrategy index_strategy)
RETURN_AUTO(detail::elwise_helper<typename std::decay<TElwiseAggregator>::type, TAllocator, TIndexStrategy, template_tensors::DimSeq<TDims...>>(
  aggregator,
  index_strategy
))

template <metal::int_... TDims, typename TAllocator = mem::alloc::heap, typename TIndexStrategy = template_tensors::ColMajor, typename TElwiseAggregator,
  ENABLE_IF(math::gt(sizeof...(TDims), 0UL))>
__host__ __device__
auto elwise(TElwiseAggregator aggregator)
RETURN_AUTO(detail::elwise_helper<typename std::decay<TElwiseAggregator>::type, TAllocator, TIndexStrategy, template_tensors::DimSeq<TDims...>>(
  aggregator,
  TIndexStrategy()
))

} // end of ns aggregator
