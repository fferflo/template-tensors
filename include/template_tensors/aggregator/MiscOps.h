#pragma once

namespace atomic {
namespace functor {

template <typename TAtomicOps>
struct addassign;
template <typename TAtomicOps>
struct inc;

} // end of ns functor
} // end of ns atomic

namespace aggregator {

namespace detail {

struct inc
{
  template <typename TCounter, typename... TArgs>
  __host__ __device__
  void operator()(TCounter& counter, TArgs&&...)
  {
    counter++;
  }
};

struct weighted_inc
{
  template <typename TCounter, typename TArg, typename TWeight>
  __host__ __device__
  void operator()(TCounter& counter, TArg&&, TWeight&& weight)
  {
    counter += util::forward<TWeight>(weight);
  }
};

template <typename TAtomicOps>
struct atomic_inc
{
  TAtomicOps ops;

  __host__ __device__
  atomic_inc()
    : ops()
  {
  }

  template <typename TData, typename... TRest>
  __host__ __device__
  auto operator()(TData& data, TRest&&...)
  RETURN_AUTO(ops.add(data, static_cast<TData>(1)))
};

} // end of ns detail

template <typename TResultType = size_t>
__host__ __device__
auto count(TResultType initial_value = 0)
RETURN_AUTO(assign<TResultType>(detail::inc(), initial_value))

namespace weighted {
template <typename TResultType>
__host__ __device__
auto count(TResultType initial_value = 0)
RETURN_AUTO(assign<TResultType>(aggregator::detail::weighted_inc(), initial_value))
} // end of ns weighted

namespace atomic {
template <typename TAtomicOps, typename TResultType = typename std::conditional<TT_IS_ON_HOST, size_t, uint32_t>::type>
__host__ __device__
auto count(TResultType initial_value = 0)
RETURN_AUTO(assign<TResultType>(detail::atomic_inc<TAtomicOps>(), initial_value))
} // end of ns atomic



template <typename TResultType>
__host__ __device__
auto sum(TResultType initial_value = 0)
RETURN_AUTO(assign<TResultType>(math::functor::addassign(), initial_value))

namespace weighted {
template <typename TResultType>
__host__ __device__
auto sum(TResultType initial_value = 0)
RETURN_AUTO(
  aggregator::map_input(
    math::functor::multiply(),
    aggregator::sum<TResultType>(initial_value)
  )
)
} // end of ns weighted

namespace atomic {
template <typename TAtomicOps, typename TResultType>
__host__ __device__
auto sum(TResultType initial_value = 0)
RETURN_AUTO(assign<TResultType>(::atomic::functor::addassign<TAtomicOps>(), initial_value))
} // end of ns atomic



template <typename TResultType>
__host__ __device__
auto prod(TResultType initial_value = 1)
RETURN_AUTO(assign<TResultType>(math::functor::multiplyassign(), initial_value))

namespace weighted {
template <typename TResultType>
__host__ __device__
auto prod(TResultType initial_value = 1)
RETURN_AUTO(
  aggregator::map_input(
    math::functor::pow(),
    aggregator::prod<TResultType>(initial_value)
  )
)
} // end of ns weighted



template <typename TResultType>
__host__ __device__
auto min(TResultType initial_value)
RETURN_AUTO(assign<TResultType>(util::functor::assign_self_mapped<math::functor::min>(), initial_value))

template <typename TResultType>
__host__ __device__
auto max(TResultType initial_value)
RETURN_AUTO(assign<TResultType>(util::functor::assign_self_mapped<math::functor::max>(), initial_value))

template <bool TDummy = false>
__host__ __device__
auto all()
RETURN_AUTO(assign<bool>(util::functor::assign_self_mapped<math::functor::landsc>(), true))

template <bool TDummy = false>
__host__ __device__
auto any()
RETURN_AUTO(assign<bool>(util::functor::assign_self_mapped<math::functor::lorsc>(), false))







#define OPERATION_T(NAME, ...) \
  template <typename TAggregator, ENABLE_IF(is_aggregator_v<TAggregator>::value)> \
  __host__ __device__ \
  auto NAME (TAggregator&& aggregator) \
  RETURN_AUTO(aggregator::map_output(__VA_ARGS__, util::forward<TAggregator>(aggregator)))

#define OPERATION_TT(NAME, ...) \
  template <typename TAggregatorLeft, typename TAggregatorRight, ENABLE_IF(is_aggregator_v<TAggregatorLeft>::value && is_aggregator_v<TAggregatorRight>::value)> \
  __host__ __device__ \
  auto NAME (TAggregatorLeft&& left, TAggregatorRight&& right) \
  RETURN_AUTO(aggregator::map_output(__VA_ARGS__, util::forward<TAggregatorLeft>(left), util::forward<TAggregatorRight>(right)))

#define OPERATION_TS(NAME, ...) \
  template <typename TAggregatorLeft, typename TNonAggregatorRight, ENABLE_IF(is_aggregator_v<TAggregatorLeft>::value && !is_aggregator_v<TNonAggregatorRight>::value)> \
  __host__ __device__ \
  auto NAME (TAggregatorLeft&& left, TNonAggregatorRight&& right) \
  RETURN_AUTO(aggregator::map_output(__VA_ARGS__, util::forward<TAggregatorLeft>(left), constant<util::store_member_t<TNonAggregatorRight&&>>(util::forward<TNonAggregatorRight>(right))))

#define OPERATION_ST(NAME, ...) \
  template <typename TNonAggregatorLeft, typename TAggregatorRight, ENABLE_IF(!is_aggregator_v<TNonAggregatorLeft>::value && is_aggregator_v<TAggregatorRight>::value)> \
  __host__ __device__ \
  auto NAME (TNonAggregatorLeft&& left, TAggregatorRight&& right) \
  RETURN_AUTO(aggregator::map_output(__VA_ARGS__, util::store_member_t<TNonAggregatorLeft&&>(util::forward<TNonAggregatorLeft>(left)), util::forward<TAggregatorRight>(right)))

OPERATION_TT(operator+, math::functor::add());
OPERATION_TS(operator+, math::functor::add());
OPERATION_ST(operator+, math::functor::add());

OPERATION_TT(operator-, math::functor::subtract());
OPERATION_TS(operator-, math::functor::subtract());
OPERATION_ST(operator-, math::functor::subtract());
OPERATION_T(operator-, math::functor::negate());

OPERATION_TT(operator*, math::functor::multiply());
OPERATION_TS(operator*, math::functor::multiply());
OPERATION_ST(operator*, math::functor::multiply());

OPERATION_TT(operator/, math::functor::divide());
OPERATION_TS(operator/, math::functor::divide());
OPERATION_ST(operator/, math::functor::divide());

#undef OPERATION_T
#undef OPERATION_TT
#undef OPERATION_TS
#undef OPERATION_ST

} // end of ns aggregator
