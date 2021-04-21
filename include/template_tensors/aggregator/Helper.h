#pragma once

namespace aggregator {

struct IsAggregator
{
};

template <typename TType>
TVALUE(bool, is_aggregator_v, std::is_base_of<IsAggregator, typename std::decay<TType>::type>::value)

template <typename TAggregator>
using resulttype_t = decltype(std::declval<TAggregator>().get());

} // end of ns aggregator
