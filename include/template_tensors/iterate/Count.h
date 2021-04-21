#pragma once

#ifdef __CUDACC__
#include <thrust/iterator/counting_iterator.h>
#else
#include <boost/iterator/counting_iterator.hpp>
#endif

namespace iterator {

#ifdef __CUDACC__
template <typename T>
using counting_iterator = thrust::counting_iterator<T>;
#else
template <typename T>
using counting_iterator = boost::counting_iterator<T>;
#endif

template <typename TCounter = size_t>
__host__ __device__
auto count(TCounter counter)
RETURN_AUTO(counting_iterator<TCounter>(counter))

} // end of ns iterator

namespace iterable {

template <typename TCounter = size_t>
__host__ __device__
auto count(TCounter begin, TCounter end)
RETURN_AUTO(
  iterable::store_iterators(::iterator::count<TCounter>(begin), ::iterator::count<TCounter>(end))
)

} // end of ns iterable
