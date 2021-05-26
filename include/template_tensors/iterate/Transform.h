#pragma once

#ifdef __CUDACC__
#include <thrust/iterator/transform_iterator.h>
#else
#include <boost/iterator/transform_iterator.hpp>
#endif

namespace iterator {

#ifdef __CUDACC__
template <typename T, typename TIterator>
using transform_iterator = thrust::transform_iterator<T, TIterator>;
#else
template <typename T, typename TIterator>
using transform_iterator = boost::transform_iterator<T, TIterator>;
#endif

template <typename TIterator, typename TFunctor>
__host__ __device__
auto transform(TIterator iterator, TFunctor&& functor)
RETURN_AUTO(transform_iterator<typename std::decay<TFunctor&&>::type, TIterator>(iterator, std::forward<TFunctor>(functor)))

namespace functor {

template <typename TFunctor>
struct Transform
{
  TFunctor functor;

  template <typename TFunctor2, ENABLE_IF(std::is_constructible<TFunctor, TFunctor2&&>::value)>
  __host__ __device__
  Transform(TFunctor2&& functor)
    : functor(std::forward<TFunctor2>(functor))
  {
  }

  template <typename TThisType, typename TIterator>
  __host__ __device__
  static auto get(TThisType&& self, TIterator&& iterator)
  RETURN_AUTO(::iterator::transform(std::forward<TIterator>(iterator), self.functor))
  FORWARD_ALL_QUALIFIERS(operator(), get)
};

template <typename TFunctor>
__host__ __device__
auto transform(TFunctor&& functor)
RETURN_AUTO(Transform<util::store_member_t<TFunctor&&>>(std::forward<TFunctor>(functor)))

} // end of ns functor

} // end of ns iterator

namespace iterable {

template <typename TIterable, typename TFunctor>
__host__ __device__
auto transform(TIterable&& in, TFunctor&& functor)
RETURN_AUTO(
  adapt(std::forward<TIterable>(in), ::iterator::functor::transform(std::forward<TFunctor>(functor)))
)

} // end of ns iterable
