#pragma once

namespace template_tensors {

namespace functor {

template <typename TElwiseEqualsOp = math::functor::eq>
struct eq;

} // end of ns functor

} // end of ns template_tensors

namespace iterable {

template <typename TForEach = for_each::AutoForEach<>, typename TInputIterable, typename TOutputIterable>
__host__ __device__
auto copy(TOutputIterable&& out, TInputIterable&& in) -> decltype(util::forward<TInputIterable>(in).begin())
{
  using Output = decltype(*util::forward<TOutputIterable>(out).begin());
  auto in_it = in.begin();
  TForEach::for_each(out.begin(), out.end(), [&](Output& output){
    output = *in_it;
    ++in_it;
  });
  ASSERT(in_it == in.end(), "Iterables have different lengths");
  return in_it;
}

template <typename TForEach = for_each::AutoForEach<>, typename TObject, typename TIterable, typename TCompare = template_tensors::functor::eq<>>
__host__ __device__
bool is_in(TObject&& object, TIterable&& iterable, TCompare&& compare = TCompare())
{
  // TODO: short_circuit for_each
  bool result = false;
  TForEach::for_each(iterable.begin(), iterable.end(), [&](const TObject& object2){
    if (compare(object, object2))
    {
      result = true;
    }
  });
  return result;
}

template <typename TForEach = for_each::AutoForEach<>, typename TIterable>
__host__ __device__
size_t distance(TIterable&& iterable)
{
  auto counter = aggregator::count<size_t>();
  TForEach::for_each(iterable.begin(), iterable.end(), counter);
  return counter.get();
}

namespace functor {

template <typename TForEach = for_each::AutoForEach<>>
struct distance
{
  template <typename TIterable>
  __host__ __device__
  auto operator()(TIterable&& iterable) const
  RETURN_AUTO(iterable::distance<TForEach>(util::forward<TIterable>(iterable)))
};

} // end of ns functor

} // end of ns iterable
