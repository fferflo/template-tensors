#pragma once

namespace atomic {

namespace functor {

template <typename TAtomicOps>
struct addassign
{
  TAtomicOps ops;

  template <typename TData, typename TAdd>
  __host__ __device__
  auto operator()(TData& data, const TAdd& add)
  RETURN_AUTO(ops.add(data, add))
};

template <typename TAtomicOps>
struct inc
{
  TAtomicOps ops;

  template <typename TData>
  __host__ __device__
  auto operator()(TData& data)
  RETURN_AUTO(ops.inc(data))
};

struct load
{
  template <typename TVariable>
  __host__ __device__
  auto operator()(TVariable&& var) const
  RETURN_AUTO(var.load())
};

} // end of ns functor

} // end of ns atomic
