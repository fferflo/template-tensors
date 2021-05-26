#pragma once

namespace iterable {

template <typename TIterable, typename TAdaptor>
class Adapt
{
private:
  TIterable m_iterable;
  TAdaptor m_adaptor;

public:
  template <typename TIterable2, typename TAdaptor2>
  __host__ __device__
  Adapt(TIterable2&& iterable, TAdaptor2&& adaptor)
    : m_iterable(std::forward<TIterable2>(iterable))
    , m_adaptor(std::forward<TAdaptor2>(adaptor))
  {
  }

  __host__ __device__
  auto begin()
  RETURN_AUTO(m_adaptor(m_iterable.begin()))

  __host__ __device__
  auto end()
  RETURN_AUTO(m_adaptor(m_iterable.end()))
};

template <typename TIterable, typename TAdaptor>
__host__ __device__
auto adapt(TIterable&& in, TAdaptor&& adaptor)
RETURN_AUTO(
  Adapt<util::store_member_t<TIterable&&>, util::store_member_t<TAdaptor&&>>
    (std::forward<TIterable>(in), std::forward<TAdaptor>(adaptor))
)

} // end of ns iterable
