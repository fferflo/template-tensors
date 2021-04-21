#pragma once

namespace iterable {

template <typename TIterator>
class StoreIterators
{
private:
  TIterator m_begin;
  TIterator m_end;

public:
  __host__ __device__
  StoreIterators(TIterator begin, TIterator end)
    : m_begin(begin)
    , m_end(end)
  {
  }

  __host__ __device__
  TIterator begin()
  {
    return m_begin;
  }

  __host__ __device__
  TIterator end()
  {
    return m_end;
  }
};

template <typename TIterator>
__host__ __device__
auto store_iterators(TIterator begin, TIterator end)
RETURN_AUTO(StoreIterators<TIterator>(begin, end))

} // end of ns iterable
