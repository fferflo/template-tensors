#pragma once

namespace iterator {

template <typename TInputIterator, typename TPredicate>
class Filter
{
private:
  TInputIterator m_current;
  TInputIterator m_end;
  TPredicate m_pred;

public:
  template <typename TInputIterator2, typename TPredicate2>
  __host__ __device__
  Filter(TInputIterator2&& begin, TInputIterator2&& end, TPredicate2&& pred)
    : m_current(util::forward<TInputIterator2>(begin))
    , m_end(util::forward<TInputIterator2>(end))
    , m_pred(util::forward<TPredicate2>(pred))
  {
    while (m_current != m_end && !pred(*m_current))
    {
      ++m_current;
    }
  }

  __host__ __device__
  Filter(TInputIterator end, TPredicate pred)
    : Filter(end, end, pred)
  {
  }

  __host__ __device__
  Filter<TInputIterator, TPredicate>& operator++()
  {
    ++m_current;
    while (m_current != m_end && !m_pred(*m_current))
    {
      ++m_current;
    }
    return *this;
  }

  __host__ __device__
  Filter<TInputIterator, TPredicate> operator++(int)
  {
    Filter<TInputIterator, TPredicate> before = *this;
    this->operator++();
    return before;
  }

  __host__ __device__
  auto operator*() const
  RETURN_AUTO(*m_current)

  __host__ __device__
  bool operator==(const Filter<TInputIterator, TPredicate>& other)
  {
    return this->m_current == other.m_current;
  }

  __host__ __device__
  bool operator!=(const Filter<TInputIterator, TPredicate>& other)
  {
    return this->m_current != other.m_current;
  }
};

template <typename TIterator, typename TPredicate>
__host__ __device__
auto filter(TIterator&& begin, TIterator&& end, TPredicate&& pred)
RETURN_AUTO(
  Filter<util::store_member_t<TIterator&&>, util::store_member_t<TPredicate&&>>
    (util::forward<TIterator>(begin), util::forward<TIterator>(end), util::forward<TPredicate>(pred))
)

} // end of ns iterator

namespace iterable {

template <typename TIterable, typename TPredicate>
class Filter
{
private:
  TIterable m_iterable;
  TPredicate m_pred;

public:
  template <typename TIterable2, typename TPredicate2>
  __host__ __device__
  Filter(TIterable2&& iterable, TPredicate2&& pred)
    : m_iterable(util::forward<TIterable2>(iterable))
    , m_pred(util::forward<TPredicate2>(pred))
  {
  }

  __host__ __device__
  auto begin()
  RETURN_AUTO(::iterator::filter(m_iterable.begin(), m_iterable.end(), m_pred))

  __host__ __device__
  auto end()
  RETURN_AUTO(::iterator::filter(m_iterable.end(), m_iterable.end(), m_pred))
};

template <typename TIterable, typename TPredicate>
__host__ __device__
auto filter(TIterable&& in, TPredicate&& pred)
RETURN_AUTO(
  Filter<util::store_member_t<TIterable&&>, util::store_member_t<TPredicate&&>>
    (util::forward<TIterable>(in), util::forward<TPredicate>(pred))
)

} // end of ns iterable
