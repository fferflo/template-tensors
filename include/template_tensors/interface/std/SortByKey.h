#pragma once

#include <boost/iterator/iterator_facade.hpp>
#include <algorithm>

namespace std {

template <typename T1, typename T2>
void swap(std::pair<T1, T2>&& t1, std::pair<T1, T2>&& t2)
{
  std::swap(std::get<0>(t1), std::get<0>(t2));
  std::swap(std::get<1>(t1), std::get<1>(t2));
}

namespace detail {

template <typename TKeyIterator, typename TValueIterator>
using SortPairValue = std::pair<
  typename std::iterator_traits<TKeyIterator>::value_type,
  typename std::iterator_traits<TValueIterator>::value_type
>;
template <typename TKeyIterator, typename TValueIterator>
using SortPairReference = std::pair<
  typename std::iterator_traits<TKeyIterator>::reference,
  typename std::iterator_traits<TValueIterator>::reference
>;

template <typename TKeyIterator, typename TValueIterator>
class SortPairIterator : public boost::iterator_facade<
  SortPairIterator<TKeyIterator, TValueIterator>,
  SortPairValue<TKeyIterator, TValueIterator>,
  std::random_access_iterator_tag,
  SortPairReference<TKeyIterator, TValueIterator>,
  typename std::iterator_traits<TKeyIterator>::difference_type>
{
public:
  SortPairIterator()
  {
  }

  SortPairIterator(TKeyIterator key_it, TValueIterator value_it)
    : m_key_it(key_it)
    , m_value_it(value_it)
  {
  }

private:
  TKeyIterator m_key_it;
  TValueIterator m_value_it;

  friend class boost::iterator_core_access;

  void increment()
  {
    ++m_key_it;
    ++m_value_it;
  }

  void decrement()
  {
    --m_key_it;
    --m_value_it;
  }

  bool equal(const SortPairIterator<TKeyIterator, TValueIterator>& other) const
  {
    return m_key_it == other.m_key_it;
  }

  SortPairReference<TKeyIterator, TValueIterator> dereference() const
  {
    return SortPairReference<TKeyIterator, TValueIterator>(*m_key_it, *m_value_it);
  }

  void advance(typename std::iterator_traits<TKeyIterator>::difference_type n)
  {
    m_key_it += n;
    m_value_it += n;
  }

  typename std::iterator_traits<TKeyIterator>::difference_type distance_to(const SortPairIterator<TKeyIterator, TValueIterator>& other) const
  {
    return std::distance(m_key_it, other.m_key_it);
  }
};

template <typename TCompareKey>
struct SortPairCompare
{
  TCompareKey compare_key;

  SortPairCompare(TCompareKey compare_key)
    : compare_key(compare_key)
  {
  }

  template <typename TPair1, typename TPair2>
  bool operator()(const TPair1& left, const TPair2& right)
  {
    return compare_key(std::get<0>(left), std::get<0>(right));
  }
};

} // namespace detail

template <typename TKeyIterator, typename TValueIterator, typename TCompareKey>
void sort_by_key(TKeyIterator keys_begin, TKeyIterator keys_end, TValueIterator values_begin, TCompareKey&& compare_key)
{
  std::sort(
    detail::SortPairIterator<TKeyIterator, TValueIterator>(keys_begin, values_begin),
    detail::SortPairIterator<TKeyIterator, TValueIterator>(keys_end, values_begin + std::distance(keys_begin, keys_end)),
    detail::SortPairCompare<util::store_member_t<TCompareKey&&>>(util::forward<TCompareKey>(compare_key))
  );
}

template <typename TKeyIterator, typename TValueIterator>
void sort_by_key(TKeyIterator keys_begin, TKeyIterator keys_end, TValueIterator values_begin)
{
  std::sort_by_key(keys_begin, keys_end, values_begin, math::functor::lt());
}

} // end of ns std
