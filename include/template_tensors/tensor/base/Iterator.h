namespace template_tensors {

namespace detail {

template <typename TTensorType, typename TIndexStrategy>
class TensorElementAtIndex
{
private:
  TTensorType tensor;
  TIndexStrategy index_strategy;

public:
  template <typename TTensorType2>
  __host__ __device__
  TensorElementAtIndex(TTensorType2&& tensor, TIndexStrategy index_strategy)
    : tensor(util::forward<TTensorType2>(tensor))
    , index_strategy(index_strategy)
  {
  }

  template <typename TTensorType2, typename TIndexStrategy2>
  friend class TensorElementAtIndex;

private:
  template <typename TThisType>
  __host__ __device__
  static auto get(TThisType&& self, size_t index)
  RETURN_AUTO(self.tensor(self.index_strategy.fromIndex(index, self.tensor.dims())))

public:
  FORWARD_ALL_QUALIFIERS(operator(), get)
};

} // end of ns detail

template <typename TIndexStrategy = template_tensors::RowMajor, typename TTensorType>
__host__ __device__
::iterator::transform_iterator<detail::TensorElementAtIndex<util::store_member_t<TTensorType&&>, TIndexStrategy>, ::iterator::counting_iterator<size_t>>
  begin(TTensorType&& tensor, TIndexStrategy index_strategy = TIndexStrategy())
{
  return ::iterator::transform_iterator<detail::TensorElementAtIndex<util::store_member_t<TTensorType&&>, TIndexStrategy>, ::iterator::counting_iterator<size_t>>(
    ::iterator::counting_iterator<size_t>(0), detail::TensorElementAtIndex<util::store_member_t<TTensorType&&>, TIndexStrategy>(util::forward<TTensorType>(tensor), index_strategy)
  );
}
// TODO: use ::iterator::count() and ::iterator::transform() here
// TODO: when constructing begin and end iterators via util::forward, tensor might be moved in first call to begin and then be unusable in call to end
template <typename TIndexStrategy = template_tensors::RowMajor, typename TTensorType>
__host__ __device__
::iterator::transform_iterator<detail::TensorElementAtIndex<util::store_member_t<TTensorType&&>, TIndexStrategy>, ::iterator::counting_iterator<size_t>>
  end(TTensorType&& tensor, TIndexStrategy index_strategy = TIndexStrategy())
{
  size_t num = template_tensors::multiplyDimensions(tensor.dims()); // TODO: should depend on index strategy?
  return ::iterator::transform_iterator<detail::TensorElementAtIndex<util::store_member_t<TTensorType&&>, TIndexStrategy>, ::iterator::counting_iterator<size_t>>(
    ::iterator::counting_iterator<size_t>(num), detail::TensorElementAtIndex<util::store_member_t<TTensorType&&>, TIndexStrategy>(util::forward<TTensorType>(tensor), index_strategy)
  );
}

template <typename TThisType>
class Iterable
{
private:
  template <typename TThisType2>
  __host__ __device__
  static auto begin1(TThisType2&& self)
  RETURN_AUTO(template_tensors::begin(static_cast<util::copy_qualifiers_t<TThisType, TThisType2&&>>(self)))

  template <typename TThisType2>
  __host__ __device__
  static auto end1(TThisType2&& self)
  RETURN_AUTO(template_tensors::end(static_cast<util::copy_qualifiers_t<TThisType, TThisType2&&>>(self)))

public:
  FORWARD_ALL_QUALIFIERS(begin, begin1)
  FORWARD_ALL_QUALIFIERS(end, end1)
};

} // end of ns tensor
