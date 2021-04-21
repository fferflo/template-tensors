namespace template_tensors {

namespace detail {

template <bool TZeroSized, size_t I>
struct StreamOutputHelper
{
  HD_WARNING_DISABLE
  template <typename TStreamType, typename TTensorType, typename... TCoords>
  __host__ __device__
  static void output(TStreamType&& stream, const TTensorType& tensor, TCoords... coords)
  {
    size_t max = tensor.template dim<I - 1>();
    util::forward<TStreamType>(stream) << "[";
    for (size_t i = 0; i < max; i++)
    {
      StreamOutputHelper<TZeroSized, I - 1>::output(util::forward<TStreamType>(stream), tensor, i, coords...);
      if (I == 1 && i < max - 1)
      {
        util::forward<TStreamType>(stream) << " ";
      }
    }
    util::forward<TStreamType>(stream) << "]";
  }
};

template <>
struct StreamOutputHelper<false, 0>
{
  HD_WARNING_DISABLE
  template <typename TStreamType, typename TTensorType, typename... TCoords>
  __host__ __device__
  static void output(TStreamType&& stream, const TTensorType& tensor, TCoords... coords)
  {
    util::forward<TStreamType>(stream) << tensor(coords...);
  }
};

template <size_t I>
struct StreamOutputHelper<true, I>
{
  HD_WARNING_DISABLE
  template <typename TStreamType, typename TTensorType, typename... TCoords>
  __host__ __device__
  static void output(TStreamType&& stream, const TTensorType& tensor, TCoords... coords)
  {
    util::forward<TStreamType>(stream) << "[]";
  }
};

} // end of ns detail

// ColMajor output
template <typename TStreamType, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
TStreamType&& operator<<(TStreamType&& stream, const TTensorType& tensor)
{
  detail::StreamOutputHelper<rows_v<TTensorType>::value == 0, math::max((size_t) 1, non_trivial_dimensions_num_v<TTensorType>::value)>::output(util::forward<TStreamType>(stream), tensor);
  return util::forward<TStreamType>(stream);
}

} // end of ns tensor
