namespace template_tensors {

namespace detail {

template <bool TZeroSized, metal::int_ I>
struct StreamOutputHelper
{
  HD_WARNING_DISABLE
  template <typename TStreamType, typename TTensorType, typename... TCoords>
  __host__ __device__
  static void output(TStreamType&& stream, const TTensorType& tensor, TCoords... coords)
  {
    dim_t max = tensor.template dim<I - 1>();
    std::forward<TStreamType>(stream) << "[";
    for (dim_t i = 0; i < max; i++)
    {
      StreamOutputHelper<TZeroSized, I - 1>::output(std::forward<TStreamType>(stream), tensor, i, coords...);
      if (I == 1 && i < max - 1)
      {
        std::forward<TStreamType>(stream) << " ";
      }
    }
    std::forward<TStreamType>(stream) << "]";
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
    std::forward<TStreamType>(stream) << tensor(coords...);
  }
};

template <metal::int_ I>
struct StreamOutputHelper<true, I>
{
  HD_WARNING_DISABLE
  template <typename TStreamType, typename TTensorType, typename... TCoords>
  __host__ __device__
  static void output(TStreamType&& stream, const TTensorType& tensor, TCoords... coords)
  {
    std::forward<TStreamType>(stream) << "[]";
  }
};

} // end of ns detail

// ColMajor output
template <typename TStreamType, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
TStreamType&& operator<<(TStreamType&& stream, const TTensorType& tensor)
{
  detail::StreamOutputHelper<rows_v<TTensorType>::value == 0, math::max(static_cast<metal::int_>(1), non_trivial_dimensions_num_v<TTensorType>::value)>::output(std::forward<TStreamType>(stream), tensor);
  return std::forward<TStreamType>(stream);
}

} // end of ns template_tensors
