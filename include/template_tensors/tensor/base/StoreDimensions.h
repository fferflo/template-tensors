namespace template_tensors {

namespace detail {

template <bool TIsStatic, typename TDimSeq>
class StoreDimensionsImpl;

template <typename TDimSeq>
class StoreDimensionsImpl<true, TDimSeq>
{
public:
  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  StoreDimensionsImpl(TDimArgTypes&&... args)
  {
  }
};

template <typename TDimSeq>
class StoreDimensionsImpl<false, TDimSeq>
{
public:
  template <typename... TDimArgTypes, ENABLE_IF(are_dim_args_v<TDimArgTypes...>::value)>
  __host__ __device__
  StoreDimensionsImpl(TDimArgTypes&&... args)
    : m_dims(toDimVector<non_trivial_dimensions_num_v<TDimSeq>::value>(util::forward<TDimArgTypes>(args)...))
  {
  }

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return getNthDimension<TIndex>(m_dims);
  }

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const volatile
  {
    return getNthDimension<TIndex>(m_dims);
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    // Compile-time constant to run-time constant conversion
    static const metal::int_ non_trivial_dimensions_num = non_trivial_dimensions_num_v<TDimSeq>::value;
    return math::lt(index, static_cast<size_t>(non_trivial_dimensions_num)) ? m_dims(index) : 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const volatile
  {
    // Compile-time constant to run-time constant conversion
    static const metal::int_ non_trivial_dimensions_num = non_trivial_dimensions_num_v<TDimSeq>::value;
    return math::lt(index, non_trivial_dimensions_num) ? m_dims(index) : 1;
  }

private:
  VectorXs<non_trivial_dimensions_num_v<TDimSeq>::value> m_dims;
};

} // end of ns detail

template <typename TDimSeq>
class StoreDimensions : public detail::StoreDimensionsImpl<is_static_v<TDimSeq>::value, TDimSeq>
{
  using detail::StoreDimensionsImpl<is_static_v<TDimSeq>::value, TDimSeq>::StoreDimensionsImpl;
};

} // end of ns tensor
