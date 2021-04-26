namespace template_tensors {

namespace detail {

template <size_t I, size_t N>
struct AreSameDimensions
{
  template <typename TVectorType1, typename... TDimArgTypes>
  __host__ __device__
  static bool test(const TVectorType1& dims1, TDimArgTypes&&... dims2)
  {
    return getNthDimension<I>(dims1) == getNthDimension<I>(util::forward<TDimArgTypes>(dims2)...) && AreSameDimensions<I + 1, N>::test(dims1, util::forward<TDimArgTypes>(dims2)...);
  }
};

template <size_t N>
struct AreSameDimensions<N, N>
{
  template <typename TVectorType1, typename... TDimArgTypes>
  __host__ __device__
  static bool test(const TVectorType1& dims1, TDimArgTypes&&... dims2)
  {
    return true;
  }
};

template <size_t I, size_t N>
struct AreSameCoordinates
{
  template <typename TVectorType1, typename... TCoordArgTypes>
  __host__ __device__
  static bool test(const TVectorType1& coords1, TCoordArgTypes&&... coords2)
  {
    return getNthCoordinate<I>(coords1) == getNthCoordinate<I>(util::forward<TCoordArgTypes>(coords2)...) && AreSameDimensions<I + 1, N>::test(coords1, util::forward<TCoordArgTypes>(coords2)...);
  }
};

template <size_t N>
struct AreSameCoordinates<N, N>
{
  template <typename TVectorType1, typename... TCoordArgTypes>
  __host__ __device__
  static bool test(const TVectorType1& coords1, TCoordArgTypes&&... coords2)
  {
    return true;
  }
};

template <size_t TMax, size_t I>
struct AreCompatibleDimensions
{
  template <typename TDimSeq, typename... TDimArgTypes>
  __host__ __device__
  static bool check(TDimArgTypes&&... dims)
  {
    return (nth_dimension_v<I, TDimSeq>::value == DYN || nth_dimension_v<I, TDimSeq>::value == template_tensors::getNthDimension<I>(util::forward<TDimArgTypes>(dims)...))
      && AreCompatibleDimensions<TMax, I + 1>::template check<TDimSeq>(util::forward<TDimArgTypes>(dims)...);
  }
};

template <size_t TMax>
struct AreCompatibleDimensions<TMax, TMax>
{
  template <typename TDimSeq, typename... TDimArgTypes>
  __host__ __device__
  static bool check(TDimArgTypes&&... dims)
  {
    return true;
  }
};

template <size_t TMax, size_t I>
struct AreCompatibleCoordinates
{
  template <typename TCoordSeq, typename... TCoordArgTypes>
  __host__ __device__
  static bool check(TCoordArgTypes&&... coords)
  {
    return (nth_coordinate_v<I, TCoordSeq>::value == DYN || nth_coordinate_v<I, TCoordSeq>::value == getNthCoordinate<I>(util::forward<TCoordArgTypes>(coords)...))
      && AreCompatibleCoordinates<TMax, I + 1>::template check<TCoordSeq>(util::forward<TCoordArgTypes>(coords)...);
  }
};

template <size_t TMax>
struct AreCompatibleCoordinates<TMax, TMax>
{
  template <typename TCoordSeq, typename... TCoordArgTypes>
  __host__ __device__
  static bool check(TCoordArgTypes&&... coords)
  {
    return true;
  }
};

} // end of ns detail





__host__ __device__
inline bool areSameDimensions()
{
  return true;
}

template <typename TVectorType1>
__host__ __device__
bool areSameDimensions(const TVectorType1&)
{
  return true;
}

template <typename TVectorType1, typename TVectorType2, typename... TRest>
__host__ __device__
bool areSameDimensions(const TVectorType1& dims1, const TVectorType2& dims2, TRest&&... rest)
{
  return detail::AreSameDimensions<0, math::max(rows_v<TVectorType1>::value, rows_v<TVectorType2>::value)>::test(dims1, dims2)
    && areSameDimensions(dims1, util::forward<TRest>(rest)...);
}

template <typename TVectorType1, typename... TDimArgTypes>
__host__ __device__
bool areSameDimensions2(const TVectorType1& dims1, TDimArgTypes&&... dims2)
{
  return detail::AreSameDimensions<0, math::max(rows_v<TVectorType1>::value, dimension_num_v<TDimArgTypes&&...>::value)>::test(dims1, util::forward<TDimArgTypes>(dims2)...);
}

__host__ __device__
inline bool areSameCoordinates()
{
  return true;
}

template <typename TVectorType1>
__host__ __device__
bool areSameCoordinates(const TVectorType1&)
{
  return true;
}

template <typename TVectorType1, typename TVectorType2, typename... TRest>
__host__ __device__
bool areSameCoordinates(const TVectorType1& dims1, const TVectorType2& dims2, TRest&&... rest)
{
  return detail::AreSameCoordinates<0, math::max(rows_v<TVectorType1>::value, rows_v<TVectorType2>::value)>::test(dims1, dims2)
    && areSameCoordinates(dims1, util::forward<TRest>(rest)...);
}

template <typename TVectorType1, typename... TCoordArgTypes>
__host__ __device__
bool areSameCoordinates2(const TVectorType1& coords1, TCoordArgTypes&&... coords2)
{
  return detail::AreSameCoordinates<0, math::max(rows_v<TVectorType1>::value, coordinate_num_v<TCoordArgTypes&&...>::value)>::test(coords1, util::forward<TCoordArgTypes>(coords2)...);
}

template <typename TDimSeq, typename... TDimArgTypes>
__host__ __device__
bool areCompatibleDimensions(TDimArgTypes&&... dim_args)
{
  const size_t MAX = math::max(dimension_num_v<TDimArgTypes...>::value, non_trivial_dimensions_num_v<TDimSeq>::value);
  return detail::AreCompatibleDimensions<MAX, 0>::template check<TDimSeq>(util::forward<TDimArgTypes>(dim_args)...);
}

template <typename TCoordSeq, typename... TCoordArgTypes>
__host__ __device__
bool areCompatibleCoordinates(TCoordArgTypes&&... coord_args)
{
  const size_t MAX = math::max(coordinate_num_v<TCoordArgTypes...>::value, non_trivial_coordinates_num_v<TCoordSeq>::value);
  return detail::AreCompatibleCoordinates<MAX, 0>::template check<TCoordSeq>(util::forward<TCoordArgTypes>(coord_args)...);
}





namespace detail {

template <size_t I>
struct CoordsAreInRange
{
  template <typename TDimArgType, typename... TCoordArgTypes>
  __host__ __device__
  static bool get(TDimArgType&& dims, TCoordArgTypes&&... coords)
  {
    auto coord = getNthCoordinate<I - 1>(util::forward<TCoordArgTypes>(coords)...);
    auto dim = getNthDimension<I - 1>(util::forward<TDimArgType>(dims));
    return coord < dim
      && CoordsAreInRange<I - 1>::get(util::forward<TDimArgType>(dims), util::forward<TCoordArgTypes>(coords)...);
  }
};

template <>
struct CoordsAreInRange<0>
{
  template <typename TDimArgType, typename... TCoordArgTypes>
  __host__ __device__
  static bool get(TDimArgType&& dims, TCoordArgTypes&&... coords)
  {
    return true;
  }
};

} // end of ns detail

template <typename TDimArgType, typename... TCoordArgTypes, ENABLE_IF(are_dim_args_v<TDimArgType>::value)>
__host__ __device__
bool coordsAreInRange(TDimArgType&& dim_arg, TCoordArgTypes&&... coords)
{
  return detail::CoordsAreInRange<coordinate_num_v<TCoordArgTypes...>::value>::get
            (util::forward<TDimArgType>(dim_arg), util::forward<TCoordArgTypes>(coords)...);
}

} // end of ns tensor