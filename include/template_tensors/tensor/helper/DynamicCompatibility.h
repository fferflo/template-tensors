namespace template_tensors {

namespace detail {

template <metal::int_ I, metal::int_ N>
struct AreSameDimensions
{
  template <typename TVectorType1, typename... TDimArgTypes>
  __host__ __device__
  static bool test(const TVectorType1& dims1, TDimArgTypes&&... dims2)
  {
    return getNthDimension<I>(dims1) == getNthDimension<I>(std::forward<TDimArgTypes>(dims2)...) && AreSameDimensions<I + 1, N>::test(dims1, std::forward<TDimArgTypes>(dims2)...);
  }
};

template <metal::int_ N>
struct AreSameDimensions<N, N>
{
  template <typename TVectorType1, typename... TDimArgTypes>
  __host__ __device__
  static bool test(const TVectorType1& dims1, TDimArgTypes&&... dims2)
  {
    return true;
  }
};

template <metal::int_ I, metal::int_ N>
struct AreSameCoordinates
{
  template <typename TVectorType1, typename... TCoordArgTypes>
  __host__ __device__
  static bool test(const TVectorType1& coords1, TCoordArgTypes&&... coords2)
  {
    return getNthCoordinate<I>(coords1) == getNthCoordinate<I>(std::forward<TCoordArgTypes>(coords2)...) && AreSameDimensions<I + 1, N>::test(coords1, std::forward<TCoordArgTypes>(coords2)...);
  }
};

template <metal::int_ N>
struct AreSameCoordinates<N, N>
{
  template <typename TVectorType1, typename... TCoordArgTypes>
  __host__ __device__
  static bool test(const TVectorType1& coords1, TCoordArgTypes&&... coords2)
  {
    return true;
  }
};

template <metal::int_ TMax, metal::int_ I>
struct AreCompatibleDimensions
{
  template <typename TDimSeq, typename... TDimArgTypes>
  __host__ __device__
  static bool check(TDimArgTypes&&... dims)
  {
    return (nth_dimension_v<I, TDimSeq>::value == DYN || nth_dimension_v<I, TDimSeq>::value == template_tensors::getNthDimension<I>(std::forward<TDimArgTypes>(dims)...))
      && AreCompatibleDimensions<TMax, I + 1>::template check<TDimSeq>(std::forward<TDimArgTypes>(dims)...);
  }
};

template <metal::int_ TMax>
struct AreCompatibleDimensions<TMax, TMax>
{
  template <typename TDimSeq, typename... TDimArgTypes>
  __host__ __device__
  static bool check(TDimArgTypes&&... dims)
  {
    return true;
  }
};

template <metal::int_ TMax, metal::int_ I>
struct AreCompatibleCoordinates
{
  template <typename TCoordSeq, typename... TCoordArgTypes>
  __host__ __device__
  static bool check(TCoordArgTypes&&... coords)
  {
    return (nth_coordinate_v<I, TCoordSeq>::value == DYN || nth_coordinate_v<I, TCoordSeq>::value == getNthCoordinate<I>(std::forward<TCoordArgTypes>(coords)...))
      && AreCompatibleCoordinates<TMax, I + 1>::template check<TCoordSeq>(std::forward<TCoordArgTypes>(coords)...);
  }
};

template <metal::int_ TMax>
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
    && areSameDimensions(dims1, std::forward<TRest>(rest)...);
}

template <typename TVectorType1, typename... TDimArgTypes>
__host__ __device__
bool areSameDimensions2(const TVectorType1& dims1, TDimArgTypes&&... dims2)
{
  return detail::AreSameDimensions<0, math::max(rows_v<TVectorType1>::value, dimension_num_v<TDimArgTypes&&...>::value)>::test(dims1, std::forward<TDimArgTypes>(dims2)...);
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
    && areSameCoordinates(dims1, std::forward<TRest>(rest)...);
}

template <typename TVectorType1, typename... TCoordArgTypes>
__host__ __device__
bool areSameCoordinates2(const TVectorType1& coords1, TCoordArgTypes&&... coords2)
{
  return detail::AreSameCoordinates<0, math::max(rows_v<TVectorType1>::value, coordinate_num_v<TCoordArgTypes&&...>::value)>::test(coords1, std::forward<TCoordArgTypes>(coords2)...);
}

template <typename TDimSeq, typename... TDimArgTypes>
__host__ __device__
bool areCompatibleDimensions(TDimArgTypes&&... dim_args)
{
  const metal::int_ MAX = math::max(dimension_num_v<TDimArgTypes...>::value, non_trivial_dimensions_num_v<TDimSeq>::value);
  return detail::AreCompatibleDimensions<MAX, 0>::template check<TDimSeq>(std::forward<TDimArgTypes>(dim_args)...);
}

template <typename TCoordSeq, typename... TCoordArgTypes>
__host__ __device__
bool areCompatibleCoordinates(TCoordArgTypes&&... coord_args)
{
  const metal::int_ MAX = math::max(coordinate_num_v<TCoordArgTypes...>::value, non_trivial_coordinates_num_v<TCoordSeq>::value);
  return detail::AreCompatibleCoordinates<MAX, 0>::template check<TCoordSeq>(std::forward<TCoordArgTypes>(coord_args)...);
}





namespace detail {

template <metal::int_ I>
struct CoordsAreInRange
{
  template <typename TDimArgType, typename... TCoordArgTypes>
  __host__ __device__
  static bool get(TDimArgType&& dims, TCoordArgTypes&&... coords)
  {
    auto coord = getNthCoordinate<I - 1>(std::forward<TCoordArgTypes>(coords)...);
    auto dim = getNthDimension<I - 1>(std::forward<TDimArgType>(dims));
    return coord < dim
      && CoordsAreInRange<I - 1>::get(std::forward<TDimArgType>(dims), std::forward<TCoordArgTypes>(coords)...);
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
            (std::forward<TDimArgType>(dim_arg), std::forward<TCoordArgTypes>(coords)...);
}

} // end of ns template_tensors
