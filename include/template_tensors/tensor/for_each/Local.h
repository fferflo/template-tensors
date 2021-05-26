namespace template_tensors {

namespace op {

namespace detail {

template <metal::int_ I, metal::int_ TCoordsRank, typename... TTensorTypes>
struct LocalForEachHelper
{
  template <typename TFunctor, typename... TCoords>
  __host__ __device__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors, TCoords... coords)
  {
    const metal::int_ MAX = util::first(std::forward<TTensorTypes>(tensors)...).template dim<I - 1>();
    for (metal::int_ i = 0; i < MAX; i++)
    {
      LocalForEachHelper<I - 1, TCoordsRank, TTensorTypes...>::for_each(std::forward<TFunctor>(func), std::forward<TTensorTypes>(tensors)..., i, coords...);
    }
  }
};

template <metal::int_ TCoordsRank, typename... TTensorTypes>
struct LocalForEachHelper<0, TCoordsRank, TTensorTypes...>
{
  HD_WARNING_DISABLE
  template <typename TFunctor, typename... TCoords>
  __host__ __device__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors, TCoords... coords)
  {
    func(toCoordVector<TCoordsRank>(coords...), tensors(coords...)...);
  }
};

template <typename... TTensorTypes>
struct LocalForEachHelper<0, DYN, TTensorTypes...>
{
  HD_WARNING_DISABLE
  template <typename TFunctor, typename... TCoords>
  __host__ __device__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors, TCoords... coords)
  {
    func(tensors(coords...)...);
  }
};

} // end of ns detail

struct LocalForEach
{
  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_for_each_available_v, math::land(mem::isOnLocal<mem::memorytype_v<TTensorTypes>::value, TIsOnHost>()...))

  template <bool TIsOnHost, typename TTensorDest, typename... TTensorSrcs>
  TVALUE(bool, is_map_available_v, is_for_each_available_v<TIsOnHost, TTensorDest, TTensorSrcs...>::value)

  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v, is_map_available_v<TIsOnHost, TTensorDest, TTensorSrc>::value)

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_parallel_v, false)

  template <metal::int_ TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes>
  __host__ __device__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors)
  {
    ASSERT(areSameDimensions(tensors.dims()...), "Incompatible dynamic dimensions");
    static_assert(math::landsc(are_compatible_dimseqs_v<dimseq_t<TTensorTypes>...>::value), "Incompatible static dimensions");

    const metal::int_ MAX_RANK = math::min(non_trivial_dimensions_num_v<TTensorTypes>::value...);
    detail::LocalForEachHelper<MAX_RANK, TCoordsRank, TTensorTypes...>::for_each(std::forward<TFunctor>(func), std::forward<TTensorTypes>(tensors)...);
  }

  TT_FOR_EACH_MAP_AND_COPY(__host__ __device__)
};

} // end of ns op

} // end of ns template_tensors
