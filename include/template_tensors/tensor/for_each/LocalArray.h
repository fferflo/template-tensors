#include <metal.hpp>

namespace template_tensors {

namespace op {

namespace detail {

HD_WARNING_DISABLE
template <typename TFunctor, typename... TArgs>
__host__ __device__
void call_without_warning(TFunctor&& functor, TArgs&&... args)
{
  functor(std::forward<TArgs>(args)...);
}

template <metal::int_ TCoordsRank>
struct LocalArrayForEachHelper
{
  HD_WARNING_DISABLE
  template <typename TForEach, typename TFunctor, typename... TTensorTypes>
  __host__ __device__
  static void for_each(TFunctor&& func, size_t size, TTensorTypes&&... tensors)
  {
    using IndexStrategy = indexstrategy_t<metal::front<metal::list<TTensorTypes...>>>;
    IndexStrategy index_strategy = util::first(std::forward<TTensorTypes>(tensors)...).getIndexStrategy();
    VectorXs<TCoordsRank> dims = util::first(std::forward<TTensorTypes>(tensors)...).template dims<TCoordsRank>();

    TForEach::template for_each<multiply_dimensions_v<combine_dimseqs_t<TTensorTypes...>>::value, mem::combine<mem::memorytype_v<TTensorTypes>::value...>()>(
      ::iterator::counting_iterator<size_t>(0),
      ::iterator::counting_iterator<size_t>(size),
      [&](size_t index){
        call_without_warning(std::forward<TFunctor>(func), index_strategy.fromIndex(index, dims), tensors.data()[index]...);
      });
  }
};

template <>
struct LocalArrayForEachHelper<DYN>
{
  HD_WARNING_DISABLE
  template <typename TForEach, typename TFunctor, typename... TTensorTypes>
  __host__ __device__
  static void for_each(TFunctor&& func, size_t size, TTensorTypes&&... tensors)
  {
    TForEach::template for_each<multiply_dimensions_v<combine_dimseqs_t<TTensorTypes...>>::value, mem::combine<mem::memorytype_v<TTensorTypes>::value...>()>(
      ::iterator::counting_iterator<size_t>(0),
      ::iterator::counting_iterator<size_t>(size),
      [&](size_t index){
        call_without_warning(std::forward<TFunctor>(func), tensors.data()[index]...);
      });
  }
};

} // end of ns detail

template <typename TForEach = for_each::AutoForEach<>>
struct LocalArrayForEach
{
  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_for_each_available_v,
       TIsOnHost
    && have_same_indexstrategy_v<TTensorTypes...>::value
    && math::land(mem::isOnLocal<mem::memorytype_v<TTensorTypes>::value, TIsOnHost>()...)
    && math::land(detail::tensor_indexstrategy_can_convert_from_index_v<TTensorTypes>::value...)
  )

  template <bool TIsOnHost, typename TTensorDest, typename... TTensorSrcs>
  TVALUE(bool, is_map_available_v, is_for_each_available_v<TIsOnHost, TTensorDest, TTensorSrcs...>::value)

  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v, is_map_available_v<TIsOnHost, TTensorDest, TTensorSrc>::value)

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_parallel_v,
    TForEach::template is_parallel_v<TIsOnHost, multiply_dimensions_v<combine_dimseqs_t<TTensorTypes...>>::value, mem::combine<mem::memorytype_v<TTensorTypes>::value...>()>::value)

  template <metal::int_ TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes>
  __host__ __device__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors)
  {
    static_assert(sizeof...(tensors) > 0, "No tensors given");
    ASSERT(math::eq(tensors.getIndexStrategy()...), "Storages must have the same indexing strategy");
    ASSERT(template_tensors::all(template_tensors::elwise(math::functor::eq(), tensors.dims()...)), "Incompatible runtime dimensions");
    static_assert(are_compatible_dimseqs_v<dimseq_t<TTensorTypes>...>::value, "Incompatible static dimensions");
    static_assert(metal::same<indexstrategy_t<TTensorTypes>...>::value,
      "Storages must have the same indexing strategy");

    detail::LocalArrayForEachHelper<TCoordsRank>::template for_each<TForEach>(
      std::forward<TFunctor>(func),
      util::first(std::forward<TTensorTypes>(tensors)...).size(),
      std::forward<TTensorTypes>(tensors)...
    );
  }

  TT_FOR_EACH_MAP_AND_COPY(__host__ __device__)
};

} // end of ns op

} // end of ns template_tensors
