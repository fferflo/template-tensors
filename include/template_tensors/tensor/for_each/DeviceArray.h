namespace template_tensors {

namespace op {

#ifdef __CUDACC__
namespace detail {

template <size_t TStep, typename TIndexStrategy, size_t TCoordsRank, typename TFunctor, typename... TElementTypes>
__global__
void kernel_for_each_array_element_with_coords(TFunctor func, TIndexStrategy index_strategy, VectorXs<TCoordsRank> dims, size_t size, TElementTypes*... arrays)
{
  for (size_t index = cuda::grid::thread_id_in_grid<1>()(); index < size; index += TStep)
  {
    func(index_strategy.fromIndex(index, dims), arrays[index]...);
  }
}

template <size_t TCoordsRank>
struct DeviceArrayForEachHelper
{
  template <size_t TBlockSize, size_t TGridSize, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, size_t size, TTensorTypes&&... tensors)
  {
    using IndexStrategy = indexstrategy_t<tmp::ts::get_t<0, tmp::ts::Sequence<TTensorTypes...>>>;

    dim3 block, grid;
    block = dim3(TBlockSize);
    grid = dim3(TGridSize);
    TT_CUDA_SAFE_CALL((detail::kernel_for_each_array_element_with_coords<TBlockSize * TGridSize, IndexStrategy, TCoordsRank><<<grid, block>>>(
      mem::toKernel(util::forward<TFunctor>(func)),
      util::first(util::forward<TTensorTypes>(tensors)...).getIndexStrategy(),
      util::first(util::forward<TTensorTypes>(tensors)...).template dims<TCoordsRank>(),
      size,
      tensors.data()...
    )));
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
};

template <size_t TStep, typename TFunctor, typename... TElementTypes>
__global__
void kernel_for_each_array_element(TFunctor func, size_t size, TElementTypes*... arrays)
{
  for (size_t index = cuda::grid::thread_id_in_grid<1>()(); index < size; index += TStep)
  {
    func(arrays[index]...);
  }
}

template <>
struct DeviceArrayForEachHelper<DYN>
{
  template <size_t TBlockSize, size_t TGridSize, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, size_t size, TTensorTypes&&... tensors)
  {
    dim3 block, grid;
    block = dim3(TBlockSize);
    grid = dim3(TGridSize);
    TT_CUDA_SAFE_CALL((detail::kernel_for_each_array_element<TBlockSize * TGridSize><<<grid, block>>>(
      mem::toKernel(util::forward<TFunctor>(func)),
      size,
      tensors.data()...
    )));
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
};



} // end of ns detail

template <size_t TBlockSize = 128, size_t TGridSize = 96> // TODO: values for TBlockSize and TGridSize
struct DeviceArrayForEach
{
  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_for_each_available_v,
       TIsOnHost
    && have_same_indexstrategy_v<TTensorTypes...>::value
    && math::land(mem::isOnDevice<mem::memorytype_v<TTensorTypes>::value, TIsOnHost>()...)
    && math::land(detail::tensor_indexstrategy_can_convert_from_index_v<TTensorTypes>::value...)
  )

  template <bool TIsOnHost, typename TTensorDest, typename... TTensorSrcs>
  TVALUE(bool, is_map_available_v, is_for_each_available_v<TIsOnHost, TTensorDest, TTensorSrcs...>::value)

  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v, is_map_available_v<TIsOnHost, TTensorDest, TTensorSrc>::value)

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_parallel_v, true)

  template <size_t TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors)
  {
    static_assert(sizeof...(tensors) > 0, "No tensors given");
    ASSERT(math::eq(tensors.getIndexStrategy()...), "Storages must have the same indexing strategy");
    ASSERT(template_tensors::all(template_tensors::elwise(math::functor::eq(), tensors.dims()...)), "Incompatible runtime dimensions");
    static_assert(are_compatible_dimseqs_v<dimseq_t<TTensorTypes>...>::value, "Incompatible static dimensions");
    static_assert(tmp::ts::are_same_v<tmp::ts::Sequence<indexstrategy_t<TTensorTypes>...>>::value,
      "Storages must have the same indexing strategy");

    detail::DeviceArrayForEachHelper<TCoordsRank>::template for_each<TBlockSize, TGridSize>(
      util::forward<TFunctor>(func),
      util::first(util::forward<TTensorTypes>(tensors)...).size(),
      util::forward<TTensorTypes>(tensors)...
    );
  }

  TT_FOR_EACH_MAP_AND_COPY(__host__)
};

#else

template <size_t TBlockSize = 128, size_t TGridSize = 96> // TODO: values for TBlockSize and TGridSize
struct DeviceArrayForEach
{
  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_for_each_available_v, false)

  template <bool TIsOnHost, typename TTensorDest, typename... TTensorSrcs>
  TVALUE(bool, is_map_available_v, false)

  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v, false)

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_parallel_v, true)

  template <size_t TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors)
  {
    static_assert(std::is_same<TFunctor, void>::value, "Cannot use DeviceArrayForEach without CUDA");
  }

  TT_FOR_EACH_MAP_AND_COPY(__host__)
};

#endif

} // end of ns op

} // end of ns tensor
