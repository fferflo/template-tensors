#include <metal.hpp>

namespace template_tensors {

namespace op {

#ifdef __CUDACC__
namespace detail {

template <metal::int_ TStep, typename TIndexStrategy, metal::int_ TCoordsRank, typename TFunctor, typename... TElementTypes>
__global__
void kernel_for_each_array_element_with_coords(TFunctor func, TIndexStrategy index_strategy, VectorXs<TCoordsRank> dims, size_t size, TElementTypes*... arrays)
{
  for (size_t index = cuda::grid::thread_id_in_grid<1>()(); index < size; index += TStep)
  {
    func(index_strategy.fromIndex(index, dims), arrays[index]...);
  }
}

template <metal::int_ TCoordsRank>
struct DeviceArrayForEachHelper
{
  template <metal::int_ TBlockSize, metal::int_ TGridSize, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, size_t size, TTensorTypes&&... tensors)
  {
    using IndexStrategy = indexstrategy_t<metal::front<metal::list<TTensorTypes...>>>;

    dim3 block, grid;
    block = dim3(TBlockSize);
    grid = dim3(TGridSize);
    TT_CUDA_SAFE_CALL((detail::kernel_for_each_array_element_with_coords<TBlockSize * TGridSize, IndexStrategy, TCoordsRank><<<grid, block>>>(
      mem::toKernel(std::forward<TFunctor>(func)),
      util::first(std::forward<TTensorTypes>(tensors)...).getIndexStrategy(),
      util::first(std::forward<TTensorTypes>(tensors)...).template dims<TCoordsRank>(),
      size,
      tensors.data()...
    )));
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
};

template <metal::int_ TStep, typename TFunctor, typename... TElementTypes>
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
  template <metal::int_ TBlockSize, metal::int_ TGridSize, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, size_t size, TTensorTypes&&... tensors)
  {
    dim3 block, grid;
    block = dim3(TBlockSize);
    grid = dim3(TGridSize);
    TT_CUDA_SAFE_CALL((detail::kernel_for_each_array_element<TBlockSize * TGridSize><<<grid, block>>>(
      mem::toKernel(std::forward<TFunctor>(func)),
      size,
      tensors.data()...
    )));
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
};



} // end of ns detail

template <metal::int_ TBlockSize = 128, metal::int_ TGridSize = 96> // TODO: values for TBlockSize and TGridSize
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

  template <metal::int_ TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors)
  {
    static_assert(sizeof...(tensors) > 0, "No tensors given");
    ASSERT(math::eq(tensors.getIndexStrategy()...), "Storages must have the same indexing strategy");
    ASSERT(template_tensors::all(template_tensors::elwise(math::functor::eq(), tensors.dims()...)), "Incompatible runtime dimensions");
    static_assert(are_compatible_dimseqs_v<dimseq_t<TTensorTypes>...>::value, "Incompatible static dimensions");
    static_assert(metal::same<indexstrategy_t<TTensorTypes>...>::value, "Storages must have the same indexing strategy");

    detail::DeviceArrayForEachHelper<TCoordsRank>::template for_each<TBlockSize, TGridSize>(
      std::forward<TFunctor>(func),
      util::first(std::forward<TTensorTypes>(tensors)...).size(),
      std::forward<TTensorTypes>(tensors)...
    );
  }

  TT_FOR_EACH_MAP_AND_COPY(__host__)
};

#else

template <metal::int_ TBlockSize = 128, metal::int_ TGridSize = 96> // TODO: values for TBlockSize and TGridSize
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

  template <metal::int_ TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors)
  {
    static_assert(std::is_same<TFunctor, void>::value, "Cannot use DeviceArrayForEach without CUDA");
  }

  TT_FOR_EACH_MAP_AND_COPY(__host__)
};

#endif

} // end of ns op

} // end of ns template_tensors
