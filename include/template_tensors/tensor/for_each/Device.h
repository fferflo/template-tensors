#include <metal.hpp>

namespace template_tensors {

namespace op {

#ifdef __CUDACC__
namespace detail {

template <typename TKernelIndexStrategy, metal::int_ TStep, metal::int_ TKernelCoordsRank, metal::int_ TCoordsRank, typename TFunctor, typename... TTensorTypes>
__global__
void kernel_for_each_element_with_coords(TFunctor func, size_t size, VectorXs<TKernelCoordsRank> kernel_dims, VectorXs<TCoordsRank> dims, TTensorTypes... tensors)
{
  for (size_t index = cuda::grid::thread_id_in_grid<1>()(); index < size; index += TStep)
  {
    VectorXs<TKernelCoordsRank> kernel_coords = TKernelIndexStrategy().fromIndex(index, kernel_dims);
    func(TKernelIndexStrategy().fromIndex(index, dims), tensors(kernel_coords)...);
  }
}
// TODO: factor out DeviceForEach for_each class out of tensor folder
template <metal::int_ TCoordsRank>
struct DeviceForEachHelper
{
  template <typename TKernelIndexStrategy, metal::int_ TBlockSize, metal::int_ TGridSize, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor func, size_t size, TTensorTypes&&... tensors)
  {
    static const metal::int_ KERNEL_COORDS_RANK = non_trivial_dimensions_num_v<metal::front<metal::list<TTensorTypes...>>>::value;

    dim3 block, grid;
    block = dim3(TBlockSize);
    grid = dim3(TGridSize);
    TT_CUDA_SAFE_CALL((detail::kernel_for_each_element_with_coords<TKernelIndexStrategy, TBlockSize * TGridSize, KERNEL_COORDS_RANK, TCoordsRank><<<grid, block>>>(
      mem::toKernel(func),
      size,
      util::first(util::forward<TTensorTypes>(tensors)...).template dims<KERNEL_COORDS_RANK>(),
      util::first(util::forward<TTensorTypes>(tensors)...).template dims<TCoordsRank>(),
      mem::toKernel(tensors)...
    )));
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
};

template <typename TKernelIndexStrategy, metal::int_ TStep, metal::int_ TKernelCoordsRank, typename TFunctor, typename... TTensorTypes>
__global__
void kernel_for_each_element(TFunctor func, size_t size, VectorXs<TKernelCoordsRank> kernel_dims, TTensorTypes... tensors)
{
  for (size_t index = cuda::grid::thread_id_in_grid<1>()(); index < size; index += TStep)
  {
    VectorXs<TKernelCoordsRank> kernel_coords = TKernelIndexStrategy().fromIndex(index, kernel_dims);
    func(tensors(kernel_coords)...);
  }
}

template <>
struct DeviceForEachHelper<DYN>
{
  template <typename TKernelIndexStrategy, metal::int_ TBlockSize, metal::int_ TGridSize, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, size_t size, TTensorTypes&&... tensors)
  {
    static const metal::int_ KERNEL_COORDS_RANK = non_trivial_dimensions_num_v<metal::front<metal::list<TTensorTypes...>>>::value;

    dim3 block, grid;
    block = dim3(TBlockSize);
    grid = dim3(TGridSize);
    TT_CUDA_SAFE_CALL((detail::kernel_for_each_element<TKernelIndexStrategy, TBlockSize * TGridSize, KERNEL_COORDS_RANK><<<grid, block>>>(
      mem::toKernel(util::forward<TFunctor>(func)),
      size,
      util::first(util::forward<TTensorTypes>(tensors)...).template dims<KERNEL_COORDS_RANK>(),
      mem::toKernel(tensors)...
    )));
    TT_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
};

} // end of ns detail

template <typename TKernelIndexStrategy = template_tensors::ColMajor, metal::int_ TBlockSize = 128, metal::int_ TGridSize = 96> // TODO: values for TKernelIndexStrategy,  TBlockSize and TGridSize
struct DeviceForEach
{
  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_for_each_available_v,
       TIsOnHost
    && math::lor(mem::isOnDevice<mem::memorytype_v<TTensorTypes>::value, TIsOnHost>()...)
  )

  template <bool TIsOnHost, typename TTensorDest, typename... TTensorSrcs>
  TVALUE(bool, is_map_available_v,
       TIsOnHost
    && mem::memorytype_v<TTensorDest>::value == mem::DEVICE
    && math::land((mem::memorytype_v<TTensorSrcs>::value != mem::HOST)...)
  )

  template <bool TIsOnHost, typename TTensorDest, typename TTensorSrc>
  TVALUE(bool, is_copy_available_v,
       TIsOnHost
    && mem::memorytype_v<TTensorDest>::value == mem::DEVICE
    && mem::memorytype_v<TTensorSrc>::value != mem::HOST
  )

  template <bool TIsOnHost, typename... TTensorTypes>
  TVALUE(bool, is_parallel_v, true)

  template <metal::int_ TCoordsRank = DYN, typename TFunctor, typename... TTensorTypes>
  __host__
  static void for_each(TFunctor&& func, TTensorTypes&&... tensors)
  {
    ASSERT(template_tensors::all(template_tensors::elwise(math::functor::eq(), tensors.dims()...)), "Incompatible runtime dimensions");
    static_assert(sizeof...(tensors) > 0, "No tensors given");
    static_assert(are_compatible_dimseqs_v<dimseq_t<TTensorTypes>...>::value, "Incompatible static dimensions");

    detail::DeviceForEachHelper<TCoordsRank>::template for_each<TKernelIndexStrategy, TBlockSize, TGridSize>(
      util::forward<TFunctor>(func),
      template_tensors::prod(util::first(util::forward<TTensorTypes>(tensors)...).dims()),
      util::forward<TTensorTypes>(tensors)...
    );
  }

  TT_FOR_EACH_MAP_AND_COPY(__host__)
};

#else

template <typename TKernelIndexStrategy = template_tensors::ColMajor, metal::int_ TBlockSize = 128, metal::int_ TGridSize = 96>  // TODO: values for TKernelIndexStrategy, TBlockSize and TGridSize
struct DeviceForEach
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
    static_assert(std::is_same<TFunctor, void>::value, "Cannot use DeviceForEach without CUDA");
  }

  TT_FOR_EACH_MAP_AND_COPY(__host__)
};

#endif

} // end of ns op

} // end of ns template_tensors
