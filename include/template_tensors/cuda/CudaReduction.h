#pragma once

#include "CudaGrid.h"

#ifdef __CUDACC__

#include <type_traits>

namespace cuda {

namespace detail {

template <size_t TSizeLeft>
struct ShuffleDown
{
  __device__
  static void shuffle(void* dest, const void* src, size_t shuffle_offset)
  {
    using ShuffleType = typename std::conditional<TSizeLeft >= sizeof(long long), long long,
                        typename std::conditional<TSizeLeft >= sizeof(long), long,
                        typename std::conditional<TSizeLeft >= sizeof(int), int,
                        void
          >::type>::type>::type;

    ShuffleType* shuffle_dest = reinterpret_cast<ShuffleType*>(dest);
    const ShuffleType* shuffle_src = reinterpret_cast<const ShuffleType*>(src);
    *shuffle_dest = __shfl_down_sync(0xFFFFFFFF, *shuffle_src, shuffle_offset);
    ShuffleDown<TSizeLeft - sizeof(ShuffleType)>::shuffle(&shuffle_dest[1], &shuffle_src[1], shuffle_offset);
  }
};

template <>
struct ShuffleDown<0>
{
  __device__
  static void shuffle(void* dest, const void* src, size_t shuffle_offset)
  {
  }
};

template <size_t TOffset = cuda::WARP_SIZE / 2>
struct ShuffleDownAll
{
  template <typename TFunctor, typename T>
  __device__
  static void shuffle(TFunctor functor, T& value)
  {
    T shuffled_value;
    ShuffleDown<sizeof(T)>::shuffle(&shuffled_value, &value, TOffset);
    functor(value, shuffled_value);
    ShuffleDownAll<TOffset / 2>::shuffle(functor, value);
  }
};

template <>
struct ShuffleDownAll<0>
{
  template <typename TFunctor, typename T>
  __device__
  static void shuffle(TFunctor functor, T& value)
  {
  }
};

} // end of ns detail


/*!
 * Reduces the given value for all threads in the current warp using the given reduction operation.
 * The result is stored inside the warp's first thread's value.
 *
 * @param reduce_op the reduction operation
 * @param value the value to be reduced
 */
template <typename TFunctor, typename T>
__device__
void warp_reduce(TFunctor functor, T& value)
{
  detail::ShuffleDownAll<>::shuffle(functor, value);
}

/*!
 * Reduces the given value for all threads in the current block using the given reduction operation.
 * The result is stored inside the block's first thread's value.
 *
 * @param reduce_op the reduction operation
 * @param value the value to be reduced
 */
template <typename TReduceOperation, typename T1, typename T2>
__device__
void block_reduce(TReduceOperation&& reduce_op, T1& value, T2&& neutral_element)
{
  static_assert(cuda::grid::MAX_WARPS_PER_BLOCK <= cuda::WARP_SIZE, "Warp size too small");
  __shared__ typename std::decay<T1>::type shared[cuda::grid::MAX_WARPS_PER_BLOCK];

  size_t warp_lane = cuda::grid::warp_lane();
  size_t warp_id = cuda::grid::warp_id_in_block();

  warp_reduce(reduce_op, value);
  if (warp_lane == 0)
  {
    shared[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0)
  {
    value = warp_lane < cuda::grid::warp_num_in_block() ? shared[warp_lane] : neutral_element;
    warp_reduce(reduce_op, value);
  }
}

} // end of ns cuda

#endif
