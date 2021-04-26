#pragma once

#include "Cuda.h"

#include <template_tensors/util/Assert.h>

#ifdef __CUDACC__

namespace cuda {

/*!
 * The number of threads inside a single CUDA warp.
 */
static const __constant__ size_t WARP_SIZE = 32;

namespace grid {

/*!
 * The x-coordinate of the maximum 1-dimensional CUDA block size.
 */
static const __constant__ size_t MAX_BLOCK_SIZE_1D_X = 512;
/*!
 * The maximum 1-dimensional CUDA block size.
 */
static const dim3 MAX_BLOCK_SIZE_1D = dim3(MAX_BLOCK_SIZE_1D_X);

/*!
 * The x-coordinate of the maximum 2-dimensional CUDA block size.
 */
static const __constant__ size_t MAX_BLOCK_SIZE_2D_X = 32;
/*!
 * The y-coordinate of the maximum 2-dimensional CUDA block size.
 */
static const __constant__ size_t MAX_BLOCK_SIZE_2D_Y = 16;
/*!
 * The maximum 2-dimensional CUDA block size.
 */
static const dim3 MAX_BLOCK_SIZE_2D = dim3(MAX_BLOCK_SIZE_2D_X, MAX_BLOCK_SIZE_2D_Y);
/*!
 * The maximum possible number of warps per CUDA block.
 */
static const __constant__ size_t MAX_WARPS_PER_BLOCK = MAX_BLOCK_SIZE_1D_X / WARP_SIZE;



/*!
 * Returns the TBlockRank-dimensional thread id inside the total grid.
 *
 * @tparam TBlockRank the number of dimensions of the grid
 * @return the thread id inside the total grid
 */
template <size_t TBlockRank>
__device__
inline template_tensors::VectorXs<TBlockRank> thread_id_in_grid();

template <>
__device__
inline template_tensors::Vector1s thread_id_in_grid<1>()
{
  return template_tensors::Vector1s(blockIdx.x * blockDim.x + threadIdx.x);
}

template <>
__device__
inline template_tensors::Vector2s thread_id_in_grid<2>()
{
  return template_tensors::Vector2s(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

template <>
__device__
inline template_tensors::Vector3s thread_id_in_grid<3>()
{
  return template_tensors::Vector3s(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
}




/*!
 * Converts the given vector to CUDA dim3 representation.
 *
 * @param dims the vector to convert
 * @return the given vector in CUDA dim3 representation
 */
template <typename TVectorType, ENABLE_IF(template_tensors::is_vector_v<TVectorType>::value)>
__host__ __device__
dim3 to_dim3(const TVectorType& dims)
{
  const size_t BLOCK_RANK = template_tensors::rows_v<TVectorType>::value;
  static_assert(0 <= BLOCK_RANK && BLOCK_RANK <= 3, "Invalid block rank");
  dim3 result;
  if (BLOCK_RANK > 0)
  {
    result.x = dims(0);
  }
  if (BLOCK_RANK > 1)
  {
    result.y = dims(1);
  }
  if (BLOCK_RANK > 2)
  {
    result.z = dims(2);
  }
  return result;
}

/*!
 * Creates a CUDA dim3 object extended only in the x-dimension.
 *
 * @param dims the x-component
 * @return the CUDA dim3 object
 */
__host__ __device__
inline dim3 to_dim3(size_t dims)
{
  dim3 result;
  result.x = dims;
  return result;
}

/*!
 * Returns the result of ceil(total / block) calculated element-wise.
 *
 * @param total the total size
 * @param block the block size
 * @return ceil(total / block)
 */
__host__ __device__
inline dim3 div_up(dim3 total, dim3 block)
{
  return dim3((total.x + block.x - 1) / block.x, (total.y + block.y - 1) / block.y, (total.z + block.z - 1) / block.z);
}

/*!
 * Returns the result of ceil(total / block) calculated element-wise.
 *
 * @param total the total size
 * @param block the block size
 * @return ceil(total / block)
 */
template <typename TVectorType, ENABLE_IF(template_tensors::is_vector_v<TVectorType>::value)>
__host__ __device__
dim3 div_up(const TVectorType& total, dim3 block)
{
  return div_up(to_dim3(total), block);
}

/*!
 * Returns the result of ceil(total / block) calculated element-wise.
 *
 * @param total the total size
 * @param block the block size
 * @return ceil(total / block)
 */
__host__ __device__
inline dim3 div_up(size_t total, size_t block)
{
  return to_dim3((total + block + 1) / block);
}

/*!
 * Returns the total number of blocks inside the CUDA grid.
 *
 * @return the total number of blocks inside the CUDA grid
 */
__device__
inline size_t block_num()
{
  return gridDim.x * gridDim.y * gridDim.z;
}

/*!
 * Returns the number of threads inside the CUDA block.
 *
 * @return the number of threads inside the CUDA block
 */
__device__
inline size_t thread_num_per_block()
{
  return blockDim.x * blockDim.y * blockDim.z;
}

/*!
 * Returns the number of threads inside the CUDA grid.
 *
 * @return the number of threads inside the CUDA grid
 */
__device__
inline size_t thread_num_in_grid()
{
  return block_num() * thread_num_per_block();
}

/*!
 * Returns the lane of the current thread inside the warp.
 *
 * @return the lane of the current thread inside the warp
 */
__device__
inline size_t warp_lane()
{
  return threadIdx.x % WARP_SIZE;
}

/*!
 * Returns the warp id of the current thread inside the block.
 *
 * @return the warp id of the current thread inside the block
 */
__device__
inline size_t warp_id_in_block()
{
  return threadIdx.x / WARP_SIZE;
}

/*!
 * Returns the number of warps inside the block.
 *
 * @return the number of warps inside the block
 */
__device__
inline size_t warp_num_in_block()
{
  return thread_num_per_block() / cuda::WARP_SIZE;
}



namespace flatten {

// TODO: template block parameters, or replace these with index strategy call?
/*!
 * Returns the flattened 1d thread id inside the thread's 3d block.
 *
 * @return the flattened thread id
 */
__device__
inline size_t thread_id_in_block()
{
  return threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

/*!
 * Returns the flattened 1d block id inside the 3d grid.
 *
 * @return the flattened thread id
 */
__device__
inline size_t block_id_in_grid()
{
  return blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
}

/*!
 * Returns the flattened 1d thread id inside the 3d grid.
 *
 * @return the flattened thread id
 */
__device__
inline size_t thread_id_in_grid()
{
  return thread_num_per_block() * block_id_in_grid() + thread_id_in_block();
}

} // end of ns flatten

} // end of ns grid

} // end of ns cuda

#endif
