#pragma once

#ifdef CEREAL_INCLUDED
#include <cereal/access.hpp>
#endif

#include <fstream>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <mutex>

#include <template_tensors/cuda/Cuda.h>
#include <template_tensors/util/Array.h>
#include <template_tensors/tmp/ValueSequence.h>
#include <template_tensors/tmp/TypeSequence.h>
#include <template_tensors/tmp/Reflection.h>
#include <template_tensors/util/Constexpr.h>
#include <template_tensors/util/Tuple.h>
#include <template_tensors/util/Assert.h>
#include <template_tensors/util/Math.h>
#include <template_tensors/util/Functor.h>
#include <template_tensors/util/Ptr.h>
#include <template_tensors/cuda/CudaMutex.h>
#include <template_tensors/cuda/CudaAtomic.h>
#include <template_tensors/util/Console.h>
#include <template_tensors/util/Dispatch.h>
#include <template_tensors/interface/std/SortByKey.h>
#include <template_tensors/interface/std/PrintChrono.h>

#include <template_tensors/for_each/Helper.h>
#include <template_tensors/for_each/Sequential.h>
#include <template_tensors/for_each/Auto.h>

#include <template_tensors/aggregator/Helper.h>
#include <template_tensors/aggregator/Constant.h>
#include <template_tensors/aggregator/Assign.h>
#include <template_tensors/aggregator/Multi.h>
#include <template_tensors/aggregator/MapInput.h>
#include <template_tensors/aggregator/MapOutput.h>
#include <template_tensors/aggregator/Filter.h>
#include <template_tensors/aggregator/MiscOps.h>

#include <template_tensors/iterate/StoreIterators.h>
#include <template_tensors/iterate/Adapt.h>
#include <template_tensors/iterate/Count.h>
#include <template_tensors/iterate/Transform.h>
#include <template_tensors/iterate/Filter.h>
#include <template_tensors/iterate/Util.h>

/*!
 * \defgroup MiscTensorOps Misc
 * \ingroup TensorOperations
 * \brief asd
 */

#include <template_tensors/tensor/helper/StaticDimsCoords.h>
#include <template_tensors/tensor/helper/StaticCompatibility.h>
#include <template_tensors/tensor/helper/DynamicDimsCoords.h>
#include <template_tensors/tensor/helper/DynamicCompatibility.h>

#include <template_tensors/tensor/helper/Concepts.h>
#include <template_tensors/tensor/helper/ElementType.h>
#include <template_tensors/tensor/helper/SubtypeHelpers.h>

#include <template_tensors/tensor/storage/indexstrategy/IndexStrategyHelper.h>
#include <template_tensors/tensor/storage/indexstrategy/ColMajor.h>
#include <template_tensors/tensor/storage/indexstrategy/RowMajor.h>
#include <template_tensors/tensor/storage/indexstrategy/SymmetricMatrixUpperTriangleRowMajor.h>
#include <template_tensors/tensor/storage/indexstrategy/SymmetricMatrixLowerTriangleRowMajor.h>
#include <template_tensors/tensor/storage/Typedefs.h>
#include <template_tensors/tensor/storage/Defines.h>
#include <template_tensors/tensor/storage/Ref.h>
#include <template_tensors/tensor/storage/IndexedPointer.h>
#include <template_tensors/tensor/storage/IndexedArray.h>

#include <template_tensors/tensor/base/StoreDimensions.h>
#include <template_tensors/tensor/base/Iterator.h>

#include <template_tensors/tensor/base/Dimensions.h>


#include <template_tensors/tensor/for_each/Helper.h>
#include <template_tensors/tensor/for_each/Local.h>
#include <template_tensors/tensor/transfer/StorageArrayCopier.h>



#include <template_tensors/tensor/storage/Singleton.h>
#include <template_tensors/tensor/shape_ops/Broadcast.h>
#include <template_tensors/tensor/shape_ops/Partial.h>

#include <template_tensors/tensor/mask/BooleanMasked.h>
#include <template_tensors/tensor/mask/IteratorMasked.h>

#include <template_tensors/tensor/base/Maskable.h>
#include <template_tensors/tensor/base/TensorBase.h>

#include <template_tensors/cuda/CudaGrid.h>
#include <template_tensors/cuda/CudaRandom.h>

#include <template_tensors/tensor/linear_algebra/Elwise.h>
#include <template_tensors/tensor/shape_ops/Concat.h>

#include <template_tensors/aggregator/Elwise.h>
#include <template_tensors/aggregator/HistogramPartial.h>
#include <template_tensors/aggregator/HistogramTotal.h>

#include <template_tensors/aggregator/StatisticsOps.h>

#include <template_tensors/tensor/shape_ops/Reduce.h>
#include <template_tensors/tensor/util/Ordering.h>

#include <template_tensors/concurrency/locking/Mutex.h>
#include <template_tensors/concurrency/locking/UniqueLock.h>
#include <template_tensors/concurrency/locking/UniqueTryLock.h>
#include <template_tensors/concurrency/locking/ConditionalUniqueTryLock.h>
#include <template_tensors/concurrency/atomic/Functor.h>
#include <template_tensors/concurrency/atomic/TryVariable.h>
#include <template_tensors/concurrency/atomic/WaitVariable.h>
#include <template_tensors/concurrency/atomic/Types.h>
#include <template_tensors/concurrency/atomic/ops/Void.h>
#include <template_tensors/concurrency/atomic/ops/TryLock.h>
#include <template_tensors/concurrency/atomic/ops/WaitLock.h>

#include <template_tensors/tensor/linear_algebra/MiscOps.h>
#include <template_tensors/tensor/transfer/Eval.h>
#include <template_tensors/tensor/util/Dummy.h>
#include <template_tensors/tensor/linear_algebra/CrossProduct.h>
#include <template_tensors/tensor/linear_algebra/MatrixProduct.h>
#include <template_tensors/tensor/linear_algebra/IdentityMatrix.h>
#include <template_tensors/tensor/linear_algebra/UnitVector.h>
#include <template_tensors/tensor/util/ElementSupplier.h>
#include <template_tensors/tensor/linear_algebra/RotationMatrix.h>
#include <template_tensors/tensor/linear_algebra/Geo.h>

#include <template_tensors/tensor/shape_ops/HeadTail.h>
#include <template_tensors/tensor/shape_ops/Flip.h>
#include <template_tensors/tensor/shape_ops/Transpose.h>
#include <template_tensors/tensor/linear_algebra/GaussianMethod.h>
#include <template_tensors/tensor/linear_algebra/solver/Helper.h>
#include <template_tensors/tensor/linear_algebra/solver/Gaussian.h>
#include <template_tensors/tensor/linear_algebra/matrix_inverse/Helper.h>
#include <template_tensors/tensor/linear_algebra/matrix_inverse/Gaussian.h>
#include <template_tensors/tensor/linear_algebra/matrix_inverse/ClosedForm.h>
#include <template_tensors/tensor/linear_algebra/matrix_inverse/Auto.h>
#include <template_tensors/tensor/for_each/DeviceArray.h>
#include <template_tensors/tensor/for_each/LocalArray.h>
#include <template_tensors/tensor/for_each/Device.h>
#include <template_tensors/tensor/storage/indexstrategy/Stride.h>
#include <template_tensors/tensor/storage/indexstrategy/MortonForLoop.h>
#include <template_tensors/tensor/storage/indexstrategy/MortonDivideAndConquer.h>
#include <template_tensors/tensor/transfer/AssignToTempMemCopyToDestCopier.h>
#include <template_tensors/tensor/transfer/MemCopyToTempAssignToDestCopier.h>
#include <template_tensors/tensor/transfer/AssignToTempMemCopyToTempAssignToDestCopier.h>
#include <template_tensors/tensor/for_each/Auto.h>
#include <template_tensors/tensor/transfer/AutoCopier.h>
#include <template_tensors/tensor/random/Random.h>

#include <template_tensors/tensor/shape_ops/Total.h>
#include <template_tensors/tensor/shape_ops/Dilate.h>
#include <template_tensors/tensor/shape_ops/Pad.h>
#include <template_tensors/tensor/shape_ops/Repeat.h>
#include <template_tensors/tensor/linear_algebra/Convolution.h>
#include <template_tensors/tensor/linear_algebra/Quaternion.h>
#include <template_tensors/tensor/util/ArgComp.h>
#include <template_tensors/tensor/util/Tuple.h>
#include <template_tensors/tensor/linear_algebra/Diag.h>
#include <template_tensors/tensor/linear_algebra/homogeneous/Vector.h>
#include <template_tensors/tensor/linear_algebra/homogeneous/Translation.h>
#include <template_tensors/tensor/linear_algebra/homogeneous/Rotation.h>

#include <template_tensors/tensor/string/StreamOutput.h>
#include <template_tensors/tensor/string/String.h>
#include <template_tensors/tensor/string/ToString.h>
#include <template_tensors/tensor/string/StringStream.h>

#include <template_tensors/interface/std/String.h>
#include <template_tensors/tensor/string/PrintStream.h>

#include <template_tensors/tensor/sparse/Iterable.h>

#include <template_tensors/interface/openmp/ForEach.h>
#include <template_tensors/interface/tbb/ForEach.h>

#include <template_tensors/interface/cv/Mat.h>
#include <template_tensors/interface/cv/Debug.h>
#include <template_tensors/interface/cv/ReadWrite.h>

#include <template_tensors/interface/eigen/Mat.h>
#include <template_tensors/interface/eigen/Decomposition.h>

#include <template_tensors/geometry/transform/Rotation.h>
#include <template_tensors/geometry/transform/AxisAngle.h>
#include <template_tensors/geometry/transform/Translation.h>
#include <template_tensors/geometry/transform/Rigid.h>
#include <template_tensors/geometry/transform/ScaledRigid.h>
#include <template_tensors/geometry/projection/Pinhole.h>
#include <template_tensors/geometry/projection/Orthographic.h>
#include <template_tensors/geometry/render/DeviceMutexRasterizer.h>
#include <template_tensors/geometry/render/DeviceRasterizer.h>
#include <template_tensors/geometry/render/Shader.h>
#include <template_tensors/geometry/render/primitives/SurfaceSplat.h>
#include <template_tensors/geometry/render/primitives/ScreenSplat.h>
#include <template_tensors/geometry/render/primitives/Triangle.h>
#include <template_tensors/geometry/Generation.h>
#include <template_tensors/geometry/RegionGrowing.h>

#include <template_tensors/file/Binvox.h>
#include <template_tensors/file/Flo.h>
#include <template_tensors/file/colmap/Metadata.h>

#include <template_tensors/numeric/LogProb.h>
#include <template_tensors/numeric/FixedFloatingPoint.h>

#include <template_tensors/interface/cublas/Gemm.h>
#include <template_tensors/interface/cudnn/ConvolutionForward.h>
#include <template_tensors/interface/std/Vector.h>
#include <template_tensors/interface/thrust/Vector.h>
#include <template_tensors/interface/tf/Tf.h>
#include <template_tensors/interface/tf2/Tf2.h>
#include <template_tensors/interface/freeimage/TensorInterface.h>
#include <template_tensors/interface/sensor_msgs/SensorMsgs.h>
#include <template_tensors/interface/pcl/PointCloud.h>
#include <template_tensors/interface/orocos_kdl/Kdl.h>
#include <template_tensors/interface/tinyply/Tinyply.h>
#include <template_tensors/interface/cnpy/Cnpy.h>
#include <template_tensors/interface/boost/python/Helper.h>
#include <template_tensors/interface/boost/python/numpy/NdArray.h>
#include <template_tensors/interface/boost/python/numpy/Dispatch.h>
#include <template_tensors/interface/boost/python/cupy/NdArray.h>
#include <template_tensors/interface/soil/Soil.h>
#include <template_tensors/interface/dlpack/SafeWrapper.h>
#include <template_tensors/interface/dlpack/PyCapsule.h>
#include <template_tensors/interface/dlpack/Tensor.h>
#include <template_tensors/interface/dlpack/Dispatch.h>
#include <template_tensors/interface/boost/python/tensorflow/EagerTensor.h>
#include <template_tensors/interface/boost/python/tensorflow/Dispatch.h>
#include <template_tensors/interface/nvof/NVOF.h>
#include <template_tensors/interface/boost/python/Tensor.h>
#include <template_tensors/interface/cppflow/CppFlow.h>

#include <template_tensors/field/CoordinateTransform.h>
#include <template_tensors/field/Interpolation.h>
#include <template_tensors/field/FromSupplier.h>
#include <template_tensors/field/Map.h>
#include <template_tensors/field/Differentiate.h>
#include <template_tensors/field/Wrap.h>

#include <template_tensors/point_cloud/nearest_neighbors/NearestNeighbors.h>
#include <template_tensors/point_cloud/nearest_neighbors/FeatureTransform.h>
#include <template_tensors/point_cloud/nearest_neighbors/GridSearch.h>
#include <template_tensors/point_cloud/SortGrid.h>

#include <template_tensors/file/slippy_maps/TileLoader.h>

#undef ELEMENT_ACCESS_PREFIX

// TODO: rename prefix TENSOR_
#undef ELEMENT_ACCESS
#undef ELEMENT_ACCESS_HOST
#undef ELEMENT_ACCESS_DEVICE

#ifndef TEMPLATE_TENSORS_DONT_ALIAS_NAMESPACE_AS_TT
namespace tt = template_tensors;
#endif
