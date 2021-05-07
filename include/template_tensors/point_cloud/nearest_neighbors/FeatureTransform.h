#include <metal.hpp>

namespace point_cloud {

namespace nearest_neighbors {

#define ThisType FeatureTransform<TNearestNeighbors, TFeatureTransform, TFeatureDistanceTransform>
template <typename TNearestNeighbors, typename TFeatureTransform, typename TFeatureDistanceTransform>
class FeatureTransform : public mem::HasMemoryType<ThisType, mem::memorytype_v<TNearestNeighbors>::value>
{
private:
  TNearestNeighbors m_nearest_neighbors;
  TFeatureTransform m_feature_transform;
  TFeatureDistanceTransform m_feature_distance_transform;

public:
  using Object = typename std::decay<TNearestNeighbors>::type::Object;

  HD_WARNING_DISABLE
  template <typename... TNearestNeighborsArgs, ENABLE_IF(std::is_constructible<TNearestNeighbors, TNearestNeighborsArgs&&...>::value)>
  __host__ __device__
  FeatureTransform(TFeatureTransform feature_transform, TFeatureDistanceTransform feature_distance_transform, TNearestNeighborsArgs&&... nearest_neighbors_args)
    : m_nearest_neighbors(util::forward<TNearestNeighborsArgs>(nearest_neighbors_args)...)
    , m_feature_transform(feature_transform)
    , m_feature_distance_transform(feature_distance_transform)
  {
  }

  HD_WARNING_DISABLE
  template <typename... TNearestNeighborsArgs, ENABLE_IF(std::is_constructible<TNearestNeighbors, TNearestNeighborsArgs&&...>::value
    && std::is_constructible<TFeatureDistanceTransform>::value && !std::is_same<TFeatureDistanceTransform, TFeatureTransform>::value)>
  __host__ __device__
  FeatureTransform(TFeatureTransform feature_transform, TNearestNeighborsArgs&&... nearest_neighbors_args)
    : FeatureTransform(feature_transform, TFeatureDistanceTransform(), util::forward<TNearestNeighborsArgs>(nearest_neighbors_args)...)
  {
  }

  HD_WARNING_DISABLE
  template <typename... TNearestNeighborsArgs, ENABLE_IF(std::is_constructible<TNearestNeighbors, TNearestNeighborsArgs&&...>::value
    && std::is_same<TFeatureDistanceTransform, TFeatureTransform>::value), bool TDummy = false>
  __host__ __device__
  FeatureTransform(TFeatureTransform feature_transform, TNearestNeighborsArgs&&... nearest_neighbors_args)
    : FeatureTransform(feature_transform, feature_transform, util::forward<TNearestNeighborsArgs>(nearest_neighbors_args)...)
  {
  }

  HD_WARNING_DISABLE
  template <typename... TNearestNeighborsArgs, ENABLE_IF(std::is_constructible<TNearestNeighbors, TNearestNeighborsArgs&&...>::value)>
  __host__ __device__
  FeatureTransform(TNearestNeighborsArgs&&... nearest_neighbors_args)
    : FeatureTransform(TFeatureTransform(), util::forward<TNearestNeighborsArgs>(nearest_neighbors_args)...)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  FeatureTransform(const FeatureTransform<TNearestNeighbors, TFeatureTransform, TFeatureDistanceTransform>& other)
    : m_nearest_neighbors(other.m_nearest_neighbors)
    , m_feature_transform(other.m_feature_transform)
    , m_feature_distance_transform(other.m_feature_distance_transform)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  FeatureTransform(FeatureTransform<TNearestNeighbors, TFeatureTransform, TFeatureDistanceTransform>&& other)
    : m_nearest_neighbors(util::move(other.m_nearest_neighbors))
    , m_feature_transform(util::move(other.m_feature_transform))
    , m_feature_distance_transform(util::move(other.m_feature_distance_transform))
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~FeatureTransform()
  {
  }

  HD_WARNING_DISABLE
  template <typename TOp, metal::int_ TRank, typename TScalar>
  __host__ __device__
  void for_each(TOp&& op, template_tensors::VectorXT<TScalar, TRank> center, template_tensors::VectorXT<TScalar, TRank> range)
  {
    m_nearest_neighbors.for_each(util::forward<TOp>(op), m_feature_transform(center), m_feature_distance_transform(range));
  }

  HD_WARNING_DISABLE
  template <typename TGetFeatures>
  __host__ __device__
  void update(TGetFeatures&& get_features)
  {
    m_nearest_neighbors.update(util::functor::compose(m_feature_transform, util::forward<TGetFeatures>(get_features)));
  }

#ifdef __CUDACC__
  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static auto toKernel(TThisType&& self)
  RETURN_AUTO(FeatureTransform<decltype(mem::toKernel(self.m_nearest_neighbors)), decltype(mem::toKernel(self.m_feature_transform)), decltype(mem::toKernel(self.m_feature_distance_transform))>
    (mem::toKernel(self.m_feature_transform), mem::toKernel(self.m_feature_distance_transform), mem::toKernel(self.m_nearest_neighbors))
  )
#endif
};
#undef ThisType

} // end of ns nearest_neighbors

} // end of ns point_cloud
