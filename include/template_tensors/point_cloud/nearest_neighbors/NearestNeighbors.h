namespace point_cloud {

namespace nearest_neighbors {

#define ThisType NearestNeighbors<TNearestNeighbors, TGetFeatures, TDistanceMetric>
template <typename TNearestNeighbors, typename TGetFeatures, typename TDistanceMetric>
class NearestNeighbors : public mem::HasMemoryType<ThisType, mem::memorytype_v<TNearestNeighbors>::value>
{
private:
  TNearestNeighbors m_nearest_neighbors;
  TGetFeatures m_get_features;
  TDistanceMetric m_distance_metric;

public:
  using Object = typename std::decay<TNearestNeighbors>::type::Object;

  HD_WARNING_DISABLE
  template <typename... TNearestNeighborsArgs, ENABLE_IF(std::is_constructible<TNearestNeighbors, TNearestNeighborsArgs&&...>::value)>
  __host__ __device__
  NearestNeighbors(TGetFeatures get_features, TDistanceMetric distance_metric, TNearestNeighborsArgs&&... nearest_neighbors_args)
    : m_nearest_neighbors(util::forward<TNearestNeighborsArgs>(nearest_neighbors_args)...)
    , m_get_features(get_features)
    , m_distance_metric(distance_metric)
  {
  }

  HD_WARNING_DISABLE
  template <typename... TNearestNeighborsArgs, ENABLE_IF(std::is_constructible<TNearestNeighbors, TNearestNeighborsArgs&&...>::value)>
  __host__ __device__
  NearestNeighbors(TGetFeatures get_features, TNearestNeighborsArgs&&... nearest_neighbors_args)
    : NearestNeighbors(get_features, TDistanceMetric(), util::forward<TNearestNeighborsArgs>(nearest_neighbors_args)...)
  {
  }

  HD_WARNING_DISABLE
  template <typename... TNearestNeighborsArgs, ENABLE_IF(std::is_constructible<TNearestNeighbors, TNearestNeighborsArgs&&...>::value)>
  __host__ __device__
  NearestNeighbors(TNearestNeighborsArgs&&... nearest_neighbors_args)
    : NearestNeighbors(TGetFeatures(), util::forward<TNearestNeighborsArgs>(nearest_neighbors_args)...)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~NearestNeighbors()
  {
  }

  HD_WARNING_DISABLE
  template <typename TOp, size_t TRank, typename TScalar, typename TPredicate>
  __host__ __device__
  void for_each(TOp&& op, template_tensors::VectorXT<TScalar, TRank> center, TScalar max_distance)
  {
    TScalar max_metric_distance = m_distance_metric(max_distance);
    m_nearest_neighbors.for_each([&](Object& object){
      TScalar metric_distance = m_distance_metric(m_get_features(object), center);
      if (metric_distance <= max_metric_distance)
      {
        op(object, metric_distance);
      }
    }, center, max_distance);
  }

  HD_WARNING_DISABLE
  template <size_t TRank, typename TScalar, typename TPredicate = util::functor::True>
  __host__ __device__
  Object* nearest(template_tensors::VectorXT<TScalar, TRank> center, TScalar max_distance, TPredicate&& pred = TPredicate())
  {
    Object* result = nullptr;
    TScalar min_metric_distance = math::consts<TScalar>::INF;

    m_nearest_neighbors.for_each([&](Object& object) mutable {
      TScalar metric_distance = m_distance_metric(m_get_features(object), center);
      if (metric_distance <= min_metric_distance && pred(object))
      { // TODO: early stop when starting grid from the middle?
        result = &object;
        min_metric_distance = metric_distance;
      }
    }, center, template_tensors::VectorXT<TScalar, TRank>(max_distance));

    if (min_metric_distance > m_distance_metric(max_distance))
    {
      result = nullptr;
    }
    return result;
  }

  HD_WARNING_DISABLE
  __host__ __device__
  void update()
  {
    m_nearest_neighbors.update(m_get_features);
  }

#ifdef __CUDACC__
  template <typename TThisType>
  __host__
  static auto toKernel(TThisType&& self)
  RETURN_AUTO(
    NearestNeighbors<decltype(mem::toKernel(self.m_nearest_neighbors)), decltype(mem::toKernel(self.m_get_features)), decltype(mem::toKernel(self.m_distance_metric))>
      (mem::toKernel(self.m_get_features), mem::toKernel(self.m_distance_metric), mem::toKernel(self.m_nearest_neighbors))
  )
#endif
};
#undef ThisType

} // end of ns nearest_neighbors

} // end of ns point_cloud