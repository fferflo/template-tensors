#include <metal.hpp>

namespace point_cloud {

namespace nearest_neighbors {

namespace detail {

template <typename TGrid>
class GridSearch : public mem::HasMemoryType<GridSearch<TGrid>, mem::memorytype_v<TGrid>::value>
{
private:
  TGrid m_grid;

public:
  using Object = typename std::remove_reference<decltype(*std::declval<TGrid>()().begin())>::type;

  HD_WARNING_DISABLE
  template <typename... TArgs, ENABLE_IF(std::is_constructible<TGrid, TArgs&&...>::value)>
  __host__ __device__
  GridSearch(TArgs&&... args)
    : m_grid(util::forward<TArgs>(args)...)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  GridSearch(const GridSearch<TGrid>& other)
    : m_grid(other.m_grid)
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  GridSearch(GridSearch<TGrid>&& other)
    : m_grid(util::move(other.m_grid))
  {
  }

  HD_WARNING_DISABLE
  __host__ __device__
  ~GridSearch()
  {
  }

  HD_WARNING_DISABLE
  template <typename TGetFeatures>
  __host__ __device__
  void update(TGetFeatures&& get_features)
  {
    m_grid.update(util::functor::compose(template_tensors::functor::static_cast_to<size_t>(), util::forward<TGetFeatures>(get_features)));
  }

  HD_WARNING_DISABLE
  template <typename TOp, metal::int_ TRank, typename TScalar>
  __host__ __device__
  void for_each(TOp&& op, template_tensors::VectorXT<TScalar, TRank> center, template_tensors::VectorXT<TScalar, TRank> range)
  {
    // TODO: search all quad cells vs checking cell distance before
    template_tensors::VectorXs<TRank> min = template_tensors::max(template_tensors::static_cast_to<size_t>(center), range) - range;
    template_tensors::VectorXs<TRank> max = template_tensors::min(template_tensors::static_cast_to<size_t>(center + range) + 1, m_grid.template dims<TRank>());
    template_tensors::VectorXs<TRank> size = max - min;
    template_tensors::for_each(util::functor::for_each(util::forward<TOp>(op)), template_tensors::head(template_tensors::offset(m_grid, min), size));
  }

#ifdef __CUDACC__
  template <typename TGridSearch>
  __host__
  static auto toKernel(TGridSearch&& grid)
  RETURN_AUTO(
    GridSearch<decltype(mem::toKernel(grid.m_grid))>(mem::toKernel(grid.m_grid))
  )
#endif
};

} // end of ns detail

template <typename TGrid, typename TGetFeatures, typename TDistanceMetric, typename TFeatureTransform, typename TFeatureDistanceTransform = TFeatureTransform>
using FeatureTransformedGridSearch = NearestNeighbors<FeatureTransform<detail::GridSearch<TGrid>, TFeatureTransform, TFeatureDistanceTransform>, TGetFeatures, TDistanceMetric>;

template <typename TGrid, typename TGetFeatures, typename TDistanceMetric>
using GridSearch = NearestNeighbors<detail::GridSearch<TGrid>, TGetFeatures, TDistanceMetric>;

} // end of ns nearest_neighbors

} // end of ns point_cloud
