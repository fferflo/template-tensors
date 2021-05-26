#if PCL_INCLUDED

// These includes are sometimes missing in pcl
#include <boost/numeric/conversion/cast.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <pcl/point_cloud.h>

namespace template_tensors {

namespace detail {

template <bool TOrganized>
struct PclElementAccess;

template <>
struct PclElementAccess<true>
{
  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.getPointCloud().at(getNthCoordinate<0>(std::forward<TCoordArgTypes>(coords)...), getNthCoordinate<1>(std::forward<TCoordArgTypes>(coords)...))
  )
};

template <>
struct PclElementAccess<false>
{
  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.getPointCloud().at(getNthCoordinate<0>(std::forward<TCoordArgTypes>(coords)...))
  )
};

} // end of ns detail

#define ThisType FromPCLWrapperTensor<TPointCloud>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::HOST, \
                                        template_tensors::DimSeq<DYN, TOrganized ? DYN : (metal::int_) 1> \
                              >
template <typename TPointCloud, bool TOrganized = false>
class FromPCLWrapperTensor : public SuperType
{
public:
  __host__
  FromPCLWrapperTensor(TPointCloud point_cloud)
    : SuperType(static_cast<dim_t>(point_cloud.width), static_cast<dim_t>(point_cloud.height))
    , m_point_cloud(point_cloud)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(detail::PclElementAccess<TOrganized>::getElement)

  template <metal::int_ TIndex>
  __host__
  dim_t getDynDim() const
  {
    return TIndex == 0 ? m_point_cloud.width : TIndex == 1 ? m_point_cloud.height : 1;
  }

  __host__
  dim_t getDynDim(size_t index) const
  {
    switch (index)
    {
      case 0: return m_point_cloud.width;
      case 1: return m_point_cloud.height;
      default: return 1;
    }
  }

  TPointCloud& getPointCloud()
  {
    return m_point_cloud;
  }

  const TPointCloud& getPointCloud() const
  {
    return m_point_cloud;
  }

private:
  TPointCloud m_point_cloud;
};
#undef SuperType
#undef ThisType

template <typename TPointType>
__host__
FromPCLWrapperTensor<pcl::PointCloud<TPointType>&> fromPcl(pcl::PointCloud<TPointType>& pcl)
{
  return FromPCLWrapperTensor<pcl::PointCloud<TPointType>&>(pcl);
}

template <typename TPointType>
__host__
FromPCLWrapperTensor<const pcl::PointCloud<TPointType>&> fromPcl(const pcl::PointCloud<TPointType>& pcl)
{
  return FromPCLWrapperTensor<const pcl::PointCloud<TPointType>&>(pcl);
}

template <typename TPointType>
__host__
FromPCLWrapperTensor<pcl::PointCloud<TPointType>> fromPcl(pcl::PointCloud<TPointType>&& pcl)
{
  return FromPCLWrapperTensor<pcl::PointCloud<TPointType>>(pcl);
}

template <typename TPointType>
__host__
FromPCLWrapperTensor<pcl::PointCloud<TPointType>> fromPcl(const pcl::PointCloud<TPointType>&& pcl)
{
  return FromPCLWrapperTensor<pcl::PointCloud<TPointType>>(pcl);
}

} // end of ns template_tensors

#endif
