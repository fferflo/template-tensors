#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

template <size_t TRank>
struct IsInDims
{
  tt::VectorXs<TRank> dims;

  __host__ __device__
  IsInDims(tt::VectorXs<TRank> dims)
    : dims(dims)
  {
  }

  __host__ __device__
  bool operator()(tt::VectorXi<TRank> pos) const
  {
    return tt::all(pos >= 0 && pos < tt::static_cast_to<int32_t>(dims));
  }
};

template <size_t TRank>
struct Neighbors
{
  tt::VectorXs<TRank> dims;

  __host__ __device__
  Neighbors(tt::VectorXs<TRank> dims)
    : dims(dims)
  {
  }

  __host__ __device__
  auto operator()(tt::VectorXi<TRank> center)
  RETURN_AUTO(
    iterable::filter(
      tt::broadcast<2 * TRank>(tt::singleton(std::move(center))) + tt::repeat<tt::DimSeq<2>>(tt::UnitVectors<int32_t, TRank>()) * tt::broadcast<2 * TRank>(tt::Vector2i(1, -1)),
      IsInDims<TRank>(dims)
    )
  )
};

HOST_DEVICE_TEST_CASE(region_grow_on_local)
{
  tt::Vector4f values(0.0, 0.1, 0.9, 1.0);
  auto regions = regionGrow<1>(values, Neighbors<1>(values.dims()), []__host__ __device__(float v1, float v2){return math::abs(v1 - v2) <= 0.5;});

  CHECK(tt::eq(regions(0), regions(1)) && !tt::eq(regions(1), regions(2)) && tt::eq(regions(2), regions(3)));
}

#ifdef __CUDACC__
HOST_TEST_CASE(region_grow_on_device)
{
  tt::Vector4f values_h(0.0, 0.1, 0.9, 1.0);
  auto values_d = mem::toDevice(values_h);

  auto regions_d = regionGrow<1>(values_d, Neighbors<1>(values_d.dims()), []__device__(float v1, float v2){return math::abs(v1 - v2) <= 0.5;});
  auto regions_h = mem::toHost(regions_d);

  CHECK(tt::eq(regions_h(0), regions_h(1)) && !tt::eq(regions_h(1), regions_h(2)) && tt::eq(regions_h(2), regions_h(3)));
}
#endif
