#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_TEST_CASE(test_assign_and_read_on_host)
{
  size_t sidelength = 10;

  // Generate objects
  tt::AllocVectorT<tt::Vector3f, mem::alloc::host_heap> objects(10);
  float start = 123;
  auto next = [&](){
    start += 81638.01823;
    start *= 838.9172938;
    start = math::fmod(start, 1e5);
    return math::fmod(start, sidelength);
  };
  for (tt::Vector3f& object : objects)
  {
    object(0) = next();
    object(1) = next();
    object(2) = next();
  }

  // Create grid
  point_cloud::SortGrid<decltype(objects)&, 3, mem::alloc::host_heap> grid(objects, tt::Vector3s(sidelength, sidelength, sidelength));
  grid.update(tt::functor::static_cast_to<size_t>());

  // Test
  CHECK(tt::sum(tt::elwise(iterable::functor::distance<>(), grid)) == objects.rows());
  for (tt::Vector3f object : objects)
  {
    CHECK(iterable::is_in(object, grid(object)));
  }
}

#ifdef __CUDACC__
HOST_TEST_CASE(test_assign_and_read_on_device)
{
  size_t sidelength = 10;

  // Generate objects
  tt::AllocVectorT<tt::Vector3f, mem::alloc::host_heap> objects_h(10);
  float start = 123;
  auto next = [&](){
    start += 81638.01823;
    start *= 838.9172938;
    start = math::fmod(start, 1e5);
    return math::fmod(start, sidelength);
  };
  for (tt::Vector3f& object : objects_h)
  {
    object(0) = next();
    object(1) = next();
    object(2) = next();
  }
  auto objects_d = mem::toDevice(objects_h);

  // Create grid
  point_cloud::SortGrid<decltype(objects_d)&, 3, mem::alloc::device> grid(objects_d, tt::Vector3s(sidelength, sidelength, sidelength));
  grid.update(tt::functor::static_cast_to<size_t>());

  // Test
  CHECK(tt::sum(mem::toHost(tt::eval(tt::elwise(iterable::functor::distance<>(), grid)))) == objects_h.rows());
  auto grid_to_kernel = mem::toKernel(grid);
  tt::for_each([grid_to_kernel]__device__(const tt::Vector3f& object){
    CHECK(iterable::is_in(object, grid_to_kernel(object)));
  }, objects_d);
}
#endif





struct Object
{
  tt::Vector3f features;
};

TT_PROCLAIM_TRIVIALLY_RELOCATABLE_NOTEMPLATE((Object));

struct GetFeatures
{
  __host__ __device__
  tt::Vector3f operator()(const Object& object) const
  {
    return object.features;
  }
};

struct DistanceMetric
{
  __host__ __device__
  float operator()(float distance) const
  {
    return distance * distance;
  }

  __host__ __device__
  float operator()(tt::Vector3f p1, tt::Vector3f p2) const
  {
    return tt::length_squared(p2 - p1);
  }
};

HOST_TEST_CASE(test_nearest_neighbors_on_host)
{
  size_t sidelength = 10;

  // Generate objects
  tt::AllocVectorT<Object, mem::alloc::host_heap> objects(10);
  float start = 123;
  auto next = [&](){
    start += 81638.01823;
    start *= 838.9172938;
    start = math::fmod(start, 1e5);
    return math::fmod(start, sidelength);
  };
  for (Object& object : objects)
  {
    object.features = tt::Vector3f(next(), next(), next());
  }

  // Create grid
  using NearestNeighbors = point_cloud::nearest_neighbors::GridSearch<point_cloud::SortGrid<decltype(objects)&, 3, mem::alloc::host_heap>, GetFeatures, DistanceMetric>;
  NearestNeighbors nearest_neighbors(objects, tt::Vector3s(sidelength, sidelength, sidelength));
  nearest_neighbors.update();

  // Test
  for (Object& object : objects)
  {
    Object* result = nearest_neighbors.nearest(object.features, 1.0f);
    CHECK(result != nullptr);
    if (result != nullptr)
    {
      CHECK(tt::eq(result->features, object.features));
    }
  }
}

#ifdef __CUDACC__
HOST_TEST_CASE(test_nearest_neighbors_on_device)
{
  size_t sidelength = 10;

  // Generate objects
  tt::AllocVectorT<Object, mem::alloc::host_heap> objects_h(10);
  float start = 123;
  auto next = [&](){
    start += 81638.01823;
    start *= 838.9172938;
    start = math::fmod(start, 1e5);
    return math::fmod(start, sidelength);
  };
  for (Object& object : objects_h)
  {
    object.features = tt::Vector3f(next(), next(), next());
  }
  auto objects_d = mem::toDevice(objects_h);

  // Create grid
  using NearestNeighbors = point_cloud::nearest_neighbors::GridSearch<point_cloud::SortGrid<decltype(objects_d)&, 3, mem::alloc::device>, GetFeatures, DistanceMetric>;
  NearestNeighbors nearest_neighbors(objects_d, tt::Vector3s(sidelength, sidelength, sidelength));
  nearest_neighbors.update();

  // Test
  auto nearest_neighbors_to_kernel = mem::toKernel(nearest_neighbors);
  tt::for_each([nearest_neighbors_to_kernel]__device__(Object& object) mutable {
    Object* result = nearest_neighbors_to_kernel.nearest(object.features, 1.0f);
    CHECK(result != nullptr);
    if (result != nullptr)
    {
      CHECK(tt::eq(result->features, object.features));
    }
  }, objects_d);
}
#endif
