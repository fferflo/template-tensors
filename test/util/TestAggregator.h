#include <HostDeviceTest.h>
#include <template_tensors/TemplateTensors.h>

HOST_DEVICE_TEST_CASE(aggregator_mean)
{
  tt::AllocMatrixT<float, mem::alloc::heap, tt::ColMajor> values(5, 1);
  tt::for_each<2>([]__host__ __device__(tt::Vector2s pos, float& el){el = pos(0) * 5 + pos(1);}, values);

  auto mean_offline = aggregator::mean_offline<float>();
  auto mean_online = aggregator::mean_online<float>();

  tt::for_each(mean_offline, values);
  tt::for_each(mean_online, values);

  CHECK(math::functor::eq_real<float>(1e-4)(mean_offline.get(), mean_online.get()));
}

HOST_DEVICE_TEST_CASE(aggregator_map_input)
{
  tt::AllocMatrixT<float, mem::alloc::heap, tt::ColMajor> values(5, 1);
  tt::for_each<2>([]__host__ __device__(tt::Vector2s pos, float& el){el = pos(0) * 5 + pos(1);}, values);

  auto mean = aggregator::mean_offline<float>() * 2;
  auto mean2 = aggregator::map_input([](float in){return in * 2;}, aggregator::mean_offline<float>());

  tt::for_each(mean, values);
  tt::for_each(mean2, values);

  CHECK(math::functor::eq_real<float>(1e-4)(mean.get(), mean2.get()));
}

HOST_DEVICE_TEST_CASE(aggregator_variance)
{
  tt::AllocMatrixT<float, mem::alloc::heap, tt::ColMajor> values(5, 1);
  tt::for_each<2>([]__host__ __device__(tt::Vector2s pos, float& el){el = pos(0) * 5 + pos(1);}, values);

  auto mean = aggregator::mean_online<float>();
  tt::for_each(mean, values);

  auto variance_online = aggregator::variance_online<float>();
  auto variance_offline = aggregator::variance_offline<float>(mean.get());

  tt::for_each(variance_offline, values);
  tt::for_each(variance_online, values);

  CHECK(math::functor::eq_real<float>(1e-4)(variance_online.get(), variance_offline.get()));
}

HOST_DEVICE_TEST_CASE(aggregator_elwise)
{
  tt::AllocMatrixT<tt::Vector2f, mem::alloc::heap, tt::ColMajor> values(5, 1);
  tt::for_each<2>([]__host__ __device__(tt::Vector2s pos, tt::Vector2f& el){
    el = tt::Vector2f(pos(0) * 5 + pos(1), pos(0) * 3 + pos(1) * 3);
  }, values);

  auto mean = aggregator::mean_online<tt::Vector2f>();
  tt::for_each(mean, values);

  auto elwise_mean1 = aggregator::elwise(aggregator::mean_online<float>(), values().dims());
  tt::for_each(elwise_mean1, values);

  auto elwise_mean2 = aggregator::elwise<2>(aggregator::mean_online<float>());
  tt::for_each(elwise_mean2, values);

  CHECK(tt::eq(mean.get(), elwise_mean1.get(), math::functor::eq_real<float>(1e-4)));
  CHECK(tt::eq(mean.get(), elwise_mean2.get(), math::functor::eq_real<float>(1e-4)));
}

HOST_DEVICE_TEST_CASE(aggregator_filter)
{
  tt::AllocVectorT<float, mem::alloc::heap, tt::ColMajor> values(10);
  tt::for_each<1>([]__host__ __device__(tt::Vector1s pos, float& el){el = pos();}, values);

  auto mean1 = aggregator::mean_offline<float>();
  tt::for_each(mean1, tt::head<5>(values));

  auto mean2 = aggregator::filter([](float in){return in < 4.5;}, aggregator::mean_offline<float>());
  tt::for_each(mean2, values);

  CHECK(math::functor::eq_real<float>(1e-4)(mean1.get(), mean2.get()));
}

HOST_DEVICE_TEST_CASE(aggregator_histogram1)
{
  tt::AllocVectorT<int, mem::alloc::heap, tt::ColMajor> values(20);
  tt::for_each<1>([]__host__ __device__(tt::Vector1s pos, int& el){el = pos() % 2;}, values);

  auto partial = aggregator::partial_histogram<2>(aggregator::count());
  auto total = aggregator::total_histogram<2>(aggregator::count());

  tt::for_each(partial, values);
  tt::for_each(total, values);

  CHECK(partial.get()(0) == partial.get()(1));
  CHECK(total.get()(0) == total.get()(1));
  CHECK(tt::eq(partial.get(), total.get()));
}

HOST_DEVICE_TEST_CASE(aggregator_histogram2)
{
  tt::AllocVectorT<int, mem::alloc::heap, tt::ColMajor> values1(20);
  tt::for_each<1>([]__host__ __device__(tt::Vector1s pos, int& el){el = pos() % 2;}, values1);

  tt::AllocVectorT<int, mem::alloc::heap, tt::ColMajor> values2(20);
  tt::for_each<1>([]__host__ __device__(tt::Vector1s pos, int& el){el = (pos() + 1) % 2;}, values2);

  auto partial = aggregator::partial_histogram<2>(aggregator::partial_histogram<2>(aggregator::count()));
  auto total = aggregator::total_histogram<2, 2>(aggregator::count());

  tt::for_each(partial, values1, values2);
  tt::for_each(total, values1, values2);

  auto partial_totalized = tt::total<1>(partial.get());

  CHECK(tt::eq(total.get(), partial_totalized));
  CHECK(total.get()(0, 0) == 0);
  CHECK(total.get()(1, 1) == 0);
  CHECK(total.get()(0, 1) == 10);
  CHECK(total.get()(1, 0) == 10);
}
