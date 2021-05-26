#include <template_tensors/TemplateTensors.h>

#include <nanobench.h>

#include <random>

#define MAX_DIM 34553
#define MAX_STRIDE 10

template <size_t TRank, typename TGenerator, typename TIndexStrategy>
void testToIndex(ankerl::nanobench::Bench& bench, std::string name, TGenerator& generator, TIndexStrategy indexstrategy)
{
  tt::VectorXs<TRank> dims = tt::random<TRank>(generator, std::uniform_int_distribution<int>(1, MAX_DIM));
  tt::VectorXs<TRank> coords = tt::random<TRank>(generator, std::uniform_int_distribution<int>(0)) % dims;
  bench.run("toIndex " + name + " " + std::to_string(TRank), [&]{
      ankerl::nanobench::doNotOptimizeAway(indexstrategy.toIndex(dims, coords));
  });
}

template <size_t TRank, typename TGenerator, typename TIndexStrategy>
void testFromIndex(ankerl::nanobench::Bench& bench, std::string name, TGenerator& generator, TIndexStrategy indexstrategy)
{
  tt::VectorXs<TRank> dims = tt::random<TRank>(generator, std::uniform_int_distribution<int>(1, MAX_DIM));
  tt::VectorXs<TRank> coords = tt::random<TRank>(generator, std::uniform_int_distribution<int>(0)) % dims;
  size_t index = indexstrategy.toIndex(dims, coords);
  bench.run("fromIndex " + name + " " + std::to_string(TRank), [&]{
      ankerl::nanobench::doNotOptimizeAway(indexstrategy.fromIndex(index, dims));
  });
}

template <size_t TRank, typename TGenerator, typename TIndexStrategy>
void testBoth(ankerl::nanobench::Bench& bench, std::string name, TGenerator&& generator, TIndexStrategy indexstrategy)
{
  testToIndex<TRank>(bench, name, generator, indexstrategy);
  testFromIndex<TRank>(bench, name, generator, indexstrategy);
}

int main(int argc, char** argv)
{
  std::random_device rd;
  std::mt19937 generator(rd());
  ankerl::nanobench::Bench bench;

  testBoth<1>(bench, "ColMajor", generator, tt::ColMajor());
  testBoth<2>(bench, "ColMajor", generator, tt::ColMajor());
  testBoth<3>(bench, "ColMajor", generator, tt::ColMajor());
  testBoth<4>(bench, "ColMajor", generator, tt::ColMajor());

  testBoth<1>(bench, "RowMajor", generator, tt::RowMajor());
  testBoth<2>(bench, "RowMajor", generator, tt::RowMajor());
  testBoth<3>(bench, "RowMajor", generator, tt::RowMajor());
  testBoth<4>(bench, "RowMajor", generator, tt::RowMajor());

  testToIndex<1>(bench, "Stride<1>", generator, tt::Stride<1>(tt::random<1>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));

  testToIndex<1>(bench, "Stride<2>", generator, tt::Stride<2>(tt::random<2>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<2>(bench, "Stride<2>", generator, tt::Stride<2>(tt::random<2>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));

  testToIndex<1>(bench, "Stride<3>", generator, tt::Stride<3>(tt::random<3>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<2>(bench, "Stride<3>", generator, tt::Stride<3>(tt::random<3>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<3>(bench, "Stride<3>", generator, tt::Stride<3>(tt::random<3>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));

  testToIndex<1>(bench, "Stride<4>", generator, tt::Stride<4>(tt::random<4>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<2>(bench, "Stride<4>", generator, tt::Stride<4>(tt::random<4>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<3>(bench, "Stride<4>", generator, tt::Stride<4>(tt::random<4>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<4>(bench, "Stride<4>", generator, tt::Stride<4>(tt::random<4>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));

  testBoth<1>(bench, "MortonDivideAndConquer<1>", generator, tt::MortonDivideAndConquer<1>());

  testBoth<1>(bench, "MortonDivideAndConquer<2>", generator, tt::MortonDivideAndConquer<2>());
  testBoth<2>(bench, "MortonDivideAndConquer<2>", generator, tt::MortonDivideAndConquer<2>());

  testBoth<1>(bench, "MortonDivideAndConquer<3>", generator, tt::MortonDivideAndConquer<3>());
  testBoth<2>(bench, "MortonDivideAndConquer<3>", generator, tt::MortonDivideAndConquer<3>());
  testBoth<3>(bench, "MortonDivideAndConquer<3>", generator, tt::MortonDivideAndConquer<3>());

  testBoth<1>(bench, "MortonDivideAndConquer<4>", generator, tt::MortonDivideAndConquer<4>());
  testBoth<2>(bench, "MortonDivideAndConquer<4>", generator, tt::MortonDivideAndConquer<4>());
  testBoth<3>(bench, "MortonDivideAndConquer<4>", generator, tt::MortonDivideAndConquer<4>());
  testBoth<4>(bench, "MortonDivideAndConquer<4>", generator, tt::MortonDivideAndConquer<4>());

  testBoth<1>(bench, "MortonForLoop<1>", generator, tt::MortonForLoop<1>());

  testBoth<1>(bench, "MortonForLoop<2>", generator, tt::MortonForLoop<2>());
  testBoth<2>(bench, "MortonForLoop<2>", generator, tt::MortonForLoop<2>());

  testBoth<1>(bench, "MortonForLoop<3>", generator, tt::MortonForLoop<3>());
  testBoth<2>(bench, "MortonForLoop<3>", generator, tt::MortonForLoop<3>());
  testBoth<3>(bench, "MortonForLoop<3>", generator, tt::MortonForLoop<3>());

  testBoth<1>(bench, "MortonForLoop<4>", generator, tt::MortonForLoop<4>());
  testBoth<2>(bench, "MortonForLoop<4>", generator, tt::MortonForLoop<4>());
  testBoth<3>(bench, "MortonForLoop<4>", generator, tt::MortonForLoop<4>());
  testBoth<4>(bench, "MortonForLoop<4>", generator, tt::MortonForLoop<4>());
}
