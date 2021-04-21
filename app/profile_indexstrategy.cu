#include <template_tensors/TemplateTensors.h>
#include <profiler/Profiler.h>

#include <random>

#define MAX_DIM 34553
#define MAX_STRIDE 10

template <size_t TRank, typename TGenerator, typename TIndexStrategy>
void testToIndex(std::string name, TGenerator& generator, TIndexStrategy indexstrategy)
{
  for (size_t r = 0; r < 100000; r++)
  {
    tt::VectorXs<TRank> dims = tt::random<TRank>(generator, std::uniform_int_distribution<int>(1, MAX_DIM));
    tt::VectorXs<TRank> coords = tt::random<TRank>(generator, std::uniform_int_distribution<int>(0)) % dims;
    profiler::makeUnpredictable(dims);
    profiler::makeUnpredictable(coords);

    PROFILE("toIndex   " << name << " " << TRank);
    profiler::doNotOptimizeAway(indexstrategy.toIndex(dims, coords));
  }
}

template <size_t TRank, typename TGenerator, typename TIndexStrategy>
void testFromIndex(std::string name, TGenerator& generator, TIndexStrategy indexstrategy)
{
  for (size_t r = 0; r < 100000; r++)
  {
    tt::VectorXs<TRank> dims = tt::random<TRank>(generator, std::uniform_int_distribution<int>(1, MAX_DIM));
    tt::VectorXs<TRank> coords = tt::random<TRank>(generator, std::uniform_int_distribution<int>(0)) % dims;
    size_t index = indexstrategy.toIndex(dims, coords);
    profiler::makeUnpredictable(dims);
    profiler::makeUnpredictable(index);

    PROFILE("fromIndex " << name << " " << TRank);
    profiler::doNotOptimizeAway(indexstrategy.fromIndex(index, dims));
  }
}

template <size_t TRank, typename TGenerator, typename TIndexStrategy>
void testBoth(std::string name, TGenerator&& generator, TIndexStrategy indexstrategy)
{
  testToIndex<TRank>(name, generator, indexstrategy);
  testFromIndex<TRank>(name, generator, indexstrategy);
}

int main(int argc, char** argv)
{
  std::random_device rd;
  std::mt19937 generator(rd());

  testBoth<1>("ColMajor", generator, tt::ColMajor());
  testBoth<2>("ColMajor", generator, tt::ColMajor());
  testBoth<3>("ColMajor", generator, tt::ColMajor());
  testBoth<4>("ColMajor", generator, tt::ColMajor());

  testBoth<1>("RowMajor", generator, tt::RowMajor());
  testBoth<2>("RowMajor", generator, tt::RowMajor());
  testBoth<3>("RowMajor", generator, tt::RowMajor());
  testBoth<4>("RowMajor", generator, tt::RowMajor());

  testToIndex<1>("Stride<1>", generator, tt::Stride<1>(tt::random<1>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));

  testToIndex<1>("Stride<2>", generator, tt::Stride<2>(tt::random<2>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<2>("Stride<2>", generator, tt::Stride<2>(tt::random<2>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));

  testToIndex<1>("Stride<3>", generator, tt::Stride<3>(tt::random<3>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<2>("Stride<3>", generator, tt::Stride<3>(tt::random<3>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<3>("Stride<3>", generator, tt::Stride<3>(tt::random<3>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));

  testToIndex<1>("Stride<4>", generator, tt::Stride<4>(tt::random<4>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<2>("Stride<4>", generator, tt::Stride<4>(tt::random<4>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<3>("Stride<4>", generator, tt::Stride<4>(tt::random<4>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));
  testToIndex<4>("Stride<4>", generator, tt::Stride<4>(tt::random<4>(generator, std::uniform_int_distribution<int>(1, MAX_STRIDE))));

  testBoth<1>("MortonDivideAndConquer<1>", generator, tt::MortonDivideAndConquer<1>());

  testBoth<1>("MortonDivideAndConquer<2>", generator, tt::MortonDivideAndConquer<2>());
  testBoth<2>("MortonDivideAndConquer<2>", generator, tt::MortonDivideAndConquer<2>());

  testBoth<1>("MortonDivideAndConquer<3>", generator, tt::MortonDivideAndConquer<3>());
  testBoth<2>("MortonDivideAndConquer<3>", generator, tt::MortonDivideAndConquer<3>());
  testBoth<3>("MortonDivideAndConquer<3>", generator, tt::MortonDivideAndConquer<3>());

  testBoth<1>("MortonDivideAndConquer<4>", generator, tt::MortonDivideAndConquer<4>());
  testBoth<2>("MortonDivideAndConquer<4>", generator, tt::MortonDivideAndConquer<4>());
  testBoth<3>("MortonDivideAndConquer<4>", generator, tt::MortonDivideAndConquer<4>());
  testBoth<4>("MortonDivideAndConquer<4>", generator, tt::MortonDivideAndConquer<4>());

  testBoth<1>("MortonForLoop<1>", generator, tt::MortonForLoop<1>());

  testBoth<1>("MortonForLoop<2>", generator, tt::MortonForLoop<2>());
  testBoth<2>("MortonForLoop<2>", generator, tt::MortonForLoop<2>());

  testBoth<1>("MortonForLoop<3>", generator, tt::MortonForLoop<3>());
  testBoth<2>("MortonForLoop<3>", generator, tt::MortonForLoop<3>());
  testBoth<3>("MortonForLoop<3>", generator, tt::MortonForLoop<3>());

  testBoth<1>("MortonForLoop<4>", generator, tt::MortonForLoop<4>());
  testBoth<2>("MortonForLoop<4>", generator, tt::MortonForLoop<4>());
  testBoth<3>("MortonForLoop<4>", generator, tt::MortonForLoop<4>());
  testBoth<4>("MortonForLoop<4>", generator, tt::MortonForLoop<4>());

  profiler::print<std::chrono::nanoseconds>();
}
