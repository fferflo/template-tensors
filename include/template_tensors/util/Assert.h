#pragma once

namespace template_tensors {

template <typename TDummy = void>
class PrintStream;

template <typename TDummy, typename T>
__host__ __device__
PrintStream<TDummy>& operator<<(PrintStream<TDummy>& stream, T&& object);

template <typename TDummy, typename T>
__host__ __device__
PrintStream<TDummy>&& operator<<(PrintStream<TDummy>&& stream, T&& object);

} // end of ns tensor



#if TT_IS_ON_DEVICE
#define TT_EXIT asm("trap;")
#else
#define TT_EXIT ::exit(EXIT_FAILURE)
#endif

#define ASSERT_(cond, str, ...) \
  do \
  { \
    if (!(cond)) \
    { \
      printf("\nAssertion '%s' failed in %s:%u!\n" str "\n", #cond, __FILE__, __LINE__, ##__VA_ARGS__); \
      TT_EXIT; \
    } \
  } while(0)

#define ASSERT_STREAM_(cond, ...) \
  do \
  { \
    if (!(cond)) \
    { \
      template_tensors::PrintStream<>() << "\nAssertion '" << #cond << "' failed in " << __FILE__<< ":" << __LINE__ << "!\n" << __VA_ARGS__ << "\n"; \
      TT_EXIT; \
    } \
  } while(0)

#ifdef DEBUG
#define ASSERT(cond, str, ...) ASSERT_(cond, str, ##__VA_ARGS__)
#define ASSERT_STREAM(cond, ...) ASSERT_STREAM_(cond, __VA_ARGS__)
#else
#define ASSERT(...)
#define ASSERT_STREAM(...)
#endif
