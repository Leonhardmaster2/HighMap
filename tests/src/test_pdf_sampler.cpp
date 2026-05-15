#include <gtest/gtest.h>

#include "highmap.hpp"

using namespace hmap;

TEST(PdfSampler, SampleReturnsValueInUnitInterval)
{
  std::vector<float> pdf = {1.f, 1.f, 1.f, 1.f};

  PdfSampler sampler(pdf, 42);

  for (int i = 0; i < 1000; ++i)
  {
    float s = sampler.sample();

    EXPECT_GE(s, 0.f);
    EXPECT_LT(s, 1.f);
  }
}

TEST(PdfSampler, SampleVectorSize)
{
  std::vector<float> pdf = {1.f, 2.f, 3.f};

  PdfSampler sampler(pdf, 42);

  std::vector<float> samples = sampler.sample(128);

  EXPECT_EQ(samples.size(), 128);
}

TEST(PdfSampler, DeterministicWithSameSeed)
{
  std::vector<float> pdf = {1.f, 2.f, 3.f, 4.f};

  PdfSampler s0(pdf, 1234);
  PdfSampler s1(pdf, 1234);

  for (int i = 0; i < 100; ++i)
  {
    float a = s0.sample();
    float b = s1.sample();

    EXPECT_FLOAT_EQ(a, b);
  }
}

TEST(PdfSampler, DifferentSeedsProduceDifferentSequences)
{
  std::vector<float> pdf = {1.f, 2.f, 3.f, 4.f};

  PdfSampler s0(pdf, 1);
  PdfSampler s1(pdf, 2);

  bool different = false;

  for (int i = 0; i < 50; ++i)
  {
    if (s0.sample() != s1.sample())
    {
      different = true;
      break;
    }
  }

  EXPECT_TRUE(different);
}

TEST(PdfSampler, UniformPdfProducesUniformBins)
{
  std::vector<float> pdf = {1.f, 1.f, 1.f, 1.f};

  PdfSampler sampler(pdf, 42);

  const int nsamples = 20000;

  std::vector<int> bins(4, 0);

  for (int i = 0; i < nsamples; ++i)
  {
    float s = sampler.sample();

    int k = std::min(3, static_cast<int>(s * 4.f));

    bins[k]++;
  }

  float expected = static_cast<float>(nsamples) / 4.f;

  for (int c : bins)
    EXPECT_NEAR(static_cast<float>(c), expected, 0.1f * expected);
}

TEST(PdfSampler, WeightedPdfFavorsLargerBins)
{
  std::vector<float> pdf = {1.f, 1.f, 10.f, 1.f};

  PdfSampler sampler(pdf, 42);

  const int nsamples = 20000;

  std::vector<int> bins(4, 0);

  for (int i = 0; i < nsamples; ++i)
  {
    float s = sampler.sample();

    int k = std::min(3, static_cast<int>(s * 4.f));

    bins[k]++;
  }

  EXPECT_GT(bins[2], bins[0]);
  EXPECT_GT(bins[2], bins[1]);
  EXPECT_GT(bins[2], bins[3]);
}

TEST(PdfSampler, SingleBinAlwaysSamplesInsideBin)
{
  std::vector<float> pdf = {1.f};

  PdfSampler sampler(pdf, 42);

  for (int i = 0; i < 100; ++i)
  {
    float s = sampler.sample();

    EXPECT_GE(s, 0.f);
    EXPECT_LT(s, 1.f);
  }
}

TEST(PdfSampler, SampleVectorDeterministic)
{
  std::vector<float> pdf = {1.f, 2.f, 3.f};

  PdfSampler s0(pdf, 99);
  PdfSampler s1(pdf, 99);

  std::vector<float> v0 = s0.sample(256);
  std::vector<float> v1 = s1.sample(256);

  ASSERT_EQ(v0.size(), v1.size());

  for (size_t i = 0; i < v0.size(); ++i)
    EXPECT_FLOAT_EQ(v0[i], v1[i]);
}
