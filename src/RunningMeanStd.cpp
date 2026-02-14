//
// Created by moinshaikh on 2/4/26.
//


#include"../include/RunningMeanStd.hpp"

#include"third_party/doctest.hpp"

namespace LunarAlighting
{

    /**
     * @brief Constructs a RunningMeanStdImpl object with a given size.
     *
     * @details Initializes the running mean and variance tracker.
     * - count: A small initial value (1e-4) is used to prevent division by zero.
     * - mean: Initialized to a zero vector of the specified size.
     * - variance: Initialized to a one vector of the specified size, representing unit variance.
     *
     * @param size The dimensionality of the observation vector to be tracked.
     */
    RunningMeanstdImpl::RunningMeanstdImpl(int size)
    : count(register_buffer("count", torch::full({1}, 1e-4, torch::kFloat))),
      mean(register_buffer("mean", torch::zeros({size}))),
      variance(register_buffer("variance", torch::ones({size}))) {}

    /**
     * @brief Constructs a RunningMeanStdImpl object from pre-existing moments.
     *
     * @details Initializes the tracker with provided mean and variance values. This is useful for
     * loading pre-trained normalization statistics.
     *
     * @param means A vector of floats representing the initial mean.
     * @param variances A vector of floats representing the initial variance.
     */
    RunningMeanstdImpl::RunningMeanstdImpl(std::vector<float> means , std::vector<float> variances) :
    count(register_buffer("count",torch::full({1},1e-4,torch::kFloat))),

    mean(register_buffer("mean", torch::from_blob(means.data(), {static_cast<long>(means.size())})
                                       .clone())),

    variance(register_buffer("variance", torch::from_blob(variances.data(), {static_cast<long>(variances.size())})
                                              .clone()))

    {

    }


    /**
     * @brief Updates the running statistics with a new batch of observations.
     *
     * @details This function first computes the mean and variance of the incoming observation batch.
     * It then calls updateFromMoments to update the running statistics in a numerically stable way.
     *
     * @param observation A tensor of new observations. It is reshaped to [-1, mean.size(0)] to handle
     *                    both single and multiple observations.
     */
    void RunningMeanstdImpl::update(torch::Tensor observation)
    {
        short dim = -1;
        observation = observation.reshape({dim,mean.size(0)});
        auto batchMeans = observation.mean(0);
        auto batchVar =observation.var(0,false,false);
        int batchCounts =  observation.size(0);
        updateFromMoments(batchMeans,batchVar,batchCounts);
    }

    /**
     * @brief Updates the running mean and variance from batch moments.
     *
     * @details This function implements a parallel algorithm for updating the mean and variance,
     * which is numerically more stable than a naive incremental update. It is based on Welford's
     * online algorithm, extended for batches.
     *
     * Let the existing statistics be (mean_A, var_A, count_A) and the new batch statistics be
     * (mean_B, var_B, count_B).
     *
     * The combined mean (mean_AB) is calculated as a weighted average:
     * mean_AB = mean_A + (count_B / (count_A + count_B)) * (mean_B - mean_A)
     *
     * The combined sum of squared differences (M2_AB) is given by:
     * M2_AB = M2_A + M2_B + (mean_B - mean_A)^2 * (count_A * count_B) / (count_A + count_B)
     *
     * where M2_A = count_A * var_A and M2_B = count_B * var_B.
     *
     * The combined variance (var_AB) is then:
     * var_AB = M2_AB / (count_A + count_B)
     *
     * @param batchMean The mean of the new batch (mean_B).
     * @param batchVariance The variance of the new batch (var_B).
     * @param batchCount The number of samples in the new batch (count_B).
     */
    void RunningMeanstdImpl::updateFromMoments(torch::Tensor batchMean, torch::Tensor batchVariance, int batchCount)
    {
        auto delta = batchMean - mean;
        auto total_count = count + batchCount;

        mean.copy_(mean + delta * batchCount / total_count);
        auto m_a = variance * count;
        auto m_b = batchVariance * batchCount;
        auto m2 = m_a + m_b + torch::pow(delta, 2) * count * batchCount / total_count;
        variance.copy_(m2 / total_count);
        count.copy_(total_count);
    }

    TEST_CASE("RunningMeanStd")
    {
        SUBCASE("Calculates mean and variance correctly")
        {
            RunningMeanstd rms(5);
            auto observations = torch::rand({3, 5});
            rms->update(observations[0]);
            rms->update(observations[1]);
            rms->update(observations[2]);

            auto expected_mean = observations.mean(0);
            auto expected_variance = observations.var(0, false, false);

            auto actual_mean = rms->get_mean();
            auto actual_variance = rms->get_variance();

            for (int i = 0; i < 5; ++i)
            {
                DOCTEST_CHECK(expected_mean[i].item().toFloat() ==
                              doctest::Approx(actual_mean[i].item().toFloat())
                                  .epsilon(0.001));
                DOCTEST_CHECK(expected_variance[i].item().toFloat() ==
                              doctest::Approx(actual_variance[i].item().toFloat())
                                  .epsilon(0.001));
            }
        }

        SUBCASE("Loads mean and variance from constructor correctly")
        {
            RunningMeanstd rms(std::vector<float>{1, 2, 3}, std::vector<float>{4, 5, 6});

            auto mean = rms->get_mean();
            auto variance = rms->get_variance();
            DOCTEST_CHECK(mean[0].item().toFloat() == doctest::Approx(1));
            DOCTEST_CHECK(mean[1].item().toFloat() == doctest::Approx(2));
            DOCTEST_CHECK(mean[2].item().toFloat() == doctest::Approx(3));
            DOCTEST_CHECK(variance[0].item().toFloat() == doctest::Approx(4));
            DOCTEST_CHECK(variance[1].item().toFloat() == doctest::Approx(5));
            DOCTEST_CHECK(variance[2].item().toFloat() == doctest::Approx(6));
        }
    }
}
