//
// Created by moinshaikh on 2/4/26.
//

#include"../include/ObservationNormalizer.hpp"
#include"../include/RunningMeanStd.hpp"
#include"third_party/doctest.hpp"

namespace LunarAlighting
{
    /**
     * @brief Constructs an ObservationNormalizerImpl with a given size and clip value.
     *
     * @details Initializes the normalizer with a clipping threshold and a `RunningMeanStd` module
     * to track observation statistics.
     *
     * @param size The dimensionality of the observations.
     * @param clip The maximum absolute value for normalized observations.
     */
    ObservationNormalizerImpl::ObservationNormalizerImpl(int size, float clip)
    : clip(register_buffer("clip", torch::full({1}, clip, torch::kFloat))),
      rms(register_module("rms", RunningMeanstd(size))) {}


    /**
     * @brief Constructs an ObservationNormalizerImpl from existing statistics.
     *
     * @details Initializes the normalizer with pre-computed mean and variance values.
     * This is useful for loading saved models or transferring statistics.
     *
     * @param means A vector of mean values.
     * @param variances A vector of variance values.
     * @param clip The maximum absolute value for normalized observations.
     */
    ObservationNormalizerImpl::ObservationNormalizerImpl(const std::vector<float> &means,
                                                     const std::vector<float> &variances,
                                                     float clip)
    : clip(register_buffer("clip", torch::full({1}, clip, torch::kFloat))),
      rms(register_module("rms", RunningMeanstd(means, variances))) {}

/**
 * @brief Constructs an ObservationNormalizerImpl by aggregating multiple other normalizers.
 *
 * @details This constructor averages the statistics (clip, mean, variance) from a list of
 * other `ObservationNormalizer` instances. It also sums up the total step count.
 * This is typically used in distributed training to synchronize statistics across workers.
 *
 * @param others A vector of other ObservationNormalizer instances to aggregate.
 */
ObservationNormalizerImpl::ObservationNormalizerImpl(const std::vector<ObservationNormalizer> &others)
    : clip(register_buffer("clip", torch::zeros({1}, torch::kFloat))),
      rms(register_module("rms", RunningMeanstd(1)))
    {
        // Calculate mean clip
        for (const auto &other : others)
        {
            clip += other->getClipValue();
        }
        clip[0] = clip[0] / static_cast<float>(others.size());

        // Calculate mean mean
        std::vector<float> mean_means(others[0]->getMean().size(), 0);
        for (const auto &other : others)
        {
            auto other_mean = other->getMean();
            for (unsigned int i = 0; i < mean_means.size(); ++i)
            {
                mean_means[i] += other_mean[i];
            }
        }
        for (auto &mean : mean_means)
        {
            mean /= others.size();
        }

        // Calculate mean variances
        std::vector<float> mean_variances(others[0]->getVariances().size(), 0);
        for (const auto &other : others)
        {
            auto other_variances = other->getVariances();
            for (unsigned int i = 0; i < mean_variances.size(); ++i)
            {
                mean_variances[i] += other_variances[i];
            }
        }
        for (auto &variance : mean_variances)
        {
            variance /= others.size();
        }

        rms = RunningMeanstd(mean_means, mean_variances);

        int total_count = std::accumulate(others.begin(), others.end(), 0,
                                          [](int accumulator, const ObservationNormalizer &other) {
                                              return accumulator + other->getStepCount();
                                          });
        rms->set_count(total_count);
    }

    /**
     * @brief Normalizes and clips an observation.
     *
     * @details Applies the normalization formula:
     * normalized_obs = (observation - mean) / sqrt(variance + epsilon)
     *
     * The result is then clipped to the range [-clip, clip] to ensure numerical stability
     * and prevent extreme values from destabilizing the network.
     *
     * @param observation The input observation tensor.
     * @return The normalized and clipped observation tensor.
     */
    torch::Tensor ObservationNormalizerImpl::processObservation(torch::Tensor &observation) const
    {
        auto normalized_obs = (observation - rms->get_mean()) /
                              torch::sqrt(rms->get_variance() + 1e-8);
        return torch::clamp(normalized_obs, -clip.item(), clip.item());
    }

    /**
     * @brief Retrieves the current running mean.
     *
     * @return A vector containing the mean values for each dimension.
     */
    std::vector<float> ObservationNormalizerImpl::getMean() const
    {
        auto mean = rms->get_mean();
        return std::vector<float>(mean.data_ptr<float>(), mean.data_ptr<float>() + mean.numel());
    }

    /**
     * @brief Retrieves the current running variance.
     *
     * @return A vector containing the variance values for each dimension.
     */
    std::vector<float> ObservationNormalizerImpl::getVariances() const
    {
        auto variance = rms->get_variance();
        return std::vector<float>(variance.data_ptr<float>(), variance.data_ptr<float>() + variance.numel());
    }

    /**
     * @brief Updates the running statistics with new observations.
     *
     * @details Delegates the update to the underlying `RunningMeanStd` module.
     *
     * @param observations A tensor of new observations to incorporate into the statistics.
     */
    void ObservationNormalizerImpl::update(torch::Tensor observations)
    {
        rms->update(observations);
    }

    TEST_CASE("ObservationNormalizer")
{
    SUBCASE("Clips values correctly")
    {
        ObservationNormalizer normalizer(7, 1);
        float observation_array[] = {-1000, -100, -10, 0, 10, 100, 1000};
        auto observation = torch::from_blob(observation_array, {7});
        auto processed_observation = normalizer->processObservation(observation);

        auto has_too_large_values = (processed_observation > 1).any().item().toBool();
        auto has_too_small_values = (processed_observation < -1).any().item().toBool();
        DOCTEST_CHECK(!has_too_large_values);
        DOCTEST_CHECK(!has_too_small_values);
    }

    SUBCASE("Normalizes values correctly")
    {
        ObservationNormalizer normalizer(5);

        float obs_1_array[] = {-10., 0., 5., 3.2, 0.};
        float obs_2_array[] = {-5., 2., 4., 3.7, -3.};
        float obs_3_array[] = {1, 2, 3, 4, 5};
        auto obs_1 = torch::from_blob(obs_1_array, {5});
        auto obs_2 = torch::from_blob(obs_2_array, {5});
        auto obs_3 = torch::from_blob(obs_3_array, {5});

        normalizer->update(obs_1);
        normalizer->update(obs_2);
        normalizer->update(obs_3);
        auto processed_observation = normalizer->processObservation(obs_3);

        DOCTEST_CHECK(processed_observation[0].item().toFloat() == doctest::Approx(1.26008659));
        DOCTEST_CHECK(processed_observation[1].item().toFloat() == doctest::Approx(0.70712887));
        DOCTEST_CHECK(processed_observation[2].item().toFloat() == doctest::Approx(-1.2240818));
        DOCTEST_CHECK(processed_observation[3].item().toFloat() == doctest::Approx(1.10914509));
        DOCTEST_CHECK(processed_observation[4].item().toFloat() == doctest::Approx(1.31322402));
    }

    SUBCASE("Loads mean and variance from constructor correctly")
    {
        ObservationNormalizer normalizer(std::vector<float>({1, 2, 3}), std::vector<float>({4, 5, 6}));

        auto mean = normalizer->getMean();
        auto variance = normalizer->getVariances();
        DOCTEST_CHECK(mean[0] == doctest::Approx(1));
        DOCTEST_CHECK(mean[1] == doctest::Approx(2));
        DOCTEST_CHECK(mean[2] == doctest::Approx(3));
        DOCTEST_CHECK(variance[0] == doctest::Approx(4));
        DOCTEST_CHECK(variance[1] == doctest::Approx(5));
        DOCTEST_CHECK(variance[2] == doctest::Approx(6));
    }

    SUBCASE("Is constructed from other normalizers correctly")
    {
        std::vector<ObservationNormalizer> normalizers;
        for (int i = 0; i < 3; ++i)
        {
            normalizers.push_back(ObservationNormalizer(3));
            for (int j = 0; j <= i; ++j)
            {
                normalizers[i]->update(torch::rand({3}));
            }
        }

        ObservationNormalizer combined_normalizer(normalizers);

        std::vector<std::vector<float>> means;
        std::transform(normalizers.begin(), normalizers.end(), std::back_inserter(means),
                       [](const ObservationNormalizer &normalizer) { return normalizer->getMean(); });
        std::vector<std::vector<float>> variances;
        std::transform(normalizers.begin(), normalizers.end(), std::back_inserter(variances),
                       [](const ObservationNormalizer &normalizer) { return normalizer->getVariances(); });

        std::vector<float> mean_means;
        for (int i = 0; i < 3; ++i)
        {
            mean_means.push_back((means[0][i] + means[1][i] + means[2][i]) / 3);
        }
        std::vector<float> mean_variances;
        for (int i = 0; i < 3; ++i)
        {
            mean_variances.push_back((variances[0][i] + variances[1][i] + variances[2][i]) / 3);
        }

        auto actual_mean_means = combined_normalizer->getMean();
        auto actual_mean_variances = combined_normalizer->getVariances();

        for (int i = 0; i < 3; ++i)
        {
            DOCTEST_CHECK(actual_mean_means[i] == doctest::Approx(mean_means[i]));
            DOCTEST_CHECK(actual_mean_variances[i] == doctest::Approx(actual_mean_variances[i]));
        }
        DOCTEST_CHECK(combined_normalizer->getStepCount() == 6);
    }
}

}
