//
// Created by moinshaikh on 2/1/26.
//
#include<math.h>
#include<cmath>
#include<limits>

#include<c10/util/ArrayRef.h>
#include<torch/torch.h>
#include"../third_party/doctest.hpp"
#include"../../include/Distribution/Normal.hpp"

namespace LunarAlighting
{
    /**
     * @brief Constructs a Normal distribution.
     *
     * @details Initializes the distribution with mean (`loc`) and standard deviation (`scale`).
     * It broadcasts the input tensors to ensure they have compatible shapes.
     * The `batch_shape` is determined from the broadcasted shape, and `event_shape` is set to empty
     * (implying independent scalar distributions).
     *
     * @param loc The mean (location) of the distribution.
     * @param scale The standard deviation (scale) of the distribution.
     */
    Normal::Normal(const torch::Tensor loc, const torch::Tensor scale)
    {
        auto broadcasted_tensors = torch::broadcast_tensors({loc,scale});
        this->loc = broadcasted_tensors[0];
        this->scale = broadcasted_tensors[1];
        this->batch_shape = this->loc.sizes().vec();
        this->event_shape = {};
    }

    /**
     * @brief Computes the entropy of the normal distribution.
     *
     * @details Calculates the differential entropy for a normal distribution.
     * The entropy \f$ H(X) \f$ represents the expected amount of information or uncertainty in the distribution.
     *
     * For a univariate normal distribution \f$ \mathcal{N}(\mu, \sigma^2) \f$, the entropy is given by:
     * \f[
     * H(X) = \frac{1}{2} \ln(2\pi e \sigma^2) = \frac{1}{2} + \frac{1}{2}\ln(2\pi) + \ln(\sigma)
     * \f]
     *
     * The implementation computes this element-wise and sums over the last dimension (`.sum(-1)`),
     * effectively computing the entropy of a multivariate normal distribution with a diagonal covariance matrix
     * (assuming independence between dimensions).
     *
     * @return A tensor containing the entropy values.
     */
    torch::Tensor Normal::entropy()
    {
        return (0.5 +0.5 * std::log(2* M_PI) + torch::log(scale)).sum(-1);
    }

    /**
     * @brief Computes the log probability density of a value.
     *
     * @details Calculates the natural logarithm of the Probability Density Function (PDF) evaluated at `value`.
     * For a normal distribution \f$ \mathcal{N}(\mu, \sigma^2) \f$, the PDF is:
     * \f[
     * P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
     * \f]
     * Taking the natural logarithm gives:
     * \f[
     * \log P(x) = -\frac{(x-\mu)^2}{2\sigma^2} - \ln(\sigma) - \frac{1}{2}\ln(2\pi)
     * \f]
     *
     * The code implements this formula term by term:
     * - \f$ -\frac{(x-\mu)^2}{2\sigma^2} \f$: The negative quadratic error (Mahalanobis distance squared).
     * - \f$ -\ln(\sigma) \f$: The log-scale normalization term.
     * - \f$ -\ln(\sqrt{2\pi}) \f$: The constant normalization term.
     *
     * @param value The input value(s) to evaluate.
     * @return A tensor of log probabilities.
     */
    torch::Tensor Normal::logProbability(torch::Tensor value)
    {
        auto variance = scale.pow(2);
        auto logScale = scale.log();
        return (-(value -loc).pow(2) /(2* variance) - logScale - std::log(std::sqrt(2 * M_PI)));
    }

    /**
     * @brief Samples from the normal distribution.
     *
     * @details Generates random samples using the reparameterization trick (implicitly handled by `at::normal`).
     * It expands the `loc` and `scale` parameters to match the requested `sample_shape` combined with the
     * distribution's batch shape.
     *
     * @param sample_shape The desired shape of the samples (e.g., {num_samples}).
     * @return A tensor of sampled values.
     */
    torch::Tensor Normal::sample(c10::ArrayRef<int64_t> sample_shape)
    {
        auto shape = extendedShape(sample_shape);
        auto no_grad_guard = torch::NoGradGuard();
        return at::normal(loc.expand(shape), scale.expand(shape));
    }

    TEST_CASE("Normal")
{
    float locs_array[] = {0, 1, 2, 3, 4, 5};
    float scales_array[] = {5, 4, 3, 2, 1, 0};
    auto locs = torch::from_blob(locs_array, {2, 3});
    auto scales = torch::from_blob(scales_array, {2, 3});
    auto dist = Normal(locs, scales);

    SUBCASE("Sampled tensors have correct shape")
    {
        CHECK(dist.sample().sizes().vec() == std::vector<int64_t>{2, 3});
        CHECK(dist.sample({20}).sizes().vec() == std::vector<int64_t>{20, 2, 3});
        CHECK(dist.sample({2, 20}).sizes().vec() == std::vector<int64_t>{2, 20, 2, 3});
        CHECK(dist.sample({1, 2, 3, 4, 5}).sizes().vec() == std::vector<int64_t>{1, 2, 3, 4, 5, 2, 3});
    }

    SUBCASE("entropy()")
    {
        auto entropies = dist.entropy();

        SUBCASE("Returns correct values")
        {
            INFO("Entropies: \n"
                 << entropies);

            CHECK(entropies[0].item().toDouble() ==
                  doctest::Approx(8.3512).epsilon(1e-3));
            CHECK(entropies[1].item().toDouble() ==
                  -std::numeric_limits<float>::infinity());
        }

        SUBCASE("Output tensor is the correct size")
        {
            CHECK(entropies.sizes().vec() == std::vector<int64_t>{2});
        }
    }

    SUBCASE("log_prob()")
    {
        float actions[2][3] = {{0, 1, 2},
                               {0, 1, 2}};
        auto actions_tensor = torch::from_blob(actions, {2, 3});
        auto log_probs = dist.logProbability(actions_tensor);

        INFO(log_probs << "\n");
        SUBCASE("Returns correct values")
        {
            CHECK(log_probs[0][0].item().toDouble() ==
                  doctest::Approx(-2.5284).epsilon(1e-3));
            CHECK(log_probs[0][1].item().toDouble() ==
                  doctest::Approx(-2.3052).epsilon(1e-3));
            CHECK(log_probs[0][2].item().toDouble() ==
                  doctest::Approx(-2.0176).epsilon(1e-3));
            CHECK(log_probs[1][0].item().toDouble() ==
                  doctest::Approx(-2.7371).epsilon(1e-3));
            CHECK(log_probs[1][1].item().toDouble() ==
                  doctest::Approx(-5.4189).epsilon(1e-3));
            CHECK(std::isnan(log_probs[1][2].item().toDouble()));
        }

        SUBCASE("Output tensor is correct size")
        {
            CHECK(log_probs.sizes().vec() == std::vector<int64_t>{2, 3});
        }
    }
}
}
