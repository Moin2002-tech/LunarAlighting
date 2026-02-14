//
// Created by moinshaikh on 1/30/26.
//

#include<ATen/core/Reduction.h>
#include<c10/util/ArrayRef.h>
#include<torch/torch.h>

#include"../../include/Distribution/Burnoulli.hpp"
#include"../third_party/doctest.hpp"


namespace LunarAlighting
{
    /**
     * @brief Constructs a Bernoulli distribution with either probabilities or logits.
     *
     * @details This constructor initializes the Bernoulli distribution. It enforces that exactly one
     * of `probs` or `logits` is provided. If `probs` are provided, they are clamped to avoid
     * numerical instability (log(0)) and converted to logits. If `logits` are provided, they are
     * converted to probabilities using the sigmoid function.
     *
     * @param probs Pointer to a tensor of probabilities (values in [0, 1]). Can be nullptr.
     * @param logits Pointer to a tensor of logits (unbounded real values). Can be nullptr.
     *
     * @throws std::runtime_error If both `probs` and `logits` are nullptr or both are non-nullptr.
     * @throws std::runtime_error If the input tensor has fewer than 1 dimension.
     */
    Bernoulli::Bernoulli(const torch::Tensor* probs,const torch::Tensor* logits)
    {

        // 1. Validate Exclusive Input
        if ((probs == nullptr) == (logits == nullptr))
            {
            throw std::runtime_error("Exactly one of 'probs' or 'logits' must be provided.");
        }

        // 2. Extract the active tensor and check dimensions
        const torch::Tensor& input_tensor = (probs != nullptr) ? *probs : *logits;
        if (input_tensor.dim() < 1)
            {
            throw std::runtime_error("Input tensor must have at least one dimension.");
        }

        // 3. Logic Branching
        if (probs != nullptr)
            {
            this->probs = *probs;
            // Standardize to logits: log(p / (1-p))
            auto eps = 1.21e-7;
            auto clamped = this->probs.clamp(eps, 1.0 - eps);
            this->logits = torch::log(clamped) - torch::log1p(-clamped);
        } else
            {
            this->logits = *logits;
            this->probs = torch::sigmoid(*logits);
        }

        // 4. Finalize shapes
        this->param =input_tensor;
        this->batch_shape = input_tensor.sizes().vec();
    }

    /**
     * @brief Computes the entropy of the distribution.
     *
     * @details Entropy is calculated using the binary cross entropy with logits function,
     * comparing the logits against the probabilities themselves. This effectively computes
     * - (p * log(p) + (1-p) * log(1-p)).
     *
     * @return A tensor containing the entropy for each element in the batch.
     */
    torch::Tensor Bernoulli::entropy()
    {
        return torch::binary_cross_entropy_with_logits(logits,
            probs,
            torch::Tensor(),
            torch::Tensor(),
            torch::Reduction::None);
    }

    /**
     * @brief Computes the log probability of a given value.
     *
     * @details Calculates the log probability of the input `value` (0 or 1) given the distribution's
     * logits. It uses binary cross entropy with logits (negated) to ensure numerical stability.
     * Broadcasting is applied if the shapes of `logits` and `value` differ.
     *
     * @param value The observed value(s) (0 or 1) for which to calculate the log probability.
     * @return A tensor of log probabilities.
     */
    torch::Tensor Bernoulli::logProbability(torch::Tensor value)
    {
        auto broadCasted_Tensors =torch::broadcast_tensors({logits,value});
        return -torch::binary_cross_entropy_with_logits(broadCasted_Tensors[0],
            broadCasted_Tensors[1],
            torch::Tensor(),
            torch::Tensor(),
            torch::Reduction::None);
    }

    /**
     * @brief Samples from the Bernoulli distribution.
     *
     * @details Generates random binary samples (0 or 1) based on the probabilities.
     * The `sampleShape` allows for generating multiple independent samples for each
     * element in the batch.
     *
     * @param sampleShape The desired shape of the samples (e.g., {num_samples}).
     * @return A tensor of sampled values with shape [sampleShape, batch_shape].
     */
    torch::Tensor Bernoulli::sample(c10::ArrayRef<int64_t> sampleShape)
    {

       auto ext_sample_shape = extendedShape(sampleShape);
        torch::NoGradGuard no_grad_guard;
        return torch::bernoulli(probs.expand(ext_sample_shape));
    }


    TEST_CASE("Bernoulli")
{
    SUBCASE("Throws when provided both probs and logits")
    {
        auto tensor = torch::Tensor();
        CHECK_THROWS(Bernoulli(&tensor, &tensor));
    }

    SUBCASE("Sampled numbers are in the right range")
    {
        float probabilities[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        auto probabilities_tensor = torch::from_blob(probabilities, {5});
        auto dist = Bernoulli(&probabilities_tensor, nullptr);

        auto output = dist.sample({100});
        auto more_than_1 = output > 1;
        auto less_than_0 = output < 0;
        CHECK(!more_than_1.any().item().toInt());
        CHECK(!less_than_0.any().item().toInt());
    }

    SUBCASE("Sampled tensors are of the right shape")
    {
        float probabilities[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        auto probabilities_tensor = torch::from_blob(probabilities, {5});
        auto dist = Bernoulli(&probabilities_tensor, nullptr);

        CHECK(dist.sample({20}).sizes().vec() == std::vector<int64_t>{20, 5});
        CHECK(dist.sample({2, 20}).sizes().vec() == std::vector<int64_t>{2, 20, 5});
        CHECK(dist.sample({1, 2, 3, 4}).sizes().vec() == std::vector<int64_t>{1, 2, 3, 4, 5});
    }

    SUBCASE("Multi-dimensional input probabilities are handled correctly")
    {
        SUBCASE("Sampled tensors are of the right shape")
        {
            float probabilities[2][4] = {{0.5, 0.5, 0.0, 0.0},
                                         {0.25, 0.25, 0.25, 0.25}};
            auto probabilities_tensor = torch::from_blob(probabilities, {2, 4});
            auto dist = Bernoulli(&probabilities_tensor, nullptr);

            CHECK(dist.sample({20}).sizes().vec() == std::vector<int64_t>{20, 2, 4});
            CHECK(dist.sample({10, 5}).sizes().vec() == std::vector<int64_t>{10, 5, 2, 4});
        }
    }


    SUBCASE("entropy()")
    {
        float probabilities[2][2] = {{0.5, 0.0},
                                     {0.25, 0.25}};
        auto probabilities_tensor = torch::from_blob(probabilities, {2, 2});
        auto dist = Bernoulli(&probabilities_tensor, nullptr);

        auto entropies = dist.entropy();

        SUBCASE("Returns correct values")
        {
            CHECK(entropies[0][0].item().toDouble() ==
                  doctest::Approx(0.6931).epsilon(1e-3));
            CHECK(entropies[0][1].item().toDouble() ==
                  doctest::Approx(0.0000).epsilon(1e-3));
            CHECK(entropies[1][0].item().toDouble() ==
                  doctest::Approx(0.5623).epsilon(1e-3));
            CHECK(entropies[1][1].item().toDouble() ==
                  doctest::Approx(0.5623).epsilon(1e-3));
        }

        SUBCASE("Output tensor is the correct size")
        {
            CHECK(entropies.sizes().vec() == std::vector<int64_t>{2, 2});
        }
    }

    SUBCASE("log_prob()")
    {
        float probabilities[2][2] = {{0.5, 0.0},
                                     {0.25, 0.25}};
        auto probabilities_tensor = torch::from_blob(probabilities, {2, 2});
        auto dist = Bernoulli(&probabilities_tensor, nullptr);

        float actions[2][2] = {{1, 0},
                               {1, 0}};
        auto actions_tensor = torch::from_blob(actions, {2, 2});
        auto log_probs = dist.logProbability(actions_tensor);

        INFO(log_probs << "\n");
        SUBCASE("Returns correct values")
        {
            CHECK(log_probs[0][0].item().toDouble() ==
                  doctest::Approx(-0.6931).epsilon(1e-3));
            CHECK(log_probs[0][1].item().toDouble() ==
                  doctest::Approx(0.0000).epsilon(1e-3));
            CHECK(log_probs[1][0].item().toDouble() ==
                  doctest::Approx(-1.3863).epsilon(1e-3));
            CHECK(log_probs[1][1].item().toDouble() ==
                  doctest::Approx(-0.2876).epsilon(1e-3));
        }

        SUBCASE("Output tensor is correct size")
        {
            CHECK(log_probs.sizes().vec() == std::vector<int64_t>{2, 2});
        }
    }
}

}
