//
// Created by moinshaikh on 1/30/26.
//

#include<c10/util/ArrayRef.h>
#include<torch/torch.h>

#include"../../include/Distribution/Categorical.hpp"
#include"../third_party/doctest.hpp"


namespace LunarAlighting
{
    /**
     * @brief Constructs a Categorical distribution from probabilities or logits.
     *
     * @details This constructor initializes the distribution by processing either the provided
     * probabilities or logits. It ensures that exactly one of the two is provided.
     * - If `probs` are provided:
     *   1. Normalizes them to sum to 1 along the last dimension.
     *   2. Clamps values to avoid numerical instability (log(0)).
     *   3. Computes `logits` as log(probs).
     * - If `logits` are provided:
     *   1. Normalizes them using log-sum-exp for numerical stability.
     *   2. Computes `probs` using softmax.
     *
     * The constructor also determines the batch shape and the number of events (categories).
     *
     * @param probs Pointer to a tensor of probabilities. Can be nullptr.
     * @param logits Pointer to a tensor of logits. Can be nullptr.
     * @throws std::runtime_error If both or neither of `probs` and `logits` are provided.
     * @throws std::runtime_error If the input tensor has fewer than 1 dimension.
     */
    Categorical::Categorical(const torch::Tensor *probs, const torch::Tensor *logits)
    {
        if ((probs == nullptr) == (logits == nullptr))
        {
            throw std::runtime_error("Either probs or logits are not provided");
        }
        if (probs != nullptr)
        {
            if (probs->dim() < 1)
            {
                throw std::runtime_error("atleast one dimenstion is required");
            }
            this->probs =*probs /probs->sum(-1,true);
            this->probs = this->probs.clamp(1.21e-7,1.-1.21e-7);
            this->logits = torch::log(this->probs);
        }
        else
        {
            if (logits->dim() < 1)
            {
               throw std::runtime_error("atleast one dimenstion is required for logits");
            }
            this->logits = *logits - logits->logsumexp(-1,true);
            this->probs = torch::softmax(this->logits,-1);
        }
        param =probs != nullptr ? * probs : *logits;
        numEvents =param.size(-1);
        if (param.dim() >= 1 ) {
            batch_shape =  param.sizes().vec();
            batch_shape.resize(batch_shape.size()-1);
        }

    }


    /**
     * @brief Computes the entropy of the categorical distribution.
     *
     * @details Entropy is calculated as -sum(p * log(p)) along the last dimension (events).
     * This measures the uncertainty of the distribution.
     *
     * @return A tensor containing the entropy for each batch element.
     */
    torch::Tensor Categorical::entropy()
    {
        auto pLogP = logits * probs;
        return -pLogP.sum(-1);
    }

    /**
     * @brief Computes the log probability of specific values (categories).
     *
     * @details Calculates the log probability of the given indices (values) under the distribution.
     * It uses `gather` to select the log-probabilities corresponding to the provided indices.
     * Broadcasting is applied to align the `value` tensor with the `logits` tensor.
     *
     * @param value A tensor of indices (categories) for which to compute log probabilities.
     *              The values should be integers in the range [0, numEvents - 1].
     * @return A tensor of log probabilities matching the shape of the input `value`.
     */
    torch::Tensor Categorical::logProbability(torch::Tensor value)
    {
        value =value.to(torch::kLong).unsqueeze(-1);
        auto broadcastedTensors = torch::broadcast_tensors({value,logits});
        value =  broadcastedTensors[0];
        value =  value.narrow(-1,0,1);
        return broadcastedTensors[1].gather(-1,value).squeeze(-1);
    }

    /**
     * @brief Samples from the categorical distribution.
     *
     * @details Generates random samples based on the probabilities of each category.
     * It uses `torch::multinomial` to perform the sampling.
     *
     * The method handles broadcasting of the probability tensor to match the requested `sampleShape`.
     * It first expands the `probs` tensor to include the sample dimensions, then flattens it
     * to 2D for `multinomial` sampling, and finally reshapes the result to the expected output shape.
     *
     * @param sampleShape The desired shape of the samples (e.g., {num_samples}).
     * @return A tensor of sampled category indices with shape [sampleShape, batch_shape].
     */
    torch::Tensor Categorical::sample(c10::ArrayRef<int64_t> sampleShape)
    {
        auto extrSampleShape = extendedShape(sampleShape);
        auto paramShape=  extrSampleShape;
        paramShape.insert(paramShape.end(),{numEvents});

        torch::Tensor probs_expanded = probs;
        for (size_t i = 0; i < sampleShape.size(); ++i) {
            probs_expanded = probs_expanded.unsqueeze(0);
        }

        auto expProbability = probs_expanded.expand(paramShape);
        torch::Tensor probs2D;
        if (probs.dim() == 1 || probs.size(0)  == 1)
        {
            probs2D = expProbability.view({-1,numEvents});

        }
        else
        {
            probs2D = expProbability.contiguous().view({-1,numEvents});

        }
        auto sample2D =  torch::multinomial(probs2D,1,true);
        return sample2D.contiguous().view(extrSampleShape);
    }

    TEST_CASE("Categorical")
    {
        SUBCASE("Throws when provided both probs and logits")
        {
            auto tensor = torch::Tensor();
            CHECK_THROWS(Categorical(&tensor, &tensor));
        }

        SUBCASE("Sampled numbers are in the right range")
        {
            float probabilities[] = {0.2, 0.2, 0.2, 0.2, 0.2};
            auto probabilities_tensor = torch::from_blob(probabilities, {5});
            auto dist = Categorical(&probabilities_tensor, nullptr);

            auto output = dist.sample({100});
            auto more_than_4 = output > 4;
            auto less_than_0 = output < 0;
            CHECK(!more_than_4.any().item().toInt());
            CHECK(!less_than_0.any().item().toInt());
        }

        SUBCASE("Sampled tensors are of the right shape")
        {
            float probabilities[] = {0.2, 0.2, 0.2, 0.2, 0.2};
            auto probabilities_tensor = torch::from_blob(probabilities, {5});
            auto dist = Categorical(&probabilities_tensor, nullptr);

            CHECK(dist.sample({20}).sizes().vec() == std::vector<int64_t>{20});
            CHECK(dist.sample({2, 20}).sizes().vec() == std::vector<int64_t>{2, 20});
            CHECK(dist.sample({1, 2, 3, 4, 5}).sizes().vec() == std::vector<int64_t>{1, 2, 3, 4, 5});
        }

        SUBCASE("Multi-dimensional input probabilities are handled correctly")
        {
            SUBCASE("Sampled tensors are of the right shape")
            {
                float probabilities[2][4] = {{0.5, 0.5, 0.0, 0.0},
                                             {0.25, 0.25, 0.25, 0.25}};
                auto probabilities_tensor = torch::from_blob(probabilities, {2, 4});
                auto dist = Categorical(&probabilities_tensor, nullptr);

                CHECK(dist.sample({20}).sizes().vec() == std::vector<int64_t>{20, 2});
                CHECK(dist.sample({10, 5}).sizes().vec() == std::vector<int64_t>{10, 5, 2});
            }

            SUBCASE("Generated tensors have correct probabilities")
            {
                float probabilities[2][4] = {{0, 1, 0, 0},
                                             {0, 0, 0, 1}};
                auto probabilities_tensor = torch::from_blob(probabilities, {2, 4});
                auto dist = Categorical(&probabilities_tensor, nullptr);

                auto output = dist.sample({5});
                auto sum = output.sum({0});

                CHECK(sum[0].item().toInt() == 5);
                CHECK(sum[1].item().toInt() == 15);
            }
        }

        SUBCASE("entropy()")
        {
            float probabilities[2][4] = {{0.5, 0.5, 0.0, 0.0},
                                         {0.25, 0.25, 0.25, 0.25}};
            auto probabilities_tensor = torch::from_blob(probabilities, {2, 4});
            auto dist = Categorical(&probabilities_tensor, nullptr);

            auto entropies = dist.entropy();

            SUBCASE("Returns correct values")
            {
                CHECK(entropies[0].item().toDouble() ==
                      doctest::Approx(0.6931).epsilon(1e-3));

                CHECK(entropies[1].item().toDouble() ==
                      doctest::Approx(1.3863).epsilon(1e-3));
            }

            SUBCASE("Output tensor is the correct size")
            {
                CHECK(entropies.sizes().vec() == std::vector<int64_t>{2});
            }
        }

        SUBCASE("log_prob()")
        {
            float probabilities[2][4] = {{0.5, 0.5, 0.0, 0.0},
                                         {0.25, 0.25, 0.25, 0.25}};
            auto probabilities_tensor = torch::from_blob(probabilities, {2, 4});
            auto dist = Categorical(&probabilities_tensor, nullptr);

            float actions[2][2] = {{0, 1},
                                   {2, 3}};
            auto actions_tensor = torch::from_blob(actions, {2, 2});
            auto log_probs = dist.logProbability(actions_tensor);

            INFO(log_probs << "\n");
            SUBCASE("Returns correct values")
            {
                CHECK(log_probs[0][0].item().toDouble() ==
                      doctest::Approx(-0.6931).epsilon(1e-3));
                CHECK(log_probs[0][1].item().toDouble() ==
                      doctest::Approx(-1.3863).epsilon(1e-3));
                CHECK(log_probs[1][0].item().toDouble() ==
                      doctest::Approx(-15.9424).epsilon(1e-3));
                CHECK(log_probs[1][1].item().toDouble() ==
                      doctest::Approx(-1.3863).epsilon(1e-3));
            }

            SUBCASE("Output tensor is correct size")
            {
                CHECK(log_probs.sizes().vec() == std::vector<int64_t>{2, 2});
            }
        }
    }
}
