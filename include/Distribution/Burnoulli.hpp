#pragma once
//
// Created by moinshaikh on 1/27/26.
//

#ifndef LUNARALIGHTINGRL_BURNOULLI_HPP
#define LUNARALIGHTINGRL_BURNOULLI_HPP

#include"Distribution.hpp"

#include<c10/util/ArrayRef.h>
#include<torch/torch.h>

namespace LunarAlighting
{
    /**
     * @class Bernoulli
     * @brief A Bernoulli probability distribution for binary events.
     *
     * The Bernoulli class represents a Bernoulli distribution over binary outcomes
     * (0 or 1). It inherits from the Distribution base class and provides operations
     * for sampling, entropy calculation, and log-probability evaluation of binary events.
     *
     * This distribution is commonly used in reinforcement learning for:
     * - Modeling binary (yes/no) action spaces in policy networks
     * - Computing probability distributions over two mutually exclusive outcomes
     * - Calculating entropy and log-probabilities for binary decision policies
     *
     * The distribution can be parameterized using either probabilities (range [0, 1])
     * or logits (unbounded real numbers), providing flexibility depending on the use
     * case and numerical stability requirements.
     */
    class Bernoulli : public Distribution
    {
    private:
        torch::Tensor probs;      ///< Probability tensor for success outcome (value 1)
        torch::Tensor logits;     ///< Log-odds tensor for success outcome
        torch::Tensor param;      ///< Primary parameterization (either probs or logits)
    public:
        /**
         * @brief Constructs a Bernoulli distribution.
         *
         * Initializes a Bernoulli distribution with either probabilities or logits.
         * At least one of the two parameters must be provided (non-null). If both are
         * provided, logits takes precedence for internal calculations.
         *
         * @param probs Pointer to a probability tensor where each element represents
         *              the probability of success (outcome = 1). Values must be in the
         *              range [0, 1]. Can be nullptr if logits is provided.
         * @param logits Pointer to a logits tensor containing log-odds for each outcome.
         *               Can be any real-valued number. Can be nullptr if probs is provided.
         *
         * @note The tensors pointed to should remain valid for the lifetime of this object.
         * @note If both probs and logits are provided, logits is used internally.
         * @note For numerical stability, logits parameterization is often preferred over probabilities.
        */
        Bernoulli(const torch::Tensor* probs,const torch::Tensor *logits);

        /**
         * @brief Computes the entropy of the Bernoulli distribution.
         *
         * Entropy measures the uncertainty or randomness in the distribution.
         * For a Bernoulli distribution, entropy is maximized when p=0.5 (maximum uncertainty)
         * and minimized when p=0 or p=1 (complete certainty).
         *
         * @return A torch::Tensor containing the entropy values. For batch distributions,
         *         returns a tensor with shape matching the batch dimensions.
         *
         * @note Entropy is always non-negative and is measured in nats (natural logarithm units).
         * @note Maximum entropy for Bernoulli is ln(2) ≈ 0.693 when p=0.5.
        */
        torch::Tensor entropy() override;

        /**
         * @brief Computes the log-probability of binary outcome(s).
         *
         * Calculates the natural logarithm of the probability for given binary values (0 or 1).
         * This is useful for policy gradient methods in reinforcement learning where
         * log-probabilities are needed for gradient computation.
         *
         * @param value A tensor containing binary outcomes (0 or 1). Can be a single value
         *              or a batch of binary values matching the distribution's batch dimensions.
         *
         * @return A torch::Tensor containing the log-probabilities corresponding to the
         *         given outcomes. Shape matches the input value tensor.
         *
         * @note For value=1, returns log(p). For value=0, returns log(1-p).
         * @note Log-probabilities are typically negative for values not equal to 1.
        */

        torch::Tensor logProbability(torch::Tensor value) override;

        /**
          * @brief Samples binary outcome(s) from the Bernoulli distribution.
          *
          * Draws random binary samples (0 or 1) from the distribution according to its
          * probability values. Used for exploration and stochastic decisions in
          * reinforcement learning agents.
          *
          * @param sample_shape Optional shape parameter specifying additional dimensions
          *                     for the samples. Default is {} for single samples.
          *
          * @return A torch::Tensor containing sampled binary outcomes (0 or 1).
          *
          * @note Sampling is a stochastic operation and returns different values on
          *       successive calls (due to random number generation).
          * @note The returned tensor contains integer values (0 or 1).
        */
        torch::Tensor sample(c10::ArrayRef<int64_t> sampleShape) override;

        inline torch::Tensor getLogits() { return logits; }
        inline torch::Tensor getProbs() { return probs; }

    };
}

#endif //LUNARALIGHTINGRL_BURNOULLI_HPP