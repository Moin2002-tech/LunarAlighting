#pragma once
//
// Created by moinshaikh on 1/27/26.
//

#ifndef LUNARALIGHTINGRL_CATEGORICAL_HPP
#define LUNARALIGHTINGRL_CATEGORICAL_HPP

#include"Distribution.hpp"
#include<c10/util/ArrayRef.h>

namespace LunarAlighting
{
    /**
    * @class Categorical
    * @brief A categorical probability distribution for discrete events.
    *
    * The Categorical class represents a categorical distribution over a discrete
    * set of events. It inherits from the Distribution base class and provides
    * operations for sampling, entropy calculation, and log-probability evaluation
    * of discrete outcomes.
    *
    * This distribution is commonly used in reinforcement learning for:
    * - Modeling discrete action spaces in policy networks
    * - Computing probability distributions over multiple discrete choices
    * - Calculating entropy and log-probabilities for policy gradient methods
    *
    * The distribution can be parameterized using either probabilities or logits,
    * providing flexibility depending on the use case and numerical stability needs.
    */

    class Categorical : public Distribution
    {
    private:
        torch::Tensor probs;      ///< Probability tensor for each event
        torch::Tensor logits;     ///< Log-odds tensor for each event
        torch::Tensor param;      ///< Primary parameterization (either probs or logits)
        int numEvents;   ///< Number of possible discrete events

    public:
        /**
         * @brief Constructs a Categorical distribution.
         *
         * Initializes a categorical distribution with either probabilities or logits.
         * At least one of the two parameters must be provided (non-null). If both are
         * provided, logits takes precedence for internal calculations.
         *
         * @param probs Pointer to a probability tensor where each element represents
         *              the probability of a discrete event. Must sum to 1 across the
         *              last dimension. Can be nullptr if logits is provided.
         * @param logits Pointer to a logits tensor containing unnormalized log-odds
         *               for each discrete event. Can be nullptr if probs is provided.
         *
         * @note The tensors pointed to should remain valid for the lifetime of this object.
         * @note If both probs and logits are provided, logits is used internally.
         */
        Categorical(const torch::Tensor *probs,const torch::Tensor *logits) ;

        /**
           * @brief Computes the entropy of the categorical distribution.
           *
           * Entropy measures the uncertainty or randomness in the distribution.
           * Higher entropy indicates greater uncertainty about which event will occur.
           *
           * @return A torch::Tensor containing the entropy values. For batch distributions,
           *         returns a tensor with shape matching the batch dimensions.
           *
           * @note Entropy is always non-negative and is measured in nats (natural logarithm units).
         */
        torch::Tensor entropy() override;

        /**
         * @brief Computes the log-probability of discrete event(s).
         *
         * Calculates the natural logarithm of the probability for given event indices.
         * This is useful for policy gradient methods in reinforcement learning where
         * log-probabilities are needed for gradient computation.
         *
         * @param value A tensor containing indices of events. Each index should be in
         *              the range [0, num_events-1]. Can be a single value or batch of values.
         *
         * @return A torch::Tensor containing the log-probabilities corresponding to the
         *         given indices. Shape matches the input value tensor.
         *
         * @note Log-probabilities are typically negative (except for probability 1.0).
         */
        torch::Tensor logProbability(torch::Tensor value) override;

        /**
         * @brief Samples event indices from the categorical distribution.
         *
         * Draws random samples from the distribution according to its probability values.
         * Used for exploration in reinforcement learning agents.
         *
         * @param sampleShape Optional shape parameter specifying additional dimensions
         *                     for the samples. Default is {} for single samples.
         *
         * @return A torch::Tensor containing sampled event indices. The returned tensor
         *         contains integer values in the range [0, num_events-1].
         *
         * @note Sampling is a stochastic operation and returns different values on
         *       successive calls (due to random number generation).
        */
        torch::Tensor sample(c10::ArrayRef<int64_t> sampleShape= {}) override;

        inline torch::Tensor get_logits() { return logits; }
        inline torch::Tensor get_param() { return param; }
        inline torch::Tensor getProbability() {return probs;}


    };
}

#endif //LUNARALIGHTINGRL_CATEGORICAL_HPP