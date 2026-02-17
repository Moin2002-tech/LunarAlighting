#pragma once
//
// Created by moinshaikh on 1/27/26.
//

#ifndef LUNARALIGHTINGRL_DISTRIBUTION_HPP
#define LUNARALIGHTINGRL_DISTRIBUTION_HPP

#include<vector>
#include<torch/torch.h>

namespace LunarAlighting
{
    /**
     * @class Distribution
     * @brief Abstract base class for probability distributions in reinforcement learning.
     *
     * The Distribution class defines the interface for probability distributions used
     * in reinforcement learning algorithms. It supports operations such as sampling,
     * entropy calculation, and log probability evaluation. This is a pure virtual base
     * class that should be implemented by specific distribution types (e.g., Gaussian,
     * Categorical, Bernoulli).
     *
     * @see Normal
     * @see Categorical
     * @see Bernoulli
    */
    class Distribution
    {
    protected:
        std::vector<int64_t> batch_shape;  ///< Shape of the batch dimension(s)
        std::vector<int64_t> event_shape;  ///< Shape of the event dimension(s)

        /**
         * @brief Computes the extended shape for sampling operations.
         *
         * Combines the sample shape, batch shape, and event shape into a single
         * extended shape for tensor operations.
         *
         * @param sampleShapes The desired shape for samples
         * @return std::vector<int64_t> The extended shape combining sample, batch, and event shapes
        */
        std::vector<int64_t> extendedShape(c10::ArrayRef<int64_t> &sampleShapes);
    public:
        /**
         * @brief Virtual destructor.
         *
         * Pure virtual destructor to ensure proper cleanup of derived classes.
         */
        virtual ~Distribution() = 0;

        /**
         * @brief Computes the entropy of the distribution.
         *
         * Entropy measures the uncertainty in the distribution. Higher entropy
         * indicates greater uncertainty.
         *
         * @return torch::Tensor A tensor containing entropy values with shape matching batch_shape
        */
        virtual torch::Tensor entropy() = 0;

        /**
         * @brief Computes the log probability density/mass of a value under this distribution.
         *
         * @param value A tensor representing the value(s) to evaluate. Should have a shape
         *              compatible with the distribution's shape parameters.
         * @return torch::Tensor A tensor of log probabilities with shape matching batch_shape
         */
        virtual torch::Tensor logProbability(torch::Tensor value) = 0;

        /**
         * @brief Generates samples from the distribution.
         *
         * @param sampleShape The desired number of samples and their dimensions (default: {})
         * @return torch::Tensor A tensor of sampled values with shape [sample_shape, batch_shape, event_shape]
        */
        virtual torch::Tensor sample(c10::ArrayRef<int64_t> sampleShape = {}) = 0;
    };

    inline Distribution::~Distribution() {

    }
}

#endif //LUNARALIGHTINGRL_DISTRIBUTION_HPP