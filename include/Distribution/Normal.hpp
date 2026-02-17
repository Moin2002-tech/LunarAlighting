#pragma once
//
// Created by moinshaikh on 1/27/26.
//

#ifndef LUNARALIGHTINGRL_NORMAL_HPP
#define LUNARALIGHTINGRL_NORMAL_HPP

#include<torch/torch.h>
#include<c10/util/ArrayRef.h>
#include"Distribution.hpp"



namespace LunarAlighting
{
    /**
     * @class Normal
     * @brief Multivariate normal (Gaussian) distribution.
     *
     * The Normal class implements a multivariate normal distribution with diagonal
     * covariance matrix. It is commonly used in continuous control reinforcement
     * learning algorithms (e.g., PPO, A2C) for policy gradient methods.
     *
     * The distribution is parameterized by mean (loc) and standard deviation (scale)
     * tensors, allowing for element-wise independent normal distributions with
     * different means and standard deviations.
     *
     * @inherits Distribution
    */
    class Normal : public Distribution
    {
    private:
        torch::Tensor loc;    ///< Mean (location) parameter of the normal distribution
        torch::Tensor scale;  ///< Standard deviation (scale) parameter of the normal distribution
    public:
        /**
         * @brief Constructs a normal distribution with given mean and standard deviation.
         *
         * @param loc The mean (location) parameter as a tensor. Can be scalar or multi-dimensional.
         * @param scale The standard deviation (scale) parameter as a tensor. Must have the same
         *              shape as loc and contain positive values.
         *
         * @note The batch_shape and event_shape are derived from the shapes of loc and scale.
        */
        Normal(const torch::Tensor loc,const torch::Tensor scale);

        /**
         * @brief Computes the entropy of the normal distribution.
         *
         * For a normal distribution, entropy = 0.5 * log(2 * pi * e * scale^2)
         *
         * @return torch::Tensor A tensor containing entropy values with shape matching batch_shape
        */
        torch::Tensor entropy() override;

        /**
         * @brief Computes the log probability density/mass of a value under this distribution.
         *
         * @param value A tensor representing the value(s) to evaluate. Should have a shape
         *              compatible with the distribution's shape parameters.
         * @return torch::Tensor A tensor of log probabilities with shape matching batch_shape
          */
        torch::Tensor logProbability(torch::Tensor value) override;

        /**
         * @brief Generates samples from the normal distribution.
         *
         * Uses the reparameterization trick: sample = loc + scale * epsilon,
         * where epsilon is a standard normal random variable.
         *
         * @param sample_shape The desired number of samples and their dimensions (default: {})
         * @return torch::Tensor Sampled values with shape [sample_shape, batch_shape, event_shape]
         */
        torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) override;


        inline torch::Tensor getLoc()
        {
            return loc;
        }

        inline torch::Tensor getScale()
        {
            return scale;
        }
    };
}

#endif //LUNARALIGHTINGRL_NORMAL_HPP