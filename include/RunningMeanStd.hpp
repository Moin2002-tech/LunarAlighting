#pragma once
//
// Created by moinshaikh on 1/27/26.
//

#ifndef LUNARALIGHTINGRL_RUNNINGMEANSTD_HPP
#define LUNARALIGHTINGRL_RUNNINGMEANSTD_HPP

#include<torch/torch.h>
#include<torch/nn.h>
#include<vector>


namespace LunarAlighting
{
    /**
     * @brief Tracks running mean and variance of observations
     *
     * This class implements an online algorithm for computing running mean and variance
     * over a stream of observations, based on the OpenAI Baselines implementation.
     * It uses numerically stable algorithms to update mean and variance incrementally.
     * Useful for normalizing observations in reinforcement learning pipelines.
    */

    class RunningMeanstdImpl : public torch::nn::Module
    {
    private:
        torch::Tensor count;    /**< Tensor storing the count of observations processed */
        torch::Tensor mean;     /**< Tensor storing the current running mean */
        torch::Tensor variance; /**< Tensor storing the current running variance */

        /**
         * @brief Updates running statistics from batch moments
         *
         * Internal helper method that updates the running mean and variance given
         * batch statistics. Uses Welford's online algorithm for numerical stability.
         *
         * @param batch_mean Mean of the current batch of observations
         * @param batch_var Variance of the current batch of observations
         * @param batch_count Number of samples in the current batch
         */
        void updateFromMoments(torch::Tensor batchMean, torch::Tensor batchVariance,int batchCount);
    public:
        /**
         *@brief Construct a RunningMeanStd tracker  for observation of a given size
         *initialize tracking Tensors with zero mean  and unit  variance
         *@param size Dimensions of observation to track
         */
        explicit RunningMeanstdImpl(int size);



        RunningMeanstdImpl(std::vector<float> means, std::vector<float> variances);

        /**
         * @brief Updates running statistics with a new observation
         *
         * Processes a single observation and updates the running mean, variance,
         * and observation count accordingly.
         *
         * @param observation New observation tensor to incorporate into statistics
         */
        void update(torch::Tensor observation);

        /**
         * @brief Gets the number of observations processed
         *
         * Returns the total count of observations that have been used to update
         * the running statistics.
         *
         * @return Count of observations processed as an integer
        */
        inline int getCount() const
        {
            return static_cast<int>(count.item().toFloat());
        }

        inline torch::Tensor get_mean() const
        {
            return mean.clone();
        }

        inline torch::Tensor get_variance() const
        {
            return variance.clone();
        }
        inline void set_count(int count)
        {
            this->count[0] = count + 1e-8;
        }
    };
    TORCH_MODULE(RunningMeanstd);
}
#endif //LUNARALIGHTINGRL_RUNNINGMEANSTD_HPP