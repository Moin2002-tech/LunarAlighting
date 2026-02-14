#pragma once
//
// Created by moinshaikh on 1/27/26.
//

#ifndef LUNARALIGHTINGRL_OBSERVATIONNORMALIZER_HPP
#define LUNARALIGHTINGRL_OBSERVATIONNORMALIZER_HPP

#include<torch/torch.h>
#include<torch/nn.h>
#include<vector>
#include"RunningMeanStd.hpp"

#include"RunningMeanStd.hpp"

namespace LunarAlighting
{
    class ObservationNormalizer;
    class ObservationNormalizerImpl : public torch::nn::Module
    {
    private:
        torch::Tensor clip; /**< Clipping value for normalized observations to bound output range */
        RunningMeanstd rms; /**< Running mean and standard deviation tracker for observations */
    public:
        /**
         * @brief Constructs an ObservationNormalizer with specified observation size
         *
         * Initializes the normalizer with zero mean and unit variance, ready to begin
         * collecting statistics from incoming observations.
         *
         * @param size Dimensionality of observations to normalize
         * @param clip Maximum absolute value for normalized observations (default: 10.0)
        */
        explicit ObservationNormalizerImpl(int size, float clip = 10.0);

        /**
         * @brief Constructs an ObservationNormalizer with provided statistics
         *
         * Initializes the normalizer with specific mean and variance values, useful for
         * loading pre-computed statistics or continuing from a checkpoint.
         *
         * @param mean Initial mean values for each observation dimension
         * @param variances Initial variance values for each observation dimension
         * @param clip Maximum absolute value for normalized observations (default: 10.0)
         */
        ObservationNormalizerImpl(const std::vector<float> &mean,const std::vector<float> &variances,float clip = 10.0);

        /**
         * @brief Constructs an ObservationNormalizer by aggregating statistics from multiple normalizers
         *
         * Combines statistics from multiple ObservationNormalizer instances into a single
         * normalizer, useful for merging parallel environment statistics.
         *
         * @param others Vector of ObservationNormalizer instances to aggregate
         */
        explicit ObservationNormalizerImpl(const std::vector<ObservationNormalizer> &others);

        /**
        * @brief Normalizes a single observation
        *
        * Applies z-score normalization (subtract mean, divide by std deviation) to the
        * observation and clips the result to the configured bounds.
        *
        * @param observation Raw observation tensor to normalize
        * @return Normalized observation tensor with clipped values
        */

        torch::Tensor processObservation(torch::Tensor &observation) const;

        std::vector<float> getMean() const;
        std::vector<float> getVariances() const;

        /**
         * @brief Updates normalization statistics with new observations
         *
         * Processes a batch of observations to update the running mean and variance
         * statistics used for future normalizations.
         *
         * @param observations Batch of observation tensors to incorporate into statistics
         */
        void update(torch::Tensor observation);

        inline float getClipValue() const
        {
            return clip.item().toFloat();
        }

        inline int  getStepCount() const {
            return rms->getCount();
        }
    };
    TORCH_MODULE (ObservationNormalizer);
}

#endif //LUNARALIGHTINGRL_OBSERVATIONNORMALIZER_HPP