#pragma once
//
// Created by moinshaikh on 1/27/26.
//

#ifndef LUNARALIGHTINGRL_FEEDFORWARDGENERATOR_HPP
#define LUNARALIGHTINGRL_FEEDFORWARDGENERATOR_HPP

#include<torch/torch.h>

#include "Generator.hpp"


namespace LunarAlighting
{
    /**
    * @class FeedForwardGenerator
    * @brief Concrete implementation of Generator for feed-forward neural network policies.
    *
    * FeedForwardGenerator manages the generation of mini-batches from collected trajectory data.
    * It implements a shuffled sampling strategy where trajectories are randomly reordered and divided
    * into fixed-size mini-batches. This is suitable for feed-forward architectures that process
    * independent samples without recurrent state.
    *
    * The generator maintains an internal index tracking position through the shuffled data,
    * allowing iterative retrieval of mini-batches until all data has been consumed.
    */
    class FeedForwardGenerator : public Generator
    {
    private:
        /** @brief Input observations from the collected trajectory. */
        torch::Tensor observations;

        /** @brief Recurrent hidden states (typically zero for feed-forward networks). */
        torch::Tensor hiddenStates;

        /** @brief Actions taken during trajectory collection. */
        torch::Tensor actions;

        /** @brief Predicted values from the critic network. */
        torch::Tensor valuePredictions;

        /** @brief Discounted cumulative returns computed from rewards. */
        torch::Tensor returns;

        /** @brief Masks for invalid transitions (e.g., at episode boundaries). */
        torch::Tensor masks;

        /** @brief Log probabilities of actions under the policy distribution. */
        torch::Tensor actionLogProbs;

        /** @brief Advantage estimates for each timestep (typically from GAE). */
        torch::Tensor advantages;

        /** @brief Shuffled indices for random sampling of the trajectory data. */
        torch::Tensor indices;

        /** @brief Current position in the shuffled data sequence. */
        int index;

    public:
        /**
         * @brief Constructor for FeedForwardGenerator.
         *
         * Initializes the generator with trajectory data and prepares for mini-batch generation.
         * The constructor internally shuffles the data indices to enable randomized sampling.
         *
         * @param miniBatchSize Size of each mini-batch to generate (number of samples per batch)
         * @param observations Tensor of observations from the trajectory (shape: [trajectory_length, ...])
         * @param hiddenStates Tensor of hidden states (typically zeros for feed-forward policies)
         * @param actions Tensor of actions taken (shape: [trajectory_length, action_dim])
         * @param valuePredictions Tensor of value predictions from critic (shape: [trajectory_length])
         * @param returns Tensor of computed returns (shape: [trajectory_length])
         * @param masks Tensor of transition validity masks (shape: [trajectory_length])
         * @param actionLogProbs Tensor of log probabilities of actions (shape: [trajectory_length])
         * @param advantages Tensor of advantage estimates (shape: [trajectory_length])
        */
        FeedForwardGenerator(
            int miniBatchSize,
           torch::Tensor observations,
           torch::Tensor hiddenStates,
           torch::Tensor actions,
           torch::Tensor valuePredictions,
           torch::Tensor returns,
           torch::Tensor masks,
           torch::Tensor actionLogProbs,
           torch::Tensor advantages);


        /**
         * @brief Check if all mini-batches have been generated.
         *
         * @return true if the internal index has reached the end of the trajectory data, false otherwise
        */
        virtual bool done() const;

        /**
         * @brief Retrieve the next mini-batch of training data.
         *
         * This method should be called only when done() returns false.
         *
         * @return MiniBatch containing the next set of training data
        */
        virtual MiniBatch next();
    };

}
#endif //LUNARALIGHTINGRL_FEEDFORWARDGENERATOR_HPP