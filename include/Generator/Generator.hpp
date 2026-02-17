#pragma once
//
// Created by moinshaikh on 1/27/26.
//



#ifndef LUNARALIGHTINGRL_GENERATOR_HPP
#define LUNARALIGHTINGRL_GENERATOR_HPP

#include<vector>
#include<torch/torch.h>

namespace LunarAlighting
{
    /**
     * @struct MiniBatch
     * @brief Container for a mini-batch of training data.
     *
     * MiniBatch encapsulates all the tensors required for a single training iteration,
     * including observations, actions, rewards, and auxiliary information used by
     * policy gradient and value-based reinforcement learning algorithms.
    */
    struct MiniBatch
    {
        torch::Tensor observations;        /**< @brief Input observations from the environment. */
        torch::Tensor hiddenStates;       /**< @brief Hidden states for recurrent neural networks. */
        torch::Tensor actions;             /**< @brief Actions taken during the trajectory. */
        torch::Tensor valuePredictions;   /**< @brief Value function predictions for each step. */
        torch::Tensor returns;             /**< @brief Discounted cumulative returns (targets for value learning). */
        torch::Tensor masks;               /**< @brief Episode end masks for handling episode boundaries. */
        torch::Tensor actionLogProbs;    /**< @brief Log probabilities of the actions under the policy. */
        torch::Tensor advantages;          /**< @brief Advantage estimates (returns - value_predictions). */

        /**
         * @brief Default constructor.
         *
         * Creates an empty MiniBatch with uninitialized tensors.
        */
        MiniBatch() {}

        /**
         * @brief Parameterized constructor.
         *
         * Initializes all MiniBatch tensors with the provided values.
         *
         * @param observations Environment observations.
         * @param hiddenStates Recurrent network hidden states.
         * @param actions Actions taken from observations.
         * @param valuePredictions Estimated state values.
         * @param returns Target returns for value function updates.
         * @param masks Masks indicating valid vs. padded steps.
         * @param actionLogProbs Log probabilities of executed actions.
         * @param advantages Advantage estimates for policy updates.
        */

        MiniBatch(
          torch::Tensor observation,
          torch::Tensor hiddenStates,
          torch::Tensor actions,
          torch::Tensor valuePredictions,
          torch::Tensor returns,
          torch::Tensor masks,
          torch::Tensor actionLogProbs,
          torch::Tensor advantages
        ) :
        observations(observation),
        hiddenStates(hiddenStates),
        actions(actions),
        valuePredictions(valuePredictions),
        returns(returns),
        masks(masks),
        actionLogProbs(actionLogProbs),
        advantages(advantages)
        {


        }

    };

    /**
     * @class Generator
     * @brief Abstract base class for generating mini-batches of training data.
     *
     * Generator defines the interface for sampling mini-batches from a larger dataset
     * of collected trajectories. Implementations include feed-forward and recurrent variants
     * for different agent architectures in reinforcement learning training loops.
     */

    class Generator
    {
    public:
        /**
        * @brief Virtual destructor for proper cleanup of derived classes.
        */
        virtual ~Generator();

        /**
         * @brief Check if there are more mini-batches to generate.
         *
         * @return true if all mini-batches have been consumed, false if more mini-batches are available
        */
        virtual bool done() const = 0;

        /**
         * @brief Retrieve the next mini-batch of training data.
         *
         * This method should be called only when done() returns false.
         *
         * @return MiniBatch containing the next set of training data
        */
        virtual MiniBatch next() = 0;
    };

    inline Generator::~Generator() {}
}

#endif //LUNARALIGHTINGRL_GENERATOR_HPP