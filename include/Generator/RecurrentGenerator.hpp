#pragma once
//
// Created by moinshaikh on 1/27/26.
//

#ifndef LUNARALIGHTINGRL_RECURRENTGENERATOR_HPP
#define LUNARALIGHTINGRL_RECURRENTGENERATOR_HPP

#include<torch/torch.h>
#include "Generator.hpp"

namespace LunarAlighting
{
    /**
     * @class RecurrentGenerator
     * @brief A mini-batch generator for recurrent neural network-based reinforcement learning algorithms.
     *
     * @details The RecurrentGenerator class extends the base Generator class to provide efficient
     * mini-batch sampling for training recurrent policies (e.g., policies with LSTM or GRU layers).
     * It manages and distributes batches of experience data collected from multiple parallel
     * environments during policy rollouts.
     *
     * This generator maintains the complete trajectory history including observations, hidden states,
     * actions, value predictions, advantages, and other learner-specific metadata. It creates mini-batches
     * by partitioning the stored experience data according to the specified number of mini-batches,
     * allowing for more stable gradient estimates and improved sample efficiency during policy optimization.
     *
     * The class is particularly useful for implementing recurrent versions of policy gradient algorithms
     * such as PPO (Proximal Policy Optimization) and A2C (Advantage Actor-Critic) that require sequence
     * information and hidden state management across multiple time steps.
     *
     * @see Generator
     * @see MiniBatch
     */
    class RecurrentGenerator : public Generator
    {
    private:
         /**
     * @brief Batch of environment observations collected from all parallel environments.
     * @details Contains the observation tensor from each environment across all time steps
     * in the current rollout trajectory.
     */
    torch::Tensor observations;

    /**
     * @brief Recurrent hidden states from the neural network.
     * @details Stores the hidden state activations (e.g., LSTM cell states or GRU activations)
     * that are necessary for recurrent policy networks to maintain temporal context.
     */
    torch::Tensor hiddenStates;

    /**
     * @brief Actions sampled from the policy during the rollout.
     * @details Contains the actions taken in each environment at each time step,
     * collected during the policy execution phase.
     */
    torch::Tensor actions;

    /**
     * @brief Value function predictions for each observation.
     * @details Stores the estimated state values (baseline estimates) computed by the critic
     * network for advantage calculation and loss computation.
     */
    torch::Tensor valuePredictions;

    /**
     * @brief Discounted cumulative returns from each state.
     * @details Contains the sum of discounted future rewards for each time step,
     * used as the target for value function training and advantage normalization.
     */
    torch::Tensor returns;

    /**
     * @brief Episode termination masks for proper advantage/return bootstrapping.
     * @details Binary masks indicating whether each state is terminal (environment episode ended).
     * Used to properly handle trajectory boundaries and prevent value bootstrapping across episode boundaries.
     */
    torch::Tensor masks;

    /**
     * @brief Logarithmic probabilities of the sampled actions under the policy.
     * @details Stores log(π(action|observation)) for each action, needed for policy gradient
     * calculations and importance weighting in algorithms like PPO.
     */
    torch::Tensor actionLogProbs;

    /**
     * @brief Estimated advantages for each state-action pair.
     * @details Contains advantage estimates (typically returns - value_predictions),
     * used to reduce variance in policy gradient estimates and improve training stability.
     */
    torch::Tensor advantages;

    /**
     * @brief Indices used for sample shuffling within mini-batches.
     * @details Stores permutation indices for randomizing the order of mini-batch samples,
     * which helps decorrelate samples and improve convergence during training.
     */
    torch::Tensor indices;

    /**
     * @brief Current position in the experience buffer for mini-batch iteration.
     * @details Tracks which mini-batch should be returned next by the next() method.
     * Incremented after each next() call until done() returns true.
     */
    int index;

    /**
     * @brief Number of environments processed per mini-batch.
     * @details Calculated as total_observations / num_mini_batch, determines the size
     * of each mini-batch returned by the next() method.
     */
    int num_envs_per_batch;
    public:
        /**
         * @brief Constructs a RecurrentGenerator with experience data for mini-batch sampling.
         *
         * @param[in] numProcesses Total number of parallel environments used to collect experience.
         * @param[in] numMiniBatchSize Number of mini-batches to partition the collected experience into.
         *                           Higher values create smaller batches, increasing gradient noise
         *                           but allowing more frequent policy updates.
         * @param[in] observations The observation tensor of shape (num_steps, num_processes, obs_dim).
         * @param[in] hiddenStates The recurrent hidden states tensor containing LSTM/GRU state information.
         * @param[in] actions The action tensor containing actions sampled from the policy.
         * @param[in] valuePredictions The critic's value function estimates for each observation.
         * @param[in] returns The discounted cumulative returns computed via n-step returns or GAE.
         * @param[in] masks Episode termination masks distinguishing terminal and non-terminal states.
         * @param[in] actionLogProbs The log-probabilities of actions under the policy.
         * @param[in] advantags Pre-computed advantage estimates for variance reduction.
         *
         * @details The constructor initializes all internal state and computes num_envs_per_batch.
         * The provided tensors are stored by reference and used to generate mini-batches throughout
         * the epoch. The generator expects all input tensors to have consistent batch dimensions.
         *
         * @see MiniBatch
        */
        RecurrentGenerator(
            int numProcesses,
            int numMiniBatchSize,
            torch::Tensor observations,
            torch::Tensor hiddenStates,
            torch::Tensor actions,
            torch::Tensor valuePredictions,
            torch::Tensor returns,
            torch::Tensor masks,
            torch::Tensor actionLogProbs,
            torch::Tensor advantags

        );

        /**
         * @brief Checks whether all mini-batches have been generated.
         *
         * @return true if all mini-batches have been iterated through; false if more mini-batches remain.
         *
         * @details This method is used in training loops to control iteration. When done() returns true,
         * the index pointer has cycled through all num_mini_batch mini-batches and the generator
         * should be reset or reconstruction for the next epoch. Typically called at the start
         * of each loop iteration to determine whether to continue sampling.
         *
         * @code
         * while (!generator.done()) {
         *     MiniBatch batch = generator.next();
         *     // Process mini-batch
         * }
         * @endcode
        */
        virtual bool done() const;

        /**
         * @brief Retrieves the next mini-batch of experience data.
         *
         * @return A MiniBatch struct containing sliced tensors for a single mini-batch iteration.
         *         The returned mini-batch contains appropriately indexed subsets of all stored
         *         experience tensors (observations, actions, returns, advantages, etc.).
         *
         * @details Each call to next() returns a new mini-batch by slicing the stored experience
         * according to the current internal index. The mini-batch size is determined by num_envs_per_batch.
         * Successive calls increment the index, cycling through mini-batches sequentially.
         * When index reaches num_mini_batch, done() returns true and next() should no longer be called
         * until the generator is reinitialized.
         *
         * @warning Calling next() after done() returns true results in undefined behavior.
         *          Always check done() before calling next() in training loops.
         *
         * @see done()
         * @see MiniBatch
         */
        virtual MiniBatch next() ;





    };
}

#endif //LUNARALIGHTINGRL_RECURRENTGENERATOR_HPP