#pragma once
//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_STORAGE_HPP
#define LUNARALIGHTINGRL_STORAGE_HPP

#include<c10/util/ArrayRef.h>
#include<torch/torch.h>

#include"Generator/Generator.hpp"
#include"Space.hpp"

namespace LunarAlighting
{
    /**
     * @brief Storage for rollouts used by on-policy algorithms
     *
     * `RolloutStorage` keeps tensors for observations, actions, rewards, value
     * predictions, returns and other per-step/per-process data required during
     * rollout collection and training. It provides utilities to insert new
     * transitions, compute returns (optionally using GAE), and create generators
     * for minibatch sampling.
     */

    class RolloutStorge
    {
    private:
        torch::Tensor observations;    /**< Observations tensor shaped [num_steps+1, num_processes, ...] */
        torch::Tensor hiddenStates;   /**< Hidden states for recurrent policies */
        torch::Tensor rewards;         /**< Rewards collected for each step */
        torch::Tensor valuePredictions; /**< Value function predictions */
        torch::Tensor returns;         /**< Computed returns (discounted rewards / GAE) */
        torch::Tensor actionLogProbs;/**< Log probabilities of chosen actions */
        torch::Tensor actions;         /**< Actions taken at each step */
        torch::Tensor masks;           /**< Masks indicating episode continuation */
        torch::Device device;         /**< Device where tensors are stored */
        int64_t numSteps;            /**< Number of steps in the rollout */
        int64_t step;/**< Current insertion index (step) */
    public:
        /**
         * @brief Constructs a RolloutStorage container
         *
         * Allocates storage tensors according to the provided shapes and parameters.
         *
         * @param numSteps Number of steps (length) of the rollout
         * @param numProcesses Number of parallel environment processes
         * @param observationsShape Shape of a single observation (per-process)
         * @param action_space Action space describing action tensor shape/type
         * @param hiddenStateSize Size of recurrent hidden state (0 for feed-forward)
         * @param device Torch device where tensors will be allocated
        */
        RolloutStorge(int64_t numSteps,
            int64_t numProcesses,
            c10::ArrayRef<int64_t> observationsShape,
            ActionSpace action_space,
            int64_t hiddenStateSize,
            torch::Device device
            );

        /**
        * @brief Constructs a combined RolloutStorage from individual storages
        *
        * Merges multiple per-process storages into a single storage on `device`.
        *
        * @param individual_storages Vector of pointers to per-process RolloutStorage
        * @param device Destination device for the merged tensors
        */
        RolloutStorge(std::vector<RolloutStorge *> individual_storages, torch::Device device);

        /**
         * @brief Prepare storage after a policy update
         *
         * Usually called after the model parameters are updated to shift the
         * last observation to the first index and reset the internal step counter.
         */
        void afterUpdate();

        /**
         * @brief Compute returns for the rollout
         *
         * Computes discounted returns or Generalized Advantage Estimation (GAE)
         * depending on `use_gae` and stores them into `returns`.
         *
         * @param next_value Estimated value for the state following the last step
         * @param use_gae If true, use GAE formula to compute advantages/returns
         * @param gamma Discount factor for future rewards
         * @param tau GAE lambda parameter (smoothing)
        */
        void computeReturns(torch::Tensor nextValue,bool useGae,float gamma,float tau);


        /**
        * @brief Creates a feed-forward minibatch generator
        *
        * Produces a `Generator` that yields minibatches from the collected
        * rollout data for feed-forward policies.
        *
        * @param advantages Tensor of computed advantages
        * @param num_mini_batch Number of minibatches to split into
        * @return Unique pointer to a configured `Generator`
        */
        std::unique_ptr<Generator> feed_forward_generator(torch::Tensor advantages,
                                                    int num_mini_batch);
        /**
         * @brief Inserts a single time-step transition into storage
         *
         * Appends the provided tensors at the current `step` index and advances
         * the internal counter.
         *
         * @param observation Observation tensor for this step
         * @param hiddenState Hidden state tensor (for recurrent policies)
         * @param action Action tensor taken at this step
         * @param actionLogProb Log probability of `action`
         * @param valuePrediction Value function prediction at this step
         * @param reward Reward received after taking the action
         * @param mask Mask indicating whether episode continues (1.0) or ended (0.0)
         */
        void insert(torch::Tensor observation,
                torch::Tensor hiddenState,
                torch::Tensor action,
                torch::Tensor actionLogProb,
                torch::Tensor valuePrediction,
                torch::Tensor reward,
                torch::Tensor mask);

        /**
         * @brief Creates a recurrent minibatch generator
         *
         * Used for recurrent policies; returns minibatches that preserve sequence
         * ordering and hidden state grouping.
         *
         * @param advantages Tensor of computed advantages
         * @param numMiniBatch Number of minibatches to split into
         * @return Unique pointer to a configured `Generator`
         */
        std::unique_ptr<Generator> recurrentGenerator(torch::Tensor advantages,
                                               int numMiniBatch);


    /**
     * @brief Sets the initial (first) observation in the storage
     *
     * Typically used to seed `observations[0]` before starting rollout
     * collection when the next observation after the last step is known.
     *
     * @param observation Observation tensor to store as the first frame
     */
    void setFirstObservation(torch::Tensor observation);

    /**
     * @brief Moves all internal tensors to the specified device
     *
     * @param device Destination torch device
     */
    void to(torch::Device device);

    /** @return Reference to the stored actions tensor */
    inline const torch::Tensor &get_actions() const
    {
        return actions;
    }

    /** @return Reference to the stored action log probabilities tensor */
    inline const torch::Tensor &get_action_log_probs() const
    {
        return actionLogProbs;
    }

    /** @return Reference to the stored hidden states tensor */
    inline const torch::Tensor &get_hidden_states() const {
        return hiddenStates;
    }

    /** @return Reference to the stored masks tensor */
    inline const torch::Tensor &get_masks() const {
        return masks;
    }

    /** @return Reference to the stored observations tensor */
    inline const torch::Tensor &get_observations() const {
        return observations;
    }

    /** @return Reference to the stored returns tensor */
    inline const torch::Tensor &get_returns() const {
        return returns;
    }

    /** @return Reference to the stored rewards tensor */
    inline const torch::Tensor &get_rewards() const {
        return rewards;
    }

    /** @return Reference to the stored value predictions tensor */
    inline const torch::Tensor &get_value_predictions() const
    {
        return valuePredictions;
    }

    /** @brief Replaces stored actions tensor */
    inline void set_actions(torch::Tensor actions) { this->actions = actions; }

    /** @brief Replaces stored action log probabilities tensor */
    inline void set_action_log_probs(torch::Tensor actionLogProbs) { this->actionLogProbs = actionLogProbs; }

    /** @brief Replaces stored hidden states tensor */
    inline void set_hidden_states(torch::Tensor hiddenStates) { this->hiddenStates = hiddenStates; }

    /** @brief Replaces stored masks tensor */
    inline void set_masks(torch::Tensor masks) { this->masks = masks; }

    /** @brief Replaces stored observations tensor */
    inline void set_observations(torch::Tensor observations) { this->observations = observations; }

    /** @brief Replaces stored returns tensor */
    inline void set_returns(torch::Tensor returns) { this->returns = returns; }

    /** @brief Replaces stored rewards tensor */
    inline void set_rewards(torch::Tensor rewards) { this->rewards = rewards; }

    /** @brief Replaces stored value predictions tensor */
    inline void set_value_predictions(torch::Tensor valuePredictions)
    {
        this->valuePredictions = valuePredictions;
    }

    };
}




#endif //LUNARALIGHTINGRL_STORAGE_HPP