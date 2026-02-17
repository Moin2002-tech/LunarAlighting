//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_POLICY_HPP
#define LUNARALIGHTINGRL_POLICY_HPP

#include<vector>
#include<memory>

#include<torch/torch.h>
#include<torch/nn.h>
#include"nnBase.hpp"
#include"OutputLayers.hpp"
#include"../ObservationNormalizer.hpp"
#include"../Space.hpp"
#include"../../include/Model/mlp_base.hpp"
#include"../../include/Model/CNNBase.hpp"

namespace LunarAlighting
{
    /**
     * @class PolicyImpl
     * @brief Core policy network for reinforcement learning agents.
     *
     * PolicyImpl represents a complete policy network that combines:
     * - A neural network backbone (NNBase) for feature extraction
     * - An output layer for action distribution generation
     * - Optional observation normalization for input preprocessing
     *
     * The policy can work with both feedforward and recurrent (GRU-based) architectures.
     * It supports continuous, discrete, and binary action spaces through polymorphic output layers.
     *
     * The policy is responsible for:
     * - Sampling actions during exploration (act method)
     * - Computing log probabilities and values for training (evaluate_actions, get_probs, get_values)
     * - Normalizing observations if configured
     *
     * @note This class uses PyTorch's nn::Module and is wrapped with TORCH_MODULE macro for proper device/dtype handling.
     */
    class PolicyImpl : public torch::nn::Module
    {
    private:
        /**
        * @brief Specifies the action space type (Discrete, Box/Continuous, or MultiBinary).
        *
        * Determines which OutputLayer implementation to use (Categorical, Normal, or Bernoulli).
        * This is set during construction and defines the structure of action distributions.
        */
        ActionSpace actionSpace;

        /**
         * @brief The neural network backbone for feature extraction.
         *
         * Can be either feedforward (MLPBase) or recurrent (CNNBase/RecurrentBase) architecture.
         * Takes observations as input and produces feature representations that feed into
         * the output layer. The base network is responsible for handling state and hidden
         * recurrent states if using RNN architectures.
        */
        std::shared_ptr<NNBase> base;

        /**
        * @brief Optional observation preprocessing and normalization.
        *
        * Tracks running mean and standard deviation of observations to normalize them
        * to zero mean and unit variance. Improves training stability and sample efficiency.
        * Can be disabled (empty) for environments where normalization is not needed.
        */
        ObservationNormalizer obervationNormalizer;

        /**
         * @brief Output layer that generates probability distributions.
         *
         * Polymorphic output layer (BernoulliOutput, CategoricalOutput, or NormalOutput)
         * that transforms base network features into action probability distributions.
         * The specific type is determined by the action_space during construction.
         */
        std::shared_ptr<OutputLayer> outputLayer;

        /**
           * @brief Forward pass for recurrent (GRU-based) policy networks.
           *
           * Internal helper method that processes inputs through a GRU-based architecture,
           * managing hidden states and masking for episode boundaries.
           *
           * @param x Input tensor of observations, shape (batch_size, num_inputs).
           * @param hxs Hidden state tensor for the GRU, shape (num_layers * num_directions, batch_size, hidden_size).
           * @param masks Mask tensor for valid timesteps, shape (batch_size, 1). Used to reset hidden states
           *              at episode boundaries (0 = reset, 1 = continue).
           *
           * @return Vector of tensors containing [output_features, new_hidden_states].
           *
           * @note This is called internally by public methods when base->is_recurrent() returns true.
           */
        std::vector<torch::Tensor> forwardGatedRecurrentUnits(torch::Tensor x,
            torch::Tensor hxs,
            torch::Tensor masks);
    public:
            /**
         * @brief Constructs a Policy network with specified architecture and configuration.
         *
         * Initializes the policy by creating the appropriate output layer based on the action space,
         * setting up the neural network backbone, and optionally enabling observation normalization.
         *
         * @param action_space The action space specification (Discrete, Box, or MultiBinary).
         *                     Determines which output layer implementation to instantiate.
         * @param base The neural network backbone (feedforward or recurrent). Must not be null.
         *            Typically an MLPBase for feedforward or CNNBase/RecurrentBase for recurrent policies.
         * @param normalize_observations If true, enables observation normalization to improve training
         *                              stability. Default is false. Can be toggled on/off later.
         *
         * @throws std::invalid_argument if base is null or action_space is invalid.
         */
        PolicyImpl(ActionSpace actionSpace,std::shared_ptr<NNBase> base,
            bool normalizeObeservation = false);

        /**
            * @brief Samples actions from the policy and computes related values.
            *
            * This is the primary interface for exploration/inference. Takes observations and returns:
            * - Sampled actions for environment execution
            * - Action log probabilities
            * - Estimated state values
            * - Updated hidden states (for recurrent policies)
            *
            * @param inputs Observation tensors, shape (batch_size, num_inputs).
            * @param rnn_hxs Hidden state tensor for RNN policies, shape (batch_size, hidden_size).
            *               Ignored for feedforward policies.
            * @param masks Mask tensor for episode boundaries, shape (batch_size, 1).
            *             0 indicates episode reset (hidden state cleared), 1 indicates continuation.
            *
            * @return Vector of tensors: [sampled_actions, action_log_probs, state_values, new_rnn_hxs]
            *         - sampled_actions: Shape (batch_size, action_dim), sampled from the policy distribution
            *         - action_log_probs: Shape (batch_size, 1), log probability of sampled actions
            *         - state_values: Shape (batch_size, 1), value estimates from the value head
            *         - new_rnn_hxs: Updated hidden states (only for recurrent policies)
            *
            * @note Actions are sampled stochastically for exploration.
            */
        std::vector<torch::Tensor> act(
            torch::Tensor input,
            torch::Tensor rnn_hxs,
            torch::Tensor masks) const;

        /**
        * @brief Evaluates the log probabilities and state values for given observations and actions.
        *
        * Used during policy training to compute log probabilities of taken actions and value targets.
        * Unlike act(), this evaluates specific actions rather than sampling new ones.
        *
        * @param inputs Observation tensors, shape (batch_size, num_inputs).
        * @param rnn_hxs Hidden state tensor for RNN policies, shape (batch_size, hidden_size).
        *               Ignored for feedforward policies.
        * @param masks Mask tensor for episode boundaries, shape (batch_size, 1).
        *             0 indicates episode reset, 1 indicates continuation.
        * @param actions The actions to evaluate, shape (batch_size, action_dim).
        *               Must be valid actions from the policy's action space.
        *
        * @return Vector of tensors: [action_log_probs, state_values, distribution_entropy]
        *         - action_log_probs: Shape (batch_size, 1), log probability of provided actions
        *         - state_values: Shape (batch_size, 1), value estimates
        *         - distribution_entropy: Shape (batch_size, 1), entropy of the policy distribution
        *
        * @note Used in actor-critic algorithms for computing policy gradients and TD targets.
        */
        std::vector<torch::Tensor> evaluateAction(
            torch::Tensor  inputs,
            torch::Tensor rnn_hxs,
            torch::Tensor masks,
            torch::Tensor actions
            ) const;

        /**
         * @brief Computes action probabilities without sampling or computing values.
         *
         * Lightweight method to get the action distribution probabilities for analysis or
         * computing behavioral metrics without the overhead of value computation.
         *
         * @param inputs Observation tensors, shape (batch_size, num_inputs).
         * @param rnn_hxs Hidden state tensor for RNN policies, shape (batch_size, hidden_size).
         *               Ignored for feedforward policies.
         * @param masks Mask tensor for episode boundaries, shape (batch_size, 1).
         *
         * @return Probability tensor representing the action distribution.
         *         Shape depends on action_space:
         *         - Discrete: (batch_size, num_actions)
         *         - Continuous: (batch_size, action_dim) containing mean values
         *         - Binary: (batch_size, 1) containing probability of positive action
         *
         * @note Does not compute values or log probabilities, making it efficient for
         *       analysis-only use cases.
         */
        torch::Tensor getProbability(torch::Tensor inputs,
            torch::Tensor rnn_hxs,
            torch::Tensor masks) const;

        /**
        * @brief Computes state value estimates without computing action probabilities.
        *
        * Efficient method for bootstrapping value estimates during trajectory collection or
        * computing value targets without policy evaluation overhead.
        *
        * @param inputs Observation tensors, shape (batch_size, num_inputs).
        * @param rnn_hxs Hidden state tensor for RNN policies, shape (batch_size, hidden_size).
        *               Ignored for feedforward policies.
        * @param masks Mask tensor for episode boundaries, shape (batch_size, 1).
        *
        * @return Value tensor of shape (batch_size, 1) containing estimated state values.
        *         Values are real numbers that estimate expected cumulative future rewards.
        *
        * @note Used for computing Temporal Difference (TD) targets and bootstrapping.
        */
        torch::Tensor getValue(
            torch::Tensor inputs,
            torch::Tensor rnn_hxs,
            torch::Tensor masks) const;


        /**
        * @brief Updates the observation normalizer with new observation statistics.
        *
        * Incrementally updates the running mean and standard deviation estimates used for
        * normalizing observations. Should be called periodically (e.g., after each environment rollout)
        * to keep normalization statistics current.
        *
        * @param observations Tensor of observation samples to incorporate, shape (num_samples, obs_dim).
        *                    Samples should be representative of current environment states.
        *
        * @note Only has effect if observation normalization was enabled during construction.
        *       Call is safe but no-op if normalization is disabled.
        */
        void updateObervationNormalizer(torch::Tensor observations);

        inline bool is_recurrent() const {
            return base->isRecurrent();
        }

        inline unsigned int getHiddenSize() const {
            return base->getHiddenSize();
        }

        inline bool usingObservationNormalizer() const
        {
            return !obervationNormalizer.is_empty();
        }


    };
    /**
     * @brief PyTorch module wrapper for PolicyImpl.
     *
     * TORCH_MODULE(Policy) creates a Policy smart pointer type that automatically handles:
     * - Device transfer (CPU/GPU)
     * - Data type conversion (float/double)
     * - Proper cloning and state management
     *
     * Usage: auto policy = Policy(action_space, base_network);
    */
    TORCH_MODULE(Policy);
}




#endif //LUNARALIGHTINGRL_POLICY_HPP