//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_MLP_BASE_HPP
#define LUNARALIGHTINGRL_MLP_BASE_HPP


#include<torch/nn.h>
#include"nnBase.hpp"



namespace LunarAlighting
{
    /**
     * @brief Multi-layer perceptron (MLP) base policy network
     *
     * `MlpBase` is a fully-connected neural network architecture that serves as the
     * backbone for actor-critic reinforcement learning policies. It maintains separate
     * sequential networks for the actor (policy) and critic (value function) with a
     * shared feature extraction trunk. Supports both feed-forward and recurrent (GRU-based)
     * architectures through inheritance from `NNBase`.
     *
     * The architecture typically consists of:
     * - A shared feature trunk (if non-recurrent)
     * - Separate actor head for action distribution
     * - Separate critic head with final linear layer for value estimation
     *
     * @note Inherits from `NNBase` to leverage recurrent infrastructure and maintain
     *       consistent interface across policy types.
     */
    class MlpBase : public NNBase
    {
    private:
        torch::nn::Sequential actor;      /**< Sequential module for actor (policy) network.
                                   Processes shared features to produce action
                                   distribution parameters. */
        torch::nn::Sequential critic;     /**< Sequential module for critic (value) network.
                                        Processes shared features through hidden layers. */
        torch::nn::Linear criticLinear;  /**< Final linear layer mapping critic hidden
                                        representation to single scalar value output. */
        unsigned int numInputs;   /**< Dimensionality of input observations.
                                        Determines size of first layer in networks. */

    public:
        /**
           * @brief Constructs an MLP-based policy network
           *
           * Initializes the actor and critic networks with fully-connected layers,
           * optionally with GRU recurrence. The networks share a common input dimension
           * but maintain separate processing pipelines for policy and value function.
           *
           * @param num_inputs Dimensionality of input observation space.
           *                   Determines the input layer size for both actor/critic.
           * @param recurrent  If true, uses GRU recurrence for temporal processing;
           *                   if false, processes each timestep independently (feed-forward).
           *                   Default: false.
           * @param hidden_size Dimension of hidden layers and GRU state.
           *                    Used for internal representation size in actor/critic networks.
           *                    Larger values increase model capacity but also computation.
           *                    Default: 64.
           *
           * @note The constructor initializes networks but does not perform weight
           *       initialization. Call init_weights() separately for custom initialization.
        */
        MlpBase(unsigned int numInputs,
            bool recurrent = false,
            unsigned int hiddenSize = 64);

        /**
        * @brief Forward pass through the MLP policy network
        *
        * Processes observations through the actor and critic networks, returning
        * action distribution parameters and value estimate. Overrides the virtual
        * `forward()` method from `NNBase`.
        *
        * @param inputs Observation tensor with shape [sequence_length, batch_size, num_inputs]
        *               or [batch_size, num_inputs] for single timesteps.
        * @param hxs Hidden state tensor for recurrent processing.
        *            Shape: [num_layers, batch_size, hidden_size] for GRU or
        *            empty tensor for feed-forward architecture.
        * @param masks Mask tensor indicating valid timesteps and episode boundaries.
        *              Used to handle variable-length sequences and reset recurrent state
        *              at episode boundaries. Shape: [sequence_length, batch_size].
        *
        * @return Vector of output tensors containing:
        *         - [0] Value estimate from critic network (scalar per sample)
        *         - [1] Actor output (action distribution parameters)
        *         - [2] Updated hidden states (for recurrent architectures)
        */
        std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                      torch::Tensor hxs,
                                      torch::Tensor masks) override;

        /**
        * @brief Gets the input observation dimensionality
        *
        * Accessor for the number of input features that this MLP expects.
        * Useful for validation and architecture inspection.
        *
        * @return Dimensionality of the input observation space (num_inputs)
        */
        inline unsigned int getNumInputs() const
        {
            return numInputs;
        }



    };
}
#endif //LUNARALIGHTINGRL_MLP_BASE_HPP