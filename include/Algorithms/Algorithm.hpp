#pragma once

//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_ALGORITHM_HPP
#define LUNARALIGHTINGRL_ALGORITHM_HPP
#include<string>
#include<vector>


#include"../Storage.hpp"

namespace LunarAlighting
{
    /**
     * @brief Data structure for storing algorithm training metrics
     *
     * `UpdateDatum` encapsulates a single scalar metric produced during a training
     * update step. Each metric is identified by a name (for logging and monitoring)
     * and contains a floating-point value representing the measurement.
     *
     * Typical uses:
     * - Policy loss from actor network
     * - Value loss from critic network
     * - Entropy bonus values
     * - Gradient norms or weight magnitudes
     * - Any other per-step training statistic
     *
     * Multiple `UpdateDatum` objects are typically collected in a vector and
     * returned from Algorithm::update() to provide comprehensive training diagnostics.
     */
    struct UpdateDatum
    {
        std::string name; /**< Identifier for this metric (e.g., "policy_loss", "value_loss").
                          Used for logging, plotting, and tensorboard integration. */
        float value;      /**< Scalar metric value. Can represent loss, reward, entropy,
                               or any other floating-point training statistic. */
    };

    /**
     * @brief Abstract base class for on-policy reinforcement learning algorithms
     *
     * `Algorithm` defines the interface for implementing on-policy RL algorithms
     * (e.g., A2C, PPO, A3C). Derived classes implement specific policy update rules
     * and loss computations while maintaining a consistent training interface.
     *
     * The algorithm pattern:
     * 1. Collect rollouts through environment interaction (managed elsewhere)
     * 2. Store rollouts in RolloutStorage
     * 3. Call update() with the storage to perform gradient updates
     * 4. Receive training metrics (losses, entropy, etc.) for monitoring
     *
     * Subclasses must override the update() method to implement their specific
     * algorithm (PPO clipping, A2C advantage calculation, etc.).
     *
     * @note This is an abstract base class; instantiation requires a concrete
     *       implementation like PPO or A2C.
     */
    class Algorithms
    {
    public:
        /**
         * @brief Virtual destructor for proper cleanup in inheritance hierarchy
         *
         * Declared as pure virtual (= 0) with inline empty implementation to force
         * derived classes to provide their own destructors. This ensures proper
         * virtual method resolution and cleanup of derived-class resources.
         */
        virtual ~Algorithms() = 0;


        /**
         * @brief Performs a single training update on accumulated rollouts
         *
         * Core method that executes the algorithm's policy and value function updates.
         * Computes losses based on collected experience in `rollouts`, performs
         * backpropagation, and applies gradient updates to network parameters.
         *
         * The specific update behavior depends on the concrete algorithm:
         * - PPO: Applies importance-weighted clipped objectives with multiple epochs
         * - A2C: Single-step advantage actor-critic updates
         * - A3C: Asynchronous advantage actor-critic with n-step returns
         *
         * @param rollouts Reference to RolloutStorage containing collected experience.
         *                 Contains observations, actions, rewards, value predictions,
         *                 and other per-step data needed for training.
         * @param decay_level Scaling factor for learning rate or weight decay (default: 1).
         *                    Values < 1 reduce learning rate, useful for annealing or
         *                    adjusting training intensity during multi-phase training.
         *                    Interpretation depends on concrete algorithm implementation.
         *
         * @return Vector of UpdateDatum objects containing training metrics from this update.
         *         Typical metrics include:
         *         - "policy_loss": Loss from the actor network
         *         - "value_loss": Loss from the critic network
         *         - "entropy": Policy entropy (for exploration tracking)
         *         - "policy_gradient_norm": Gradient magnitude for monitoring
         *         - Algorithm-specific metrics (e.g., "ppo_clip_fraction" for PPO)
         *
         * @note The update() method is pure virtual and must be implemented by derived
         *       classes. Each algorithm implementation defines its own loss computation,
         *       clipping strategies, and optimization schedules.
         *
         * @see PPO for clipped policy optimization example
         * @see A2C for advantage actor-critic example
         */
        virtual std::vector<UpdateDatum> update(RolloutStorge &rolloutStorage, float decayLevel = 1) = 0;
    };
    inline Algorithms::~Algorithms() {}
}


#endif //LUNARALIGHTINGRL_ALGORITHM_HPP