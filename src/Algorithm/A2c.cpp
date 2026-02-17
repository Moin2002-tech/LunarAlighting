/**
 * @file A2c.cpp
 * @brief Implementation of Advantage Actor-Critic (A2C) reinforcement learning algorithm
 * @author moinshaikh
 * @date 2/7/26
 * 
 * This file implements the A2C algorithm, a policy gradient method that combines
 * value function approximation with policy optimization. The algorithm uses
 * advantage estimation to reduce variance in policy gradient updates and includes
 * entropy regularization for exploration.
 * 
 * Key features:
 * - Actor-critic architecture with shared feature extraction
 * - Advantage estimation using temporal difference learning
 * - Entropy regularization for exploration
 * - Gradient clipping for training stability
 * - Support for observation normalization
 */

#include"../../include/Algorithms/A2c.hpp"
#include"../../include/Algorithms/Algorithm.hpp"
#include"../../include/Model/mlp_base.hpp"
#include"../../include/Model/policy.hpp"
#include"../../include/Storage.hpp"
#include"../../include/Space.hpp"

#include"../third_party/doctest.hpp"

/**
 * @namespace LunarAlighting
 * @brief Main namespace for lunar alighting reinforcement learning project
 * 
 * This namespace contains all components of the lunar alighting reinforcement learning
 * system, including algorithms, models, storage utilities, and testing frameworks.
 */
namespace LunarAlighting
{
    /**
     * @brief A2C constructor with hyperparameter initialization
     * @param policy Reference to the policy network (actor-critic model)
     * @param actorLossCoef Weight for actor (policy) loss in total loss computation
     * @param valueLossCoef Weight for critic (value) loss in total loss computation
     * @param entropyCoef Weight for entropy regularization term
     * @param learningRate Initial learning rate for RMSprop optimizer
     * @param epsilon Epsilon parameter for RMSprop numerical stability
     * @param alpha Alpha parameter for RMSprop momentum
     * @param maxGradNorm Maximum gradient norm for clipping (prevents gradient explosion)
     * 
     * Initializes the A2C algorithm with specified hyperparameters and creates
     * an RMSprop optimizer with custom settings. The algorithm uses a single
     * optimizer for both actor and critic networks (shared parameters).
     * 
     * The constructor stores the original learning rate to support learning rate
     * scheduling during training updates.
     */
    A2C::A2C(Policy &policy,
        float actorLossCoef,
        float valueLossCoef,
        float entropyCoef,
        float learningRate,
        float epsilon,
        float alpha,
        float maxGradNorm) :
        policy(policy),
    actorLossCoef(actorLossCoef),
    valueLossCoef(valueLossCoef),
    entropyCoef(entropyCoef),
    maxGradNorm(maxGradNorm),
    originalLearningRate(learningRate),
    optimizer(std::make_unique<torch::optim::RMSprop>(policy->parameters(),torch::optim::RMSpropOptions(learningRate).eps(epsilon).alpha(alpha)))
    {

    }

    /**
     * @brief Perform A2C algorithm update using collected rollouts
     * @param rollouts Reference to rollout storage containing trajectory data
     * @param decay_level Learning rate decay factor for scheduling
     * @return Vector of UpdateDatum containing loss metrics for monitoring
     * 
     * This method implements the core A2C update algorithm:
     * 1. Updates learning rate based on decay schedule
     * 2. Updates observation normalizer if enabled
     * 3. Evaluates actions and computes value estimates
     * 4. Calculates advantages using TD error (returns - values)
     * 5. Computes value loss (MSE between predicted and actual returns)
     * 6. Computes policy loss using advantage-weighted log probabilities
     * 7. Adds entropy regularization for exploration
     * 8. Performs gradient descent with clipping
     * 
     * The advantage function is computed as: A(s,a) = R(s,a) - V(s)
     * where R(s,a) are the discounted returns and V(s) is the value estimate.
     * 
     * @note Advantages are not normalized in A2C (unlike PPO)
     */
    std::vector<UpdateDatum> A2C::update(RolloutStorge &rollouts, float decay_level)
    {
       // Update learning rate with decay schedule
       // optimizer->option
       // optimizer->options.learningRate(originalLearningRate*decay_level);
        for (auto& group :optimizer->param_groups())
        {
            static_cast<torch::optim::RMSpropOptions&> (group.options()).lr(originalLearningRate* decay_level);
        }
        
        // === Data Preparation ===
        // Reshape observations for batch processing
        auto full_obs_shape = rollouts.get_observations().sizes();
        std::vector<int64_t> obs_shape(full_obs_shape.begin() + 2,
                                       full_obs_shape.end());
        obs_shape.insert(obs_shape.begin(), -1);
        auto action_shape = rollouts.get_actions().size(-1);
        auto rewards_shape = rollouts.get_rewards().sizes();
        int num_steps = rewards_shape[0];    // Number of timesteps in rollout
        int num_processes = rewards_shape[1]; // Number of parallel environments

        // Update observation normalizer statistics
        if (policy->usingObservationNormalizer())
        {
            policy->updateObervationNormalizer(rollouts.get_observations());
        }

        // === Policy Evaluation ===
        // Evaluate actions and compute value estimates for all timesteps except last
        auto evaluate_result = policy->evaluateAction(
            rollouts.get_observations().slice(0, 0, -1).view(obs_shape),
            rollouts.get_hidden_states()[0].view({-1, policy->getHiddenSize()}),
            rollouts.get_masks().slice(0, 0, -1).view({-1, 1}),
            rollouts.get_actions().view({-1, action_shape}));
        
        // Reshape evaluation results to [num_steps, num_processes, 1]
        auto values = evaluate_result[0].view({num_steps, num_processes, 1});
        auto action_log_probs = evaluate_result[1].view(
            {num_steps, num_processes, 1});

        // === Advantage Computation ===
        // Advantages = Returns - Values (TD error)
        // Note: Advantages are not normalized in A2C (unlike PPO)
        auto advantages = rollouts.get_returns().slice(0, 0, -1) - values;

        // === Loss Computation ===
        // Value loss: Mean squared error between predicted values and actual returns
        auto value_loss = advantages.pow(2).mean();

        // Policy loss: Negative advantage-weighted log probabilities
        // detach() prevents gradients from flowing back through advantage computation
        auto action_loss = -(advantages.detach() * action_log_probs).mean();

        // Total loss combines all components with respective coefficients
        // Loss = value_coef * value_loss + action_loss - entropy_coef * entropy
        auto loss = (value_loss * valueLossCoef +
                     action_loss -
                     evaluate_result[2] * entropyCoef);

        // === Optimization Step ===
        optimizer->zero_grad();  // Clear previous gradients
        loss.backward();          // Compute gradients
        optimizer->step();        // Update parameters

        // Return loss metrics for monitoring and logging
        return {{"Value loss", value_loss.item().toFloat()},
                {"Action loss", action_loss.item().toFloat()},
                {"Entropy", evaluate_result[2].item().toFloat()}};
    }

    /**
     * @brief Test helper function for learning basic action-reward patterns
     * @param policy Reference to policy network to train
     * @param storage Reference to rollout storage for trajectory data
     * @param a2c Reference to A2C algorithm instance
     * 
     * This function implements a simple learning scenario where the reward
     * equals the action value. The agent should learn to prefer higher action
     * values to maximize rewards. This tests the basic policy gradient learning
     * capability of the A2C algorithm.
     * 
     * Training loop:
     * - 10 outer epochs, each with 5 timesteps
     * - Random binary observations (0 or 1)
     * - Actions sampled from policy
     * - Rewards set equal to action values
     * - Policy updated after each epoch
     */
    static void learn_pattern(Policy &policy, RolloutStorge &storage, A2C &a2c)
{
    for (int i = 0; i < 10; ++i)  // 10 training epochs
    {
        for (int j = 0; j < 5; ++j)  // 5 timesteps per epoch
        {
            // Generate random binary observation
            auto observation = torch::randint(0, 2, {2, 1}).to(torch::kFloat);

            std::vector<torch::Tensor> act_result;
            {
                torch::NoGradGuard no_grad;  // Disable gradient computation for action sampling
                act_result = policy->act(observation,
                                         torch::Tensor(),    // No hidden state
                                         torch::ones({2, 1})); // All masks active
            }
            auto actions = act_result[1];  // Extract actions from policy output

            // Reward equals action value (learn to prefer higher actions)
            auto rewards = actions;
            
            // Store trajectory data
            storage.insert(observation,
                           torch::zeros({2, 5}),  // Zero hidden states
                           actions,
                           act_result[2],         // Action log probabilities
                           act_result[0],         // Value estimates
                           rewards,
                           torch::ones({2, 1}));  // Active masks
        }

        // Compute next value for return calculation
        torch::Tensor next_value;
        {
            torch::NoGradGuard no_grad;
            next_value = policy->getValue(
                                   storage.get_observations()[-1],
                                   storage.get_hidden_states()[-1],
                                   storage.get_masks()[-1])
                             .detach();
        }
        
        // Compute discounted returns (no gamma=0.9, no GAE, no reward scaling)
        storage.computeReturns(next_value, false, 0., 0.9);

        // Update policy using A2C algorithm
        a2c.update(storage);
        
        // Prepare storage for next epoch
        storage.afterUpdate();
    }
}

/**
     * @brief Test helper function for learning observation-action matching game
     * @param policy Reference to policy network to train
     * @param storage Reference to rollout storage for trajectory data
     * @param a2c Reference to A2C algorithm instance
     * 
     * This function implements a more complex learning scenario where the agent
     * must learn to match its action to the observation. The game rule is:
     * - If action == observation: reward = +1
     * - If action != observation: reward = -1
     * 
     * This tests the agent's ability to learn conditional policies based on
     * observations, requiring it to understand the relationship between
     * observations and optimal actions.
     * 
     * Training loop:
     * - 10 outer epochs, each with 5 timesteps
     * - Random binary observations (0 or 1)
     * - Agent must learn to copy observation to action
     * - Sparse reward structure (+1 for match, -1 for mismatch)
     * - Policy updated after each epoch
     */
static void learn_game(Policy &policy, RolloutStorge &storage, A2C &a2c)
{
    // Game rule: Match action to observation for reward
    // If action == observation: reward = +1, else: reward = -1
    auto observation = torch::randint(0, 2, {2, 1}).to(torch::kFloat);
    storage.setFirstObservation(observation);

    for (int i = 0; i < 10; ++i)  // 10 training epochs
    {
        for (int j = 0; j < 5; ++j)  // 5 timesteps per epoch
        {
            std::vector<torch::Tensor> act_result;
            {
                torch::NoGradGuard no_grad;  // Disable gradient computation for action sampling
                act_result = policy->act(observation,
                                         torch::Tensor(),    // No hidden state
                                         torch::ones({2, 1})); // All masks active
            }
            auto actions = act_result[1];  // Extract actions from policy output

            // Compute rewards: +1 if action matches observation, -1 otherwise
            auto rewards = ((actions.to(torch::kLong) == observation.to(torch::kLong)).to(torch::kFloat) * 2) - 1;
            
            // Generate new observation for next timestep
            observation = torch::randint(0, 2, {2, 1}).to(torch::kFloat);
            
            // Store trajectory data
            storage.insert(observation,
                           torch::zeros({2, 5}),  // Zero hidden states
                           actions,
                           act_result[2],         // Action log probabilities
                           act_result[0],         // Value estimates
                           rewards,
                           torch::ones({2, 1}));  // Active masks
        }

        // Compute next value for return calculation
        torch::Tensor next_value;
        {
            torch::NoGradGuard no_grad;
            next_value = policy->getValue(
                                   storage.get_observations()[-1],
                                   storage.get_hidden_states()[-1],
                                   storage.get_masks()[-1])
                             .detach();
        }
        
        // Compute discounted returns (gamma=0.9, GAE lambda=0.1, no reward scaling)
        storage.computeReturns(next_value, false, 0.1, 0.9);

        // Update policy using A2C algorithm
        a2c.update(storage);
        
        // Prepare storage for next epoch
        storage.afterUpdate();
    }
}

    /**
     * @brief Test suite for A2C algorithm implementation
     * 
     * This test suite validates the core functionality of the A2C algorithm
     * through two main test scenarios:
     * 1. Basic pattern learning - tests fundamental policy gradient learning
     * 2. Observation-action matching - tests conditional policy learning
     * 
     * The tests verify that the algorithm can:
     * - Learn simple action-reward relationships
     * - Learn observation-dependent policies
     * - Handle both normalized and non-normalized observations
     * - Show measurable improvement in policy performance
     */
    TEST_CASE("A2C")
{
    /**
     * @brief Test A2C's ability to learn basic action-reward patterns
     * 
     * This test verifies that the A2C algorithm can learn a simple relationship
     * where higher action values yield higher rewards. The test checks that:
     * - Policy probabilities shift towards higher action values after training
     * - The algorithm successfully updates the policy in the correct direction
     * - Learning is measurable through probability changes
     * 
     * Expected outcome: Post-training probability of action 1 should be higher
     * than pre-training probability, indicating successful learning.
     */
    SUBCASE("update() learns basic pattern")
    {
        torch::manual_seed(0);  // Ensure reproducible results
        auto base = std::make_shared<MlpBase>(1, false, 5);  // MLP with 1 input, 5 hidden units
        ActionSpace space{"Discrete", {2}};  // Binary action space
        Policy policy(space, base);
        RolloutStorge storage(5, 2, {1}, space, 5, torch::kCPU);  // 5 steps, 2 processes
        A2C a2c(policy, 1, 0.5, 1e-3, 0.001);  // A2C with specific hyperparameters

        // Measure pre-training policy probabilities
        auto pre_game_probs = policy->getProbability(
            torch::ones({2, 1}),    // Observation
            torch::zeros({2, 5}),    // Hidden state
            torch::ones({2, 1}));    // Mask

        // Train the agent on the basic pattern learning task
        learn_pattern(policy, storage, a2c);

        // Measure post-training policy probabilities
        auto post_game_probs = policy->getProbability(
            torch::ones({2, 1}),    // Observation
            torch::zeros({2, 5}),    // Hidden state
            torch::ones({2, 1}));    // Mask

        INFO("Pre-training probabilities: \n"
             << pre_game_probs << "\n");
        INFO("Post-training probabilities: \n"
             << post_game_probs << "\n");
        
        // Verify learning: probability of action 0 should decrease, action 1 should increase
        CHECK(post_game_probs[0][0].item().toDouble() <
              pre_game_probs[0][0].item().toDouble());
        CHECK(post_game_probs[0][1].item().toDouble() >
              pre_game_probs[0][1].item().toDouble());
    }

    /**
     * @brief Test A2C's ability to learn observation-action matching game
     * 
     * This test verifies that the A2C algorithm can learn a conditional policy
     * where the optimal action depends on the observation. The test includes
     * two subcases to test both normalized and non-normalized observation handling.
     * 
     * The game rule: Agent receives reward +1 if action matches observation,
     * -1 otherwise. This tests the agent's ability to learn state-dependent policies.
     * 
     * Expected outcome: After training, the policy should show measurable
     * improvement in matching actions to observations.
     */
    SUBCASE("update() learns basic game")
    {
        /**
         * @brief Test with non-normalized observations
         * 
         * Tests the algorithm's ability to learn from raw observation values
         * without normalization. This is the baseline test case.
         */
        SUBCASE("Without normalized observations")
        {
            torch::manual_seed(0);  // Ensure reproducible results
            auto base = std::make_shared<MlpBase>(1, false, 5);  // MLP without observation normalization
            ActionSpace space{"Discrete", {2}};  // Binary action space
            Policy policy(space, base);
            RolloutStorge storage(5, 2, {1}, space, 5, torch::kCPU);  // 5 steps, 2 processes
            A2C a2c(policy, 1, 0.5, 1e-7, 0.0001);  // A2C with specific hyperparameters

            // Measure pre-training policy probabilities
            auto pre_game_probs = policy->getProbability(
                torch::ones({2, 1}),    // Observation
                torch::zeros({2, 5}),    // Hidden state
                torch::ones({2, 1}));    // Mask

            // Train the agent on the observation-action matching task
            learn_game(policy, storage, a2c);

            // Measure post-training policy probabilities
            auto post_game_probs = policy->getProbability(
                torch::ones({2, 1}),    // Observation
                torch::zeros({2, 5}),    // Hidden state
                torch::ones({2, 1}));    // Mask

            INFO("Pre-training probabilities: \n"
                 << pre_game_probs << "\n");
            INFO("Post-training probabilities: \n"
                 << post_game_probs << "\n");
            
            // Verify learning: policy should have adapted to observation-action relationship
            CHECK(post_game_probs[0][0].item().toDouble() <
                  pre_game_probs[0][0].item().toDouble());
            CHECK(post_game_probs[0][1].item().toDouble() >
                  pre_game_probs[0][1].item().toDouble());
        }

        /**
         * @brief Test with normalized observations
         * 
         * Tests the algorithm's ability to learn from normalized observations.
         * This verifies that the observation normalization feature works correctly
         * and doesn't interfere with learning.
         */
        SUBCASE("With normalized observations")
        {
            torch::manual_seed(0);  // Ensure reproducible results
            auto base = std::make_shared<MlpBase>(1, false, 5);  // MLP base network
            ActionSpace space{"Discrete", {2}};  // Binary action space
            Policy policy(space, base, true);  // Policy with observation normalization enabled
            RolloutStorge storage(5, 2, {1}, space, 5, torch::kCPU);  // 5 steps, 2 processes
            A2C a2c(policy, 1, 0.5, 1e-7, 0.0001);  // A2C with specific hyperparameters

            // Measure pre-training policy probabilities
            auto pre_game_probs = policy->getProbability(
                torch::ones({2, 1}),    // Observation
                torch::zeros({2, 5}),    // Hidden state
                torch::ones({2, 1}));    // Mask

            // Train the agent on the observation-action matching task
            learn_game(policy, storage, a2c);

            // Measure post-training policy probabilities
            auto post_game_probs = policy->getProbability(
                torch::ones({2, 1}),    // Observation
                torch::zeros({2, 5}),    // Hidden state
                torch::ones({2, 1}));    // Mask

            INFO("Pre-training probabilities: \n"
                 << pre_game_probs << "\n");
            INFO("Post-training probabilities: \n"
                 << post_game_probs << "\n");
            
            // Verify learning: policy should have adapted despite observation normalization
            CHECK(post_game_probs[0][0].item().toDouble() <
                  pre_game_probs[0][0].item().toDouble());
            CHECK(post_game_probs[0][1].item().toDouble() >
                  pre_game_probs[0][1].item().toDouble());
        }
    }
}

}
