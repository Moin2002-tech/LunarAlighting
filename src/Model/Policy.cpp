/**
 * @file Policy.cpp
 * @brief Implementation of neural network policy for reinforcement learning agents
 * @author moinshaikh
 * @date 2/6/26
 * 
 * This file implements the PolicyImpl class, which serves as the core neural network
 * for reinforcement learning agents. It combines a base neural network with appropriate
 * output layers to handle different action space types (Discrete, Box, MultiBinary).
 * 
 * Key features:
 * - Support for multiple action space types (Discrete, Box, MultiBinary)
 * - Optional observation normalization for improved training stability
 * - Actor-critic architecture with shared feature extraction
 * - Recurrent and non-recurrent network support
 * - Comprehensive action sampling and evaluation methods
 */

#include<torch/torch.h>

#include"../../include/Model/policy.hpp"
#include"../../include/Distribution/Categorical.hpp"
#include"../../include/Model/mlp_base.hpp"
#include"../../include/Model/OutputLayers.hpp"
#include"../../include/ObservationNormalizer.hpp"
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
     * @brief PolicyImpl constructor with action space and network configuration
     * @param actionSpace The action space specification (type and shape)
     * @param base Shared pointer to the base neural network for feature extraction
     * @param normalizeObeservation Whether to enable observation normalization
     * 
     * Constructs a policy network by combining a base neural network with an appropriate
     * output layer based on the action space type. The constructor handles:
     * 
     * - Action space type detection and output layer selection:
     *   - "Discrete": CategoricalOutput for discrete actions
     *   - "Box": NormalOutput for continuous actions
     *   - "MultiBinary": BernoulliOutput for binary action vectors
     * 
     * - Optional observation normalization (only supported with MlpBase):
     *   - Normalizes observations to zero mean and unit variance
     *   - Improves training stability and convergence
     * 
     * - Module registration for proper PyTorch parameter tracking
     * 
     * @throws std::runtime_error if unsupported action space type is provided
     */
    PolicyImpl::PolicyImpl(ActionSpace actionSpace,
        std::shared_ptr<NNBase> base,
        bool normalizeObeservation) :
     actionSpace(actionSpace),
     base(register_module("base",base)),
     obervationNormalizer(nullptr)
    {
        // Determine output layer based on action space type
        int numOutput = actionSpace.shape[0];
        if (actionSpace.type == "Discrete")
        {
            // Discrete actions: use categorical distribution
            outputLayer = std::make_shared<CategoricalOutput>(base->getOutputSize(),numOutput);
        }
        else if (actionSpace.type == "Box")
        {
            // Continuous actions: use normal distribution
            outputLayer = std::make_shared<NormalOutput>(base->getOutputSize(),numOutput);
        }
        else if (actionSpace.type == "MultiBinary")
        {
            // Binary action vectors: use Bernoulli distribution
             outputLayer = std::make_shared<BernoulliOutput>(base->getOutputSize(),numOutput);
        }
        else
        {
            throw std::runtime_error("Unsupported action space type");
        }
        
        // Register output layer as a submodule
        register_module("output",outputLayer);
        
        // Initialize observation normalizer if requested
        if (normalizeObeservation)
        {
            // Normalized observations only supported for MlpBase
            auto mlp_base = static_cast<MlpBase *>(base.get());
            obervationNormalizer = register_module("ObservationNormalizer",
                ObservationNormalizer(mlp_base->getNumInputs()));
        }
    }

    /**
     * @brief Sample actions from the policy network
     * @param input Input observations tensor [batch_size, obs_dim]
     * @param rnn_hxs Recurrent hidden states tensor [batch_size, hidden_size]
     * @param masks Episode continuation masks tensor [batch_size, 1]
     * @return Vector of tensors containing [value, action, log_probs, next_hidden_states]
     * 
     * This method performs forward pass through the policy network and samples actions
     * from the resulting probability distribution. It returns all necessary components
     * for reinforcement learning training:
     * 
     * Process:
     * 1. Normalize observations if normalizer is enabled
     * 2. Forward pass through base network to get features and value estimates
     * 3. Create distribution from features using appropriate output layer
     * 4. Sample actions from the distribution
     * 5. Compute log probabilities of sampled actions
     * 6. Format outputs based on action space type
     * 
     * Output format:
     * - Discrete: Actions and log probs have shape [batch_size, 1]
     * - Continuous/MultiBinary: Log probs summed across action dimensions
     */
    std::vector<torch::Tensor> PolicyImpl::act(torch::Tensor input,
        torch::Tensor rnn_hxs,
        torch::Tensor masks) const
    {
        // Apply observation normalization if enabled
        if (obervationNormalizer)
        {
            input = obervationNormalizer->processObservation(input);
        }

        // Forward pass through base network
        // Returns [value, features, next_hidden_states]
        auto base_output = base->forward(input, rnn_hxs, masks);
        
        // Create action distribution from network features
        auto dist = outputLayer->forward(base_output[1]);

        // Sample actions from the distribution
        auto action = dist->sample();
        
        // Compute log probabilities of sampled actions
        auto action_log_probs = dist->logProbability(action);

        // Format outputs based on action space type
        if (actionSpace.type == "Discrete")
        {
            // For discrete actions, add dimension for consistency
            action = action.unsqueeze(-1);
            action_log_probs = action_log_probs.unsqueeze(-1);
        }
        else
        {
            // For continuous/multi-binary, sum log probs across action dimensions
            action_log_probs = dist->logProbability(action).sum(-1, true);
        }

        // Return all components needed for training
        return {base_output[0], // value estimate
                action,         // sampled action
                action_log_probs, // log probability of action
                base_output[2]}; // next hidden states
    }

    /**
     * @brief Evaluate actions and compute policy metrics for training
     * @param inputs Input observations tensor [batch_size, obs_dim]
     * @param rnn_hxs Recurrent hidden states tensor [batch_size, hidden_size]
     * @param masks Episode continuation masks tensor [batch_size, 1]
     * @param actions Actions to evaluate tensor [batch_size, action_dim]
     * @return Vector of tensors containing [value, log_probs, entropy, next_hidden_states]
     * 
     * This method evaluates specific actions (typically from stored rollouts) and computes
     * the metrics needed for policy gradient training. Unlike act(), this method doesn't
     * sample new actions but evaluates existing ones.
     * 
     * Process:
     * 1. Normalize observations if normalizer is enabled
     * 2. Forward pass through base network to get features and value estimates
     * 3. Create distribution from features using appropriate output layer
     * 4. Compute log probabilities of the given actions
     * 5. Compute distribution entropy for exploration regularization
     * 6. Format outputs based on action space type
     * 
     * Key difference from act():
     * - Evaluates existing actions rather than sampling new ones
     * - Returns entropy for policy optimization
     * - Used during training updates, not action selection
     */
    std::vector<torch::Tensor> PolicyImpl::evaluateAction(torch::Tensor inputs,
                                                        torch::Tensor rnn_hxs,
                                                        torch::Tensor masks,
                                                        torch::Tensor actions) const
    {
        // Apply observation normalization if enabled
        if (obervationNormalizer)
        {
            inputs = obervationNormalizer->processObservation(inputs);
        }

        // Forward pass through base network
        // Returns [value, features, next_hidden_states]
        auto base_output = base->forward(inputs, rnn_hxs, masks);
        
        // Create action distribution from network features
        auto dist = outputLayer->forward(base_output[1]);

        // Compute log probabilities of the given actions
        torch::Tensor action_log_probs;
        if (actionSpace.type == "Discrete")
        {
            // For discrete actions, squeeze and reshape for proper computation
            action_log_probs = dist->logProbability(actions.squeeze(-1))
                                   .view({actions.size(0), -1})
                                   .sum(-1)
                                   .unsqueeze(-1);
        }
        else
        {
            // For continuous/multi-binary, sum log probs across action dimensions
            action_log_probs = dist->logProbability(actions).sum(-1, true);
        }

        // Compute distribution entropy for exploration regularization
        auto entropy = dist->entropy().mean();

        // Return all components needed for policy gradient training
        return {base_output[0], // value estimate
                action_log_probs, // log probability of evaluated actions
                entropy,         // distribution entropy
                base_output[2]}; // next hidden states
    }

    /**
     * @brief Get action probabilities from categorical distribution
     * @param inputs Input observations tensor [batch_size, obs_dim]
     * @param rnn_hxs Recurrent hidden states tensor [batch_size, hidden_size]
     * @param masks Episode continuation masks tensor [batch_size, 1]
     * @return Action probabilities tensor [batch_size, num_actions]
     * 
     * This method returns the raw probability distribution over actions for discrete
     * action spaces. It's primarily used for debugging, monitoring, and analysis
     * rather than training. The method only works with categorical distributions
     * (discrete action spaces).
     * 
     * @note This method assumes the action space is discrete and will cast the
     * distribution to Categorical type. For continuous action spaces, this method
     * is not applicable.
     */
    torch::Tensor PolicyImpl::getProbability(torch::Tensor inputs,
                                    torch::Tensor rnn_hxs,
                                    torch::Tensor masks) const {
        // Apply observation normalization if enabled
        if (obervationNormalizer)
        {
            inputs = obervationNormalizer->processObservation(inputs);
        }

        // Forward pass through base network
        auto base_output = base->forward(inputs, rnn_hxs, masks);
        
        // Create action distribution from network features
        auto dist = outputLayer->forward(base_output[1]);

        // Return probability distribution (only valid for categorical/discrete actions)
        return static_cast<Categorical *>(dist.get())->getProbability();
    }

    /**
     * @brief Get value estimates for given observations
     * @param inputs Input observations tensor [batch_size, obs_dim]
     * @param rnn_hxs Recurrent hidden states tensor [batch_size, hidden_size]
     * @param masks Episode continuation masks tensor [batch_size, 1]
     * @return Value estimates tensor [batch_size, 1]
     * 
     * This method performs a forward pass through the base network and returns
     * only the value estimates (critic output). It's used when only value
     * function evaluation is needed, without action sampling or policy evaluation.
     * 
     * Common use cases:
     * - Computing next values for return calculation
     * - Value function evaluation during testing
     * - Advantage computation in policy gradient algorithms
     */
    torch::Tensor PolicyImpl::getValue(torch::Tensor inputs,
                                     torch::Tensor rnn_hxs,
                                     torch::Tensor masks) const
    {
        // Apply observation normalization if enabled
        if (obervationNormalizer)
        {
            inputs = obervationNormalizer->processObservation(inputs);
        }

        // Forward pass through base network and return value estimates
        auto base_output = base->forward(inputs, rnn_hxs, masks);

        return base_output[0]; // Return only the value component
    }

    /**
     * @brief Update observation normalizer statistics with new observations
     * @param observations New observations tensor for updating normalizer statistics
     * 
     * This method updates the running statistics (mean, variance, count) of the
     * observation normalizer using new batch of observations. This is typically
     * called during training to adapt the normalizer to the data distribution.
     * 
     * The normalizer maintains running statistics to normalize observations to
     * zero mean and unit variance, which improves training stability.
     * 
     * @pre Observation normalizer must be initialized (normalizeObeservation=true)
     * @throws Assertion error if normalizer is not initialized
     */
    void PolicyImpl::updateObervationNormalizer(torch::Tensor observations)
    {
        // Ensure normalizer is properly initialized
        assert(!obervationNormalizer.is_empty());
        
        // Update running statistics with new observations
        obervationNormalizer->update(observations);
    }

    /**
     * @brief Test suite for Policy implementation
     * 
     * This comprehensive test suite validates the Policy class functionality
     * for both recurrent and non-recurrent network configurations. The tests
     * verify correct tensor shapes, network behavior, and output consistency.
     * 
     * Test coverage includes:
     * - Network configuration and sanity checks
     * - Action sampling with proper tensor shapes
     * - Action evaluation with log probabilities and entropy
     * - Value function estimation
     * - Probability distribution extraction
     * - Both recurrent and non-recurrent architectures
     */
    TEST_CASE("Policy")
{
    /**
     * @brief Test recurrent policy network configuration and operations
     * 
     * Tests the Policy class with a recurrent base network (MlpBase with recurrent=True).
     * Verifies that recurrent features work correctly and all methods produce
     * properly shaped outputs.
     */
    SUBCASE("Recurrent")
    {
        // Create recurrent policy with 3D observations, 5 discrete actions
        auto base = std::make_shared<MlpBase>(3, true, 10);  // 3 inputs, recurrent, 10 hidden units
        Policy policy(ActionSpace{"Discrete", {5}}, base);

        /**
         * @brief Verify recurrent network configuration
         * 
         * Tests basic network properties to ensure recurrent features are properly enabled.
         */
        SUBCASE("Sanity checks")
        {
            CHECK(policy->is_recurrent() == true);   // Should be recurrent
            CHECK(policy->getHiddenSize() == 10);    // Hidden size should match base network
        }

        /**
         * @brief Test action sampling output tensor shapes
         * 
         * Verifies that the act() method returns tensors with correct shapes:
         * - Value: [batch_size, 1]
         * - Actions: [batch_size, 1] 
         * - Log probs: [batch_size, 1]
         * - Hidden states: [batch_size, hidden_size]
         */
        SUBCASE("act() output tensors are correct shapes")
        {
            // Test input tensors
            auto inputs = torch::rand({4, 3});      // 4 samples, 3 dimensions
            auto rnn_hxs = torch::rand({4, 10});     // 4 samples, 10 hidden units
            auto masks = torch::zeros({4, 1});      // 4 samples, 1 mask (all active)
            auto outputs = policy->act(inputs, rnn_hxs, masks);

            REQUIRE(outputs.size() == 4);  // Should return 4 tensors

            // Verify value tensor shape [4, 1]
            INFO("Value: \n"
                 << outputs[0] << "\n");
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Verify actions tensor shape [4, 1]
            INFO("Actions: \n"
                 << outputs[1] << "\n");
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 1);

            // Verify log probabilities tensor shape [4, 1]
            INFO("Log probs: \n"
                 << outputs[2] << "\n");
            CHECK(outputs[2].size(0) == 4);
            CHECK(outputs[2].size(1) == 1);

            // Verify hidden states tensor shape [4, 10]
            INFO("Hidden states: \n"
                 << outputs[3] << "\n");
            CHECK(outputs[3].size(0) == 4);
            CHECK(outputs[3].size(1) == 10);
        }

        /**
         * @brief Test action evaluation output tensor shapes
         * 
         * Verifies that the evaluateAction() method returns tensors with correct shapes:
         * - Value: [batch_size, 1]
         * - Log probs: [batch_size, 1]
         * - Entropy: scalar (0-dimensional)
         * - Hidden states: [batch_size, hidden_size]
         */
        SUBCASE("evaluate_actions() output tensors are correct shapes")
        {
            // Test input tensors
            auto inputs = torch::rand({4, 3});      // 4 samples, 3 dimensions
            auto rnn_hxs = torch::rand({4, 10});     // 4 samples, 10 hidden units
            auto masks = torch::zeros({4, 1});      // 4 samples, 1 mask (all active)
            auto actions = torch::randint(5, {4, 1}); // 4 samples, 1 action (0-4)
            auto outputs = policy->evaluateAction(inputs, rnn_hxs, masks, actions);

            REQUIRE(outputs.size() == 4);  // Should return 4 tensors

            // Verify value tensor shape [4, 1]
            INFO("Value: \n"
                 << outputs[0] << "\n");
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Verify log probabilities tensor shape [4, 1]
            INFO("Log probs: \n"
                 << outputs[1] << "\n");
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 1);

            // Verify entropy tensor is scalar (0-dimensional)
            INFO("Entropy: \n"
                 << outputs[2] << "\n");
            CHECK(outputs[2].sizes().size() == 0);

            // Verify hidden states tensor shape [4, 10]
            INFO("Hidden states: \n"
                 << outputs[3] << "\n");
            CHECK(outputs[3].size(0) == 4);
            CHECK(outputs[3].size(1) == 10);
        }

        /**
         * @brief Test value function output tensor shape
         * 
         * Verifies that the getValue() method returns value estimates with correct shape.
         */
        SUBCASE("get_values() output tensor is correct shapes")
        {
            // Test input tensors
            auto inputs = torch::rand({4, 3});      // 4 samples, 3 dimensions
            auto rnn_hxs = torch::rand({4, 10});     // 4 samples, 10 hidden units
            auto masks = torch::zeros({4, 1});      // 4 samples, 1 mask (all active)
            auto outputs = policy->getValue(inputs, rnn_hxs, masks);

            // Verify value tensor shape [4, 1]
            CHECK(outputs.size(0) == 4);
            CHECK(outputs.size(1) == 1);
        }

        /**
         * @brief Test probability distribution output tensor shape
         * 
         * Verifies that the getProbability() method returns action probabilities
         * with correct shape for discrete action spaces.
         */
        SUBCASE("get_probs() output tensor is correct shapes")
        {
            // Test input tensors
            auto inputs = torch::rand({4, 3});      // 4 samples, 3 dimensions
            auto rnn_hxs = torch::rand({4, 10});     // 4 samples, 10 hidden units
            auto masks = torch::zeros({4, 1});      // 4 samples, 1 mask (all active)
            auto outputs = policy->getProbability(inputs, rnn_hxs, masks);

            // Verify probabilities tensor shape [4, 5] (4 samples, 5 actions)
            CHECK(outputs.size(0) == 4);
            CHECK(outputs.size(1) == 5);
        }
    }

    /**
     * @brief Test non-recurrent policy network configuration and operations
     * 
     * Tests the Policy class with a non-recurrent base network (MlpBase with recurrent=False).
     * Verifies that non-recurrent features work correctly and all methods produce
     * properly shaped outputs. The test structure mirrors the recurrent tests for
     * comprehensive coverage.
     */
    SUBCASE("Non-recurrent")
    {
        // Create non-recurrent policy with 3D observations, 5 discrete actions
        auto base = std::make_shared<MlpBase>(3, false, 10);  // 3 inputs, non-recurrent, 10 hidden units
        Policy policy(ActionSpace{"Discrete", {5}}, base);

        /**
         * @brief Verify non-recurrent network configuration
         * 
         * Tests basic network properties to ensure non-recurrent features are properly configured.
         */
        SUBCASE("Sanity checks")
        {
            CHECK(policy->is_recurrent() == false);  // Should not be recurrent
        }

        /**
         * @brief Test action sampling output tensor shapes for non-recurrent network
         * 
         * Verifies that the act() method returns tensors with correct shapes:
         * - Value: [batch_size, 1]
         * - Actions: [batch_size, 1] 
         * - Log probs: [batch_size, 1]
         * - Hidden states: [batch_size, hidden_size] (still present but not used in computation)
         */
        SUBCASE("act() output tensors are correct shapes")
        {
            // Test input tensors
            auto inputs = torch::rand({4, 3});      // 4 samples, 3 dimensions
            auto rnn_hxs = torch::rand({4, 10});     // 4 samples, 10 hidden units (ignored for non-recurrent)
            auto masks = torch::zeros({4, 1});      // 4 samples, 1 mask (all active)
            auto outputs = policy->act(inputs, rnn_hxs, masks);

            REQUIRE(outputs.size() == 4);  // Should return 4 tensors

            // Verify value tensor shape [4, 1]
            INFO("Value: \n"
                 << outputs[0] << "\n");
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Verify actions tensor shape [4, 1]
            INFO("Actions: \n"
                 << outputs[1] << "\n");
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 1);

            // Verify log probabilities tensor shape [4, 1]
            INFO("Log probs: \n"
                 << outputs[2] << "\n");
            CHECK(outputs[2].size(0) == 4);
            CHECK(outputs[2].size(1) == 1);

            // Verify hidden states tensor shape [4, 10] (maintained for interface consistency)
            INFO("Hidden states: \n"
                 << outputs[3] << "\n");
            CHECK(outputs[3].size(0) == 4);
            CHECK(outputs[3].size(1) == 10);
        }

        /**
         * @brief Test action evaluation output tensor shapes for non-recurrent network
         * 
         * Verifies that the evaluateAction() method returns tensors with correct shapes:
         * - Value: [batch_size, 1]
         * - Log probs: [batch_size, 1]
         * - Entropy: scalar (0-dimensional)
         * - Hidden states: [batch_size, hidden_size]
         */
        SUBCASE("evaluate_actions() output tensors are correct shapes")
        {
            // Test input tensors
            auto inputs = torch::rand({4, 3});      // 4 samples, 3 dimensions
            auto rnn_hxs = torch::rand({4, 10});     // 4 samples, 10 hidden units (ignored for non-recurrent)
            auto masks = torch::zeros({4, 1});      // 4 samples, 1 mask (all active)
            auto actions = torch::randint(5, {4, 1}); // 4 samples, 1 action (0-4)
            auto outputs = policy->evaluateAction(inputs, rnn_hxs, masks, actions);

            REQUIRE(outputs.size() == 4);  // Should return 4 tensors

            // Verify value tensor shape [4, 1]
            INFO("Value: \n"
                 << outputs[0] << "\n");
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Verify log probabilities tensor shape [4, 1]
            INFO("Log probs: \n"
                 << outputs[1] << "\n");
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 1);

            // Verify entropy tensor is scalar (0-dimensional)
            INFO("Entropy: \n"
                 << outputs[2] << "\n");
            CHECK(outputs[2].sizes().size() == 0);

            // Verify hidden states tensor shape [4, 10]
            INFO("Hidden states: \n"
                 << outputs[3] << "\n");
            CHECK(outputs[3].size(0) == 4);
            CHECK(outputs[3].size(1) == 10);
        }

        /**
         * @brief Test value function output tensor shape for non-recurrent network
         * 
         * Verifies that the getValue() method returns value estimates with correct shape.
         * The behavior should be identical to recurrent networks for value estimation.
         */
        SUBCASE("get_values() output tensor is correct shapes")
        {
            // Test input tensors
            auto inputs = torch::rand({4, 3});      // 4 samples, 3 dimensions
            auto rnn_hxs = torch::rand({4, 10});     // 4 samples, 10 hidden units (ignored for non-recurrent)
            auto masks = torch::zeros({4, 1});      // 4 samples, 1 mask (all active)
            auto outputs = policy->getValue(inputs, rnn_hxs, masks);

            // Verify value tensor shape [4, 1]
            CHECK(outputs.size(0) == 4);
            CHECK(outputs.size(1) == 1);
        }

        /**
         * @brief Test probability distribution output tensor shape for non-recurrent network
         * 
         * Verifies that the getProbability() method returns action probabilities
         * with correct shape for discrete action spaces. Behavior should be identical
         * to recurrent networks for probability extraction.
         */
        SUBCASE("get_probs() output tensor is correct shapes")
        {
            // Test input tensors
            auto inputs = torch::rand({4, 3});      // 4 samples, 3 dimensions
            auto rnn_hxs = torch::rand({4, 10});     // 4 samples, 10 hidden units (ignored for non-recurrent)
            auto masks = torch::zeros({4, 1});      // 4 samples, 1 mask (all active)
            auto outputs = policy->getProbability(inputs, rnn_hxs, masks);

            // Verify probabilities tensor shape [4, 5] (4 samples, 5 actions)
            CHECK(outputs.size(0) == 4);
            CHECK(outputs.size(1) == 5);
        }
    }
}

}
