//
// Created by moinshaikh on 2/2/26.
//

#include<memory>
#include<vector>

#include"../include/Generator/FeedForwardGenerator.hpp"
#include"../include/Generator/RecurrentGenerator.hpp"
#include"../include/Storage.hpp"
#include"../include/Space.hpp"
#include"third_party/doctest.hpp"

namespace LunarAlighting
{
    /**
     * @brief Constructs a RolloutStorage object.
     *
     * @details Initializes all storage tensors (observations, hidden states, rewards, etc.) with zeros.
     * The shapes are determined by the number of steps, number of processes, and the dimensions of the
     * observation and action spaces.
     *
     * @param numSteps The number of steps per rollout.
     * @param numProcesses The number of parallel environments.
     * @param observationsShape The shape of a single observation.
     * @param action_space The action space definition (used to determine action tensor shape and type).
     * @param hiddenStateSize The size of the hidden state vector (for recurrent policies).
     * @param device The device (CPU/GPU) where tensors will be allocated.
     */
    RolloutStorge::RolloutStorge(int64_t numSteps,
        int64_t numProcesses,
        c10::ArrayRef<int64_t> observationsShape,
        ActionSpace action_space,
        int64_t hiddenStateSize,
        torch::Device device) : device(device),numSteps(numSteps),step(0)
    {
        std::vector<int64_t> observation_Shape{numSteps+1,numProcesses};
        observation_Shape.insert(observation_Shape.end(),observationsShape.begin(),observationsShape.end());

        observations =  torch::zeros(observation_Shape,torch::TensorOptions(device));
        hiddenStates = torch::zeros({numSteps + 1, numProcesses,
                              hiddenStateSize},
                             torch::TensorOptions(device));

        rewards =  torch::zeros({numSteps,numProcesses,1},torch::TensorOptions(device));

        valuePredictions = torch::zeros({numSteps+1,numProcesses,1},torch::TensorOptions(device));

        returns = torch::zeros({numSteps+1,numProcesses,1},torch::TensorOptions(device));
        actionLogProbs = torch::zeros({numSteps,numProcesses,1},torch::TensorOptions(device));
        int numAction;
        if (action_space.type == "Discrete")
        {
           numAction  = 1;
        }
        else
        {
            numAction = action_space.shape[0];
        }
        actions = torch::zeros({numSteps, numProcesses, numAction}, torch::TensorOptions(device));
        if (action_space.type == "Discrete")
        {
            actions = actions.to(torch::kLong);
        }
        masks = torch::ones({numSteps + 1, numProcesses, 1}, torch::TensorOptions(device));
    }

    /**
     * @brief Constructs a RolloutStorage by merging multiple individual storages.
     *
     * @details Concatenates the tensors from a list of `RolloutStorage` pointers along the process dimension (dim 1).
     * This is useful when collecting rollouts from distributed workers and aggregating them for a central update.
     *
     * @param individual_storages A vector of pointers to individual RolloutStorage objects.
     * @param device The device where the merged tensors will be stored.
     */
    RolloutStorge::RolloutStorge(std::vector<RolloutStorge *> individual_storages, torch::Device device) :
    device(device),
    numSteps(individual_storages[0]->get_rewards().size(0)),
        step(0)
        {std::vector<torch::Tensor> observations_vec;
        std::transform(individual_storages.begin(), individual_storages.end(),
                       std::back_inserter(observations_vec),
                       [](RolloutStorge *storage){ return storage->get_observations(); });
        observations = torch::cat(observations_vec, 1);

        std::vector<torch::Tensor> hidden_states_vec;
        std::transform(individual_storages.begin(), individual_storages.end(),
                       std::back_inserter(hidden_states_vec),
                       [](RolloutStorge *storage) { return storage->get_hidden_states(); });
        hiddenStates = torch::cat(hidden_states_vec, 1);

        std::vector<torch::Tensor> rewards_vec;
        std::transform(individual_storages.begin(), individual_storages.end(),
                       std::back_inserter(rewards_vec),
                       [](RolloutStorge *storage) { return storage->get_rewards(); });
        rewards = torch::cat(rewards_vec, 1);

        std::vector<torch::Tensor> value_predictions_vec;
        std::transform(individual_storages.begin(), individual_storages.end(),
                       std::back_inserter(value_predictions_vec),
                       [](RolloutStorge *storage) { return storage->get_value_predictions(); });
        valuePredictions = torch::cat(value_predictions_vec, 1);

        std::vector<torch::Tensor> returns_vec;
        std::transform(individual_storages.begin(), individual_storages.end(),
                       std::back_inserter(returns_vec),
                       [](RolloutStorge *storage) { return storage->get_returns(); });
        returns = torch::cat(returns_vec, 1);

        std::vector<torch::Tensor> action_log_probs_vec;
        std::transform(individual_storages.begin(), individual_storages.end(),
                       std::back_inserter(action_log_probs_vec),
                       [](RolloutStorge *storage) { return storage->get_action_log_probs(); });
        actionLogProbs = torch::cat(action_log_probs_vec, 1);

        std::vector<torch::Tensor> actions_vec;
        std::transform(individual_storages.begin(), individual_storages.end(),
                       std::back_inserter(actions_vec),
                       [](RolloutStorge *storage) { return storage->get_actions(); });
        actions = torch::cat(actions_vec, 1);

        std::vector<torch::Tensor> masks_vec;
        std::transform(individual_storages.begin(), individual_storages.end(),
                       std::back_inserter(masks_vec),
                       [](RolloutStorge *storage) { return storage->get_masks(); });
        masks = torch::cat(masks_vec, 1);
    }


    /**
     * @brief Resets the storage for the next rollout.
     *
     * @details Copies the last observation, hidden state, and mask to the first position (index 0).
     * This ensures continuity between rollouts, as the start of the new rollout corresponds to the
     * end of the previous one.
     */
    void RolloutStorge::afterUpdate() {
            observations[0].copy_(observations[-1]);
            hiddenStates[0].copy_(hiddenStates[-1]);
            masks[0].copy_(masks[-1]);
    }

    /**
     * @brief Computes returns using Generalized Advantage Estimation (GAE) or standard discounted returns.
     *
     * @details
     * If `useGae` is true, it computes returns based on the GAE formula:
     * \f[
     * A_t^{GAE} = \delta_t + (\gamma \lambda) A_{t+1}^{GAE}
     * \f]
     * where \f$ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \f$ is the temporal difference error.
     * The return is then \f$ R_t = A_t^{GAE} + V(s_t) \f$.
     *
     * If `useGae` is false, it computes standard discounted returns:
     * \f[
     * R_t = r_t + \gamma R_{t+1}
     * \f]
     *
     * @param nextValue The estimated value of the state immediately following the rollout.
     * @param useGae Whether to use Generalized Advantage Estimation.
     * @param gamma The discount factor \f$ \gamma \f$.
     * @param tau The GAE parameter \f$ \lambda \f$ (only used if `useGae` is true).
     */
    void RolloutStorge::computeReturns(torch::Tensor nextValue,bool useGae,float gamma,float tau)
    {
        if (useGae)
        {
            // SAFETY NOTE: Ensure your tensor supports [-1] indexing or use .index({-1})
            // Usually safer to use valuePredictions[rewards.size(0)] if allocated that way.
            valuePredictions[-1] = nextValue;

            // This holds the running GAE for all processes.
            // Size: [num_processes, 1]
            torch::Tensor gae = torch::zeros({rewards.size(1), 1}, torch::TensorOptions(device));

            for (int step = rewards.size(0) - 1; step >= 0; --step)
            {
                // Calculate delta for the current step
                auto delta = rewards[step] + gamma * valuePredictions[step + 1] * masks[step + 1] - valuePredictions[step];

                // Update the running GAE
                // ERROR WAS HERE: Do not use [step] on 'gae'
                gae = delta + gamma * tau * masks[step + 1] * gae;

                // Store the result in returns
                returns[step] = gae + valuePredictions[step];
            }
        }
        else
        {
            returns[-1] = nextValue;
            for (int step = rewards.size(0) - 1; step >= 0; --step)
            {
                returns[step] = returns[step + 1] * gamma * masks[step + 1] + rewards[step];
            }
        }
    }

    /**
     * @brief Creates a generator for feed-forward policy training.
     *
     * @details Flattens the time and process dimensions to create minibatches of independent samples.
     * Checks if the total number of samples is sufficient for the requested number of minibatches.
     *
     * @param advantages The computed advantages tensor.
     * @param num_mini_batch The number of minibatches to generate.
     * @return A unique_ptr to a FeedForwardGenerator.
     * @throws std::runtime_error If the batch size is smaller than the number of minibatches.
     */
    std::unique_ptr<Generator> RolloutStorge::feed_forward_generator(torch::Tensor advantages,
                                                  int num_mini_batch)
    {
        auto numStep =actions.size(0);
        auto numProcesses = actions.size(1);
        auto batchSizes = numProcesses * numStep;
        if (batchSizes < num_mini_batch)
        {
            throw std::runtime_error("PPO needs the number of processes (" +
                            std::to_string(numProcesses) +
                            ") * the number of steps (" +
                            std::to_string(numStep) + ") = " +
                            std::to_string(numProcesses * numStep) +
                            " to be greater than or equal to the number of minibatches (" +
                            std::to_string(num_mini_batch) +
                            ")");
        }
        auto miniBatchSize =  batchSizes / num_mini_batch;
        return std::make_unique<FeedForwardGenerator>(
                                                       miniBatchSize,
                                                       observations,
                                                       hiddenStates,
                                                       actions,
                                                       valuePredictions,
                                                       returns,
                                                       masks,
                                                       actionLogProbs,
                                                       advantages
                                                       );
    }

    /**
     * @brief Inserts a transition into the storage.
     *
     * @details Copies the provided tensors into the storage at the current step index.
     * Increments the step index (modulo numSteps).
     *
     * @param observation The observation at the current step.
     * @param hidden_state The hidden state at the current step.
     * @param action The action taken.
     * @param action_log_prob The log probability of the action.
     * @param value_prediction The value prediction for the current state.
     * @param reward The reward received.
     * @param mask The mask indicating if the episode continues (1) or ends (0).
     */
    void RolloutStorge::insert(torch::Tensor observation,
                            torch::Tensor hidden_state,
                            torch::Tensor action,
                            torch::Tensor action_log_prob,
                            torch::Tensor value_prediction,
                            torch::Tensor reward,
                            torch::Tensor mask)
{
    observations[step + 1].copy_(observation);
    hiddenStates[step + 1].copy_(hidden_state);
    actions[step].copy_(action);
    actionLogProbs[step].copy_(action_log_prob);
    valuePredictions[step].copy_(value_prediction);
    rewards[step].copy_(reward);
    masks[step + 1].copy_(mask);

    step = (step + 1) % numSteps;
}

/**
 * @brief Creates a generator for recurrent policy training.
 *
 * @details Creates minibatches that preserve the temporal sequence of observations for each process.
 * This is necessary for training RNNs/LSTMs where hidden states depend on previous inputs.
 *
 * @param advantages The computed advantages tensor.
 * @param num_mini_batch The number of minibatches to generate.
 * @return A unique_ptr to a RecurrentGenerator.
 * @throws std::runtime_error If the number of processes is smaller than the number of minibatches.
 */
std::unique_ptr<Generator> RolloutStorge::recurrentGenerator(
    torch::Tensor advantages, int num_mini_batch)
{
    auto num_processes = actions.size(1);
    if (num_processes < num_mini_batch)
    {
        throw std::runtime_error("PPO needs the number of processes (" +
                                 std::to_string(num_processes) +
                                 ") to be greater than or equal to the number of minibatches (" +
                                 std::to_string(num_mini_batch) +
                                 ")");
    }
    return std::make_unique<RecurrentGenerator>(
        num_processes,
        num_mini_batch,
        observations,
        hiddenStates,
        actions,
        valuePredictions,
        returns,
        masks,
        actionLogProbs,
        advantages);
}

/**
 * @brief Sets the initial observation.
 *
 * @details Copies the initial observation into the first slot (index 0) of the observations tensor.
 * This is typically done at the beginning of the very first rollout.
 *
 * @param observation The initial observation tensor.
 */
void RolloutStorge::setFirstObservation(torch::Tensor observation)
{
    observations[0].copy_(observation);
}

/**
 * @brief Moves all storage tensors to the specified device.
 *
 * @param device The target device (e.g., CPU or CUDA).
 */
void RolloutStorge::to(torch::Device device)
{
    this->device = device;
    observations = observations.to(device);
    hiddenStates = hiddenStates.to(device);
    rewards = rewards.to(device);
    valuePredictions = valuePredictions.to(device);
    returns = returns.to(device);
    actionLogProbs = actionLogProbs.to(device);
    actions = actions.to(device);
    masks = masks.to(device);
}


    TEST_CASE("RolloutStorage")
{
    SUBCASE("Initializes tensors to correct sizes")
    {
        RolloutStorge storage(3, 5, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);

        CHECK(storage.get_observations().size(0) == 4);
        CHECK(storage.get_observations().size(1) == 5);
        CHECK(storage.get_observations().size(2) == 5);
        CHECK(storage.get_observations().size(3) == 2);

        CHECK(storage.get_hidden_states().size(0) == 4);
        CHECK(storage.get_hidden_states().size(1) == 5);
        CHECK(storage.get_hidden_states().size(2) == 10);

        CHECK(storage.get_rewards().size(0) == 3);
        CHECK(storage.get_rewards().size(1) == 5);
        CHECK(storage.get_rewards().size(2) == 1);

        CHECK(storage.get_value_predictions().size(0) == 4);
        CHECK(storage.get_value_predictions().size(1) == 5);
        CHECK(storage.get_value_predictions().size(2) == 1);

        CHECK(storage.get_returns().size(0) == 4);
        CHECK(storage.get_returns().size(1) == 5);
        CHECK(storage.get_returns().size(2) == 1);

        CHECK(storage.get_action_log_probs().size(0) == 3);
        CHECK(storage.get_action_log_probs().size(1) == 5);
        CHECK(storage.get_action_log_probs().size(2) == 1);

        CHECK(storage.get_actions().size(0) == 3);
        CHECK(storage.get_actions().size(1) == 5);
        CHECK(storage.get_actions().size(2) == 1);

        CHECK(storage.get_masks().size(0) == 4);
        CHECK(storage.get_masks().size(1) == 5);
        CHECK(storage.get_masks().size(2) == 1);
    }

    SUBCASE("Initializes actions to correct type")
    {
        SUBCASE("Long")
        {
            RolloutStorge storage(3, 5, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);

            CHECK(storage.get_actions().dtype() == torch::kLong);
        }

        SUBCASE("Float")
        {
            RolloutStorge storage(3, 5, {5, 2}, ActionSpace{"Box", {3}}, 10, torch::kCPU);

            CHECK(storage.get_actions().dtype() == torch::kFloat);
        }
    }

    SUBCASE("to() doesn't crash")
    {
        RolloutStorge storage(3, 4, {5}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);
        storage.to(torch::kCPU);
    }

    SUBCASE("insert() inserts values")
    {
        RolloutStorge storage(3, 4, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);
        storage.insert(torch::rand({4, 5, 2}) + 1,
                       torch::rand({4, 10}) + 1,
                       torch::randint(1, 3, {4, 1}),
                       torch::rand({4, 1}) + 1,
                       torch::rand({4, 1}) + 1,
                       torch::rand({4, 1}) + 1,
                       torch::zeros({4, 1}));

        INFO("Observations: \n"
             << storage.get_observations() << "\n");
        CHECK(storage.get_observations()[1][0][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Hidden states: \n"
             << storage.get_hidden_states() << "\n");
        CHECK(storage.get_hidden_states()[1][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Actions: \n"
             << storage.get_actions() << "\n");
        CHECK(storage.get_actions()[0][0][0].item().toInt() != 0);
        INFO("Action log probs: \n"
             << storage.get_action_log_probs() << "\n");
        CHECK(storage.get_action_log_probs()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Value predictions: \n"
             << storage.get_value_predictions() << "\n");
        CHECK(storage.get_value_predictions()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Rewards: \n"
             << storage.get_rewards() << "\n");
        CHECK(storage.get_rewards()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Masks: \n"
             << storage.get_masks() << "\n");
        CHECK(storage.get_masks()[1][0][0].item().toInt() != 1);
    }

    SUBCASE("compute_returns()")
    {
        RolloutStorge storage(3, 2, {4}, ActionSpace{"Discrete", {3}}, 5, torch::kCPU);

        std::vector<float> value_preds{0, 1};
        std::vector<float> rewards{0, 1};
        std::vector<float> masks{1, 1};
        storage.insert(torch::zeros({2, 4}),
                       torch::zeros({2, 5}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(value_preds.data(), {2, 1}),
                       torch::from_blob(rewards.data(), {2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        value_preds = {1, 2};
        rewards = {1, 2};
        masks = {1, 0};
        storage.insert(torch::zeros({2, 4}),
                       torch::zeros({2, 5}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(value_preds.data(), {2, 1}),
                       torch::from_blob(rewards.data(), {2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        value_preds = {2, 3};
        rewards = {2, 3};
        masks = {1, 1};
        storage.insert(torch::zeros({2, 4}),
                       torch::zeros({2, 5}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(value_preds.data(), {2, 1}),
                       torch::from_blob(rewards.data(), {2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));

        SUBCASE("Gives correct results without GAE")
        {
            std::vector<float> next_values{0, 1};
            storage.computeReturns(torch::from_blob(&next_values[0], {2, 1}),
                                    false, 0.6, 0.6);

            INFO("Masks: \n"
                 << storage.get_masks() << "\n");
            INFO("Rewards: \n"
                 << storage.get_rewards() << "\n");
            INFO("Returns: \n"
                 << storage.get_returns() << "\n");
            CHECK(storage.get_returns()[0][0].item().toDouble() ==
                  doctest::Approx(1.32));
            CHECK(storage.get_returns()[0][1].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][0].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][1].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][0].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][1].item().toDouble() ==
                  doctest::Approx(3.6));
            CHECK(storage.get_returns()[3][0].item().toDouble() ==
                  doctest::Approx(0));
            CHECK(storage.get_returns()[3][1].item().toDouble() ==
                  doctest::Approx(1));
        }

        SUBCASE("Gives correct results with GAE")
        {
            std::vector<float> next_values{0, 1};
                storage.computeReturns(torch::from_blob(&next_values[0], {2, 1}),
                                    true, 0.6, 0.6);

            INFO("Masks: \n"
                 << storage.get_masks() << "\n");
            INFO("Rewards: \n"
                 << storage.get_rewards() << "\n");
            INFO("Value predictions: \n"
                 << storage.get_value_predictions() << "\n");
            INFO("Returns: \n"
                 << storage.get_returns() << "\n");
            CHECK(storage.get_returns()[0][0].item().toDouble() ==
                  doctest::Approx(1.032));
            CHECK(storage.get_returns()[0][1].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][0].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][1].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][0].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][1].item().toDouble() ==
                  doctest::Approx(3.6));
            CHECK(storage.get_returns()[3][0].item().toDouble() ==
                  doctest::Approx(0));
            CHECK(storage.get_returns()[3][1].item().toDouble() ==
                  doctest::Approx(0));
        }
    }

    SUBCASE("after_update() copies last observation, moves hidden state and mask to "
            "the 0th timestep")
    {
        RolloutStorge storage(3, 2, {3}, ActionSpace{"Discrete", {3}}, 2, torch::kCPU);

        std::vector<float> obs{0, 1, 2, 1, 2, 3};
        std::vector<float> hidden_states{0, 1, 0, 1};
        std::vector<float> masks{0, 1};
        storage.insert(torch::from_blob(obs.data(), {2, 3}),
                       torch::from_blob(hidden_states.data(), {2, 2}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        obs = {0, 1, 2, 1, 2, 3};
        hidden_states = {0, 1, 0, 1};
        masks = {0, 1};
        storage.insert(torch::from_blob(obs.data(), {2, 3}),
                       torch::from_blob(hidden_states.data(), {2, 2}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        obs = {5, 6, 7, 7, 8, 9};
        hidden_states = {1, 2, 3, 4};
        masks = {0, 0};
        storage.insert(torch::from_blob(obs.data(), {2, 3}),
                       torch::from_blob(hidden_states.data(), {2, 2}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        storage.afterUpdate();

        INFO("Observations: \n"
             << storage.get_observations() << "\n");
        CHECK(storage.get_observations()[0][0][1].item().toDouble() ==
              doctest::Approx(6));
        INFO("Hidden_states: \n"
             << storage.get_hidden_states() << "\n");
        CHECK(storage.get_hidden_states()[0][0][1].item().toDouble() ==
              doctest::Approx(2));
        INFO("Masks: \n"
             << storage.get_masks() << "\n");
        CHECK(storage.get_masks()[0][0][0].item().toDouble() ==
              doctest::Approx(0));
    }

    SUBCASE("Can create feed-forward generator")
    {
        RolloutStorge storage(3, 5, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);
        auto generator = storage.feed_forward_generator(torch::rand({3, 5, 1}), 5);
        generator->next();
    }

    SUBCASE("Can create recurrent generator")
    {
        RolloutStorge storage(3, 5, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);
        auto generator = storage.recurrentGenerator(torch::rand({3, 5, 1}), 5);
        generator->next();
    }

    SUBCASE("Can combine multiple storages into one")
    {
        std::vector<RolloutStorge> storages;
        for (int i = 0; i < 5; ++i)
        {
            storages.push_back({3, 1, {4}, ActionSpace{"Discrete", {3}}, 5, torch::kCPU});

            std::vector<float> value_preds{1};
            std::vector<float> rewards{1};
            std::vector<float> masks{1};
            storages[i].insert(torch::rand({1, 4}),
                               torch::rand({1, 5}),
                               torch::rand({1, 1}),
                               torch::rand({1, 1}),
                               torch::from_blob(value_preds.data(), {1, 1}),
                               torch::from_blob(rewards.data(), {1, 1}),
                               torch::from_blob(masks.data(), {1, 1}));
            value_preds = {2};
            rewards = {2};
            masks = {0};
            storages[i].insert(torch::rand({1, 4}),
                               torch::rand({1, 5}),
                               torch::rand({1, 1}),
                               torch::rand({1, 1}),
                               torch::from_blob(value_preds.data(), {1, 1}),
                               torch::from_blob(rewards.data(), {1, 1}),
                               torch::from_blob(masks.data(), {1, 1}));
            value_preds = {3};
            rewards = {3};
            masks = {1};
            storages[i].insert(torch::rand({1, 4}),
                               torch::rand({1, 5}),
                               torch::rand({1, 1}),
                               torch::rand({1, 1}),
                               torch::from_blob(value_preds.data(), {1, 1}),
                               torch::from_blob(rewards.data(), {1, 1}),
                               torch::from_blob(masks.data(), {1, 1}));
        }

        std::vector<RolloutStorge *> pointers;
        std::transform(storages.begin(), storages.end(), std::back_inserter(pointers),
                       [](RolloutStorge &storage) { return &storage; });

        RolloutStorge combined_storage(pointers, torch::kCPU);

        CHECK(combined_storage.get_observations().size(0) == 4);
        CHECK(combined_storage.get_observations().size(1) == 5);
        CHECK(combined_storage.get_hidden_states().size(0) == 4);
        CHECK(combined_storage.get_hidden_states().size(1) == 5);
        CHECK(combined_storage.get_hidden_states().size(2) == 5);
        CHECK(combined_storage.get_rewards().size(0) == 3);
        CHECK(combined_storage.get_rewards().size(1) == 5);
        CHECK(combined_storage.get_rewards().size(2) == 1);
        CHECK(combined_storage.get_value_predictions().size(0) == 4);
        CHECK(combined_storage.get_value_predictions().size(1) == 5);
        CHECK(combined_storage.get_value_predictions().size(2) == 1);
        CHECK(combined_storage.get_returns().size(0) == 4);
        CHECK(combined_storage.get_returns().size(1) == 5);
        CHECK(combined_storage.get_returns().size(2) == 1);
        CHECK(combined_storage.get_action_log_probs().size(0) == 3);
        CHECK(combined_storage.get_action_log_probs().size(1) == 5);
        CHECK(combined_storage.get_action_log_probs().size(2) == 1);
        CHECK(combined_storage.get_actions().size(0) == 3);
        CHECK(combined_storage.get_actions().size(1) == 5);
        CHECK(combined_storage.get_actions().size(2) == 1);
        CHECK(combined_storage.get_masks().size(0) == 4);
        CHECK(combined_storage.get_masks().size(1) == 5);
        CHECK(combined_storage.get_masks().size(2) == 1);

        for (int i = 0; i < 5; ++i)
        {
            CHECK((combined_storage.get_observations().narrow(1, i, 1) == storages[i].get_observations())
                      .all()
                      .item()
                      .toBool());
        }
    }
}





}
