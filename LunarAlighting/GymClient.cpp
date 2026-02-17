//
// Created by moinshaikh on 2/9/26.
//



#include<string>
#include<chrono>

#include"../src/third_party/doctest.hpp"


#include<spdlog/spdlog.h>

#include<ATen/Parallel.h>

#include"../include/LunarAlighting.hpp"

#include"../src/third_party/doctest.hpp"

#include"Communication.hpp"
#include"Request.hpp"

using namespace LunarAlighting;
// Algorithm hyperparameters
const std::string algorithm = "PPO";
const float actorLossCoef = 1.0;
const int batchSize = 40;
const float clipParam = 0.2;
const float discountFactor = 0.99;
const float entropyCoef = 1e-3;
const float gae = 0.9;
const float klTarget = 0.5;
const float learningRate = 1e-3;
const int logInterval = 10;
const int maxFrames = 10e+7;
const int numEpoch = 3;
const int numMiniBatch = 20;
const int rewardAverageWindowSize = 10;
const float rewardClipValue = 100; // Post scaling
const bool use_gae = true;
const bool use_lr_decay = false;
const float valueLossCoef = 0.5;


const std::string envName = "LunarAlighting-v1";
const int  numEnvs = 8;
const float renderRewardThresHold= 160;

//model hypermeter
const int hiddenSize = 64;
const bool recurrent = false;
const bool useCuda = false;

std::vector<float> flattenVector(std::vector<float> const &input)
{
    return input;
}
template<typename T>
std::vector<float> flattenVector(std::vector<std::vector<T>> const& input)
{
    std::vector<float> output;
    for (auto const &elements : input)
    {
        auto subVector = flattenVector(elements);
        output.reserve(output.size()+ subVector.size());
        output.insert(output.end(),subVector.cbegin(),subVector.cend());

    }

    return output;
}


//int main(int argc, char *argv[])
TEST_CASE("GymClient")
{
     spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("%^[%T %7l] %v%$");
    at::set_num_threads(1);
    torch::manual_seed(0);

    torch::Device device = useCuda ? torch::kCUDA : torch::kCPU ;

    spdlog::info("Connecting to the Gym Environment");
    GymClient::Communicator communicator("tcp://127.0.0.1:10201");

    spdlog::info("Creating Environment");

    auto makeParams =  std::make_shared<GymClient::makeParam> ();

    makeParams->envName =envName;
    makeParams->numEnv =numEnvs;

    GymClient::Request<GymClient::makeParam> make_request("make",makeParams);

    communicator.sendRequest(make_request);

    auto make_response = communicator.getResponse<GymClient::MakeResponse>();
    if (make_response) {
        spdlog::info(make_response->result);
    } else {
        spdlog::error("Failed to get make response from gym server");
        return; // Exit the test if server doesn't respond
    }

    GymClient::Request<GymClient::infoParam> info_request("info", std::make_shared<GymClient::infoParam>());
    communicator.sendRequest(info_request);
    auto env_info = communicator.getResponse<GymClient::InfoResponse>();
    if (!env_info) {
        spdlog::error("Failed to get info response from gym server");
        return; // Exit the test if server doesn't respond
    }
    //spdlog::info("Action space: {} - [{}]", env_info->actionSpaceShape,
                 //env_info->actionSpaceShape);
    spdlog::info("Observation space: {} - [{}]", env_info->observationSpaceType,
                 env_info->observationSpaceType);

    spdlog::info("Resetting environment");

    auto reset_param = std::make_shared<GymClient::resetParam>();
    GymClient::Request<GymClient::resetParam> reset_request("reset", reset_param);

    communicator.sendRequest(reset_request);

    auto observation_shape = env_info->observationSpaceShape;
    observation_shape.insert(observation_shape.begin(), numEnvs);
    torch::Tensor observation;
    std::vector<float> observation_vec;

    if (env_info->observationSpaceShape.size() > 1)
    {
        observation_vec = flattenVector(communicator.getResponse<GymClient::CnnResetResponse>()->observation);
        observation = torch::from_blob(observation_vec.data(), observation_shape).to(device);
    }
    else
    {
        observation_vec = flattenVector(communicator.getResponse<GymClient::MlpResetResponse>()->observation);
        observation = torch::from_blob(observation_vec.data(), observation_shape).to(device);
    }

    std::shared_ptr<NNBase> base;
    if (env_info->observationSpaceShape.size() == 1)
    {
        base = std::make_shared<MlpBase>(env_info->observationSpaceShape[0], recurrent, hiddenSize);
    }
    else
    {
        base = std::make_shared<CnnBase>(env_info->observationSpaceShape[0], recurrent, hiddenSize);
    }
    base->to(device);
    ActionSpace space{env_info->actionSpaceType, env_info->actionSpaceShape};
    Policy policy(nullptr);
    if (env_info->observationSpaceShape.size() == 1)
    {
        // With observation normalization
        policy = Policy(space, base, true);
    }
    else
    {
        // Without observation normalization
        policy = Policy(space, base, true);
    }
    policy->to(device);
    RolloutStorge storage(batchSize, numEnvs, env_info->observationSpaceShape, space, hiddenSize, device);
    std::unique_ptr<Algorithms> algo;
    if (algorithm == "A2C")
    {
        algo = std::make_unique<A2C>(policy, actorLossCoef, valueLossCoef, entropyCoef, learningRate);
    }
    else if (algorithm == "PPO")
    {
        algo = std::make_unique<PPO>(policy,
                                     clipParam,
                                     numEpoch,
                                     numMiniBatch,
                                     actorLossCoef,
                                     valueLossCoef,
                                     entropyCoef,
                                     learningRate,
                                     1e-8,
                                     0.5,
                                     klTarget);
    }


    storage.setFirstObservation(observation);

    std::vector<float> running_rewards(numEnvs);
    int episode_count = 0;
    bool render = false;
    std::vector<float> reward_history(rewardAverageWindowSize);

    RunningMeanstd returns_rms(1);
    auto returns = torch::zeros({numEnvs});

    auto start_time = std::chrono::high_resolution_clock::now();

    int num_updates = maxFrames / (batchSize * numEnvs);

    for (int update= 0; update < num_updates ; ++update)
    {
        for (int step = 0; step < batchSize; ++step)
        {
            std::vector<torch::Tensor> actResult;
            {
            torch::NoGradGuard noGrad;
                torch::NoGradGuard no_grad;
                actResult = policy->act(storage.get_observations()[step],
                                         storage.get_hidden_states()[step],
                                         storage.get_masks()[step]);
            }
            auto actionTensor = actResult[0].cpu().to(torch::kFloat);
            float *actionArray = actionTensor.data_ptr<float>();
            std::vector<std::vector<float>> action(numEnvs);
            for (int i = 0;i< numEnvs;++i)
            {
                if (space.type == "Discrete")
                {
                    action[i] ={actionArray[i]};
                }
                else
                {
                    for (int j = 0; j< env_info->actionSpaceShape[0];++j)
                    {
                        action[i].push_back(actionArray[i * env_info->actionSpaceShape[0] + j]);
                    }

                }
            }

            auto stepParams = std::make_shared<GymClient::stepParam>();
            stepParams->action =  action;
            stepParams->render =  render;
            GymClient::Request<GymClient::stepParam> step_request("step",stepParams);
            communicator.sendRequest(step_request);
            std::vector<float> rewards;
            std::vector<float> realRewards;
            std::vector<std::vector<bool>> dones_vec;
            if (env_info->observationSpaceShape.size() > 1)
            {
                auto step_result = communicator.getResponse<GymClient::CnnStepResponse>();
                observation_vec = flattenVector(step_result->observation);
                observation = torch::from_blob(observation_vec.data(), observation_shape).to(device);
                auto raw_reward_vec = flattenVector(step_result->real_reward);
                auto reward_tensor = torch::from_blob(raw_reward_vec.data(), {numEnvs}, torch::kFloat);
                returns = returns * discountFactor + reward_tensor;
                returns_rms->update(returns);
                reward_tensor = torch::clamp(reward_tensor / torch::sqrt(returns_rms->get_variance() + 1e-8),
                                             -rewardClipValue, rewardClipValue);
                rewards = std::vector<float>(reward_tensor.data_ptr<float>(), reward_tensor.data_ptr<float>() + reward_tensor.numel());
                realRewards = flattenVector(step_result->real_reward);
                dones_vec = step_result->done;
            }
            else
            {
                auto step_result = communicator.getResponse<GymClient::MlpStepResponse>();
                observation_vec = flattenVector(step_result->observation);
                observation = torch::from_blob(observation_vec.data(), observation_shape).to(device);
                auto raw_reward_vec = flattenVector(step_result->real_reward);
                auto reward_tensor = torch::from_blob(raw_reward_vec.data(), {numEnvs}, torch::kFloat);
                returns = returns * discountFactor + reward_tensor;
                returns_rms->update(returns);
                reward_tensor = torch::clamp(reward_tensor / torch::sqrt(returns_rms->get_variance() + 1e-8),
                                             -rewardClipValue, rewardClipValue);
                rewards = std::vector<float>(reward_tensor.data_ptr<float>(), reward_tensor.data_ptr<float>() + reward_tensor.numel());
                realRewards = flattenVector(step_result->real_reward);
                dones_vec = step_result->done;
            }
            for (int i = 0;i<numEnvs;++i)
            {
                running_rewards[i] += realRewards[i];
                if (dones_vec[i][0])
                {
                    reward_history[episode_count % rewardAverageWindowSize] =  running_rewards[i];
                    running_rewards[i] = 0;
                    returns[i] = 0;
                    episode_count++;
                }
            }
            auto dones = torch::zeros({numEnvs,1},torch::TensorOptions(device));
            for (int i= 0;i<numEnvs;++i)
            {
                dones[i][0] = static_cast<int> (dones_vec[i][0]);
            }
            storage.insert(observation,
                           actResult[3],
                           actResult[1],
                           actResult[2],
                           actResult[0],
                           torch::from_blob(rewards.data(), {numEnvs, 1}).to(device),
                           1 - dones);
        }
        torch::Tensor nextValues;
        {
            torch::NoGradGuard noGradGuard;
            nextValues = policy->getValue(storage.get_observations()[-1],
                storage.get_hidden_states()[-1],
                storage.get_masks()[-1]).detach();
        }
        storage.computeReturns(nextValues,use_gae,discountFactor,gae);

        float decay_level;
        if (use_lr_decay)
        {
            decay_level = 1. - static_cast<float>(update) / num_updates;
        }
        else
        {
            decay_level = 1;
        }
        auto update_data = algo->update(storage, decay_level);
        storage.afterUpdate();

        if (update % logInterval == 0 && update > 0)
        {
            auto total_steps = (update + 1) * batchSize * numEnvs;
            auto run_time = std::chrono::high_resolution_clock::now() - start_time;
            auto run_time_secs = std::chrono::duration_cast<std::chrono::seconds>(run_time);
            auto fps = total_steps / (run_time_secs.count() + 1e-9);
            spdlog::info("---");
            spdlog::info("Update: {}/{}", update, num_updates);
            spdlog::info("Total frames: {}", total_steps);
            spdlog::info("FPS: {}", fps);

            for (const auto &datum : update_data)
            {
                spdlog::info("{}: {}", datum.name, datum.value);
            }
            float average_reward = std::accumulate(reward_history.begin(), reward_history.end(), 0);
            average_reward /= episode_count < rewardAverageWindowSize ? episode_count : rewardAverageWindowSize;
            spdlog::info("Reward: {}", average_reward);
            render = average_reward >= renderRewardThresHold;
        }
    }
}




