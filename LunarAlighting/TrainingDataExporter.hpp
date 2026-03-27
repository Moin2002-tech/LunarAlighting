//
// Training Data Exporter for Lunar Alighting RL
// Sends training data to Python data logger via ZMQ
//

#ifndef LUNARALIGHTINGRL_TRAINING_DATA_EXPORTER_HPP
#define LUNARALIGHTINGRL_TRAINING_DATA_EXPORTER_HPP

#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <chrono>


namespace LunarAlighting {

struct TrainingUpdateData {
    int update;
    int total_frames;
    float fps;
    float average_reward;
    int episode_count;
    float policy_loss;
    float value_loss;
    float entropy;
    float success_rate;
    
    nlohmann::json to_json() const {
        return nlohmann::json{
            {"update", update},
            {"total_frames", total_frames},
            {"fps", fps},
            {"average_reward", average_reward},
            {"episode_count", episode_count},
            {"policy_loss", policy_loss},
            {"value_loss", value_loss},
            {"entropy", entropy},
            {"success_rate", success_rate}
        };
    }
};

struct EpisodeData {
    int episode;
    float reward;
    int length;
    bool success;
    bool crash;
    float final_altitude;
    float final_velocity;
    float fuel_used;
    
    nlohmann::json to_json() const {
        return nlohmann::json{
            {"episode", episode},
            {"reward", reward},
            {"length", length},
            {"success", success},
            {"crash", crash},
            {"final_altitude", final_altitude},
            {"final_velocity", final_velocity},
            {"fuel_used", fuel_used}
        };
    }
};
/*
class TrainingDataExporter {
private:
    zmq::context_t context;
    zmq::socket_t socket;
    std::string endpoint;
    bool enabled;
    
public:
    TrainingDataExporter(const std::string& zmq_endpoint = "tcp://127.0.0.1:10202") 
        : context(1), socket(context, ZMQ_PUB), endpoint(zmq_endpoint), enabled(true) {
        try {
            socket.connect(endpoint);
            std::cout << "📊 Training Data Exporter connected to " << endpoint << std::endl;
        } catch (const zmq::error_t& e) {
            std::cerr << "Failed to bind ZMQ socket: " << e.what() << std::endl;
            enabled = false;
        }
    }
    
    ~TrainingDataExporter() {
        if (enabled) {
            socket.close();
            context.close();
        }
    }
    
    void send_training_update(const TrainingUpdateData& data) {
        if (!enabled) return;
        
        try {
            nlohmann::json json_data = data.to_json();
            std::string message = "training_update " + json_data.dump();
            zmq::message_t zmq_message(message.begin(), message.end());
            socket.send(zmq_message);
            std::cout << "📤 Sent training update: " << data.update << std::endl;
        } catch (const zmq::error_t& e) {
            std::cerr << "Error sending training update: " << e.what() << std::endl;
        }
    }
    
    void send_episode_data(const EpisodeData& data) {
        if (!enabled) return;
        
        try {
            nlohmann::json json_data = data.to_json();
            std::string message = "episode " + json_data.dump();
            zmq::message_t zmq_message(message.begin(), message.end());
            socket.send(zmq_message);
            std::cout << "📤 Sent episode data: " << data.episode << " reward=" << data.reward << std::endl;
        } catch (const zmq::error_t& e) {
            std::cerr << "Error sending episode data: " << e.what() << std::endl;
        }
    }
    
    bool is_enabled() const { return enabled; }
};
*/

    class TrainingDataExporter {
private:
    zmq::context_t context;
    zmq::socket_t socket;
    bool enabled;

public:
    explicit TrainingDataExporter(const std::string& zmq_endpoint = "tcp://127.0.0.1:10202")
        : context(1), socket(context, ZMQ_PUSH), enabled(false)  // ← ZMQ_PUSH
    {
        try {
            // Set send timeout so we never block if Python isn't ready
            int timeout_ms = 1000;
            socket.setsockopt(ZMQ_SNDTIMEO, &timeout_ms, sizeof(timeout_ms));
            socket.connect(zmq_endpoint);  // C++ connects, Python binds
            enabled = true;
            std::cout << "📊 Training Data Exporter connected to " << zmq_endpoint << std::endl;
        } catch (const zmq::error_t& e) {
            std::cerr << "Failed to connect ZMQ socket: " << e.what() << std::endl;
        }
    }

    ~TrainingDataExporter() {
        // zmq::socket_t and context_t are RAII — no manual close needed
    }

    void send_training_update(const TrainingUpdateData& data) {
        if (!enabled) return;
        try {
            std::string message = "training_update " + data.to_json().dump();
            zmq::message_t zmq_message(message.begin(), message.end());
            socket.send(zmq_message);
            std::cout << "📤 Sent training update: " << data.update << std::endl;
        } catch (const zmq::error_t& e) {
            std::cerr << "Error sending training update: " << e.what() << std::endl;
        }
    }

    void send_episode_data(const EpisodeData& data) {
        if (!enabled) return;
        try {
            std::string message = "episode " + data.to_json().dump();
            zmq::message_t zmq_message(message.begin(), message.end());
            socket.send(zmq_message);
            std::cout << "📤 Sent episode data: " << data.episode
                      << " reward=" << data.reward << std::endl;
        } catch (const zmq::error_t& e) {
            std::cerr << "Error sending episode data: " << e.what() << std::endl;
        }
    }

    bool is_enabled() const { return enabled; }
};
} // namespace LunarAlighting

#endif // LUNARALIGHTINGRL_TRAINING_DATA_EXPORTER_HPP


