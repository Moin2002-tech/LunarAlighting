//
// Created by moinshaikh on 2/10/26.
// Real-time SDL3 lunar landing simulation with RL control
//

#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <chrono>

#include <SDL3/SDL.h>
#include <torch/torch.h>
#include <spdlog/spdlog.h>

#include "../include/LunarAlighting.hpp"
#include "../include/Model/policy.hpp"
#include "../include/Model/mlp_base.hpp"
#include "../include/Space.hpp"
#include "../src/third_party/doctest.hpp"

using namespace LunarAlighting;

// Custom bitmap font data
struct FONT
{
    static constexpr std::array<std::array<uint8_t, 7>,128> generate()
    {
        std::array<std::array<uint8_t, 7>,128> FONT_DATA = {};
        FONT_DATA['0'] = {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E};
        FONT_DATA['1'] = {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E};
        FONT_DATA['2'] = {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F};
        FONT_DATA['3'] = {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E};
        FONT_DATA['4'] = {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02};
        FONT_DATA['5'] = {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E};
        FONT_DATA['6'] = {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E};
        FONT_DATA['7'] = {0x1F,0x01,0x02,0x04,0x08,0x08,0x08};
        FONT_DATA['8'] = {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E};
        FONT_DATA['9'] = {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C};
        FONT_DATA['.'] = {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C};
        FONT_DATA[':'] = {0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00};
        FONT_DATA['-'] = {0x00,0x00,0x00,0x1F,0x00,0x00,0x00};
        FONT_DATA['+'] = {0x00,0x04,0x04,0x1F,0x04,0x04,0x00};
        FONT_DATA['%'] = {0x18,0x19,0x02,0x04,0x08,0x13,0x03};
        FONT_DATA['/'] = {0x01,0x02,0x02,0x04,0x08,0x08,0x10};
        FONT_DATA['A'] = {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11};
        FONT_DATA['B'] = {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E};
        FONT_DATA['C'] = {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E};
        FONT_DATA['D'] = {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C};
        FONT_DATA['E'] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F};
        FONT_DATA['F'] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10};
        FONT_DATA['G'] = {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F};
        FONT_DATA['H'] = {0x11,0x11,0x11,0x1F,0x11,0x11,0x11};
        FONT_DATA['I'] = {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E};
        FONT_DATA['J'] = {0x07,0x02,0x02,0x02,0x02,0x12,0x0C};
        FONT_DATA['K'] = {0x11,0x12,0x14,0x18,0x14,0x12,0x11};
        FONT_DATA['L'] = {0x10,0x10,0x10,0x10,0x10,0x10,0x1F};
        FONT_DATA['M'] = {0x11,0x1B,0x15,0x15,0x11,0x11,0x11};
        FONT_DATA['N'] = {0x11,0x11,0x19,0x15,0x13,0x11,0x11};
        FONT_DATA['O'] = {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E};
        FONT_DATA['P'] = {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10};
        FONT_DATA['Q'] = {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D};
        FONT_DATA['R'] = {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11};
        FONT_DATA['S'] = {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E};
        FONT_DATA['T'] = {0x1F,0x04,0x04,0x04,0x04,0x04,0x04};
        FONT_DATA['U'] = {0x11,0x11,0x11,0x11,0x11,0x11,0x0E};
        FONT_DATA['V'] = {0x11,0x11,0x11,0x11,0x11,0x0A,0x04};
        FONT_DATA['W'] = {0x11,0x11,0x11,0x15,0x15,0x15,0x0A};
        FONT_DATA['X'] = {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11};
        FONT_DATA['Y'] = {0x11,0x11,0x11,0x0A,0x04,0x04,0x04};
        FONT_DATA['Z'] = {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F};
        FONT_DATA['a'] = {0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F};
        FONT_DATA['b'] = {0x10,0x10,0x16,0x19,0x11,0x11,0x1E};
        FONT_DATA['c'] = {0x00,0x00,0x0E,0x10,0x10,0x11,0x0E};
        FONT_DATA['d'] = {0x01,0x01,0x0D,0x13,0x11,0x11,0x0F};
        FONT_DATA['e'] = {0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E};
        FONT_DATA['f'] = {0x06,0x09,0x08,0x1C,0x08,0x08,0x08};
        FONT_DATA['g'] = {0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E};
        FONT_DATA['h'] = {0x10,0x10,0x16,0x19,0x11,0x11,0x11};
        FONT_DATA['i'] = {0x04,0x00,0x0C,0x04,0x04,0x04,0x0E};
        FONT_DATA['j'] = {0x02,0x00,0x06,0x02,0x02,0x12,0x0C};
        FONT_DATA['k'] = {0x10,0x10,0x12,0x14,0x18,0x14,0x12};
        FONT_DATA['l'] = {0x0C,0x04,0x04,0x04,0x04,0x04,0x0E};
        FONT_DATA['m'] = {0x00,0x00,0x1A,0x15,0x15,0x11,0x11};
        FONT_DATA['n'] = {0x00,0x00,0x16,0x19,0x11,0x11,0x11};
        FONT_DATA['o'] = {0x00,0x00,0x0E,0x11,0x11,0x11,0x0E};
        FONT_DATA['p'] = {0x00,0x00,0x1E,0x11,0x1E,0x10,0x10};
        FONT_DATA['q'] = {0x00,0x00,0x0D,0x13,0x0F,0x01,0x01};
        FONT_DATA['r'] = {0x00,0x00,0x16,0x19,0x10,0x10,0x10};
        FONT_DATA['s'] = {0x00,0x00,0x0E,0x10,0x0E,0x01,0x1E};
        FONT_DATA['t'] = {0x08,0x08,0x1C,0x08,0x08,0x09,0x06};
        FONT_DATA['u'] = {0x00,0x00,0x11,0x11,0x11,0x13,0x0D};
        FONT_DATA['v'] = {0x00,0x00,0x11,0x11,0x11,0x0A,0x04};
        FONT_DATA['w'] = {0x00,0x00,0x11,0x11,0x15,0x15,0x0A};
        FONT_DATA['x'] = {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11};
        FONT_DATA['y'] = {0x00,0x00,0x11,0x11,0x0F,0x01,0x0E};
        FONT_DATA['z'] = {0x00,0x00,0x1F,0x02,0x04,0x08,0x1F};
        FONT_DATA[' '] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00};
        FONT_DATA['['] = {0x0E,0x08,0x08,0x08,0x08,0x08,0x0E};
        FONT_DATA[']'] = {0x0E,0x02,0x02,0x02,0x02,0x02,0x0E};
        FONT_DATA['('] = {0x02,0x04,0x08,0x08,0x08,0x04,0x02};
        FONT_DATA[')'] = {0x08,0x04,0x02,0x02,0x02,0x04,0x08};
        FONT_DATA['<'] = {0x02,0x04,0x08,0x10,0x08,0x04,0x02};
        FONT_DATA['>'] = {0x08,0x04,0x02,0x01,0x02,0x04,0x08};
        FONT_DATA['='] = {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00};
        FONT_DATA['|'] = {0x04,0x04,0x04,0x04,0x04,0x04,0x04};

        return FONT_DATA;
    }
};

inline constexpr auto FONT_DATA = FONT::generate();

// Physics constants for lunar environment
constexpr float GRAVITY = 1.62f; // Moon gravity (m/s^2)
constexpr float THRUST_MAIN = 13.0f; // Main engine thrust (m/s^2)
constexpr float THRUST_SIDE = 4.0f; // Side engine thrust (m/s^2)
constexpr float MAX_LANDING_VELOCITY = 2.0f; // Max safe landing velocity (m/s)
constexpr float MAX_ANGLE = 0.2f; // Max safe landing angle (radians)

// Rocket physics constants
constexpr float ROCKET_DRY_MASS = 1000.0f; // kg (without fuel)
constexpr float ROCKET_FUEL_MASS = 500.0f; // kg (full fuel)
constexpr float ROCKET_MOMENT_OF_INERTIA = 5000.0f; // kg⋅m²
constexpr float DRAG_COEFFICIENT = 0.1f; // Air resistance (minimal on moon)
constexpr float ANGULAR_DAMPING = 0.5f; // Angular velocity damping
constexpr float ENGINE_TORQUE_ARM = 1.5f; // Distance from center to side engines (m)

// Fire engine effects
constexpr int MAX_PARTICLES = 200;
constexpr float PARTICLE_LIFETIME = 1.5f; // seconds
constexpr float FLAME_VARIATION = 0.3f; // flame flicker variation

// Screen settings - reduced for better compatibility
constexpr int SCREEN_WIDTH = 800;  // Reduced from 1200
constexpr int SCREEN_HEIGHT = 600;  // Reduced from 800
constexpr int GROUND_HEIGHT = 100;
constexpr int SCALE = 100; // Pixels per meter

// Colors
#define BLACK SDL_Color({0, 0, 0, 255})
#define WHITE SDL_Color({255, 255, 255, 255})
#define RED SDL_Color({255, 0, 0, 255})
#define GREEN SDL_Color({0, 255, 0, 255})
#define BLUE SDL_Color({100, 149, 237, 255})
#define YELLOW SDL_Color({255, 255, 0, 255})
#define GRAY SDL_Color({128, 128, 128, 255})
#define DARK_GRAY SDL_Color({64, 64, 64, 255})
#define MOON_SURFACE SDL_Color({169, 169, 169, 255})
#define CRATER_COLOR SDL_Color({105, 105, 105, 255})
#define FLAME_ORANGE SDL_Color({255, 165, 0, 255})
#define FLAME_RED SDL_Color({255, 69, 0, 255})
#define FLAME_YELLOW SDL_Color({255, 255, 0, 255})

struct Particle {
    float x, y;
    float vx, vy;
    float lifetime;
    float max_lifetime;
    float size;
    SDL_Color color;
    int type; // 0=flame, 1=smoke
};

class Rocket {
private:
    // Physics state
    float x, y;           // Position (meters)
    float vx, vy;         // Velocity (m/s)
    float angle;          // Angle (radians)
    float angular_vel;    // Angular velocity (rad/s)
    float fuel;           // Fuel remaining (0-1)
    
    // Enhanced physics
    float mass;           // Current mass (kg)
    float thrust_force;   // Current thrust force (N)
    float torque;         // Current torque (N⋅m)
    
    // Visual properties
    int width, height;
    SDL_Color color;
    
    // Landing legs state
    bool left_leg_contact;
    bool right_leg_contact;
    
    // Landing reset timer
    float landing_timer;
    bool should_reset;
    
    // Fire engine effects
    std::vector<Particle> particles;
    float flame_intensity;
    float engine_noise;
    
    // Engine states
    bool main_engine_on;
    bool left_engine_on;
    bool right_engine_on;
    
    // Physics helper methods
    float getCurrentMass() const {
        return ROCKET_DRY_MASS + (fuel * ROCKET_FUEL_MASS);
    }
    
    float getCurrentMomentOfInertia() const {
        return ROCKET_MOMENT_OF_INERTIA + (fuel * ROCKET_FUEL_MASS * 0.5f); // Simplified
    }
    
public:
    Rocket(float start_x, float start_y) 
        : x(start_x), y(start_y), vx(0.0f), vy(0.0f), 
          angle(0.0f), angular_vel(0.0f), fuel(1.0f),
          mass(getCurrentMass()), thrust_force(0.0f), torque(0.0f),
          width(20), height(40), color(WHITE),
          left_leg_contact(false), right_leg_contact(false),
          landing_timer(0.0f), should_reset(false),
          flame_intensity(0.0f), engine_noise(0.0f),
          main_engine_on(false), left_engine_on(false), right_engine_on(false) {
        particles.reserve(MAX_PARTICLES);
    }
    
    void update(float dt, int action, const std::vector<float>& terrain_heights) {
        // Update current mass
        mass = getCurrentMass();
        
        // Check if rocket should reset after landing
        if (should_reset) {
            return; // Don't update physics if waiting for reset
        }
        
        // Apply gravity (F = mg, a = F/m = g)
        vy -= GRAVITY * dt;
        
        // Apply drag (minimal on moon but still there)
        float drag_force_x = -DRAG_COEFFICIENT * vx * std::abs(vx) / mass;
        float drag_force_y = -DRAG_COEFFICIENT * vy * std::abs(vy) / mass;
        vx += drag_force_x * dt;
        vy += drag_force_y * dt;
        
        // Apply thrust based on action (allow thrust even when landed for landing thrust)
        if (fuel > 0.0f) {
            switch(action) {
                case 1: // Fire left engine
                    {
                        float thrust = THRUST_SIDE * mass;
                        float thrust_x = thrust * std::sin(angle);
                        float thrust_y = thrust * std::cos(angle);
                        vx += thrust_x * dt / mass;
                        vy += thrust_y * dt / mass;
                        
                        // Apply torque from off-center thrust
                        torque += thrust * ENGINE_TORQUE_ARM;
                        fuel -= 0.001f * dt;
                        left_engine_on = true;
                        createEngineParticles(-1, 0.7f);
                    }
                    break;
                case 2: // Fire main engine
                    {
                        float thrust = THRUST_MAIN * mass;
                        float thrust_x = thrust * std::sin(angle);
                        float thrust_y = thrust * std::cos(angle);
                        vx += thrust_x * dt / mass;
                        vy += thrust_y * dt / mass;
                        
                        thrust_force = thrust;
                        fuel -= 0.003f * dt;
                        main_engine_on = true;
                        createEngineParticles(0, 1.0f);
                    }
                    break;
                case 3: // Fire right engine
                    {
                        float thrust = THRUST_SIDE * mass;
                        float thrust_x = thrust * std::sin(angle);
                        float thrust_y = thrust * std::cos(angle);
                        vx += thrust_x * dt / mass;
                        vy += thrust_y * dt / mass;
                        
                        // Apply torque from off-center thrust
                        torque -= thrust * ENGINE_TORQUE_ARM;
                        fuel -= 0.001f * dt;
                        right_engine_on = true;
                        createEngineParticles(1, 0.7f);
                    }
                    break;
                case 0: // Do nothing
                default:
                    break;
            }
        }
        
        // Update angular motion with torque and damping
        float moment_of_inertia = getCurrentMomentOfInertia();
        float angular_acceleration = torque / moment_of_inertia;
        angular_vel += angular_acceleration * dt;
        angular_vel *= (1.0f - ANGULAR_DAMPING * dt); // Apply damping
        
        // Update flame intensity with realistic flickering
        if (main_engine_on) {
            flame_intensity = 0.8f + 0.2f * std::sin(SDL_GetTicks() * 0.015f) + 
                            0.1f * std::sin(SDL_GetTicks() * 0.037f);
        } else {
            flame_intensity = 0.0f;
        }
        
        // Update position
        x += vx * dt;
        y += vy * dt;
        angle += angular_vel * dt;
        
        // Keep angle in reasonable range
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
        
        // Update particles
        updateParticles(dt);
        
        // Check ground collision with terrain
        int rocket_screen_x = static_cast<int>(x * SCALE);
        if (rocket_screen_x >= 0 && rocket_screen_x < SCREEN_WIDTH && !terrain_heights.empty()) {
            float terrain_y = terrain_heights[rocket_screen_x] / SCALE;
            
            // Get rocket bottom position considering angle
            float rocket_bottom = y - (height/2/SCALE) * std::cos(angle);
            
            if (rocket_bottom <= terrain_y) {
                // Prevent rocket from going through terrain
                y = terrain_y + (height/2/SCALE) * std::cos(angle);
                
                // Apply impact physics
                if (std::abs(vy) > 0.1f) {
                    // Bounce with energy loss
                    vy = -vy * 0.2f;
                    vx *= 0.8f; // Friction
                    angular_vel *= 0.5f; // Angular impact
                } else {
                    vy = 0.0f;
                    vx *= 0.9f; // Ground friction
                    angular_vel *= 0.8f; // Ground friction for rotation
                }
                
                // Check leg contacts based on angle and terrain
                float left_leg_x = rocket_screen_x - (width/2) * std::cos(angle) * SCALE;
                float right_leg_x = rocket_screen_x + (width/2) * std::cos(angle) * SCALE;
                
                float left_leg_y = y - (height/2/SCALE) * std::cos(angle) - (width/2/SCALE) * std::sin(angle);
                float right_leg_y = y - (height/2/SCALE) * std::cos(angle) + (width/2/SCALE) * std::sin(angle);
                
                // Get terrain heights at leg positions
                int left_leg_screen_x = static_cast<int>(left_leg_x);
                int right_leg_screen_x = static_cast<int>(right_leg_x);
                
                left_leg_contact = false;
                right_leg_contact = false;
                
                if (left_leg_screen_x >= 0 && left_leg_screen_x < SCREEN_WIDTH) {
                    float left_terrain_y = terrain_heights[left_leg_screen_x] / SCALE;
                    left_leg_contact = (left_leg_y <= left_terrain_y);
                }
                
                if (right_leg_screen_x >= 0 && right_leg_screen_x < SCREEN_WIDTH) {
                    float right_terrain_y = terrain_heights[right_leg_screen_x] / SCALE;
                    right_leg_contact = (right_leg_y <= right_terrain_y);
                }
                
                // If both legs are on ground, consider it landed and reset after delay
                if (left_leg_contact && right_leg_contact) {
                    // Apply ground friction to horizontal movement
                    vx *= 0.95f;
                    
                    // Allow small thrust for landing adjustments but limit movement
                    if (std::abs(vx) < 0.1f && std::abs(vy) < 0.1f) {
                        landing_timer += dt;
                        
                        // Reset after 2 seconds of landing anywhere
                        if (landing_timer > 2.0f) {
                            should_reset = true;
                        }
                    } else {
                        landing_timer = 0.0f; // Reset timer if still moving
                    }
                } else {
                    landing_timer = 0.0f; // Reset timer if not properly landed
                }
            }
        }
        
        // Wall boundaries with energy loss
        if (x < 0) { 
            x = 0; 
            vx = std::abs(vx) * 0.3f; // Energy loss on impact
            angular_vel *= -0.5f; // Spin from impact
        }
        if (x > SCREEN_WIDTH / SCALE) { 
            x = SCREEN_WIDTH / SCALE; 
            vx = -std::abs(vx) * 0.3f;
            angular_vel *= -0.5f;
        }
    }
    
    void createEngineParticles(int engine_offset, float intensity) {
        if (particles.size() >= MAX_PARTICLES) return;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle_spread(-0.3f, 0.3f);
        std::uniform_real_distribution<float> velocity_spread(3.0f, 8.0f);
        std::uniform_real_distribution<float> size_spread(2.0f, 8.0f);
        
        // Particle count based on thrust force
        int particles_to_create = static_cast<int>(intensity * thrust_force / 1000.0f);
        particles_to_create = std::min(particles_to_create, 8);
        
        for (int i = 0; i < particles_to_create && particles.size() < MAX_PARTICLES; i++) {
            Particle p;
            
            // Position at engine nozzle with some spread
            float nozzle_offset = (height/2/SCALE) + 0.1f;
            float nozzle_x = x - nozzle_offset * std::sin(angle) + engine_offset * (width/2/SCALE) * std::cos(angle);
            float nozzle_y = y - nozzle_offset * std::cos(angle) - engine_offset * (width/2/SCALE) * std::sin(angle);
            
            // Add position variation
            nozzle_x += (gen() % 100 - 50) / 1000.0f;
            nozzle_y += (gen() % 100 - 50) / 1000.0f;
            
            p.x = nozzle_x;
            p.y = nozzle_y;
            
            // Velocity opposite to thrust direction with realistic spread
            float particle_angle = angle + M_PI + angle_spread(gen) + engine_offset * 0.5f;
            float base_velocity = velocity_spread(gen) * intensity;
            
            // Add rocket velocity to particle velocity
            p.vx = vx + base_velocity * std::sin(particle_angle);
            p.vy = vy + base_velocity * std::cos(particle_angle);
            
            // Lifetime and size based on intensity
            p.lifetime = p.max_lifetime = PARTICLE_LIFETIME * (0.5f + intensity * 0.5f);
            p.size = size_spread(gen) * intensity;
            
            // Color based on temperature and intensity
            float temp_factor = intensity;
            if (temp_factor > 0.8f) {
                p.color = FLAME_YELLOW;
                p.type = 0;
            } else if (temp_factor > 0.5f) {
                p.color = FLAME_ORANGE;
                p.type = 0;
            } else if (temp_factor > 0.3f) {
                p.color = FLAME_RED;
                p.type = 1; // smoke
            } else {
                p.color = DARK_GRAY;
                p.type = 1; // more smoke
            }
            
            particles.push_back(p);
        }
    }
    
    void updateParticles(float dt) {
        for (auto it = particles.begin(); it != particles.end();) {
            it->lifetime -= dt;
            
            if (it->lifetime <= 0) {
                it = particles.erase(it);
            } else {
                // Update particle physics with gravity
                it->x += it->vx * dt;
                it->y += it->vy * dt;
                
                // Apply reduced gravity to particles (hot gases rise)
                if (it->type == 0) { // flame particles
                    it->vy -= GRAVITY * dt * 0.2f; // Less affected by gravity
                } else { // smoke particles
                    it->vy -= GRAVITY * dt * 0.05f; // Even less affected
                }
                
                // Apply drag to particles
                it->vx *= (1.0f - 0.5f * dt);
                it->vy *= (1.0f - 0.3f * dt);
                
                // Fade out and cool down
                float alpha = it->lifetime / it->max_lifetime;
                it->color.a = static_cast<Uint8>(255 * alpha);
                
                // Color transition from hot to cool
                if (it->type == 0 && alpha < 0.5f) {
                    it->color = FLAME_RED;
                    it->type = 1; // becomes smoke
                }
                
                ++it;
            }
        }
    }
    
    void draw(SDL_Renderer* renderer) {
        // Convert to screen coordinates
        int screen_x = static_cast<int>(x * SCALE);
        int screen_y = SCREEN_HEIGHT - static_cast<int>(y * SCALE);
        
        // Save current transform
        SDL_FPoint center = {static_cast<float>(screen_x), static_cast<float>(screen_y)};
        
        // Draw engine flames and particles
        drawEngineEffects(renderer);
        
        // Draw rocket body (rotated rectangle)
        SDL_FRect rocket_rect = {
            static_cast<float>(screen_x - width/2),
            static_cast<float>(screen_y - height/2),
            static_cast<float>(width),
            static_cast<float>(height)
        };
        
        // Create rotated points for rocket body
        SDL_FPoint points[4];
        for (int i = 0; i < 4; i++) {
            float px, py;
            if (i == 0) { px = -width/2; py = -height/2; }
            else if (i == 1) { px = width/2; py = -height/2; }
            else if (i == 2) { px = width/2; py = height/2; }
            else { px = -width/2; py = height/2; }
            
            // Rotate point
            float rotated_x = px * std::cos(angle) - py * std::sin(angle);
            float rotated_y = px * std::sin(angle) + py * std::cos(angle);
            
            points[i] = {screen_x + rotated_x, screen_y + rotated_y};
        }
        
        // Draw rocket body with metallic look
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
        for (int i = 0; i < 4; i++) {
            SDL_RenderLine(renderer, points[i].x, points[i].y, 
                          points[(i+1)%4].x, points[(i+1)%4].y);
        }
        
        // Draw rocket details
        SDL_SetRenderDrawColor(renderer, RED.r, RED.g, RED.b, RED.a);
        // Nose cone
        float nose_x = screen_x - (height/2 + 5) * std::sin(angle);
        float nose_y = screen_y - (height/2 + 5) * std::cos(angle);
        SDL_RenderLine(renderer, screen_x, screen_y, nose_x, nose_y);
        
        // Draw landing legs
        SDL_SetRenderDrawColor(renderer, GREEN.r, GREEN.g, GREEN.b, GREEN.a);
        
        // Left leg
        float left_leg_x = screen_x - (width/2) * std::cos(angle);
        float left_leg_y = screen_y - (height/2) * std::cos(angle) - (width/2) * std::sin(angle);
        SDL_RenderLine(renderer, left_leg_x, left_leg_y, left_leg_x, left_leg_y + 10);
        
        // Right leg
        float right_leg_x = screen_x + (width/2) * std::cos(angle);
        float right_leg_y = screen_y - (height/2) * std::cos(angle) + (width/2) * std::sin(angle);
        SDL_RenderLine(renderer, right_leg_x, right_leg_y, right_leg_x, right_leg_y + 10);
    }
    
    void drawEngineEffects(SDL_Renderer* renderer) {
        // Draw particles first (background)
        for (const auto& particle : particles) {
            int px = static_cast<int>(particle.x * SCALE);
            int py = SCREEN_HEIGHT - static_cast<int>(particle.y * SCALE);
            int size = static_cast<int>(particle.size);
            
            SDL_SetRenderDrawColor(renderer, particle.color.r, particle.color.g, 
                                  particle.color.b, particle.color.a);
            
            if (particle.type == 0) {
                // Flame particle - draw as glowing circle
                for (int i = -size/2; i <= size/2; i++) {
                    for (int j = -size/2; j <= size/2; j++) {
                        if (i*i + j*j <= (size/2)*(size/2)) {
                            SDL_RenderPoint(renderer, px + i, py + j);
                        }
                    }
                }
            } else {
                // Smoke particle - draw as faded square
                SDL_FRect smoke_rect = {static_cast<float>(px - size/2), 
                                       static_cast<float>(py - size/2),
                                       static_cast<float>(size), 
                                       static_cast<float>(size)};
                SDL_RenderFillRect(renderer, &smoke_rect);
            }
        }
        
        // Draw engine flames
        if (fuel > 0.0f) {
            int screen_x = static_cast<int>(x * SCALE);
            int screen_y = SCREEN_HEIGHT - static_cast<int>(y * SCALE);
            
            // Main engine flame
            if (main_engine_on) {
                float flame_length = 20 * flame_intensity;
                float flame_width = 8 * flame_intensity;
                
                // Flame core (white-yellow)
                SDL_SetRenderDrawColor(renderer, FLAME_YELLOW.r, FLAME_YELLOW.g, 
                                      FLAME_YELLOW.b, 255);
                float core_x = screen_x - (height/2 + flame_length/2) * std::sin(angle);
                float core_y = screen_y - (height/2 + flame_length/2) * std::cos(angle);
                SDL_RenderLine(renderer, screen_x, screen_y, core_x, core_y);
                
                // Flame outer (orange-red)
                SDL_SetRenderDrawColor(renderer, FLAME_ORANGE.r, FLAME_ORANGE.g, 
                                      FLAME_ORANGE.b, 200);
                for (int i = -1; i <= 1; i++) {
                    float outer_x = screen_x - (height/2 + flame_length) * std::sin(angle) + i * flame_width * std::cos(angle);
                    float outer_y = screen_y - (height/2 + flame_length) * std::cos(angle) - i * flame_width * std::sin(angle);
                    SDL_RenderLine(renderer, screen_x - i * 2 * std::cos(angle), 
                                  screen_y + i * 2 * std::sin(angle), outer_x, outer_y);
                }
            }
            
            // Left engine flame
            if (left_engine_on) {
                drawSideEngineFlame(renderer, screen_x, screen_y, -1);
            }
            
            // Right engine flame
            if (right_engine_on) {
                drawSideEngineFlame(renderer, screen_x, screen_y, 1);
            }
        }
    }
    
    void drawSideEngineFlame(SDL_Renderer* renderer, int screen_x, int screen_y, int side) {
        float flame_length = 12 * flame_intensity;
        
        // Calculate side engine position
        float engine_x = screen_x - (height/2) * std::sin(angle) + side * (width/2) * std::cos(angle);
        float engine_y = screen_y - (height/2) * std::cos(angle) - side * (width/2) * std::sin(angle);
        
        // Draw flame
        SDL_SetRenderDrawColor(renderer, FLAME_ORANGE.r, FLAME_ORANGE.g, 
                              FLAME_ORANGE.b, 200);
        float flame_end_x = engine_x - flame_length * std::sin(angle + side * 0.3f);
        float flame_end_y = engine_y - flame_length * std::cos(angle + side * 0.3f);
        SDL_RenderLine(renderer, engine_x, engine_y, flame_end_x, flame_end_y);
        
        // Flame glow
        SDL_SetRenderDrawColor(renderer, FLAME_YELLOW.r, FLAME_YELLOW.g, 
                              FLAME_YELLOW.b, 150);
        SDL_RenderLine(renderer, engine_x, engine_y, 
                      (engine_x + flame_end_x) / 2, (engine_y + flame_end_y) / 2);
    }
    
    // Get observation for RL model
    torch::Tensor getObservation(const std::vector<float>& terrain_heights) {
        // Normalize observations similar to gym environment
        float norm_x = (x - SCREEN_WIDTH/(2*SCALE)) / (SCREEN_WIDTH/(2*SCALE));
        
        // Get terrain height at rocket position for normalization
        int rocket_screen_x = static_cast<int>(x * SCALE);
        float current_terrain_height = GROUND_HEIGHT / SCALE; // default
        if (rocket_screen_x >= 0 && rocket_screen_x < SCREEN_WIDTH && !terrain_heights.empty()) {
            current_terrain_height = terrain_heights[rocket_screen_x] / SCALE;
        }
        
        float norm_y = (y - current_terrain_height) / ((SCREEN_HEIGHT-50)/SCALE); // normalized to terrain
        float norm_vx = vx / 5.0f; // Normalize to [-1, 1] range
        float norm_vy = vy / 5.0f;
        float norm_angle = angle / M_PI; // Normalize to [-1, 1]
        float norm_angular_vel = angular_vel / 2.0f;
        
        return torch::tensor({{norm_x, norm_y, norm_vx, norm_vy, 
                              norm_angle, norm_angular_vel, 
                              static_cast<float>(left_leg_contact), 
                              static_cast<float>(right_leg_contact)}});
    }
    
    bool isLanded() const { return left_leg_contact && right_leg_contact; }
    bool isCrashed() const {
        return isLanded() && (std::abs(vy) > MAX_LANDING_VELOCITY || 
                             std::abs(angle) > MAX_ANGLE);
    }
    bool isSuccess() const {
        return isLanded() && !isCrashed();
    }
    
    bool shouldReset() const { return should_reset; }
    
    void reset(float start_x, float start_y) {
        x = start_x;
        y = start_y;
        vx = vy = 0.0f;
        angle = angular_vel = 0.0f;
        fuel = 1.0f;
        mass = getCurrentMass();
        thrust_force = torque = 0.0f;
        left_leg_contact = right_leg_contact = false;
        landing_timer = 0.0f;
        should_reset = false;
        main_engine_on = left_engine_on = right_engine_on = false;
        flame_intensity = 0.0f;
        particles.clear();
    }
    float getX() const { return x; }
    float getY() const { return y; }
    float getVX() const { return vx; }
    float getVY() const { return vy; }
    float getAngle() const { return angle; }
    float getFuel() const { return fuel; }
    float getLandingTimer() const { return landing_timer; }
};

class LunarSimulation {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    
    std::unique_ptr<Rocket> rocket;
    Policy policy;
    torch::Device device;
    
    float target_x;
    int target_width;
    
    bool running;
    bool paused;
    bool show_predictions;
    
    // Performance tracking
    int episodes;
    int successful_landings;
    float total_reward;
    
    // Moon terrain
    std::vector<float> terrain_heights;
    std::vector<std::pair<float, float>> craters; // x, radius
    
    void generateMoonTerrain() {
        // Generate base terrain with enhanced variation for bumpier surface
        terrain_heights.resize(SCREEN_WIDTH + 1);
        
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for consistent terrain
        std::uniform_real_distribution<float> height_variation(-0.4f, 0.4f);
        std::uniform_real_distribution<float> rough_variation(-0.2f, 0.2f);
        
        // Create more varied and bumpier terrain using multiple sine waves and noise
        for (int x = 0; x <= SCREEN_WIDTH; x++) {
            float base_height = GROUND_HEIGHT;
            
            // Add large rolling hills
            base_height += 15 * std::sin(x * 0.008f);
            base_height += 10 * std::sin(x * 0.015f);
            
            // Add medium bumps for more texture
            base_height += 5 * std::sin(x * 0.05f);
            base_height += 3 * std::sin(x * 0.08f);
            
            // Add small rough variations for bumpiness
            base_height += height_variation(gen) * 8;
            base_height += rough_variation(gen) * 2;
            
            // Add occasional sharp bumps
            if (x % 50 == 0) {
                base_height += (gen() % 100 - 50) / 10.0f;
            }
            
            terrain_heights[x] = base_height;
        }
        
        // Generate craters
        std::uniform_real_distribution<float> crater_x(100.0f, SCREEN_WIDTH - 100.0f);
        std::uniform_real_distribution<float> crater_radius(20.0f, 80.0f);
        
        for (int i = 0; i < 8; i++) {
            float cx = crater_x(gen);
            float radius = crater_radius(gen);
            craters.push_back({cx, radius});
            
            // Modify terrain around crater
            for (int x = static_cast<int>(cx - radius); x <= static_cast<int>(cx + radius) && x <= SCREEN_WIDTH; x++) {
                if (x >= 0) {
                    float dist = std::abs(x - cx);
                    if (dist < radius) {
                        // Create crater depression
                        float crater_depth = (1.0f - (dist / radius)) * 15;
                        terrain_heights[x] -= crater_depth;
                        // Ensure minimum height
                        terrain_heights[x] = std::max(50.0f, terrain_heights[x]);
                    }
                }
            }
        }
        
        // Smooth the terrain
        for (int iteration = 0; iteration < 2; iteration++) {
            std::vector<float> smoothed = terrain_heights;
            for (int x = 1; x < SCREEN_WIDTH; x++) {
                smoothed[x] = (terrain_heights[x-1] + terrain_heights[x] + terrain_heights[x+1]) / 3.0f;
            }
            terrain_heights = smoothed;
        }
    }
    
public:
    LunarSimulation() : window(nullptr), renderer(nullptr), device(torch::kCPU), policy(nullptr),
                       running(true), paused(false), show_predictions(true),
                       episodes(0), successful_landings(0), total_reward(0.0f) {
        
        // Generate moon terrain
        generateMoonTerrain();
        
        // Initialize RL model
        const int hiddenSize = 64;
        const bool recurrent = false;
        
        auto base = std::make_shared<MlpBase>(8, recurrent, hiddenSize);
        base->to(device);
        
        ActionSpace space{"Discrete", {4}};
        policy = Policy(space, base, true);
        policy->to(device);
        
        // Initialize rocket at random position
        resetEpisode();
        
        // Set landing target
        target_x = SCREEN_WIDTH / (2 * SCALE);
        target_width = 60 / SCALE; // 60cm target zone
    }
    
    bool init() {
        // Initialize SDL with video subsystem
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            spdlog::error("SDL initialization failed: {}", SDL_GetError());
            return false;
        }
        
        // Set hints for better rendering
        SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
        SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0"); // Disable vsync for better compatibility
        
        window = SDL_CreateWindow("Lunar Landing RL Simulation", 
                                  SCREEN_WIDTH, SCREEN_HEIGHT, 
                                  SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
        if (!window) {
            spdlog::error("Window creation failed: {}", SDL_GetError());
            SDL_Quit();
            return false;
        }
        
        renderer = SDL_CreateRenderer(window, nullptr);
        if (!renderer) {
            spdlog::error("Renderer creation failed: {}", SDL_GetError());
            SDL_DestroyWindow(window);
            window = nullptr;
            SDL_Quit();
            return false;
        }
        
        // Make window visible and bring to front
        SDL_ShowWindow(window);
        SDL_RaiseWindow(window);
        
        return true;
    }
    
    void resetEpisode() {
        // Random starting position
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> x_dist(1.0f, (SCREEN_WIDTH/SCALE) - 1.0f);
        std::uniform_real_distribution<float> y_dist(3.0f, 5.0f);
        
        if (!rocket) {
            rocket = std::make_unique<Rocket>(x_dist(gen), y_dist(gen));
        } else {
            rocket->reset(x_dist(gen), y_dist(gen));
        }
        episodes++;
    }
    
    void handleEvents() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch(event.type) {
                case SDL_EVENT_QUIT:
                    running = false;
                    break;
                case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                    if (event.window.windowID == SDL_GetWindowID(window)) {
                        running = false;
                    }
                    break;
                case SDL_EVENT_KEY_DOWN:
                    switch(event.key.key) {
                        case SDLK_SPACE:
                            paused = !paused;
                            break;
                        case SDLK_R:
                            resetEpisode();
                            break;
                        case SDLK_P:
                            show_predictions = !show_predictions;
                            break;
                        case SDLK_ESCAPE:
                            running = false;
                            break;
                    }
                    break;
            }
        }
    }
    
    void update(float dt) {
        if (paused || rocket->isLanded()) return;
        
        // Get RL prediction
        torch::NoGradGuard no_grad;
        auto observation = rocket->getObservation(terrain_heights).to(device);
        auto hidden_states = torch::zeros({1, 64}).to(device);
        auto masks = torch::ones({1, 1}).to(device);
        
        auto results = policy->act(observation, hidden_states, masks);
        int action = results[0].item<int>();
        
        // Update rocket physics
        rocket->update(dt, action, terrain_heights);
        
        // Check if rocket should reset
        if (rocket->shouldReset()) {
            resetEpisode();
        }
        
        // Check episode end
        if (rocket->isLanded()) {
            if (rocket->isSuccess()) {
                successful_landings++;
                total_reward += 100.0f;
            } else {
                total_reward -= 100.0f;
            }
        }
    }
    
    void draw() {
        // Clear screen
        SDL_SetRenderDrawColor(renderer, BLACK.r, BLACK.g, BLACK.b, BLACK.a);
        SDL_RenderClear(renderer);
        
        // Draw stars
        drawStars();
        
        // Draw moon terrain
        drawMoonTerrain();
        
        // Draw landing target
        drawLandingTarget();
        
        // Draw rocket
        rocket->draw(renderer);
        
        // Draw UI
        drawUI();
        
        SDL_RenderPresent(renderer);
    }
    
    void drawStars() {
        SDL_SetRenderDrawColor(renderer, WHITE.r, WHITE.g, WHITE.b, WHITE.a);
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for consistent stars
        std::uniform_int_distribution<int> x_dist(0, SCREEN_WIDTH);
        std::uniform_int_distribution<int> y_dist(0, SCREEN_HEIGHT - 150);
        
        for (int i = 0; i < 150; i++) {
            int x = x_dist(gen);
            int y = y_dist(gen);
            int brightness = 128 + (gen() % 128);
            SDL_SetRenderDrawColor(renderer, brightness, brightness, brightness, 255);
            SDL_RenderPoint(renderer, x, y);
        }
    }
    
    void drawMoonTerrain() {
        // Draw main terrain surface
        for (int x = 0; x < SCREEN_WIDTH - 1; x++) {
            int y1 = SCREEN_HEIGHT - static_cast<int>(terrain_heights[x]);
            int y2 = SCREEN_HEIGHT - static_cast<int>(terrain_heights[x + 1]);
            
            // Main surface color
            SDL_SetRenderDrawColor(renderer, MOON_SURFACE.r, MOON_SURFACE.g, MOON_SURFACE.b, 255);
            SDL_RenderLine(renderer, x, y1, x + 1, y2);
            
            // Fill below surface
            SDL_SetRenderDrawColor(renderer, MOON_SURFACE.r, MOON_SURFACE.g, MOON_SURFACE.b, 255);
            for (int y = y1; y < SCREEN_HEIGHT; y++) {
                SDL_RenderPoint(renderer, x, y);
            }
        }
        
        // Draw crater details
        for (const auto& crater : craters) {
            drawCrater(crater.first, crater.second);
        }
        
        // Add surface texture
        std::random_device rd;
        std::mt19937 gen(123);
        std::uniform_int_distribution<int> texture_x(0, SCREEN_WIDTH);
        std::uniform_int_distribution<int> brightness(100, 140);
        
        for (int i = 0; i < 500; i++) {
            int x = texture_x(gen);
            int terrain_y = SCREEN_HEIGHT - static_cast<int>(terrain_heights[x]);
            
            SDL_SetRenderDrawColor(renderer, brightness(gen), brightness(gen), brightness(gen), 200);
            SDL_RenderPoint(renderer, x, terrain_y + 1);
        }
    }
    
    void drawCrater(float cx, float radius) {
        // Draw crater rim and interior
        for (int angle = 0; angle < 360; angle += 5) {
            float rad = angle * M_PI / 180.0f;
            
            // Outer rim
            float rim_x = cx + radius * std::cos(rad);
            float rim_y = SCREEN_HEIGHT - terrain_heights[static_cast<int>(cx)] - radius * 0.3f * std::sin(rad);
            
            SDL_SetRenderDrawColor(renderer, CRATER_COLOR.r, CRATER_COLOR.g, CRATER_COLOR.b, 200);
            SDL_RenderPoint(renderer, static_cast<int>(rim_x), static_cast<int>(rim_y));
            
            // Inner shadow
            if (radius > 30) {
                float inner_x = cx + radius * 0.7f * std::cos(rad);
                float inner_y = SCREEN_HEIGHT - terrain_heights[static_cast<int>(cx)] - radius * 0.2f * std::sin(rad);
                
                SDL_SetRenderDrawColor(renderer, DARK_GRAY.r, DARK_GRAY.g, DARK_GRAY.b, 150);
                SDL_RenderPoint(renderer, static_cast<int>(inner_x), static_cast<int>(inner_y));
            }
        }
    }
    
    void drawTargetText(int x, int y) {
        // Simple text rendering for "TARGET" using the bitmap font
        std::string text = "TARGET";
        int char_width = 8;
        int char_height = 7;
        int scale = 1;
        
        SDL_SetRenderDrawColor(renderer, YELLOW.r, YELLOW.g, YELLOW.b, 255);
        
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            if (c >= 128) c = '?';
            
            const auto& font_data = FONT_DATA[c];
            
            for (int row = 0; row < char_height; row++) {
                uint8_t byte = font_data[row];
                for (int col = 0; col < 5; col++) {
                    if (byte & (0x10 >> col)) {
                        for (int sx = 0; sx < scale; sx++) {
                            for (int sy = 0; sy < scale; sy++) {
                                SDL_RenderPoint(renderer, 
                                              x + i * (char_width + 1) * scale + col * scale + sx,
                                              y + row * scale + sy);
                            }
                        }
                    }
                }
            }
        }
    }
    
    void drawLandingTarget() {
        int target_screen_x = static_cast<int>(target_x * SCALE);
        int target_screen_width = static_cast<int>(target_width * SCALE);
        
        // Find terrain height at target location
        int terrain_y = SCREEN_HEIGHT - static_cast<int>(terrain_heights[target_screen_x]);
        
        // Draw landing pad with enhanced visibility
        SDL_SetRenderDrawColor(renderer, GREEN.r, GREEN.g, GREEN.b, 255);
        SDL_FRect target = {static_cast<float>(target_screen_x - target_screen_width/2),
                           static_cast<float>(terrain_y - 8),
                           static_cast<float>(target_screen_width), 8.0f};
        SDL_RenderFillRect(renderer, &target);
        
        // Draw landing pad border for better visibility
        SDL_SetRenderDrawColor(renderer, YELLOW.r, YELLOW.g, YELLOW.b, 255);
        SDL_FRect target_border = {static_cast<float>(target_screen_x - target_screen_width/2),
                                  static_cast<float>(terrain_y - 8),
                                  static_cast<float>(target_screen_width), 8.0f};
        SDL_RenderRect(renderer, &target_border);
        
        // Draw landing pad markers
        // Left marker
        SDL_RenderLine(renderer, target_screen_x - target_screen_width/2 - 15, terrain_y - 15,
                      target_screen_x - target_screen_width/2 - 15, terrain_y + 5);
        SDL_RenderLine(renderer, target_screen_x - target_screen_width/2 - 20, terrain_y - 10,
                      target_screen_x - target_screen_width/2 - 10, terrain_y - 10);
        
        // Right marker
        SDL_RenderLine(renderer, target_screen_x + target_screen_width/2 + 15, terrain_y - 15,
                      target_screen_x + target_screen_width/2 + 15, terrain_y + 5);
        SDL_RenderLine(renderer, target_screen_x + target_screen_width/2 + 10, terrain_y - 10,
                      target_screen_x + target_screen_width/2 + 20, terrain_y - 10);
        
        // Draw target zone indicator text
        drawTargetText(target_screen_x, terrain_y - 25);
    }
    
    void drawUI() {
        // Bitmap font rendering function
        auto renderText = [this](const std::string& text, int x, int y, SDL_Color color) {
            int char_width = 8;
            int char_height = 7;
            int scale = 2;
            
            for (size_t i = 0; i < text.size(); i++) {
                char c = text[i];
                if (c >= 128) c = '?'; // Handle unsupported chars
                
                const auto& font_data = FONT_DATA[c];
                
                for (int row = 0; row < char_height; row++) {
                    uint8_t byte = font_data[row];
                    for (int col = 0; col < 5; col++) {
                        if (byte & (0x10 >> col)) {
                            SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
                            SDL_FRect pixel = {
                                static_cast<float>(x + i * (char_width + 1) * scale + col * scale),
                                static_cast<float>(y + row * scale),
                                static_cast<float>(scale),
                                static_cast<float>(scale)
                            };
                            SDL_RenderFillRect(renderer, &pixel);
                        }
                    }
                }
            }
        };
        
        int y_offset = 10;
        
        // Episode info
        renderText("Episode: " + std::to_string(episodes), 10, y_offset, WHITE);
        y_offset += 25;
        renderText("Success Rate: " + std::to_string(episodes > 0 ? (successful_landings * 100 / episodes) : 0) + "%", 
                  10, y_offset, WHITE);
        y_offset += 25;
        
        // Rocket state
        renderText("Position: (" + std::to_string(rocket->getX()).substr(0, 4) + ", " + 
                  std::to_string(rocket->getY()).substr(0, 4) + ")", 10, y_offset, WHITE);
        y_offset += 25;
        renderText("Velocity: (" + std::to_string(rocket->getVX()).substr(0, 4) + ", " + 
                  std::to_string(rocket->getVY()).substr(0, 4) + ")", 10, y_offset, WHITE);
        y_offset += 25;
        renderText("Angle: " + std::to_string(rocket->getAngle() * 180 / M_PI).substr(0, 4) + "°", 
                  10, y_offset, WHITE);
        y_offset += 25;
        renderText("Fuel: " + std::to_string(rocket->getFuel() * 100).substr(0, 3) + "%", 10, y_offset, WHITE);
        y_offset += 25;
        
        // Landing status
        if (rocket->isLanded()) {
            if (rocket->isSuccess()) {
                renderText("SUCCESSFUL LANDING!", 10, y_offset, GREEN);
                y_offset += 25;
                float countdown = 3.0f - rocket->getLandingTimer();
                renderText("Reset in: " + std::to_string(static_cast<int>(countdown)) + "s", 10, y_offset, YELLOW);
            } else {
                renderText("CRASH LANDING!", 10, y_offset, RED);
                y_offset += 25;
                float countdown = 3.0f - rocket->getLandingTimer();
                renderText("Reset in: " + std::to_string(static_cast<int>(countdown)) + "s", 10, y_offset, YELLOW);
            }
        } else if (std::abs(rocket->getVY()) > MAX_LANDING_VELOCITY) {
            renderText("TOO FAST!", 10, y_offset, YELLOW);
        } else if (std::abs(rocket->getAngle()) > MAX_ANGLE) {
            renderText("BAD ANGLE!", 10, y_offset, YELLOW);
        }
        
        // Controls
        y_offset = SCREEN_HEIGHT - 50;
        renderText("SPACE:Pause R:Reset P:Pred ESC:Quit", 10, y_offset, WHITE);
        
        // RL predictions (if enabled)
        if (show_predictions && !rocket->isLanded()) {
            torch::NoGradGuard no_grad;
            auto observation = rocket->getObservation(terrain_heights).to(device);
            auto hidden_states = torch::zeros({1, 64}).to(device);
            auto masks = torch::ones({1, 1}).to(device);
            
            auto results = policy->act(observation, hidden_states, masks);
            int action = results[0].item<int>();
            auto value = results[2].item<float>();
            auto probs = policy->getProbability(observation, hidden_states, masks);
            
            int pred_x = SCREEN_WIDTH - 200;
            int pred_y = 10;
            
            renderText("RL Predictions:", pred_x, pred_y, WHITE);
            pred_y += 20;
            
            const std::vector<std::string> action_names = {"Nothing", "Left", "Main", "Right"};
            for (int i = 0; i < 4; i++) {
                SDL_Color color = (i == action) ? GREEN : WHITE;
                renderText(action_names[i] + ":" + std::to_string(probs[0][i].item<float>()).substr(0, 3), 
                          pred_x, pred_y, color);
                pred_y += 15;
            }
            
            renderText("Value:" + std::to_string(value).substr(0, 3), pred_x, pred_y + 5, WHITE);
        }
        
        // Pause indicator
        if (paused) {
            renderText("PAUSED", SCREEN_WIDTH/2 - 40, 50, YELLOW);
        }
    }
    
    void run() {
        if (!init()) return;
        
        auto last_time = std::chrono::high_resolution_clock::now();
        
        while (running) {
            auto current_time = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(current_time - last_time).count();
            last_time = current_time;
            
            // Handle events first
            handleEvents();
            
            // Update physics
            update(dt);
            
            // Draw everything
            draw();
            
            // Ensure the window is visible (only call occasionally, not every frame)
            static int frame_count = 0;
            frame_count++;
            if (frame_count % 60 == 0) { // Only show/raise window every second
                SDL_ShowWindow(window);
                SDL_RaiseWindow(window);
            }
            
            // Cap frame rate to ~60 FPS
            SDL_Delay(16);
        }
        
        cleanup();
    }
    
    void cleanup() {
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
        
        spdlog::info("Simulation ended. Episodes: {}, Success Rate: {:.1f}%", 
                    episodes, episodes > 0 ? (successful_landings * 100.0f / episodes) : 0.0f);
    }
};


TEST_CASE("LunarAlightingSimulationTraining")
{
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("%^[%T %7l] %v%$");
    
    torch::manual_seed(42);
    
    spdlog::info("Starting Lunar Landing RL Simulation");
    spdlog::info("=====================================");
    
    try {
        LunarSimulation simulation;
        simulation.run();
    } catch (const std::exception& e) {
        spdlog::error("Simulation failed: {}", e.what());
    }
}
