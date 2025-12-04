/**
 * @file mujoco_sim_node.cpp
 * @brief MuJoCo simulation node with ROS2 interface and GLFW visualization
 * 
 * This node runs MuJoCo simulation with passive viewer (like Python's launch_passive)
 * and provides ROS2 topics for:
 * - Publishing: joint_states, imu, odometry
 * - Subscribing: joint control commands (position, velocity, torque, gains)
 * - Services: reset, pause/unpause
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include "mujoco_ros_msgs/msg/joint_control_cmd.hpp"

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>

// Mouse interaction state
struct MouseState {
    bool button_left = false;
    bool button_middle = false;
    bool button_right = false;
    double lastx = 0;
    double lasty = 0;
};

class MujocoSimNode : public rclcpp::Node {
public:
    MujocoSimNode() : Node("mujoco_sim") {
        // Declare parameters
        this->declare_parameter<std::string>("model_path", "");
        this->declare_parameter<std::string>("robot_name", "robot");
        this->declare_parameter<double>("sim_rate", 1000.0);
        this->declare_parameter<double>("publish_rate", 500.0);
        this->declare_parameter<std::vector<std::string>>("joint_names", std::vector<std::string>{});
        this->declare_parameter<std::string>("base_link_name", "base_link");
        this->declare_parameter<bool>("headless", false);

        // Get parameters
        model_path_ = this->get_parameter("model_path").as_string();
        robot_name_ = this->get_parameter("robot_name").as_string();
        sim_rate_ = this->get_parameter("sim_rate").as_double();
        publish_rate_ = this->get_parameter("publish_rate").as_double();
        joint_names_ = this->get_parameter("joint_names").as_string_array();
        base_link_name_ = this->get_parameter("base_link_name").as_string();
        headless_ = this->get_parameter("headless").as_bool();

        if (model_path_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Parameter 'model_path' is required!");
            throw std::runtime_error("Missing required parameter: model_path");
        }

        // Load MuJoCo model
        if (!loadModel(model_path_)) {
            throw std::runtime_error("Failed to load MuJoCo model");
        }

        // Initialize visualization if not headless
        if (!headless_) {
            if (!initVisualization()) {
                throw std::runtime_error("Failed to initialize visualization");
            }
        }

        // Initialize command storage
        num_actuators_ = model_->nu;
        target_positions_.resize(num_actuators_, 0.0);
        target_velocities_.resize(num_actuators_, 0.0);
        target_torques_.resize(num_actuators_, 0.0);
        kp_gains_.resize(num_actuators_, 0.0);
        kd_gains_.resize(num_actuators_, 0.0);

        // Build joint mapping if joint names provided
        if (!joint_names_.empty()) {
            buildJointMapping();
        } else {
            buildDefaultJointMapping();
        }

        // Publishers
        joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/mujoco/joint_states", 10);
        imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>(
            "/mujoco/imu", 10);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/mujoco/odom", 10);

        // Subscriber for combined joint control command
        joint_cmd_sub_ = this->create_subscription<mujoco_ros_msgs::msg::JointControlCmd>(
            "/mujoco/joint_cmd", 10,
            std::bind(&MujocoSimNode::jointCmdCallback, this, std::placeholders::_1));

        // Services
        pause_srv_ = this->create_service<std_srvs::srv::SetBool>(
            "/mujoco/pause",
            std::bind(&MujocoSimNode::pauseCallback, this, 
                     std::placeholders::_1, std::placeholders::_2));
        
        reset_srv_ = this->create_service<std_srvs::srv::Empty>(
            "/mujoco/reset",
            std::bind(&MujocoSimNode::resetCallback, this,
                     std::placeholders::_1, std::placeholders::_2));

        // Start simulation thread
        sim_running_ = true;
        sim_paused_ = false;
        sim_thread_ = std::thread(&MujocoSimNode::simulationLoop, this);

        RCLCPP_INFO(this->get_logger(), "MuJoCo simulation node started");
        RCLCPP_INFO(this->get_logger(), "  Model: %s", model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Actuators: %d", num_actuators_);
        RCLCPP_INFO(this->get_logger(), "  Joints: %zu", joint_names_.size());
        RCLCPP_INFO(this->get_logger(), "  Sim rate: %.1f Hz", sim_rate_);
        RCLCPP_INFO(this->get_logger(), "  Publish rate: %.1f Hz", publish_rate_);
        RCLCPP_INFO(this->get_logger(), "  Visualization: %s", headless_ ? "disabled" : "enabled");
    }

    ~MujocoSimNode() {
        sim_running_ = false;
        if (sim_thread_.joinable()) {
            sim_thread_.join();
        }
        
        // Cleanup visualization
        if (!headless_) {
            mjr_freeContext(&context_);
            mjv_freeScene(&scene_);
            if (window_) {
                glfwDestroyWindow(window_);
            }
            glfwTerminate();
        }
        
        if (data_) mj_deleteData(data_);
        if (model_) mj_deleteModel(model_);
    }

private:
    bool loadModel(const std::string& path) {
        char error[1000] = "";
        
        bool is_mjb = (path.size() >= 4 && 
                       path.compare(path.size() - 4, 4, ".mjb") == 0);
        if (is_mjb) {
            model_ = mj_loadModel(path.c_str(), nullptr);
        } else {
            model_ = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
        }

        if (!model_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s", error);
            return false;
        }

        data_ = mj_makeData(model_);
        if (!data_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create data");
            mj_deleteModel(model_);
            model_ = nullptr;
            return false;
        }

        // Find base body
        base_body_id_ = mj_name2id(model_, mjOBJ_BODY, base_link_name_.c_str());
        if (base_body_id_ < 0) {
            base_body_id_ = mj_name2id(model_, mjOBJ_BODY, "Trunk");
        }
        if (base_body_id_ < 0) {
            base_body_id_ = 1;
            RCLCPP_WARN(this->get_logger(), "Base body not found, using body ID 1");
        }

        // Find IMU sensors
        imu_sensor_id_ = mj_name2id(model_, mjOBJ_SENSOR, "imu_quat");
        gyro_sensor_id_ = mj_name2id(model_, mjOBJ_SENSOR, "imu_gyro");
        accel_sensor_id_ = mj_name2id(model_, mjOBJ_SENSOR, "imu_accel");

        if (imu_sensor_id_ < 0 || gyro_sensor_id_ < 0 || accel_sensor_id_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "Required IMU sensors not found. Please ensure 'imu_quat', 'imu_gyro', and 'imu_accel' are defined in the model.");
            return false;
        }

        mj_resetData(model_, data_);
        mj_forward(model_, data_);

        return true;
    }

    bool initVisualization() {
        // Initialize GLFW
        if (!glfwInit()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize GLFW");
            return false;
        }

        // Create window
        window_ = glfwCreateWindow(1200, 900, "MuJoCo Simulation", nullptr, nullptr);
        if (!window_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create GLFW window");
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);

        // Initialize MuJoCo visualization
        mjv_defaultCamera(&camera_);
        mjv_defaultOption(&opt_);
        mjv_defaultScene(&scene_);
        mjr_defaultContext(&context_);

        mjv_makeScene(model_, &scene_, 2000);
        mjr_makeContext(model_, &context_, mjFONTSCALE_150);

        // Set up camera
        camera_.lookat[0] = 0;
        camera_.lookat[1] = 0;
        camera_.lookat[2] = 0.8;
        camera_.distance = 4.0;
        camera_.azimuth = 90;
        camera_.elevation = -20;

        // Set up callbacks
        glfwSetWindowUserPointer(window_, this);
        
        glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int button, int action, int mods) {
            auto* node = static_cast<MujocoSimNode*>(glfwGetWindowUserPointer(w));
            node->mouseButtonCallback(button, action, mods);
        });
        
        glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double xpos, double ypos) {
            auto* node = static_cast<MujocoSimNode*>(glfwGetWindowUserPointer(w));
            node->mouseMoveCallback(xpos, ypos);
        });
        
        glfwSetScrollCallback(window_, [](GLFWwindow* w, double xoffset, double yoffset) {
            auto* node = static_cast<MujocoSimNode*>(glfwGetWindowUserPointer(w));
            node->scrollCallback(yoffset);
        });

        // Release context for other threads
        glfwMakeContextCurrent(nullptr);

        return true;
    }

    void mouseButtonCallback(int button, int action, int /*mods*/) {
        mouse_.button_left = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
        mouse_.button_middle = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
        mouse_.button_right = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

        glfwGetCursorPos(window_, &mouse_.lastx, &mouse_.lasty);
    }

    void mouseMoveCallback(double xpos, double ypos) {
        if (!mouse_.button_left && !mouse_.button_middle && !mouse_.button_right) {
            return;
        }

        double dx = xpos - mouse_.lastx;
        double dy = ypos - mouse_.lasty;
        mouse_.lastx = xpos;
        mouse_.lasty = ypos;

        int width, height;
        glfwGetWindowSize(window_, &width, &height);

        bool shift = (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

        mjtMouse action;
        if (mouse_.button_right) {
            action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        } else if (mouse_.button_left) {
            action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        } else {
            action = mjMOUSE_ZOOM;
        }

        mjv_moveCamera(model_, action, dx / height, dy / height, &scene_, &camera_);
    }

    void scrollCallback(double yoffset) {
        mjv_moveCamera(model_, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scene_, &camera_);
    }

    void render() {
        if (headless_ || !window_) return;

        glfwMakeContextCurrent(window_);

        // Get framebuffer size
        int width, height;
        glfwGetFramebufferSize(window_, &width, &height);

        // Update scene
        mjv_updateScene(model_, data_, &opt_, nullptr, &camera_, mjCAT_ALL, &scene_);

        // Render
        mjrRect viewport = {0, 0, width, height};
        mjr_render(viewport, &scene_, &context_);

        // Swap buffers
        glfwSwapBuffers(window_);
        glfwPollEvents();

        glfwMakeContextCurrent(nullptr);
    }

    void buildJointMapping() {
        joint_qpos_indices_.resize(joint_names_.size(), -1);
        joint_qvel_indices_.resize(joint_names_.size(), -1);
        actuator_indices_.resize(joint_names_.size(), -1);

        for (size_t i = 0; i < joint_names_.size(); ++i) {
            const std::string& name = joint_names_[i];
            
            int joint_id = mj_name2id(model_, mjOBJ_JOINT, name.c_str());
            if (joint_id >= 0) {
                joint_qpos_indices_[i] = model_->jnt_qposadr[joint_id];
                joint_qvel_indices_[i] = model_->jnt_dofadr[joint_id];
            } else {
                RCLCPP_WARN(this->get_logger(), "Joint '%s' not found", name.c_str());
            }

            int act_id = mj_name2id(model_, mjOBJ_ACTUATOR, name.c_str());
            if (act_id < 0) {
                act_id = mj_name2id(model_, mjOBJ_ACTUATOR, (name + "_actuator").c_str());
            }
            if (act_id >= 0) {
                actuator_indices_[i] = act_id;
            } else {
                RCLCPP_WARN(this->get_logger(), "Actuator for '%s' not found", name.c_str());
            }
        }
    }

    void buildDefaultJointMapping() {
        for (int i = 0; i < model_->njnt; ++i) {
            if (model_->jnt_type[i] == mjJNT_HINGE) {
                const char* name = mj_id2name(model_, mjOBJ_JOINT, i);
                if (name) {
                    joint_names_.push_back(name);
                    joint_qpos_indices_.push_back(model_->jnt_qposadr[i]);
                    joint_qvel_indices_.push_back(model_->jnt_dofadr[i]);
                    
                    int act_id = mj_name2id(model_, mjOBJ_ACTUATOR, name);
                    actuator_indices_.push_back(act_id);
                }
            }
        }
        RCLCPP_INFO(this->get_logger(), "Auto-detected %zu joints", joint_names_.size());
    }

    void simulationLoop() {
        using clock = std::chrono::steady_clock;
        
        double sim_dt = 1.0 / sim_rate_;
        double pub_dt = 1.0 / publish_rate_;
        
        auto sim_start = clock::now();
        double sim_time = 0.0;
        double last_pub_time = 0.0;

        while (sim_running_) {
            // Check if window should close
            if (!headless_ && window_ && glfwWindowShouldClose(window_)) {
                sim_running_ = false;
                rclcpp::shutdown();
                break;
            }

            if (sim_paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                sim_start = clock::now();
                sim_time = data_->time;
                last_pub_time = sim_time;
                
                // Still render when paused
                render();
                continue;
            }

            // Apply PD control
            applyControl();

            // Step simulation
            mj_step(model_, data_);

            // Render (like viewer.sync() in Python)
            render();

            // Publish at specified rate
            if (data_->time - last_pub_time >= pub_dt) {
                publishState();
                last_pub_time = data_->time;
            }

            // Real-time sync
            sim_time += sim_dt;
            auto target_time = sim_start + std::chrono::duration<double>(sim_time);
            auto now = clock::now();
            
            if (now < target_time) {
                std::this_thread::sleep_until(target_time);
            }
        }
    }

    void applyControl() {
        std::lock_guard<std::mutex> lock(cmd_mutex_);

        for (size_t i = 0; i < joint_names_.size(); ++i) {
            if (actuator_indices_[i] < 0) continue;

            double q_des = (i < target_positions_.size()) ? target_positions_[i] : 0.0;
            double dq_des = (i < target_velocities_.size()) ? target_velocities_[i] : 0.0;
            double tau_ff = (i < target_torques_.size()) ? target_torques_[i] : 0.0;
            double kp = (i < kp_gains_.size()) ? kp_gains_[i] : 0.0;
            double kd = (i < kd_gains_.size()) ? kd_gains_[i] : 0.0;

            double q_cur = 0.0, dq_cur = 0.0;
            if (joint_qpos_indices_[i] >= 0) {
                q_cur = data_->qpos[joint_qpos_indices_[i]];
            }
            if (joint_qvel_indices_[i] >= 0) {
                dq_cur = data_->qvel[joint_qvel_indices_[i]];
            }

            double tau = kp * (q_des - q_cur) + kd * (dq_des - dq_cur) + tau_ff;
            data_->ctrl[actuator_indices_[i]] = tau;
        }
    }

    void publishState() {
        auto now = this->now();

        // Publish joint states
        sensor_msgs::msg::JointState js_msg;
        js_msg.header.stamp = now;
        js_msg.header.frame_id = "";

        for (size_t i = 0; i < joint_names_.size(); ++i) {
            js_msg.name.push_back(joint_names_[i]);
            
            double pos = 0.0, vel = 0.0, eff = 0.0;
            if (joint_qpos_indices_[i] >= 0) {
                pos = data_->qpos[joint_qpos_indices_[i]];
            }
            if (joint_qvel_indices_[i] >= 0) {
                vel = data_->qvel[joint_qvel_indices_[i]];
            }
            if (actuator_indices_[i] >= 0) {
                eff = data_->actuator_force[actuator_indices_[i]];
            }
            
            js_msg.position.push_back(pos);
            js_msg.velocity.push_back(vel);
            js_msg.effort.push_back(eff);
        }

        joint_state_pub_->publish(js_msg);

        // Publish IMU
        sensor_msgs::msg::Imu imu_msg;
        imu_msg.header.stamp = now;
        imu_msg.header.frame_id = "base_link";

        // Orientation from sensor
        if (imu_sensor_id_ >= 0) {
            int adr = model_->sensor_adr[imu_sensor_id_];
            imu_msg.orientation.w = data_->sensordata[adr];
            imu_msg.orientation.x = data_->sensordata[adr + 1];
            imu_msg.orientation.y = data_->sensordata[adr + 2];
            imu_msg.orientation.z = data_->sensordata[adr + 3];
        }

        // Angular velocity from sensor
        if (gyro_sensor_id_ >= 0) {
            int adr = model_->sensor_adr[gyro_sensor_id_];
            imu_msg.angular_velocity.x = data_->sensordata[adr];
            imu_msg.angular_velocity.y = data_->sensordata[adr + 1];
            imu_msg.angular_velocity.z = data_->sensordata[adr + 2];
        }

        // Linear acceleration from sensor
        if (accel_sensor_id_ >= 0) {
            int adr = model_->sensor_adr[accel_sensor_id_];
            imu_msg.linear_acceleration.x = data_->sensordata[adr];
            imu_msg.linear_acceleration.y = data_->sensordata[adr + 1];
            imu_msg.linear_acceleration.z = data_->sensordata[adr + 2];
        }

        imu_pub_->publish(imu_msg);

        // Publish odometry
        if (base_body_id_ >= 0) {
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = now;
            odom_msg.header.frame_id = "odom";
            odom_msg.child_frame_id = "base_link";

            int pos_adr = base_body_id_ * 3;
            int quat_adr = base_body_id_ * 4;

            odom_msg.pose.pose.position.x = data_->xpos[pos_adr];
            odom_msg.pose.pose.position.y = data_->xpos[pos_adr + 1];
            odom_msg.pose.pose.position.z = data_->xpos[pos_adr + 2];

            odom_msg.pose.pose.orientation.w = data_->xquat[quat_adr];
            odom_msg.pose.pose.orientation.x = data_->xquat[quat_adr + 1];
            odom_msg.pose.pose.orientation.y = data_->xquat[quat_adr + 2];
            odom_msg.pose.pose.orientation.z = data_->xquat[quat_adr + 3];

            odom_msg.twist.twist.linear.x = data_->qvel[0];
            odom_msg.twist.twist.linear.y = data_->qvel[1];
            odom_msg.twist.twist.linear.z = data_->qvel[2];
            odom_msg.twist.twist.angular.x = data_->qvel[3];
            odom_msg.twist.twist.angular.y = data_->qvel[4];
            odom_msg.twist.twist.angular.z = data_->qvel[5];

            odom_pub_->publish(odom_msg);
        }
    }

    // Combined joint command callback
    void jointCmdCallback(const mujoco_ros_msgs::msg::JointControlCmd::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        
        if (!msg->position.empty()) {
            target_positions_ = msg->position;
        }
        if (!msg->velocity.empty()) {
            target_velocities_ = msg->velocity;
        }
        if (!msg->torque.empty()) {
            target_torques_ = msg->torque;
        }
        if (!msg->kp.empty()) {
            kp_gains_ = msg->kp;
        }
        if (!msg->kd.empty()) {
            kd_gains_ = msg->kd;
        }
    }

    // Service callbacks
    void pauseCallback(const std_srvs::srv::SetBool::Request::SharedPtr req,
                       std_srvs::srv::SetBool::Response::SharedPtr res) {
        sim_paused_ = req->data;
        res->success = true;
        res->message = sim_paused_ ? "Simulation paused" : "Simulation resumed";
        RCLCPP_INFO(this->get_logger(), "%s", res->message.c_str());
    }

    void resetCallback(const std_srvs::srv::Empty::Request::SharedPtr,
                       std_srvs::srv::Empty::Response::SharedPtr) {
        bool was_paused = sim_paused_;
        sim_paused_ = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        mj_resetData(model_, data_);
        mj_forward(model_, data_);

        {
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            std::fill(target_positions_.begin(), target_positions_.end(), 0.0);
            std::fill(target_velocities_.begin(), target_velocities_.end(), 0.0);
            std::fill(target_torques_.begin(), target_torques_.end(), 0.0);
        }

        sim_paused_ = was_paused;
        RCLCPP_INFO(this->get_logger(), "Simulation reset");
    }

    // MuJoCo model and data
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    int base_body_id_ = -1;
    int imu_sensor_id_ = -1;
    int gyro_sensor_id_ = -1;
    int accel_sensor_id_ = -1;

    // Visualization
    GLFWwindow* window_ = nullptr;
    mjvCamera camera_;
    mjvOption opt_;
    mjvScene scene_;
    mjrContext context_;
    MouseState mouse_;
    bool headless_ = false;

    // Parameters
    std::string model_path_;
    std::string robot_name_;
    std::string base_link_name_;
    double sim_rate_;
    double publish_rate_;
    std::vector<std::string> joint_names_;
    int num_actuators_ = 0;

    // Joint mapping
    std::vector<int> joint_qpos_indices_;
    std::vector<int> joint_qvel_indices_;
    std::vector<int> actuator_indices_;

    // Command storage
    std::mutex cmd_mutex_;
    std::vector<double> target_positions_;
    std::vector<double> target_velocities_;
    std::vector<double> target_torques_;
    std::vector<double> kp_gains_;
    std::vector<double> kd_gains_;

    // Simulation thread
    std::thread sim_thread_;
    std::atomic<bool> sim_running_{false};
    std::atomic<bool> sim_paused_{false};

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

    // Subscribers
    rclcpp::Subscription<mujoco_ros_msgs::msg::JointControlCmd>::SharedPtr joint_cmd_sub_;

    // Services
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr pause_srv_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<MujocoSimNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("mujoco_sim"), "Error: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
