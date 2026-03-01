import gym
from gym import spaces
import numpy as np
import pygame


class BipedalRobotEnv(gym.Env):
    """
    Custom Environment for 2D bipedal robot.

    This environment simulates a simplified bipedal robot in a 2D plane,
    where the robot is modeled as a single leg with a torso. The robot
    must learn to walk by applying torques to hip and knee joints and
    controlling swing force.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super(BipedalRobotEnv, self).__init__()

        # Physical parameters
        self.gravity = 9.81  # m/s^2
        self.dt = 0.01  # simulation time step (s)
        self.control_dt = 0.05  # control time step (s), may be multiple simulation steps
        self.control_steps_per_action = int(self.control_dt / self.dt)

        # Robot parameters
        self.torso_mass = 10.0  # kg
        self.thigh_mass = 2.5  # kg
        self.calf_mass = 2.0  # kg
        self.foot_mass = 1.0  # kg

        self.torso_length = 0.6  # m
        self.thigh_length = 0.4  # m
        self.calf_length = 0.4  # m
        self.foot_length = 0.2  # m

        # Initial conditions
        self.init_height = 1.0  # m, initial height of torso
        self.desired_height = 0.9  # m, target height for torso

        # Joint limits
        self.hip_angle_limit = np.pi / 3  # rad
        self.knee_angle_limit = np.pi / 2  # rad

        # Force limits
        self.max_hip_torque = 100.0  # Nm
        self.max_knee_torque = 100.0  # Nm
        self.max_foot_force = 300.0  # N

        # Termination conditions
        self.max_torso_tilt = np.deg2rad(15)  # 15 degrees
        self.min_height_factor = 0.3  # fraction of desired height
        self.max_steps_without_swing = 10

        # Reward function parameters
        self.reward_params = {
            'w1': 1.0,  # balance reward weight
            'w2': 0.2,  # energy reward weight
            'w3': 0.8,  # gait reward weight
            'w4': 0.7,  # safety reward weight
            'alpha': 2.0,  # torso tilt penalty factor
            'beta': 1.0,  # height error penalty factor
            'gamma': 0.003,  # energy consumption factor
            'delta': 0.1,  # foot force penalty factor
            'eta': 0.2,  # knee angle penalty factor
            'stride_period': 1.0  # desired stride period (s)
        }

        # State normalization parameters
        self.state_means = np.zeros(11)
        self.state_stds = np.ones(11)

        # Define action space - [hip_torque, knee_torque, swing_force]
        # Values are normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Define observation space - 11-dimensional state vector
        # [torso_x, torso_x_dot, torso_z, torso_z_dot,
        #  hip_angle, hip_angle_dot, knee_angle, knee_angle_dot,
        #  foot_force_x, foot_force_z, foot_phase]
        # Using larger bounds for safety
        high = np.array([
            np.inf,  # torso_x
            np.inf,  # torso_x_dot
            np.inf,  # torso_z
            np.inf,  # torso_z_dot
            np.pi,  # hip_angle
            np.inf,  # hip_angle_dot
            np.pi,  # knee_angle
            np.inf,  # knee_angle_dot
            np.inf,  # foot_force_x
            np.inf,  # foot_force_z
            1.0  # foot_phase (0 or 1)
        ])

        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        # Visualization parameters
        self.screen = None
        self.clock = None
        self.render_mode = render_mode
        self.scale = 100  # pixels per meter
        self.screen_width = 1200
        self.screen_height = 400

        # Initialize state variables
        self.state = None
        self.steps_since_last_swing = 0
        self.current_time = 0.0
        self.last_hip_torque = 0.0
        self.last_knee_torque = 0.0
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Returns:
            observation (np.array): initial state
        """
        super().reset(seed=seed)

        # Reset the state with some randomness for robustness
        noise_scale = 0.1

        # Initial position near the origin with small random perturbation
        init_torso_x = self.np_random.uniform(-0.1, 0.1)
        init_torso_z = self.init_height + self.np_random.uniform(-0.05, 0.05)

        # Initial velocities close to zero
        init_torso_x_dot = self.np_random.uniform(-0.1, 0.1)
        init_torso_z_dot = self.np_random.uniform(-0.1, 0.1)

        # Initial joint angles for a natural standing pose
        init_hip_angle = np.deg2rad(self.np_random.uniform(-5, 5))
        init_knee_angle = np.deg2rad(self.np_random.uniform(5, 15))

        # Initial joint velocities close to zero
        init_hip_angle_dot = self.np_random.uniform(-0.1, 0.1)
        init_knee_angle_dot = self.np_random.uniform(-0.1, 0.1)

        # Initial foot forces - start with the foot on the ground
        init_foot_force_x = 0.0
        init_foot_force_z = (self.torso_mass + self.thigh_mass + self.calf_mass + self.foot_mass) * self.gravity
        init_foot_phase = 1.0  # 1 = foot on ground, 0 = foot in air

        self.state = np.array([
            init_torso_x,
            init_torso_x_dot,
            init_torso_z,
            init_torso_z_dot,
            init_hip_angle,
            init_hip_angle_dot,
            init_knee_angle,
            init_knee_angle_dot,
            init_foot_force_x,
            init_foot_force_z,
            init_foot_phase
        ], dtype=np.float32)

        # Reset tracking variables
        self.steps_since_last_swing = 0
        self.current_time = 0.0
        self.last_hip_torque = 0.0
        self.last_knee_torque = 0.0

        # Return initial observation
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Returns the current state observation, possibly normalized.
        """
        # For simplicity, we're not doing state normalization here
        # In a real implementation, you would normalize the state
        return self.state.astype(np.float32)

    def _normalize_action(self, action):
        """
        Scale the normalized action values to actual torque and force values.
        """
        # Clip action to ensure it's within bounds
        action = np.clip(action, -1.0, 1.0)

        # Scale to actual values
        hip_torque = action[0] * self.max_hip_torque
        knee_torque = action[1] * self.max_knee_torque
        swing_force = max(0, action[2]) * 1.0  # Normalize to [0, 1]

        # Add some smoothing to avoid sudden torque changes
        hip_torque = 0.8 * hip_torque + 0.2 * self.last_hip_torque
        knee_torque = 0.8 * knee_torque + 0.2 * self.last_knee_torque

        self.last_hip_torque = hip_torque
        self.last_knee_torque = knee_torque

        return hip_torque, knee_torque, swing_force

    def _calculate_reward(self, state, action, next_state):
        """
        Calculate the reward based on the reward function components.

        Args:
            state (np.array): Current state
            action (np.array): Action taken
            next_state (np.array): Resulting state

        Returns:
            float: Calculated reward
        """
        # Unpack state variables
        torso_x = state[0]
        torso_z = state[2]
        hip_angle = state[4]
        knee_angle = state[6]
        foot_force_z = state[9]
        foot_phase = state[10]

        # Unpack actions
        hip_torque, knee_torque, swing_force = self._normalize_action(action)

        # Balance reward
        balance_reward = (
                np.exp(-self.reward_params['alpha'] * abs(torso_x)) +
                np.exp(-self.reward_params['beta'] * abs(torso_z - self.desired_height))
        )

        # Energy reward (negative to penalize high energy consumption)
        energy_reward = -self.reward_params['gamma'] * (hip_torque ** 2 + knee_torque ** 2)

        # Gait reward - encourages periodic motion when foot is on ground
        gait_reward = 0
        if foot_phase > 0.5:  # Foot is on ground
            gait_reward = np.cos(2 * np.pi * self.current_time / self.reward_params['stride_period'])

        # Safety reward (negative to penalize unsafe states)
        safety_reward = (
                -self.reward_params['delta'] * max(0, foot_force_z - self.max_foot_force) -
                self.reward_params['eta'] * max(0, abs(knee_angle) - self.knee_angle_limit)
        )

        # Combine rewards with weights
        total_reward = (
                self.reward_params['w1'] * balance_reward +
                self.reward_params['w2'] * energy_reward +
                self.reward_params['w3'] * gait_reward +
                self.reward_params['w4'] * safety_reward
        )

        return total_reward

    def _check_termination(self, state):
        """
        Check if episode should terminate based on termination conditions.

        Returns:
            bool: True if episode should terminate
        """
        torso_x = state[0]  # Torso tilt angle
        torso_z = state[2]  # Torso height

        # Condition 1: Torso tilt exceeds limit
        if abs(torso_x) > self.max_torso_tilt:
            return True

        # Condition 2: Torso height is too low
        if torso_z < self.min_height_factor * self.desired_height:
            return True

        # Condition 3: Too many steps without swing
        if self.steps_since_last_swing > self.max_steps_without_swing:
            return True

        return False

    def _dynamics_step(self, hip_torque, knee_torque, swing_force):
        """
        Simulate the robot dynamics for one time step.

        Args:
            hip_torque (float): Torque applied to hip joint
            knee_torque (float): Torque applied to knee joint
            swing_force (float): Force applied for leg swing [0, 1]

        Returns:
            np.array: Next state after applying dynamics
        """
        # Unpack state
        torso_x, torso_x_dot, torso_z, torso_z_dot = self.state[0:4]
        hip_angle, hip_angle_dot, knee_angle, knee_angle_dot = self.state[4:8]
        foot_force_x, foot_force_z, foot_phase = self.state[8:11]

        # Simplified dynamics - this would be replaced with proper physics in a real implementation
        # Here we're just approximating the effect of forces and torques

        # Check if we're in stance phase (foot on ground)
        is_stance_phase = foot_phase > 0.5

        # Calculate leg length based on joint angles
        leg_extension = (
                self.thigh_length * np.cos(hip_angle) +
                self.calf_length * np.cos(hip_angle + knee_angle)
        )

        # Calculate foot position
        foot_x = torso_x - self.thigh_length * np.sin(hip_angle) - self.calf_length * np.sin(hip_angle + knee_angle)
        foot_z = torso_z - leg_extension

        # Apply forces to update velocities
        total_mass = self.torso_mass + self.thigh_mass + self.calf_mass + self.foot_mass

        # Apply gravity
        accel_z = -self.gravity

        # Apply ground reaction force if in stance phase
        if is_stance_phase and foot_z <= 0.01:  # Small threshold for ground contact
            # Ground reaction - simplified normal force to counter gravity
            ground_reaction_z = total_mass * self.gravity
            foot_force_z = ground_reaction_z

            # Apply hip and knee torques to produce horizontal force
            horizontal_force = hip_torque / (self.thigh_length + self.calf_length)
            foot_force_x = horizontal_force

            # Update accelerations
            accel_x = horizontal_force / total_mass
            accel_z = 0  # Assuming perfect ground reaction

            # Update joint velocities based on torques
            hip_accel = hip_torque / (self.thigh_mass * self.thigh_length ** 2) - knee_torque / (
                        self.thigh_mass * self.thigh_length * self.calf_length)
            knee_accel = knee_torque / (self.calf_mass * self.calf_length ** 2)

            hip_angle_dot += hip_accel * self.dt
            knee_angle_dot += knee_accel * self.dt

            # Check for swing action
            if swing_force > 0.5:  # Threshold for initiating swing
                # Transition to swing phase
                foot_phase = 0.0
                self.steps_since_last_swing = 0
            else:
                self.steps_since_last_swing += 1
        else:
            # Swing phase dynamics
            foot_force_x = 0.0
            foot_force_z = 0.0

            # Calculate joint accelerations from torques
            hip_accel = hip_torque / (self.thigh_mass * self.thigh_length ** 2)
            knee_accel = knee_torque / (self.calf_mass * self.calf_length ** 2)

            hip_angle_dot += hip_accel * self.dt
            knee_angle_dot += knee_accel * self.dt

            # Apply torso accelerations
            accel_x = 0.0  # No horizontal acceleration in swing phase

            # Check for ground contact
            if foot_z <= 0.01 and torso_z_dot < 0:
                # Transition to stance phase
                foot_phase = 1.0
                # Apply a small vertical impulse on contact
                accel_z = -torso_z_dot / self.dt

        # Update velocities
        torso_x_dot += accel_x * self.dt
        torso_z_dot += accel_z * self.dt

        # Apply damping to joint velocities
        hip_angle_dot *= 0.95  # Damping factor
        knee_angle_dot *= 0.95  # Damping factor

        # Update positions
        torso_x += torso_x_dot * self.dt
        torso_z += torso_z_dot * self.dt
        hip_angle += hip_angle_dot * self.dt
        knee_angle += knee_angle_dot * self.dt

        # Apply joint limits
        hip_angle = np.clip(hip_angle, -self.hip_angle_limit, self.hip_angle_limit)
        knee_angle = np.clip(knee_angle, -self.knee_angle_limit, self.knee_angle_limit)

        # Ensure torso doesn't go below ground
        torso_z = max(torso_z, leg_extension)

        # Update current time
        self.current_time += self.dt

        # Create new state
        next_state = np.array([
            torso_x, torso_x_dot, torso_z, torso_z_dot,
            hip_angle, hip_angle_dot, knee_angle, knee_angle_dot,
            foot_force_x, foot_force_z, foot_phase
        ])

        return next_state

    def step(self, action):
        """
        Take a step in the environment by applying an action.

        Args:
            action (np.array): Action to take [hip_torque, knee_torque, swing_force]

        Returns:
            observation (np.array): Next state
            reward (float): Reward for the action
            terminated (bool): Whether the episode is done
            truncated (bool): Whether the episode is truncated
            info (dict): Additional information
        """
        if self.state is None:
            return self.reset()[0], 0.0, False, False, {}

        # Save current state
        old_state = self.state.copy()

        # Normalize action from [-1, 1] to actual torques and forces
        hip_torque, knee_torque, swing_force = self._normalize_action(action)

        # Simulate multiple physics steps per control step
        for _ in range(self.control_steps_per_action):
            self.state = self._dynamics_step(hip_torque, knee_torque, swing_force)

            # Check for termination
            if self._check_termination(self.state):
                break

        # Calculate reward
        reward = self._calculate_reward(old_state, action, self.state)

        # Check termination
        terminated = self._check_termination(self.state)
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode is None:
            return

        # Initialize screen if not done already
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        # Fill background
        self.screen.fill((255, 255, 255))

        # Draw ground
        ground_height = self.screen_height // 2 + 50
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (0, ground_height),
            (self.screen_width, ground_height),
            2
        )

        # Extract state
        torso_x, _, torso_z, _, hip_angle, _, knee_angle, _, _, _, foot_phase = self.state

        # Calculate positions for rendering
        screen_center_x = self.screen_width // 2

        # Torso position
        torso_screen_x = screen_center_x + int(torso_x * self.scale)
        torso_screen_y = ground_height - int(torso_z * self.scale)

        # Hip position (bottom of torso)
        hip_screen_x = torso_screen_x
        hip_screen_y = torso_screen_y

        # Knee position
        knee_screen_x = hip_screen_x - int(self.thigh_length * np.sin(hip_angle) * self.scale)
        knee_screen_y = hip_screen_y + int(self.thigh_length * np.cos(hip_angle) * self.scale)

        # Ankle position
        ankle_screen_x = knee_screen_x - int(self.calf_length * np.sin(hip_angle + knee_angle) * self.scale)
        ankle_screen_y = knee_screen_y + int(self.calf_length * np.cos(hip_angle + knee_angle) * self.scale)

        # Draw torso
        pygame.draw.circle(self.screen, (0, 0, 255), (torso_screen_x, torso_screen_y), 10)

        # Draw thigh
        pygame.draw.line(
            self.screen,
            (255, 0, 0),
            (hip_screen_x, hip_screen_y),
            (knee_screen_x, knee_screen_y),
            5
        )

        # Draw calf
        pygame.draw.line(
            self.screen,
            (0, 255, 0),
            (knee_screen_x, knee_screen_y),
            (ankle_screen_x, ankle_screen_y),
            5
        )

        # Draw foot, with color based on phase
        foot_color = (0, 0, 0) if foot_phase > 0.5 else (100, 100, 100)
        pygame.draw.circle(self.screen, foot_color, (ankle_screen_x, ankle_screen_y), 5)

        # Draw state information
        font = pygame.font.Font(None, 24)
        state_text = f"Torso Height: {torso_z:.2f}m, Hip Angle: {np.rad2deg(hip_angle):.1f}°, Knee Angle: {np.rad2deg(knee_angle):.1f}°"
        text_surface = font.render(state_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(1.0 / self.dt)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        """
        Clean up resources.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None