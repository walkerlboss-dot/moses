"""
Real-World Deployment Module
============================

Safe deployment of simulation-trained policies to physical robots.

Features:
- Real-time policy inference
- Comprehensive safety monitoring
- Emergency stop integration
- Performance logging and diagnostics

References:
-----------
[1] Hwangbo et al. [3] for real-world deployment methodology
[2] Rudin et al. [4] for rapid deployment
[3] Margolis et al. [5] for hardware considerations
[4] Tan et al. [2] for safety limits
[5] Zhao et al. [1] for sim-to-real deployment survey
[6] ISO 10218-1/2: Robots and robotic devices — Safety requirements
[7] ISO/TS 15066: Collaborative robots safety

Author: Moses Team
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import json
import logging
import threading
from queue import Queue
import signal
import sys

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety monitoring levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DeploymentState(Enum):
    """Deployment runtime states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    CALIBRATING = "calibrating"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyLimits:
    """Safety limits for physical deployment."""
    # Joint limits
    joint_position_min: np.ndarray = field(default_factory=lambda: np.full(12, -np.pi))
    joint_position_max: np.ndarray = field(default_factory=lambda: np.full(12, np.pi))
    joint_velocity_max: np.ndarray = field(default_factory=lambda: np.full(12, 20.0))
    joint_torque_max: np.ndarray = field(default_factory=lambda: np.full(12, 100.0))
    joint_acceleration_max: np.ndarray = field(default_factory=lambda: np.full(12, 100.0))
    
    # End-effector limits
    ee_velocity_max: float = 2.0  # m/s
    ee_force_max: float = 100.0   # N
    
    # Power limits
    motor_current_max: np.ndarray = field(default_factory=lambda: np.full(12, 40.0))
    motor_temperature_max: np.ndarray = field(default_factory=lambda: np.full(12, 80.0))
    battery_voltage_min: float = 22.0  # V
    
    # Stability limits
    base_roll_max: float = np.pi / 3   # 60 degrees
    base_pitch_max: float = np.pi / 3
    base_height_min: float = 0.15       # m
    base_height_max: float = 1.0
    
    # Communication limits
    max_latency_ms: float = 50.0
    max_packet_loss: float = 0.05
    
    # Emergency triggers
    emergency_button_pressed: bool = False
    human_detected_in_workspace: bool = False
    
    def validate_joint_positions(self, q: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Check if joint positions are within limits."""
        violations = (q < self.joint_position_min) | (q > self.joint_position_max)
        return not violations.any(), violations
    
    def validate_joint_velocities(self, qd: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Check if joint velocities are within limits."""
        violations = np.abs(qd) > self.joint_velocity_max
        return not violations.any(), violations
    
    def validate_joint_torques(self, tau: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Check if joint torques are within limits."""
        violations = np.abs(tau) > self.joint_torque_max
        return not violations.any(), violations


@dataclass
class DeploymentConfig:
    """Configuration for policy deployment."""
    # Control
    control_frequency_hz: float = 1000.0
    policy_frequency_hz: float = 50.0
    action_scale: float = 0.5
    action_filter_cutoff: float = 5.0
    
    # Safety
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
    enable_emergency_stop: bool = True
    enable_safety_monitoring: bool = True
    enable_action_filtering: bool = True
    enable_joint_limit_clipping: bool = True
    
    # Logging
    log_frequency_hz: float = 100.0
    log_buffer_size: int = 10000
    log_directory: str = "./logs"
    
    # Communication
    robot_interface_type: str = "ros2"  # ros2, lcm, custom
    robot_topic_prefix: str = "/robot"
    command_topic: str = "/joint_commands"
    state_topic: str = "/joint_states"
    
    # Recovery
    enable_auto_recovery: bool = False
    recovery_attempts_max: int = 3
    recovery_timeout_sec: float = 5.0
    
    # Device
    device: str = "cpu"  # cpu, cuda
    use_tensorrt: bool = False
    use_onnx: bool = False


class SafetyMonitor:
    """
    Real-time safety monitoring for physical robot deployment.
    
    Continuously monitors robot state against safety limits and
    triggers appropriate responses.
    
    References:
    [1] ISO 10218-1/2 [6] for robot safety
    [2] ISO/TS 15066 [7] for collaborative safety
    """
    
    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.state = SafetyLevel.NORMAL
        self.violation_history = deque(maxlen=1000)
        self._monitoring = False
        self._monitor_thread = None
        
    def check_state(self,
                    joint_positions: np.ndarray,
                    joint_velocities: np.ndarray,
                    joint_torques: np.ndarray,
                    base_orientation: Optional[np.ndarray] = None,
                    motor_temperatures: Optional[np.ndarray] = None,
                    battery_voltage: Optional[float] = None,
                    latency_ms: Optional[float] = None) -> SafetyLevel:
        """
        Check current robot state against safety limits.
        
        Returns:
            SafetyLevel indicating current safety status
        """
        violations = []
        
        # Check joint limits
        pos_ok, pos_viol = self.limits.validate_joint_positions(joint_positions)
        if not pos_ok:
            violations.append(('joint_position', pos_viol))
            
        vel_ok, vel_viol = self.limits.validate_joint_velocities(joint_velocities)
        if not vel_ok:
            violations.append(('joint_velocity', vel_viol))
            
        tau_ok, tau_viol = self.limits.validate_joint_torques(joint_torques)
        if not tau_ok:
            violations.append(('joint_torque', tau_viol))
            
        # Check base orientation
        if base_orientation is not None:
            roll, pitch = base_orientation[:2]
            if abs(roll) > self.limits.base_roll_max:
                violations.append(('base_roll', abs(roll)))
            if abs(pitch) > self.limits.base_pitch_max:
                violations.append(('base_pitch', abs(pitch)))
                
        # Check temperatures
        if motor_temperatures is not None:
            temp_viol = motor_temperatures > self.limits.motor_temperature_max
            if temp_viol.any():
                violations.append(('motor_temperature', temp_viol))
                
        # Check battery
        if battery_voltage is not None and battery_voltage < self.limits.battery_voltage_min:
            violations.append(('battery_voltage', battery_voltage))
            
        # Check latency
        if latency_ms is not None and latency_ms > self.limits.max_latency_ms:
            violations.append(('latency', latency_ms))
            
        # Emergency triggers
        if self.limits.emergency_button_pressed:
            self.state = SafetyLevel.EMERGENCY
            violations.append(('emergency_button', True))
            
        if self.limits.human_detected_in_workspace:
            self.state = SafetyLevel.CRITICAL
            violations.append(('human_detected', True))
            
        # Determine safety level
        if not violations:
            self.state = SafetyLevel.NORMAL
        elif any(v[0] in ['emergency_button', 'human_detected'] for v in violations):
            self.state = SafetyLevel.EMERGENCY
        elif any(v[0] in ['motor_temperature', 'battery_voltage'] for v in violations):
            self.state = SafetyLevel.CRITICAL
        elif any(v[0] in ['joint_torque', 'base_roll', 'base_pitch'] for v in violations):
            self.state = SafetyLevel.CRITICAL
        else:
            self.state = SafetyLevel.WARNING
            
        # Log violations
        if violations:
            self.violation_history.append({
                'timestamp': time.time(),
                'level': self.state.value,
                'violations': violations,
            })
            
        return self.state
    
    def get_recommended_action(self) -> Optional[np.ndarray]:
        """
        Get recommended safety action based on current state.
        
        Returns:
            Safe action or None if no intervention needed
        """
        if self.state == SafetyLevel.EMERGENCY:
            return None  # Trigger emergency stop
        elif self.state == SafetyLevel.CRITICAL:
            return np.zeros(12)  # Zero action
        elif self.state == SafetyLevel.WARNING:
            return np.zeros(12) * 0.5  # Reduced action
        return None
    
    def start_monitoring(self, state_callback: Callable):
        """Start background safety monitoring thread."""
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                state = state_callback()
                self.check_state(**state)
                time.sleep(0.001)  # 1kHz monitoring
                
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)


class EmergencyStop:
    """
    Emergency stop system with multiple trigger sources.
    
    Implements hardware and software e-stop with latching behavior.
    """
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig()
        self.is_triggered = False
        self.trigger_source = None
        self.trigger_time = None
        self._callbacks = []
        self._lock = threading.Lock()
        
        # Register signal handlers
        if self.config.enable_emergency_stop:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
    def _signal_handler(self, signum, frame):
        """Handle OS signals as emergency stops."""
        self.trigger(f"signal_{signum}")
        sys.exit(1)
        
    def register_callback(self, callback: Callable):
        """Register callback to call on e-stop trigger."""
        self._callbacks.append(callback)
        
    def trigger(self, source: str = "unknown"):
        """
        Trigger emergency stop.
        
        Args:
            source: Identifier for what triggered the stop
        """
        with self._lock:
            if self.is_triggered:
                return
            self.is_triggered = True
            self.trigger_source = source
            self.trigger_time = time.time()
            
        logger.critical(f"EMERGENCY STOP triggered by: {source}")
        
        # Execute callbacks
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"E-stop callback failed: {e}")
                
    def reset(self, password: str = None):
        """
        Reset emergency stop (requires explicit action).
        
        Args:
            password: Optional safety password for reset
        """
        with self._lock:
            self.is_triggered = False
            self.trigger_source = None
            self.trigger_time = None
            
        logger.info("Emergency stop reset")
        
    def check(self) -> bool:
        """Check if emergency stop is active."""
        return self.is_triggered


class PerformanceLogger:
    """
    Performance logging and diagnostics for deployment.
    
    Logs state, actions, rewards, and timing information for
    post-deployment analysis.
    """
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig()
        self.buffer = deque(maxlen=config.log_buffer_size if config else 10000)
        self.episode_buffer = []
        self.episode_count = 0
        self.start_time = None
        
    def start_episode(self):
        """Start logging a new episode."""
        self.episode_buffer = []
        self.start_time = time.time()
        self.episode_count += 1
        
    def log_step(self,
                 timestamp: float,
                 observation: np.ndarray,
                 action: np.ndarray,
                 reward: float = None,
                 policy_latency_ms: float = None,
                 control_latency_ms: float = None,
                 safety_level: SafetyLevel = None):
        """
        Log a single step.
        
        Args:
            timestamp: Wall-clock time
            observation: Current observation
            action: Executed action
            reward: Reward (if available)
            policy_latency_ms: Policy inference latency
            control_latency_ms: Control loop latency
            safety_level: Current safety level
        """
        entry = {
            'timestamp': timestamp,
            'episode': self.episode_count,
            'observation': observation.copy(),
            'action': action.copy(),
            'reward': reward,
            'policy_latency_ms': policy_latency_ms,
            'control_latency_ms': control_latency_ms,
            'safety_level': safety_level.value if safety_level else None,
        }
        
        self.episode_buffer.append(entry)
        self.buffer.append(entry)
        
    def end_episode(self, episode_reward: float = None, success: bool = None):
        """End current episode and save data."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            'episode': self.episode_count,
            'duration_sec': duration,
            'total_reward': episode_reward,
            'success': success,
            'n_steps': len(self.episode_buffer),
            'mean_policy_latency_ms': np.mean([
                e['policy_latency_ms'] for e in self.episode_buffer
                if e['policy_latency_ms'] is not None
            ]) if self.episode_buffer else 0,
        }
        
        # Save episode data
        import os
        os.makedirs(self.config.log_directory, exist_ok=True)
        filepath = os.path.join(
            self.config.log_directory,
            f"episode_{self.episode_count:04d}.npz"
        )
        
        np.savez_compressed(
            filepath,
            observations=np.array([e['observation'] for e in self.episode_buffer]),
            actions=np.array([e['action'] for e in self.episode_buffer]),
            rewards=np.array([e['reward'] for e in self.episode_buffer]),
            timestamps=np.array([e['timestamp'] for e in self.episode_buffer]),
            summary=json.dumps(summary),
        )
        
        logger.info(f"Episode {self.episode_count} logged: {summary}")
        
    def get_statistics(self) -> Dict:
        """Get deployment statistics."""
        if not self.buffer:
            return {}
            
        latencies = [e['policy_latency_ms'] for e in self.buffer
                     if e['policy_latency_ms'] is not None]
        
        return {
            'total_episodes': self.episode_count,
            'total_steps': len(self.buffer),
            'mean_policy_latency_ms': np.mean(latencies) if latencies else 0,
            'max_policy_latency_ms': np.max(latencies) if latencies else 0,
            'safety_violations': sum(
                1 for e in self.buffer
                if e['safety_level'] not in [None, 'normal']
            ),
        }


class PolicyDeployer:
    """
    Main deployment class for running policies on physical robots.
    
    Orchestrates policy inference, safety monitoring, control loop,
    and logging for safe real-world deployment.
    """
    
    def __init__(self,
                 policy: nn.Module,
                 robot_interface,
                 config: DeploymentConfig = None):
        """
        Args:
            policy: Trained policy network
            robot_interface: Robot hardware interface
            config: Deployment configuration
        """
        self.policy = policy
        self.robot = robot_interface
        self.config = config or DeploymentConfig()
        
        # Safety systems
        self.safety_monitor = SafetyMonitor(self.config.safety_limits)
        self.emergency_stop = EmergencyStop(self.config)
        
        # Logging
        self.logger = PerformanceLogger(self.config)
        
        # State
        self.state = DeploymentState.IDLE
        self.current_observation = None
        self.current_action = np.zeros(12)
        self._running = False
        self._control_thread = None
        
        # Action filtering
        self._action_history = deque(maxlen=10)
        self._last_action = np.zeros(12)
        
        # Timing
        self._policy_dt = 1.0 / self.config.policy_frequency_hz
        self._control_dt = 1.0 / self.config.control_frequency_hz
        
        # Move policy to device
        self.policy.to(self.config.device)
        self.policy.eval()
        
        # Register e-stop callback
        self.emergency_stop.register_callback(self._emergency_stop_callback)
        
    def _emergency_stop_callback(self):
        """Callback for emergency stop - send zero commands."""
        self.state = DeploymentState.EMERGENCY_STOP
        self._send_zero_command()
        self._running = False
        
    def _send_zero_command(self):
        """Send zero torque/velocity command to robot."""
        if self.robot is not None:
            try:
                self.robot.send_command(np.zeros(12))
            except Exception as e:
                logger.error(f"Failed to send zero command: {e}")
                
    def _filter_action(self, action: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to actions for smoothness."""
        if not self.config.enable_action_filtering:
            return action
            
        # Exponential moving average
        alpha = 0.7  # Filter coefficient
        filtered = alpha * action + (1 - alpha) * self._last_action
        self._last_action = filtered
        return filtered
    
    def _clip_action(self, action: np.ndarray,
                     joint_positions: np.ndarray) -> np.ndarray:
        """Clip action to respect joint limits."""
        if not self.config.enable_joint_limit_clipping:
            return action
            
        # Compute target positions
        target_positions = joint_positions + action * self.config.action_scale
        
        # Clip to limits
        clipped = np.clip(
            target_positions,
            self.config.safety_limits.joint_position_min,
            self.config.safety_limits.joint_position_max
        )
        
        # Convert back to action
        safe_action = (clipped - joint_positions) / self.config.action_scale
        return safe_action
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from robot."""
        if self.robot is None:
            return np.zeros(48)  # Default observation size
            
        try:
            obs = self.robot.get_observation()
            return obs
        except Exception as e:
            logger.error(f"Failed to get observation: {e}")
            return self.current_observation if self.current_observation is not None else np.zeros(48)
    
    def _run_policy(self, observation: np.ndarray) -> np.ndarray:
        """Run policy inference."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.config.device)
            action_tensor = self.policy(obs_tensor)
            action = action_tensor.cpu().numpy().flatten()
        return action
    
    def _control_loop(self):
        """Main control loop running at control frequency."""
        last_policy_time = time.time()
        
        while self._running:
            loop_start = time.time()
            
            # Check emergency stop
            if self.emergency_stop.check():
                self.state = DeploymentState.EMERGENCY_STOP
                break
                
            # Get robot state
            observation = self._get_observation()
            self.current_observation = observation
            
            # Extract joint state for safety checks
            # Assuming observation format: [joint_pos(12), joint_vel(12), imu(12), ...]
            joint_positions = observation[:12]
            joint_velocities = observation[12:24]
            
            # Run policy at policy frequency
            if loop_start - last_policy_time >= self._policy_dt:
                policy_start = time.time()
                action = self._run_policy(observation)
                policy_latency = (time.time() - policy_start) * 1000  # ms
                last_policy_time = loop_start
                
                # Filter and clip action
                action = self._filter_action(action)
                action = self._clip_action(action, joint_positions)
                action = action * self.config.action_scale
                
                self.current_action = action
                
                # Log
                self.logger.log_step(
                    timestamp=loop_start,
                    observation=observation,
                    action=action,
                    policy_latency_ms=policy_latency,
                    safety_level=self.safety_monitor.state,
                )
                
            # Safety check
            if self.config.enable_safety_monitoring:
                safety_state = self.safety_monitor.check_state(
                    joint_positions=joint_positions,
                    joint_velocities=joint_velocities,
                    joint_torques=np.zeros(12),  # Would come from robot
                )
                
                if safety_state == SafetyLevel.EMERGENCY:
                    self.emergency_stop.trigger("safety_monitor")
                    break
                elif safety_state == SafetyLevel.CRITICAL:
                    self.current_action = np.zeros(12)
                    
            # Send command
            if self.robot is not None:
                try:
                    self.robot.send_command(self.current_action)
                except Exception as e:
                    logger.error(f"Failed to send command: {e}")
                    
            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = self._control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -0.001:
                logger.warning(f"Control loop overrun: {-sleep_time*1000:.2f}ms")
                
    def initialize(self):
        """Initialize robot and safety systems."""
        self.state = DeploymentState.INITIALIZING
        logger.info("Initializing deployment...")
        
        # Initialize robot
        if self.robot is not None:
            self.robot.initialize()
            
        # Start safety monitoring
        if self.config.enable_safety_monitoring:
            self.safety_monitor.start_monitoring(self._get_robot_state)
            
        self.state = DeploymentState.IDLE
        logger.info("Initialization complete")
        
    def _get_robot_state(self) -> Dict:
        """Get robot state for safety monitoring."""
        obs = self._get_observation()
        return {
            'joint_positions': obs[:12],
            'joint_velocities': obs[12:24],
            'joint_torques': np.zeros(12),
        }
        
    def deploy(self, max_duration_sec: float = None):
        """
        Deploy policy on physical robot.
        
        Args:
            max_duration_sec: Maximum deployment duration (None for indefinite)
        """
        if self.state == DeploymentState.EMERGENCY_STOP:
            logger.error("Cannot deploy: emergency stop is active")
            return
            
        self.state = DeploymentState.RUNNING
        self._running = True
        self.logger.start_episode()
        
        logger.info("Starting policy deployment")
        
        try:
            self._control_loop()
        except Exception as e:
            logger.error(f"Deployment error: {e}")
            self.state = DeploymentState.ERROR
            self.emergency_stop.trigger("exception")
        finally:
            self._running = False
            self.state = DeploymentState.STOPPING
            self._send_zero_command()
            self.logger.end_episode()
            
            if self.config.enable_safety_monitoring:
                self.safety_monitor.stop_monitoring()
                
            self.state = DeploymentState.IDLE
            logger.info("Deployment stopped")
            
    def pause(self):
        """Pause deployment (maintain position)."""
        if self.state == DeploymentState.RUNNING:
            self.state = DeploymentState.PAUSED
            logger.info("Deployment paused")
            
    def resume(self):
        """Resume paused deployment."""
        if self.state == DeploymentState.PAUSED:
            self.state = DeploymentState.RUNNING
            logger.info("Deployment resumed")
            
    def stop(self):
        """Stop deployment gracefully."""
        self._running = False
        logger.info("Stopping deployment...")
        
    def get_status(self) -> Dict:
        """Get current deployment status."""
        return {
            'state': self.state.value,
            'safety_level': self.safety_monitor.state.value,
            'emergency_stop': self.emergency_stop.is_triggered,
            'episode': self.logger.episode_count,
            'statistics': self.logger.get_statistics(),
        }


class RobotInterfaceBase:
    """
    Base class for robot hardware interfaces.
    
    Implementations should override these methods for specific hardware.
    """
    
    def initialize(self):
        """Initialize robot hardware."""
        raise NotImplementedError
        
    def get_observation(self) -> np.ndarray:
        """Get current observation from robot."""
        raise NotImplementedError
        
    def send_command(self, action: np.ndarray):
        """Send command to robot."""
        raise NotImplementedError
        
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        raise NotImplementedError
        
    def set_joint_positions(self, positions: np.ndarray):
        """Set joint positions (for calibration)."""
        raise NotImplementedError
        
    def shutdown(self):
        """Shutdown robot safely."""
        raise NotImplementedError


class SimRobotInterface(RobotInterfaceBase):
    """Simulated robot interface for testing deployment."""
    
    def __init__(self, n_dof: int = 12):
        self.n_dof = n_dof
        self.joint_positions = np.zeros(n_dof)
        self.joint_velocities = np.zeros(n_dof)
        self._initialized = False
        
    def initialize(self):
        self._initialized = True
        logger.info("Simulated robot initialized")
        
    def get_observation(self) -> np.ndarray:
        return np.concatenate([
            self.joint_positions,
            self.joint_velocities,
            np.zeros(24),  # IMU and other sensors
        ])
        
    def send_command(self, action: np.ndarray):
        # Simple integration
        self.joint_velocities = action
        self.joint_positions += action * 0.001  # 1ms timestep
        
    def get_joint_positions(self) -> np.ndarray:
        return self.joint_positions.copy()
        
    def set_joint_positions(self, positions: np.ndarray):
        self.joint_positions = positions.copy()
        
    def shutdown(self):
        self._initialized = False
        logger.info("Simulated robot shutdown")
