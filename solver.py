import pinocchio
import numpy as np
from scipy.optimize import minimize, Bounds
from typing import List, Optional, Tuple, Union

class RobotKinematics:
    def __init__(self, urdf_path: str, ee_targets: dict, config: Optional[dict] = None):
        """
        Initializes the robot model for kinematic computations.
        Arguments:
            urdf_path: Path to the URDF file for the robot model.
            ee_targets: Dictionary specifying end-effector link names and orientation constraints. 
                        Example: {"gripper": True}
        """
        # Configuration parameters for the IK solver
        self.solver_config = {"ftol": 1e-5, 
                     "disp": False, 
                     "eps": 1e-8, 
                     "maxiter": 5000}
        if config is not None:
            self.solver_config.update(config)
        
        # Load the robot model from URDF
        self.robot_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.robot_data = self.robot_model.createData()

        # Extract frame IDs and orientation constraints
        self.target_frame_ids = [self.robot_model.getFrameId(name) for name in ee_targets.keys()]
        self.orientation_constraints = {frame_id: list(ee_targets.values())[i] 
                                        for i, frame_id in enumerate(self.target_frame_ids)}
        
        # Joint limits and bounds for optimization
        self.joint_bounds = Bounds(self.robot_model.lowerPositionLimit, self.robot_model.upperPositionLimit)

        # Placeholder for desired end-effector poses
        self.desired_poses = []

    def __str__(self) -> str:
        """
        Returns a string summary of the kinematics solver and its configuration.
        """
        summary = "Robot Kinematics Solver:\n"
        summary += f" Generalized Coordinates: {self.robot_model.nq}\n"
        summary += f" Generalized Velocities: {self.robot_model.nv}\n"
        summary += " Solver Configuration:\n"
        for key, value in self.solver_config.items():
            summary += f"   {key}: {value}\n"
        return summary

    def _compute_loss(self, joint_angles: np.ndarray) -> float:
        """
        Computes the optimization loss based on the current joint configuration.
        Arguments:
            joint_angles: The robot's joint angles as a NumPy array.
        Returns:
            Scalar loss value representing the total error across all targets.
        """
        pinocchio.framesForwardKinematics(self.robot_model, self.robot_data, joint_angles)
        total_error = 0.0
        
        for idx, frame_id in enumerate(self.target_frame_ids):
            if self.orientation_constraints[frame_id]:
                delta_transform = self.desired_poses[idx].actInv(self.robot_data.oMf[frame_id])
                error = pinocchio.log(delta_transform).vector
            else:
                error = self.robot_data.oMf[frame_id].translation - np.asarray(self.desired_poses[idx])
            total_error += np.linalg.norm(error) ** 2
        return total_error

    def compute_forward_kinematics(self, joint_angles: Union[np.ndarray, List[float]]) -> List[np.ndarray]:
        """
        Computes the forward kinematics for the given joint configuration.
        Arguments:
            joint_angles: Array or list of joint positions.
        Returns:
            A list of end-effector poses in [x, y, z, qx, qy, qz, qw] format.
        """
        pinocchio.framesForwardKinematics(self.robot_model, self.robot_data, np.asarray(joint_angles))
        pinocchio.updateFramePlacements(self.robot_model, self.robot_data)
        return [pinocchio.SE3ToXYZQUAT(self.robot_data.oMf[frame_id]) for frame_id in self.target_frame_ids]

    def solve_ik(self, target_poses: List[List[float]], 
                 initial_guess: Optional[np.ndarray] = None, 
                 tolerance: float = 1e-3, 
                 max_attempts: int = 1) -> Tuple[np.ndarray, bool, float]:
        """
        Solves the inverse kinematics problem to find joint angles for the desired poses.
        Arguments:
            target_poses: List of desired end-effector poses.
            initial_guess: Initial joint configuration for the solver. If None, a random guess is used.
            tolerance: Acceptable loss threshold for the solution.
            max_attempts: Maximum attempts to find a solution.
        Returns:
            A tuple containing:
            - Joint configuration as a NumPy array.
            - Boolean indicating success.
            - The final loss value.
        """
        attempts = 0
        best_solution = None
        best_loss = float("inf")
        success = False

        # Convert target poses into appropriate formats
        self.desired_poses = []
        for i, frame_id in enumerate(self.target_frame_ids):
            if self.orientation_constraints[frame_id]:
                self.desired_poses.append(pinocchio.XYZQUATToSE3(target_poses[i]))
            else:
                self.desired_poses.append(target_poses[i])
        
        # Calculate initial loss
        if initial_guess is not None:
            current_loss = self._compute_loss(initial_guess)
        else:
            current_loss = float("inf")

        while attempts < max_attempts and current_loss > tolerance:
            # Generate random initial guess if necessary
            if attempts > 0 or initial_guess is None:
                initial_guess = np.random.uniform(low=self.robot_model.lowerPositionLimit, 
                                                  high=self.robot_model.upperPositionLimit)
            
            # Run optimization
            result = minimize(self._compute_loss, initial_guess, method="SLSQP", 
                              options=self.solver_config, bounds=self.joint_bounds)
            
            # Update best solution if the current result is better
            current_loss = result.fun
            if current_loss < best_loss:
                best_solution = result.x
                best_loss = current_loss
                success = result.success
            
            print(f"Attempt: {attempts + 1}  Loss: {current_loss:.6f}  Initial Guess: {initial_guess}")
            attempts += 1

        return best_solution, success and (best_loss < tolerance), best_loss
