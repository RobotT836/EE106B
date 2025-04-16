import numpy as np
import mujoco as mj
import dm_control
from utils import *

# Inspired by: https://alefram.github.io/posts/Basic-inverse-kinematics-in-Mujoco
class LevenbergMarquardtIK:
    
    def __init__(self, model: dm_control.mujoco.wrapper.core.MjModel, 
                 data: dm_control.mujoco.wrapper.core.MjData, 
                 step_size: int, 
                 tol: int, 
                 alpha: int, 
                 jacp: np.array, 
                 jacr: np.array, 
                 damping: int, 
                 max_steps: int, 
                 physics: dm_control.mjcf.physics.Physics):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping = damping
        self.max_steps = max_steps
        self.physics = physics
    
    def calculate(self, target_positions: np.array, 
                  target_orientations: np.array, 
                  body_ids: list, 
                  evaluating=False):
        """
        Calculates joint angles given target positions and orientations by solving inverse kinematics.
        Uses the Levenbeg-Marquardt method for nonlinear optimization. 

        Parameters
        ----------
        target_positions: 3xn np.array containing n desired x,y,z positions
        target_orientations: 4xn np.array containing n desired quaternion orientations
        body_ids: list of length n containing the ids for every body

        Returns
        -------
        new_qpos: np.array of size self.physics.data.qpos containing desired positions in joint space

        Tips: 
            -To access the body id you can use: self.model.body([insert name of body]).id 
            -You should consider using clip_to_valid_state in utils.py to ensure that joint poisitons
            are possible 
        """
        #YOUR CODE HERE
        self.data.qpos = self.physics.data.qpos.copy()
        mj.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_ids).xpos
        error = np.subtract(target_positions, current_pose)

        while (np.linalg.norm(error) >= self.tol):
            #calculate jacobian
            # jac = []
            for i in range(target_positions.shape[1]):
                mj.mj_jac(self.model, self.data, self.jacp, self.jacr, target_positions[:, i], body_ids[i])
                #calculate delta of joint q
                # jac.append(self.jacp)

                n = self.jacp.shape[1]
                I = np.identity(n)
                product = self.jacp.T @ self.jacp + self.damping * I
            
                if np.isclose(np.linalg.det(product), 0):
                    j_inv = np.linalg.pinv(product) @ self.jacp.T
                else:
                    j_inv = np.linalg.inv(product) @ self.jacp.T
                
                delta_q = j_inv @ error
                #compute next step
                self.data.qpos += self.step_size * delta_q
                #check limits
                clip_to_valid_state(self.physics, self.data.qpos) 
                #compute forward kinematics

            mj.mj_forward(self.model, self.data) 
            #calculate new error
            error = np.subtract(target_positions[i], self.data.body(body_ids).xpos)


