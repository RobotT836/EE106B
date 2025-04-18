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
        self.physics.data.qpos = self.physics.data.qpos.copy()
        mj.mj_forward(self.model.ptr, self.data.ptr)
        current_pose = np.array([self.physics.data.xpos[bid] for bid in body_ids])
        error = np.subtract(target_positions, current_pose)

        while (np.linalg.norm(error) >= self.tol) and self.max_steps > 0:
            #calculate jacobian
            error = np.array([0])
            jac = np.zeros(29)

            # print(np.array(target_positions).shape[1])

            for i in range(np.array(target_positions).shape[0]):
                ## calculate the jacobian
                mj.mj_jac(self.model.ptr, self.data.ptr, self.jacp, self.jacr, target_positions[i, :], body_ids[i])

                #calculate delta of joint q
                jac_all = np.vstack((self.jacp, self.jacr))
                jac = np.vstack((jac, jac_all))
                ##calc position and quat error

                pos_error = target_positions[i, :] - self.physics.data.xpos[body_ids[i]]
                quat_error = quaternion_error_naive(self.physics.data.xquat[body_ids[i]], target_orientations[i, :])
                all_error = np.hstack((pos_error, quat_error))
                error = np.hstack((error, all_error))
                

            error = error[1:]
            jac = jac[1:, :]
            # 30x29 jacobian after all 4 fingers and palm, thus 6x29 after one finger/palm and 3x29 per jacobian translation/rotation
            # 24x1 for error
            n = jac.shape[1]
            I = np.identity(n)
            product = jac.T @ jac + self.damping * I
        
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jac.T
            else:
                j_inv = np.linalg.inv(product) @ jac.T
            
            delta_q = j_inv @ error
            #compute next step
            self.physics.data.qpos[1:] += self.step_size * delta_q
            #check limits
            q = clip_to_valid_state(self.physics, self.physics.data.qpos)
            self.physics.data.qpos[:] = q
                
            mj.mj_forward(self.model.ptr, self.data.ptr) #self.physics.forward()
            self.max_steps -= 1
            #calculate new error
            


