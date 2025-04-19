import numpy as np
from scipy.optimize import linprog, minimize
import AllegroHandEnv
from AllegroHandEnv import AllegroHandEnvSphere
import dm_control
import mujoco as mj
import grasp_synthesis
import types

"""
Note: this code gives a suggested structure for implementing grasp synthesis.
You may decide to follow it or not. 
"""

def synthesize_grasp(env: grasp_synthesis.AllegroHandEnv, 
                         q_h_init: np.array,
                         fingertip_names: list[str], 
                         max_iters=1000, 
                         lr=0.1):
    """
    Given an initial hand joint configuration, q_h_init, return adjusted joint angles that are touching
    the object and approximate force closure. This is algorithm 1 in the project specification.

    Parameters
    ----------
    env: AllegroHandEnv instance (can use to access physics)
    q_h_init: array of joint positions for the hand
    max_iters: maximum number of iterations for the optimization
    lr: learning rate for the gradient step

    Output
    ------
    New joint angles after contact and force closure adjustment
    """
    #YOUR CODE HERE

    i = 0
    q = q_h_init
    made_contact = False
    while i < max_iters:
        distances = []
        for f in fingertip_names:
            pos = env.physics.data.xpos[env.physics.model.name2id(f, 'body')].copy()
            d = env.sphere_surface_distance(pos, env.sphere_center, env.sphere_radius)
            distances.append(d)
        made_contact = all(abs(d) < 1e-2 for d in distances)
        grad = numeric_gradient(joint_space_objective, q, env, fingertip_names, made_contact)
        q_new = q - lr * grad
        
        improvement = joint_space_objective(env, q, fingertip_names, made_contact) - joint_space_objective(env, q_new, fingertip_names, made_contact)
        if improvement > 0:
            q = q_new
            if improvement < 1e-6:
                break
        i += 1
    return q

def joint_space_objective(env: grasp_synthesis.AllegroHandEnvSphere, 
                          q_h: np.array,
                          fingertip_names: list[str], 
                          in_contact: bool, 
                          beta=10.0, 
                          friction_coeff=0.5, 
                          num_friction_cone_approx=4,
                          Qp_thresh=1e-5):
    """
    This function minimizes an objective such that the distance from the origin
    in wrench space as well as distance from fingers to object surface is minimized.
    This is algorithm 2 in the project specification. 

    Parameters
    ----------
    env: AllegroHandEnv instance (can use to access physics)
    q_h: array of joint positions for the hand
    fingertip_names: names of the fingertips as defined in the MJCF
    in_contact: helper variable to determine if the fingers are in contact with the object
    beta: weight coefficient on the surface penalty 
    friction_coeff: Friction coefficient for the ball
    num_friction_cone_approx: number of approximation vectors in the friction cone

    
    Output
    ------
    fc_loss + (beta * d) as written in algorithm 2
    """
    env.set_configuration(q_h)
    #YOUR CODE HERE
    beta = 10
    finger_ids = [env.physics.model.name2id(name, 'body') for name in fingertip_names]
    fingertip_names = np.array(['sawyer/allegro_right/ff_tip_rubber', 'sawyer/allegro_right/mf_tip_rubber', 'sawyer/allegro_right/rf_tip_rubber', 'sawyer/allegro_right/th_tip_rubber'])

    pos = env.physics.data.xpos[finger_ids].copy()
    
    d = np.sum(max(0, env.sphere_surface_distance(p, env.sphere_center, env.sphere_radius)**2) for p in pos)
    #print(d)

    if not in_contact:
        return beta * d
    else:
        norms = env.get_contact_normals(env.physics.data.contact[1:])
        FC = build_friction_cones(norms, friction_coeff, num_friction_cone_approx)

        # positions = []
        # for i in range(4):
        #     contact = env.physics.data.contact[i]
        #     print(f"Finger ids: {finger_ids}")
        #     print("0 ", contact.geom[0])
        #     print("1 ", contact.geom[1])
        #     if contact.geom[0] in finger_ids or contact.geom[1] in finger_ids:
        #         positions.append(c.pos.copy())

        # positions = np.array(positions)
        positions = env.get_contact_positions(fingertip_names)
        print(positions.shape)
        print(positions[0])
        
        G = build_grasp_matrix(positions, FC)

        print("G shape:", G.shape)
        # print("Any NaN in G?", np.any(np.isnan(G)))
        # print("Any inf in G?", np.any(np.isinf(G)))
        print("norms shape:", norms.shape)
        print("FC length:", len(FC))

        fc_loss = optimize_necessary_condition(G, env, beta, d)
        if fc_loss < Qp_thresh:
            fc_loss = optimize_sufficient_condition(G)
        print("Force closure loss:", fc_loss)
        return fc_loss + (beta * d)


def numeric_gradient(function: types.FunctionType, 
                     q_h: np.array, 
                     env: grasp_synthesis.AllegroHandEnv, 
                     fingertip_names: list[str], 
                     in_contact: bool, 
                     eps=0.01):
    """
    This function approximates the gradient of the joint_space_objective

    Parameters
    ----------
    function: function we are taking the gradient of
    q_h: joint configuration of the hand 
    env: AllegroHandEnv instance 
    fingertip_names: names of the fingertips as defined in the MJCF
    in_contact: helper variable to determine if the fingers are in contact with the object
    eps: hyperparameter for the delta of the gradient 

    Output
    ------
    Approximate gradient of the inputted function
    """
    baseline = function(env, q_h, fingertip_names, in_contact)
    grad = np.zeros_like(q_h)
    for i in range(len(q_h)):
        q_h_pert = q_h.copy()
        q_h_pert[i] += eps
        val_pert = function(env, q_h_pert, fingertip_names, in_contact)
        grad[i] = (val_pert - baseline) / eps
    return grad


def build_friction_cones(normal: np.array, mu=0.5, num_approx=4):
    """
    This function builds a discrete friction cone around each normal vector. 

    Parameters
    ----------
    normal: nx3 np.array where n is the number of normal directions
        normal directions for each contact
    mu: friction coefficient
    num_approx: number of approximation vectors in the friction cone

    Output
    ------
    friction_cone_vectors: array of discretized friction cones represented 
    as vectors
    """
    #YOUR CODE HERE

    FCs =[]
    for n in normal:
        norm = n / np.linalg.norm(n)
        FC_vectors = []
        for i in range(num_approx):
            theta = 2 * np.pi * i / num_approx
            ortho = np.random.randn(3)
            ortho -= ortho.dot(norm) * norm
            ortho = ortho / np.linalg.norm(ortho)
            vec = (
                np.cos(np.arctan(mu)) * norm +
                np.sin(np.arctan(mu)) * (
                    np.cos(theta) * ortho +
                    np.sin(theta) * np.cross(norm, ortho)
                )
            )
            FC_vectors.append(vec)
        FCs.append(FC_vectors)
    return FCs



def build_grasp_matrix(positions: np.array, friction_cones: list, origin=np.zeros(3)):
    """
    Builds a grasp map containing wrenches along the discretized friction cones. 

    Parameters
    ----------
    positions: nx3 np.array of contact positions where n is the number of contacts
    firction_cone: a list of lists as outputted by build_friction_cones. 
    origin: the torque reference. In this case, it's the object center.
    
    Return a 2D numpy array G with shape (6, sum_of_all_cone_directions).
    """
    #YOUR CODE HERE
    grasp = []
    for pos, cones in zip(positions, friction_cones):
        rel = pos - origin

        for vec in cones:
            T = np.cross(rel, vec)
            wrench = np.hstack([vec, T])
            grasp.append(wrench)

    return np.array(grasp).T

def optimize_necessary_condition(G: np.array, env: grasp_synthesis.AllegroHandEnv, beta: float, D: float):
    """
    Returns the result of the L2 optimization on the distance from wrench origin to the
    wrench space of G

    Parameters
    ----------
    G: grasp matrix
    env: AllegroHandEnv instance (can use to access physics)

    Returns the minimum of the objective

    Hint: use scipy.optimize.minimize
    """
    #YOUR CODE HERE
    def objective(a):
        return np.linalg.norm(G @ a) + beta * D

    x0 = np.zeros(G.shape[1])
    bounds = [(0, None) for _ in range(G.shape[1])]

    res = minimize(objective, x0, method='SLSQP', bounds=bounds)

    return res.fun


def optimize_sufficient_condition(G: np.array, K=20):
    """
    Runs the optimization from the project spec to evaluate Q- distance. 

    Parameters
    ----------
    G: grasp matrix
    K: number of approximations to the norm ball

    Returns the Q- value

    Hints:
        -Use scipy.optimize.linprog
        -Here's a resource with the basics: https://realpython.com/linear-programming-python/
        -You'll have to find a way to represent the alpha's for the constraints
            -Consider including the alphas in the linprog objective with coefficients 0 
        -For the optimization method, do method='highs'
    """
    #YOUR CODE HERE

    vecs = np.random.randn(6, K)
    unitvecs = vecs / np.linalg.norm(vecs, axis=0)

    Qval = np.inf

    for i in range(K):
        c = np.zeros(G.shape[1] + 1)
        c[-1] = -1

        A_eq = np.hstack([G, unitvecs[:, i:i+1]])
        b_eq = np.zeros(6)

        bounds = [(0, None) for _ in range(G.shape[1])] + [(None, None)]

        sol = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if sol.success:
            Qval = min(Qval, -sol.fun)

    return Qval
