#!/usr/bin/env python
"""
Starter script for lab1. 
Author: Chris Correa
"""
import copy
import sys
import argparse
import time
import numpy as np
import signal
import roslaunch
import rospkg
import matplotlib.pyplot as plt

from paths.trajectories import LinearTrajectory, CircularTrajectory, PolygonalTrajectory
from paths.paths import MotionPath
from controllers.controllers import (
    WorkspaceVelocityController, 
    PDJointVelocityController, 
    PDJointTorqueController, 
    FeedforwardJointVelocityController
)
from utils.utils import *
from path_planner import PathPlanner

from trac_ik_python.trac_ik import IK

import rospy
import tf
import tf2_ros
import intera_interface
import moveit_commander
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sawyer_pykdl import sawyer_kinematics

#RUN THESE
#  cd ~/ros_workspaces/proj1/src/proj1_pkg/scripts
#  cd ../../..; catkin_make; source devel/setup.bash; cd src/proj1_pkg/scripts
#  python main.py -task polygon -ar_marker 9 -c jointspace



def tuck():
    """
    Tuck the robot arm to the start position. Use with caution
    """
    if input('Would you like to tuck the arm? (y/n): ') == 'y':
        rospack = rospkg.RosPack()
        path = rospack.get_path('intera_examples')
        launch_path = path + '/launch/sawyer_tuck.launch'
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
        launch.start()
    else:
        print('Canceled. Not tucking the arm.')



def lookup_tag(tag_number, t = rospy.Duration(10.0)):
    """
    Given an AR tag number, this returns the position of the AR tag in the robot's base frame.
    You can use either this function or try starting the scripts/tag_pub.py script.  More info
    about that script is in that file.  

    Parameters
    ----------
    tag_number : int

    Returns
    -------
    3x' :obj:`numpy.ndarray`
        tag position
    """

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    to_frame = 'ar_marker_{}'.format(tag_number)

    try:
        trans = tfBuffer.lookup_transform('base', to_frame, rospy.Time(0), t)
        tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
        return np.array(tag_pos)
    except Exception as e:
        print(e)
        print("Retrying ...")
        return None
    



def get_trajectory(limb, kin, ik_solver, tag_pos, args):
    """
    Returns an appropriate robot trajectory for the specified task.  You should 
    be implementing the path functions in paths.py and call them here
    
    Parameters
    ----------
    task : string
        name of the task.  Options: line, circle, square
    tag_pos : 3x' :obj:`numpy.ndarray`
        
    Returns
    -------
    :obj:`moveit_msgs.msg.RobotTrajectory`
    """
    num_way = args.num_way
    controller_name = args.controller_name
    task = args.task

    # target_position = tag_pos[0]
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    try:
        trans = tfBuffer.lookup_transform('base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
        current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
        print("Current Position:", current_position)
    except Exception as e:
        print(e)
        return None

    if task == 'line':
        target_pos = tag_pos[0]
        target_pos[2] = current_position[2] #linear path moves to a Z position above AR Tag.
        trajectory = LinearTrajectory(current_position, target_pos, 1)
    elif task == 'circle':
        target_pos = tag_pos[0]
        target_pos[2] = current_position[2]
        trajectory = CircularTrajectory(target_pos, radius=0.1, total_time=15)
    elif task == 'polygon':
        #assert lin is not []
        target_pos = tag_pos[0]
        target_pos[2] = current_position[2]
        points = [[0,0,0], [.15,0, 0], [.15,.15, 0],[0, .15, 0]]
        trajectory = PolygonalTrajectory( target_pos + points, 10)

    else:
        raise ValueError('task {} not recognized'.format(task))
    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(num_way, controller_name!='workspace')

def get_controller(controller_name, limb, kin):
    """
    Gets the correct controller from controllers.py

    Parameters
    ----------
    controller_name : string

    Returns
    -------
    :obj:`Controller`
    """
    if controller_name == 'workspace':
        # YOUR CODE HERE
        Kp = None
        Kv = None
        controller = WorkspaceVelocityController(limb, kin, Kp, Kv)
    elif controller_name == 'jointspace':
        # YOUR CODE HERE
        Kp = 0.2 * np.array([0.4, 2, 1.7, 1.5, 2, 2, 3])
        Kv = 0.01 * np.array([2, 1, 2, 0.5, 0.8, 0.8, 0.8])
        controller = PDJointVelocityController(limb, kin, Kp, Kv)
    elif controller_name == 'torque':
        # YOUR CODE HERE
        Kp = None
        Kv = None
        controller = PDJointTorqueController(limb, kin, Kp, Kv)
    elif controller_name == 'open_loop':
        controller = FeedforwardJointVelocityController(limb, kin)
    else:
        raise ValueError('Controller {} not recognized'.format(controller_name))
    return controller


def main():
    """
    Examples of how to run me:
    python scripts/main.py --help <------This prints out all the help messages
    and describes what each parameter is
    python scripts/main.py -t line -ar 1 -c workspace --log
    python scripts/main.py -t circle -ar 2 -c jointspace --log
    python scripts/main.py -t polygon -ar 3 -c torque --log

    You can also change the rate, timeout if you want
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('-ar_marker', '-ar', nargs='+', help=
        'Which AR marker to use.  Default: 1'
    )
    parser.add_argument('-controller_name', '-c', type=str, default='moveit', 
        help='Options: moveit, open_loop, jointspace, workspace, or torque.  Default: moveit'
    )
    parser.add_argument('-rate', type=int, default=200, help="""
        This specifies how many ms between loops.  It is important to use a rate
        and not a regular while loop because you want the loop to refresh at a
        constant rate, otherwise you would have to tune your PD parameters if 
        the loop runs slower / faster.  Default: 200"""
    )
    parser.add_argument('-timeout', type=int, default=None, help=
        """after how many seconds should the controller terminate if it hasn\'t already.  
        Default: None"""
    )
    parser.add_argument('-num_way', type=int, default=300, help=
        'How many waypoints for the :obj:`moveit_msgs.msg.RobotTrajectory`.  Default: 300'
    )
    parser.add_argument('--log', action='store_true', help='plots controller performance')
    args = parser.parse_args()


    rospy.init_node('moveit_node')
    # this is used for sending commands (velocity, torque, etc) to the robot
    ik_solver = IK("base", "right_hand")
    limb = intera_interface.Limb('right')
    # this is used to get the dynamics (inertia matrix, manipulator jacobian, etc) from the robot
    # in the current position, UNLESS you specify other joint angles.  see the source code
    # https://github.com/ucb-ee106/baxter_pykdl/blob/master/src/sawyer_pykdl/sawyer_pykdl.py
    # for info on how to use each method
    kin = sawyer_kinematics('right')

    #GET INITIAL TAG POS
    tag_pos = [lookup_tag(marker) for marker in args.ar_marker]
    orig_tag_pos = copy.deepcopy(tag_pos)
        
    
    # Get an appropriate RobotTrajectory for the task (circular, linear, or square)
    # If the controller is a workspace controller, this should return a trajectory where the
    # positions and velocities are workspace positions and velocities.  If the controller
    # is a jointspace or torque controller, it should return a trajectory where the positions
    # and velocities are the positions and velocities of each joint.
    robot_trajectory = get_trajectory(limb, kin, ik_solver, tag_pos, args)

    # This is a wrapper around MoveIt! for you to use.  We use MoveIt! to go to the start position
    # of the trajectory
    planner = PathPlanner('right_arm')
    if args.controller_name == "workspace":
        pose = create_pose_stamped_from_pos_quat(
            robot_trajectory.joint_trajectory.points[0].positions,
            [0, 1, 0, 0],
            'base'
        )
        plan = planner.plan_to_pose(pose)
        planner.execute_plan(plan[1])
    else:
        start = robot_trajectory.joint_trajectory.points[0].positions
        print("START:", robot_trajectory.joint_trajectory.points[0].positions)
        
        while not rospy.is_shutdown():
            try:
                # limb.move_to_joint_positions(joint_array_to_dict(start, limb), timeout=7.0, threshold=0.0001) # ONLY FOR EMERGENCIES!!!
                plan = planner.plan_to_joint_pos(start)
                planner.execute_plan(plan[1])
                break
            except moveit_commander.exception.MoveItCommanderException as e:
                print(e)
                print("Failed planning, retrying...") 


    if args.controller_name == "moveit":
        # by publishing the trajectory to the move_group/display_planned_path topic, you should 
        # be able to view it in RViz.  You will have to click the "loop animation" setting in 
        # the planned path section of MoveIt! in the menu on the left side of the screen.
        pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
        disp_traj = DisplayTrajectory()
        disp_traj.trajectory.append(robot_trajectory)
        # disp_traj.trajectory_start = planner._group.get_current_joint_values()
        disp_traj.trajectory_start = RobotState()
        pub.publish(disp_traj)

        try:
            input('Press <Enter> to execute the trajectory using MOVEIT')
        except KeyboardInterrupt:
            sys.exit()
        # uses MoveIt! to execute the trajectory.  make sure to view it in RViz before running this.
        # the lines above will display the trajectory in RViz
        planner.execute_plan(robot_trajectory)
    else:
        # Project 1 PART B
        controller = get_controller(args.controller_name, limb, kin)
        try:
            input('Press <Enter> to execute the trajectory using YOUR OWN controller')
        except KeyboardInterrupt:
            sys.exit()

        times = list()
        actual_positions = list()
        actual_velocities = list()
        target_positions = list()
        target_velocities = list()


            # execute the path using your own controller.
        if args.task == 'polygon':
            done = controller.execute_path(
                times, 
                actual_positions,
                actual_velocities,
                target_positions,
                target_velocities,
                robot_trajectory, 
                rate=args.rate, 
                timeout=args.timeout, 
                log=args.log
            )
            ik_solver = IK("base", "right_hand")
            kin = sawyer_kinematics('right')
            robot_trajectory = get_trajectory(limb, kin, ik_solver, orig_tag_pos, args)

            if args.log:
                controller.plot_results(
                    times,
                    actual_positions, 
                    actual_velocities, 
                    target_positions, 
                    target_velocities
                )

        else:

            try:
                while not rospy.is_shutdown():
                    done = controller.execute_path(
                        times, 
                        actual_positions,
                        actual_velocities,
                        target_positions,
                        target_velocities,
                        robot_trajectory, 
                        rate=args.rate, 
                        timeout=args.timeout, 
                        log=args.log,
                    )
                    ik_solver = IK("base", "right_hand")
                    kin = sawyer_kinematics('right')
                    
                    tag_pos = [lookup_tag(marker, rospy.Duration(2.0)) for marker in args.ar_marker]
                    robot_trajectory = get_trajectory(limb, kin, ik_solver, tag_pos, args)
            except (KeyboardInterrupt, SystemExit, rospy.ROSException, rospy.ROSInterruptException):
                m = 0
                for i in range(len(times)-1):
                    if times[i] < times[i+1]:
                        m += times[i]
                    times[i] += m
                times[-1] += m


                controller.plot_results(
                    times,
                    actual_positions, 
                    actual_velocities, 
                    target_positions, 
                    target_velocities
                )
                sys.exit(0)
            m = 0
            for i in range(len(times)-1):
                if times[i] < times[i+1]:
                    m += times[i]
                times[i] += m
            times[-1] += m


            controller.plot_results(
                times,
                actual_positions, 
                actual_velocities, 
                target_positions, 
                target_velocities
            )
            sys.exit(0)
            
        if not done:
            print('Failed to move to position')
            sys.exit(0)


if __name__ == "__main__":
    tuck()
    
    main()
