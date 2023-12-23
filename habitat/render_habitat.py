import os

import cv2
import numpy as np
import argparse
import random
import quaternion as qt
import json
from tqdm import tqdm

import habitat_sim
import habitat_sim.agent
import habitat_sim.bindings as hab_bind

from habitat_sim.agent import AgentState

FORWARD_KEY = 'w'
LEFT_KEY="a"
RIGHT_KEY="d"
BACKWARD_KEY="s"
UP_KEY="u"
DOWN_KEY="i"

FINISH="f"

def make_cfg(settings):
    sim_cfg = hab_bind.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    sensors = {
        "color_sensor": {  # active if sim_settings["color_sensor"]
            "sensor_type": hab_bind.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        }
    }


    # Creat sensor
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = hab_bind.CameraSensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]
                sensor_spec.sensor_subtype = hab_bind.SensorSubType.PINHOLE
                if not settings["silent"]:
                    print("==== Initialized Sensor Spec: =====")
                    print("Sensor uuid: ", sensor_spec.uuid)
                    print("Sensor type: ", sensor_spec.sensor_type)
                    print("Sensor position: ", sensor_spec.position)
                    print("===================================")

                sensor_specs.append(sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "move_up": habitat_sim.agent.ActionSpec(
            "move_up", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "move_down": habitat_sim.agent.ActionSpec(
            "move_down", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
    }

    # override action space to no-op to test physics
    if sim_cfg.enable_physics:
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
            )
        }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def id_to_lable(semantic_observation, scene_dict):
    for id in np.unique(semantic_observation):
        if scene_dict['id_to_label'][id] < 0:
            semantic_observation[semantic_observation==id] = 0
        elif scene_dict['id_to_label'][id] == 0:
            print('Warning: unexpected id 0 occured, considered as unlabeled...')
            print(scene_dict)
        else:
            semantic_observation[semantic_observation==id] = scene_dict['id_to_label'][id]

    return semantic_observation

def load_pose(pose_file):
    position = np.array()
    rotation = qt.quaternion()
    return position, rotation

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="room_0")

args = parser.parse_args()

data_path = "/home/star/Dataset/Replica/replica"
scene = args.scene

with open(os.path.join(data_path, scene, 'habitat', 'info_semantic.json'), 'r') as f:
    state_dict = json.load(f)

settings = {}

# set simulator parameters
settings['width'] = 800
settings['height'] = 800
settings['sensor_height'] = 0.0
settings['color_sensor'] = True
settings['silent'] = True
settings['scene'] = os.path.join(data_path, scene, "habitat", "mesh_semantic.ply")

cfg = make_cfg(settings)
simulator = habitat_sim.Simulator(cfg)

print(simulator._sensors['color_sensor']._sensor_object.hfov)
agent = simulator.get_agent(0)
observations = simulator.get_sensor_observations()
cv2.imshow("rgb", cv2.cvtColor(observations["color_sensor"], cv2.COLOR_BGR2RGB))

output_dir = "/home/star/Dataset/Replica/replica_generated_msp"
output_path = os.path.join(output_dir, scene)

eqr_path = os.path.join(output_path, "eqr")
fisheye_path = os.path.join(output_path, "fisheye")
os.makedirs(eqr_path, exist_ok=True)
os.makedirs(fisheye_path, exist_ok=True)

index = 0
agent_pose = []
while True:
    key = cv2.waitKey(0)
    if key == ord(FORWARD_KEY):
        observations = simulator.step("move_forward")
        print("move forward")
    elif key == ord(LEFT_KEY):
        observations = simulator.step("move_left")
        print("move_left")
    elif key == ord(RIGHT_KEY):
        observations = simulator.step("move_right")
        print("move_right")
    elif key == ord(BACKWARD_KEY):
        observations = simulator.step("move_backward")
        print("move backward")
    elif key == ord(UP_KEY):
        observations = simulator.step("move_up")
        print("move up")
    elif key == ord(DOWN_KEY):
        observations = simulator.step("move_down")
        print("move down")
    elif key == 82:
        observations = simulator.step("look_up")
        print("look up")
    elif key == 84:
        observations = simulator.step("look_down")
        print("look down")
    elif key == 81:
        observations = simulator.step("turn_left")
        print("turn left")
    elif key == 83:
        observations = simulator.step("turn_right")
        print("turn right")
    elif key == 27:
        break
    else:
        continue

    rgb = observations["color_sensor"]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.imshow("rgb", rgb)

    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states["color_sensor"]
    agent_pose.append((sensor_state.position, sensor_state.rotation))
    index += 1

    print("Position is {}, rotation is {}".format(sensor_state.position, sensor_state.rotation))
    print("This is No.{} image captured".format(index))

def render_agent_pose(pose):
    trans, quat = pose
    agent_state = AgentState()
    agent_state.position = trans
    agent_state.rotation = quat
    agent.set_state(agent_state)
    
    observations = simulator.get_sensor_observations()
    rgb = observations["color_sensor"]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    return rgb

cube_names = ["front", "left", "right", "up", "down", "back"]
cube_rots = [qt.from_rotation_matrix(np.array([[1,0,0],[0,1,0],[0,0,1]])), 
             qt.from_rotation_matrix(np.array([[0,0,1],[0,1,0],[-1,0,0]])), 
             qt.from_rotation_matrix(np.array([[0,0,-1],[0,1,0],[1,0,0]])), 
             qt.from_rotation_matrix(np.array([[1,0,0],[0,0,-1],[0,1,0]])), 
             qt.from_rotation_matrix(np.array([[1,0,0],[0,0,1],[0,-1,0]])), 
             qt.from_rotation_matrix(np.array([[-1,0,0],[0,1,0],[0,0,-1]]))]

def pose2np(trans, quat):
    np_pose = np.eye(4)
    np_pose[:3, 3] = trans
    np_pose[:3, :3] = qt.as_rotation_matrix(quat)
    
    return np_pose

def np2pose(np_pose):
    trans = np_pose[:3, 3]
    quat = qt.from_rotation_matrix(np_pose[:3, :3])
    
    return trans, quat

def render_cube_map(path, pose):
    trans, quat = pose
    for cube_num, cube_name in enumerate(cube_names):
        cube_rot = cube_rots[cube_num]
        cube_world_rot = quat*cube_rot
        rgb = render_agent_pose((trans, cube_world_rot))
        
        rgb_name = os.path.join(path, cube_name + ".png")
        cv2.imwrite(rgb_name, rgb)
    
    # camera to world pose
    pose = pose2np(trans, quat)
    pose_name = os.path.join(path, "xyz.txt")
    np.savetxt(pose_name, pose)

inner_size = 2
outer_size = 2
rig_size = 0.2

def render_target_eqr(index, pose):
    trans, _ = pose
    path = os.path.join(eqr_path, str(index))
    if not os.path.exists(path):
        os.makedirs(path)
    
    for num in range(inner_size):
        num_path = os.path.join(path, str(num))
        if not os.path.exists(num_path):
            os.makedirs(num_path)
        
        theta = 2 * np.pi * random.random()
        phi = np.arccos(2 * random.random() - 1)
        r = rig_size * random.random()

        rand_trans = np.zeros(3)
        rand_trans[0] = r * np.sin(phi) * np.cos(theta) + trans[0]
        rand_trans[1] = r * np.sin(phi) * np.sin(theta) + trans[1]
        rand_trans[2] = r * np.cos(phi) + trans[2]
        
        rand_quat = qt.from_euler_angles(
            random.random() * 2 * np.pi, random.random() * 2 *  np.pi, random.random() * 2 *  np.pi
        )

        render_cube_map(num_path, (rand_trans, rand_quat))
        
    for num in range(inner_size, inner_size + outer_size):
        num_path = os.path.join(path, str(num))
        if not os.path.exists(num_path):
            os.makedirs(num_path)
        
        theta = 2 * np.pi * random.random()
        phi = np.arccos(2 * random.random() - 1)
        r = rig_size * (1 + 0.5 * random.random())

        rand_trans = np.zeros(3)
        rand_trans[0] = r * np.sin(phi) * np.cos(theta) + trans[0]
        rand_trans[1] = r * np.sin(phi) * np.sin(theta) + trans[1]
        rand_trans[2] = r * np.cos(phi) + trans[2]
        
        rand_quat = qt.from_euler_angles(
            random.random() * 2 * np.pi, random.random() * 2 *  np.pi, random.random() * 2 *  np.pi
        )

        render_cube_map(num_path, (rand_trans, rand_quat))

fisheye_poses = [np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-rig_size],[0,0,0,1]]), 
                 np.array([[0,0,-1,rig_size],[0,1,0,0,],[1,0,0,0],[0,0,0,1]]), 
                 np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,rig_size],[0,0,0,1]]), 
                 np.array([[0,0,1,-rig_size],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])]

def render_fisheye(index, pose):
    trans, quat = pose
    path = os.path.join(fisheye_path, str(index))
    if not os.path.exists(path):
        os.makedirs(path)
    
    for num, fisheye_pose in enumerate(fisheye_poses):
        num_path = os.path.join(path, str(num))
        if not os.path.exists(num_path):
            os.makedirs(num_path)
        
        pose = pose2np(trans, quat)
        fisheye_pose = pose.dot(fisheye_pose)
        fisheye_trans, fisheye_quat = np2pose(fisheye_pose)
        
        render_cube_map(num_path, (fisheye_trans, fisheye_quat))

for num, pose in enumerate(tqdm(agent_pose)):
    render_target_eqr(num, pose)
    render_fisheye(num, pose)