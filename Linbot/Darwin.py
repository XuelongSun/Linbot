import sys
import os
from collections import deque
import time

import math 
import numpy as np
import cv2

from controller import Motion
from webots import WebotsRunner
from walking import Walking

# motion and gait module
libraryPath = os.path.join(os.environ.get("WEBOTS_HOME"), 'projects', 'robots',
                           'robotis', 'darwin-op', 'libraries', 'python39')
libraryPath = libraryPath.replace('/', os.sep)
sys.path.append(libraryPath)
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager


class DarwinOP(WebotsRunner):
    def __init__(self, name):
        super(DarwinOP, self).__init__(name)
        # kinematics
        self.kinematics = {
            'leg_length': 219.5,
            'thigh_length': 93.0,
            'calf_length': 93.0,
            'ankle_length': 33.5
        }
        
        self.motor_names = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                            'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL',
                            'PelvR', 'PelvL', 'LegUpperR', 'LegUpperL',
                            'LegLowerR', 'LegLowerL', 'AnkleR', 'AnkleL',
                            'FootR', 'FootL', 'Neck', 'Head')
        # when you look the face of the robot:
        # pitch: arrpoaching you is +; roll: clockwise is +;
        # when you look the head of the robot from the sky, yaw: z-axis conterclockwise is +
        self.motor_direction = {
            'ShoulderR': -1,
            'ShoulderL': 1,
            'Neck': 1,
            'Head': 1,
            'ArmUpperR': 1,
            'ArmUpperL': 1,
            'ArmLowerR': -1,
            'ArmLowerL': 1,
            'PelvYR': -1,
            'PelvYL': -1,
            'PelvR': -1,
            'PelvL': -1,
            'LegUpperR': 1,
            'LegUpperL': -1,
            'LegLowerR': 1,
            'LegLowerL': -1,
            'AnkleR': -1,
            'AnkleL': 1,
            'FootR': 1,
            'FootL': 1,
        }

        # sensors
        self.position_sensor_names = [n + 'S' for n in self.motor_names]
        self.inertial_sensor_names = ['Gyro', 'Accelerometer']

        # official gait and motion manager, we may use ourselves' gait algorithm
        self.motion_manager = RobotisOp2MotionManager(self.robot)
        self.gait_manager = RobotisOp2GaitManager(self.robot, "config_offical.ini")
        self.gait_manager.setBalanceEnable(True)
        self.initial_position = None
        
        # self-build walking algorithm
        self.walker = Walking()
        # set gait dt the same as the robot time step
        self.walker.dt = self.time_step
        self.walk_speed = 20 # cooresponding to 'x_move_amp'
        
        # flag determine which gait algorithm we use
        self.use_self_gait = False
        
        # related to fallen detection
        self.face_up_count = 0
        self.face_down_count = 0
        self.fallen_acc_threshold = 80
        self.fallen_acc_steps = 20
        
        # self-motion related
        self.estimate_x = 0
        self.estimate_z = 0
        # use dqueue to store data for fast moving average
        self.yaw_history = deque([0], maxlen=self.walker.loop_number)
        self.heading = 0

        # attitude 
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.robottime = 0
        
    def RegisterMotions(self):
        self.HeadsUp = Motion('Motion/HeadsUp.motion')
        self.HeadsDown = Motion('Motion/HeadsDown.motion')
        self.ClimbObstacle = Motion('Motion/ClimbObstacle_2.motion')
        self.ClimbStair = Motion('Motion/stair.motion')
        self.DownStair = Motion('Motion/stair_down_all_test.motion')
        self.StepLeft_small = Motion('Motion/MoveLsmall.motion')
        self.StepRight_small = Motion('Motion/MoveRsmall.motion')
        self.StepLeft_small_1 = Motion('Motion/MoveLsmall_1.motion')
        self.StepRight_small_1 = Motion('Motion/MoveRsmall_1.motion')
        print('Motions Registered.')
        
    def play_user_motion_sync(self, motion):
        motion.play()
        while not motion.isOver():
            self.my_step()
        return 0
    
    def initialize(self):
        print('Initializing robot...')
        self.position_sensors = self.get_sensor(self.position_sensor_names)
        self.inertial_sensors = self.get_sensor(self.inertial_sensor_names)
        self.camera = self.get_sensor(['Camera'])['Camera']
        self.camera_height = self.camera.getHeight()
        self.camera_width = self.camera.getWidth()
        self.camera_fov = self.camera.getFov()
        self.fx = self.fy = self.camera_width/ 2 / math.tan(self.camera_fov/2)
        print('fx',self.fx)
        self.camera_internalMat = np.array([[ 92.59826332,  0.         ,    79.23232356],
                                            [ 0.         ,  75.85286424,    59.47240082],
                                            [ 0.         ,  0.         ,    1.         ]])
        self.camera_v_rot = np.array([[-0.45775242], [ 0.00344399], [ 0.00954099]])
        self.camera_v_trans = np.array([[-0.1189015 ], [-0.21213912], [ 0.424298  ]])

        self.motors = self.get_actuator(self.motor_names)
        self.my_step()
        
        # stand up
        self.motion_manager.playPage(1)
        self.wait(200)
        # get position sensors values
        self.initial_position = self.get_position_sensor_values()
        print(self.initial_position)
        
        self.loop_count = 0
        print('Successfully initialized.')
        
        return self.sensors, self.actuators

    def get_position_sensor_values(self):
        position = dict.fromkeys(self.position_sensor_names)
        for k, v in self.position_sensors.items():
            position[k] = v.getValue()
        return position
            
    def set_motor_position(self, motor_name, value, offset=0):
        # direction
        value *= self.motor_direction[motor_name]
        # offset, can get from position sensors of a specific gesture
        value += offset
        # clip
        min_p = self.motors[motor_name].getMinPosition()
        max_P = self.motors[motor_name].getMaxPosition()
        if value < min_p:
            value = min_p
        if value > max_P:
            value = max_P

        # set motor position
        self.motors[motor_name].setPosition(value)

        return value

    def get_yaw_velocity(self):
        gyro_values = self.inertial_sensors['Gyro'].getValues()
        # lookup table of this gyro:
        # [-27.925 0 0, 27.925 1024 0]
        v = (gyro_values[2] - 512.0) * 0.05454 + 0.000771
        return v
    

    def estimate_self_heading(self):
        # update heading estimated from the Gyro
        yaw = self.yaw_history[-1] + self.get_yaw_velocity()
        self.yaw_history.append(yaw)
        self.heading = (np.mean(self.yaw_history)) % (np.pi*2)
        return self.heading   
    def get_attitude(self):
        alpha = 0.1

        # base on World coordinate :  x_positive: left, y_positive: up, z_positive: forward
        # acc: [x , -z , -y] 
        acc = np.array(self.inertial_sensors['Accelerometer'].getValues())
        # gyro: [z, -x, y]
        gyro = np.array(self.inertial_sensors['Gyro'].getValues())

        acc = (acc -512)/1024 * 39.24 * 2
        ax = acc[0]
        ay = -acc[2]
        az = -acc[1]

        gyro = (gyro -512)/1024 * 27.925 * 2
        gx = -gyro[1]
        gy = gyro[2]
        gz = gyro[0]

        dt= self.time_step/1000
        self.robottime = self.robot.getTime()
        # # Get estimated angles from raw accelerometer data
        roll_hat_acc = math.atan2(ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        pitch_hat_acc = math.atan2(az, math.sqrt(ax ** 2.0 + ay ** 2.0))
        # print('roll_hat_acc', roll_hat_acc,'pitch_hat_acc', pitch_hat_acc)
        
        # # Calculate Euler angle derivatives 
        roll_dot = gz + math.sin(self.roll) * math.tan(self.pitch) * gx + math.cos(self.roll) * math.tan(self.pitch) * gy
        pitch_dot = math.cos(self.roll) * gx - math.sin(self.pitch) * gy
        yaw_dot = math.sin(self.roll)/ math.cos(self.pitch)*gx + math.cos(self.roll) / math.cos(self.pitch) * gy
        # print('roll_dot\t', roll_dot, 'pitch_dot\t', pitch_dot, 'yaw_dot\t', yaw_dot)

        # # Update complimentary filter
        # self.roll = (1 - alpha) * (self.roll + dt * roll_dot) + alpha * roll_hat_acc
        # self.pitch = (1 - alpha) * (self.pitch + dt * pitch_dot) + alpha * pitch_hat_acc
        self.roll = self.roll + dt * roll_dot
        self.pitch = self.pitch + dt * pitch_dot
        self.yaw = self.yaw + dt * yaw_dot
        # print('roll', self.roll,'\tpitch', self.pitch,'\tyaw', self.yaw)

        if self.yaw > np.pi:
            self.yaw -= np.pi * 2
            # print('time:', self.robottime)
        if self.yaw< np.pi:
            self.yaw += np.pi * 2
            # print('time:', self.robottime)

        return self.roll, self.pitch, self.yaw

        
    def estimate_self_position(self):
        # stepsize related to the gait x move
        step_size = self.walker.param_move['x_move_amp'] / 1000
        self.estimate_x += step_size*np.cos(self.heading)
        self.estimate_z += step_size*np.sin(self.heading)
        return self.estimate_x, self.estimate_z
    
    def set_head_pitch(self, angle=0):
        # -0.36rad(-20deg, down) to 0.96rad(55deg up)
        v = self.set_motor_position('Head', np.deg2rad(angle))
        return v
    
    def set_head_yaw(self, angle=0):
        # -1.81rad(-100deg, right) to 1.81rad(100deg left)
        v = self.set_motor_position('Neck', np.deg2rad(angle))
        return v
        
    def keyboard_process(self):
        key = 0
        key = self.keyboard.getKey()
        if key == self.keyboard.UP:
            if self.use_self_gait:
                self.walker.param_move['x_move_amp'] += 0.2
                print('x_move_amp:',self.walker.param_move['x_move_amp'])
            else:
                self.gait_manager.setXAmplitude(0.5)
                print('forward')
        elif key == self.keyboard.DOWN:
            if self.use_self_gait:
                self.walker.param_move['x_move_amp'] -= 0.2
                print('x_move_amp:',self.walker.param_move['x_move_amp'])
            else:
                self.gait_manager.setXAmplitude(-0.5)
                print('back')
        elif key == self.keyboard.LEFT:
            if self.use_self_gait:
                self.walker.param_move['a_move_amp'] += 0.1
                print('a_move_amp:',self.walker.param_move['a_move_amp'])
            else:
                self.gait_manager.setAAmplitude(0.5)
                print('left')
        elif key == self.keyboard.RIGHT:
            if self.use_self_gait:
                self.walker.param_move['a_move_amp'] -= 0.1
                print('a_move_amp:',self.walker.param_move['a_move_amp'])
            else:
                self.gait_manager.setAAmplitude(-0.5)
                print('right')
        elif key == ord("Y"):
            self.walker.param_move['y_move_amp'] += 0.1
            print('y_move_amp:',self.walker.param_move['y_move_amp'])
        elif key == ord("Z"):
            self.walker.param_move['y_move_amp'] -= 0.1
            print('y_move_amp:',self.walker.param_move['y_move_amp'])
        elif key == ord("R"):
            self.walker.param_move['z_move_amp'] = 25
            self.walker.param_move['x_move_amp'] = 5
            self.walker.param_move['y_move_amp'] = 0
            self.walker.param_move['a_move_amp'] = 0
            print(self.walker.param_move)

        elif key == ord("C"):
            img = self.get_img()
            localtime = time.localtime(time.time())
            imgname = '.\\pic\\' +time.strftime("%Y-%m-%d %H_%M_%S", time.localtime()) +'.png'
            self.camera.saveImage(imgname, 100)
        elif key == ord("W"):
            # head up
            motor_position = self.get_position_sensor_values()
            head_position = np.rad2deg(motor_position['HeadS']) + 1
            self.set_head_pitch(head_position)
            
        elif key == ord("A"):
            # head left
            motor_position = self.get_position_sensor_values()
            neck_position = np.rad2deg(motor_position['NeckS']) + 1
            self.set_head_yaw(neck_position)
            pass
        
        elif key == ord("S"):
            # head down
            motor_position = self.get_position_sensor_values()
            head_position = np.rad2deg(motor_position['HeadS']) - 1
            self.set_head_pitch(head_position)
            
        elif key == ord("D"):
            # head right
            motor_position = self.get_position_sensor_values()
            neck_position = np.rad2deg(motor_position['NeckS']) - 1
            self.set_head_yaw(neck_position)
            pass
        
        else:
            if not self.use_self_gait: 
                self.gait_manager.setXAmplitude(0)
                self.gait_manager.setAAmplitude(0)

        return key

    def set_head_pitch(self, angle=0):
        # -0.36rad(-20deg, down) to 0.96rad(55deg up)
        # <-0.2rad(<-11deg, feet will be seen)
        v = self.set_motor_position('Head', np.deg2rad(angle))
        return v
    
        
    def get_img(self, hsv=False):
        # get current image
        img = np.array(self.camera.getImageArray(), dtype=np.uint8)
        if hsv:
            # convert to HSV for color recognition
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return img

    def save_img(self, filename='img.jpg', img=None):
        try:
            if img is None:
                self.camera.saveImage(filename, quality=50)
            else:
                cv2.imwrite(filename, img)
            return True
        except:
            return False
    
    def check_if_fallen(self):
        acc = self.inertial_sensors['Accelerometer'].getValues()
        if acc[1] < 512.0 - self.fallen_acc_threshold:
            self.face_up_count += 1
        else:
            self.face_up_count = 0

        if acc[1] > 512.0 + self.fallen_acc_threshold:
            self.face_down_count += 1
        else:
            self.face_down_count = 0
        
        # if fallen, gets up
        if self.face_up_count > self.fallen_acc_steps:
            return 'Fallen face up'
        elif self.face_down_count > self.fallen_acc_steps:
            return 'Fallen face down'
        else:
            return None
    
    def ready_to_walk(self):
        if self.use_self_gait:
            # use the first value of motors from the walking as the get-ready pose
            self.walker.timer = 0 
            self.walker.period_counter = 0
            self.walk_step()
            # wait until stable
            self.wait(100)
            # reset timer
            self.walker.timer = 0 
            self.walker.period_counter = 0
        else:
            self.motion_manager.playPage(9)
            self.wait(100)
            self.gait_manager.start()

    def walk_step(self):
        if self.use_self_gait:
            if self.loop_count % (self.walker.param_time['period_time']*4 / self.walker.dt) == 0:
                if self.walker.period_counter < 2:
                    # smooth start
                    self.walker.param_move['x_move_amp'] = 0
                else:
                    self.walker.param_move['x_move_amp'] = self.walk_speed
                
                self.walker.update_param_move()
                self.walker.update_param_time()
                # print('update walking parameters at {}'.format(self.loop_count))
                
            # calculate endpoint
            eps, pel_l, pel_r, hip_pitch, arm_l, arm_r = self.walker.computer_endpoint_total()
            
            # calculate motor angles: 
            # left
            angles_L = self.walker.computer_ik(eps['xl'], eps['yl'], eps['zl'], c=eps['cl'])
            # right
            angles_R = self.walker.computer_ik(eps['xr'], eps['yr'], eps['zr'], c=eps['cr'])
            
            if (angles_L is False) or (angles_R is False):
                # walk time control
                self.walker.timer += self.walker.dt
                if self.walker.timer >= self.walker.param_time['period_time']:
                    self.walker.timer = 0
                    self.walker.period_counter += 1
                return False
            
            # pel offset
            angles_L[1] += pel_l
            angles_R[1] += pel_r
            angles_L[2] -= hip_pitch
            angles_R[2] -= hip_pitch
            
            # balance control:
            gyro_value = self.inertial_sensors['Gyro'].getValues()
            bal_offset = self.walker.balance_control(np.array(gyro_value)-512.0)
            motor_name = ['PelvY', 'Pelv', 'LegUpper', 'LegLower', 'Ankle', 'Foot']
            for i,d in enumerate([1,3,4,5]):
                angles_L[d] -= bal_offset[i] * self.motor_direction[motor_name[d] + 'L']
                angles_R[d] -= bal_offset[i] * self.motor_direction[motor_name[d] + 'R']
            
            # set motor
            # HIP_YAW HIP_ROLL HIP_PITCH KNEE_PITCH ANKLE_PITCH ANKLE_ROLL
            for d in ['L', 'R']:
                for i, s in enumerate(['PelvY', 'Pelv', 'LegUpper', 'LegLower', 'Ankle', 'Foot']):
                    offset = self.initial_position[s + d + 'S']
                    eval("self.set_motor_position('{}{}', angles_{}[{}], offset=offset)".format(s,d,d,i))
            self.set_motor_position('ShoulderR', arm_r)
            self.set_motor_position('ShoulderL', arm_l)
            
            # walk time control
            self.walker.timer += self.walker.dt
            if self.walker.timer >= self.walker.param_time['period_time']:
                self.walker.timer = 0
                self.walker.period_counter += 1
        else:
            self.gait_manager.step(self.time_step)

        return True

