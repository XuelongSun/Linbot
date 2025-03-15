from os import walk
# from matplotlib.pyplot import flag
import numpy as np
import cv2
import math
from numpy.lib.function_base import select
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager

from types import prepare_class
from numpy.lib.function_base import average, select
from controller import Motion, Robot
#from managers import RobotisOp2GaitManager, RobotisOp2MotionManager


class LevelsHandle:
    def __init__(self, robot, debug_data=None):
        self.robot = robot

        self.level_names = {
            0: 'start',
            1: 'door',
            2: 'trap',
            3: 'stairs',
            4: 'kick_ball',
            5: 'H_bar',
            6: 'mine',
            7: 'bridge',
            8: 'end',
            None:None
        }
        
        # HSV ranges for color recognition
        self.floor_hsv_range = {
            'blue step':[(103,113,122),(111,184,215)],
            'green trap':[(62,77,200),(70,100,255)],
            'red step':[(0,200,222),(10,255,240)],
            'brick':[(12,80,172),(25,190,215)],
            'white tile':[(11,8,220),(25,21,231)],
            'gray floor':[(30,4,0),(45,10,210)],
            'end_floor':[(100,0,109),(147,73,176)],
            'start_floor':[(35,75,60),(57,200,255)],
            'green bridge/step':[(62,20,104),(74,110,255)],
        }
        self.floor_with_shadow_hsv_range = {
            'blue step':[(70,140,84),(112,255,100)],
            'green trap':[(62,77,28),(70,100,50)],
            'red step':[(0,10,0),(10,255,70)],
            'brick':[(12,26,0),(57,105,70)],
            'white tile':[(17,52,76),(125,119,152)],
            'gray floor':[(100,50,60),(117,103,249)],
            'end_floor':[(107,89,38),(118,138,77)],
            'start_floor':[(56,46,38),(59,133,161)],
            'green bridge/step':[(80,63,50),(92,160,70)],
        }       
        self.object_hsv_range = {
            'yellow bar':[(20,160,20),(30,255,255)], 
            'blue H-bar/hole':[(118,136,0),(140,255,255)],
            # 'ball':[(45,171,95),(113,255,172)],
            'ball':[(105,116,44),(120,238,176)],
            # 'ball':[(105,180,91),(120,238,176)], # this can get rid of the blue_step
            'orange AD board':[(0,80,0),(75,190,226)],
            'hole':[(117,146,234),(123,245,243)],
            'door/mine':[(0,0,0),(0,0,10)], # add hsv range of the black door/mine
        }
        
        self.object_with_shadow_hsv_range = {
            'blue H-bar/hole':[(118,136,50),(140,255,110)],
            'ball':[(100,138,50),(123,255,80)],
        }
                
        self.ground_hsv_range = [(2,0,73),(14,164,255)]
        
        # level state
        self.current_level = None  # None means unknown, number correspond to level names
        self.current_level_floor = None
        self.next_level_floor = None
        self.passed_level_start = False
        self.passed_level_end = False
        # data stored for analysis
        self.data = debug_data
        
        # levels handle global data
        self.handle_global_data = {}
        for k,v in self.level_names.items():
            self.handle_global_data[v] = {}

        self.handle_global_data['kick_ball']['con_angle'] = 0
        self.handle_global_data['kick_ball']['head_pitch'] = 0
        self.handle_global_data['kick_ball']['head_yaw'] = 0
        self.handle_global_data['kick_ball']['status'] = 'firstCall'
        self.handle_global_data['kick_ball']['ball_img_size'] = 0
        self.handle_global_data['kick_ball']['hole_img_size'] = 0
        self.handle_global_data['kick_ball']['robot_time'] = 0
        self.handle_global_data['kick_ball']['wait_time'] = 0
        self.handle_global_data['kick_ball']['head_ctrl'] = False
        self.handle_global_data['kick_ball']['walk_ctrl'] = False
        self.handle_global_data['kick_ball']['isInPlanedPath'] = False
        self.handle_global_data['kick_ball']['end_turn_dir'] = 0

        self.handle_global_data['door']['prepared'] = 'ZeroStage'
        self.handle_global_data['door']['forward_len'] = 0
        self.handle_global_data['door']['counter_len'] = 0
        self.handle_global_data['door']['State'] = 'Adjusting'
        self.handle_global_data['door']['bar_appear'] = 'empty'
        
        self.handle_global_data['end']['GateFlag'] = 0
        self.handle_global_data['end']['head_pitch'] = 30
        self.handle_global_data['end']['phase'] = 0
        
        self.handle_global_data['stairs']['StartFlag'] = 0
        self.handle_global_data['stairs']['Hist_H_Threshold'] = 100
        self.handle_global_data['stairs']['Hist_V_Threshold'] = 50        
        self.handle_global_data['stairs']['UpStair_cnt'] = 0
        self.handle_global_data['stairs']['Head_Pitch'] = 0.1
        self.handle_global_data['stairs']['end'] = False

        self.handle_global_data['H_bar']['StartFlag'] = 0
        self.handle_global_data['H_bar']['Hist_H_Threshold'] = 100
        self.handle_global_data['H_bar']['Hist_V_Threshold'] = 24
        self.handle_global_data['H_bar']['Head_Pitch'] = 0.1

        self.handle_global_data['bridge']['forward_len'] = 0

    def My_Limit(self,value,min,max):
        if value > max:
            return max
        elif value < min:
            return min
        else:
            return value
    
    def slow_walk(self, walk_time):
        while walk_time >1:
            walk_time -= self.robot.time_step
            self.robot.gait_manager.step(self.robot.time_step)
            self.robot.my_step()

    def headdown_stop (self, robot):
        robot.gait_manager.stop()
        robot.set_head_pitch(np.rad2deg(-0.36))
        self.robot.set_head_yaw(0)
        robot.wait(500)
    
    def Clear_Level_Flags_Stair(self):
        self.handle_global_data['stairs']['StartFlag'] = 0
        self.handle_global_data['stairs']['Hist_H_Threshold'] = 100
        self.handle_global_data['stairs']['Hist_V_Threshold'] = 50
        self.handle_global_data['stairs']['UpStair_cnt'] = 0
        self.handle_global_data['stairs']['Head_Pitch'] = 0.1
        self.handle_global_data['stairs']['end'] = True
        # if self.robot.gait_manager.is
        # self.robot.gait_manager.start()

    def Calc_Histogram (self, img_filtered,level):
        isEmpty = False
        (w,h) = img_filtered.shape
        Histgram_Vertical = np.zeros(w)
        Histgram_Horizontal = np.zeros(h)
        for row in range(h):
            for col in range(w):
                if img_filtered[col][row] > 1:
                    Histgram_Horizontal [row] += 1
        for col in range(w):
            for row in range(h):
                if img_filtered[col][row] >1:
                    Histgram_Vertical [col] += 1

        if level == 'stairs':
            Array_hor = np.where(Histgram_Horizontal>self.handle_global_data['stairs']['Hist_H_Threshold'])
            Array_ver = np.where(Histgram_Vertical>self.handle_global_data['stairs']['Hist_V_Threshold'])
        elif level == 'H_bar':
            Array_hor = np.where(Histgram_Horizontal>self.handle_global_data['H_bar']['Hist_H_Threshold'])
            Array_ver = np.where(Histgram_Vertical>self.handle_global_data['H_bar']['Hist_V_Threshold'])

        if len(Array_hor[0])==0:    #exclude the none case
            Average_H = 0
            Average_V = 0
            isEmpty = True
            # raise Exception('Cannot find the stair/H-bar')
            print('Cannot find the stair/H-bar')
            self.handle_global_data['stairs']['StartFlag'] = 0
            self.handle_global_data['H_bar']['StartFlag'] = 0    
            return (Average_H,Average_V,isEmpty)
        else:
            Average_H = average(Array_hor)  # reference to y index of center point
        if len(Array_ver[0])==0:
            Average_V=0
            Average_V = 0
            isEmpty = True
            # raise Exception('Cannot find the stair/H-bar')
            print('Cannot find the stair/H-bar')
            self.handle_global_data['stairs']['StartFlag'] = 0 
            self.handle_global_data['H_bar']['StartFlag'] = 0 
            return (Average_H,Average_V,isEmpty)
        else:
            Average_V = average(Array_ver)  # reference to x index of center point

        return (Average_H,Average_V,isEmpty)
        
    def levels_recognition(self):
        current_level = None
        img = self.robot.get_img(hsv=True)
        floor_k = None
        kernel = np.ones((5, 5), np.uint8)
        # recognize the current level
        print('acc_value: ', self.robot.inertial_sensors['Accelerometer'].getValues()[1])
        for k,v in self.floor_hsv_range.items():
            img_filtered = cv2.inRange(img, v[0], v[1])
            pixel_size = cv2.countNonZero(img_filtered)
            floor_area = np.where(img_filtered > 1)
            if pixel_size > 1000:
                if floor_area[1].max() >= img_filtered.shape[1]-30:
                    floor_k = k
                    img_floor = img_filtered.copy()
                    if (k == 'blue step') and (not self.handle_global_data['stairs']['end']):
                        # stairs and slop
                        if self.robot.inertial_sensors['Accelerometer'].getValues()[1] > 600:
                            break
                        else:
                            current_level = 3
                            break
                    if k == 'green trap':
                        # trap
                        current_level = 2
                        break
                    if (k == 'green bridge/step') and (self.current_level != 3):
                        # bridge
                        current_level = 7
                        break
                    if k == 'end_floor':
                        # end
                        current_level = 8
                        break
                    if k in ['brick', 'white tile', 'gray floor']:
                        # various levels, need further recognize
                        ball_range = self.object_hsv_range['ball']
                        # ball_range_shadow = self.object_with_shadow_hsv_range['ball']
                        img_filtered_ball = cv2.inRange(img, ball_range[0], ball_range[1])
                        # img_filtered_ball_s = cv2.inRange(img, ball_range_shadow[0], ball_range_shadow[1])
                        # img_filtered_ball = cv2.bitwise_or(img_filtered_ball, img_filtered_ball_s)
                        img_filtered_ball = cv2.morphologyEx(img_filtered_ball, cv2.MORPH_OPEN, kernel)
                        pixel_size_ball = cv2.countNonZero(img_filtered_ball)
                        if  pixel_size_ball > 60:
                            # kick-ball
                            if self.handle_global_data['kick_ball']['status'] != 'kickBallCplt':
                                # Identify if it is a staircase
                                stair_flag = False
                                bule_step_hsv = self.floor_hsv_range['blue step']
                                img_filtered = cv2.inRange(img, bule_step_hsv[0],bule_step_hsv[1])
                                pixel_size = cv2.countNonZero(img_filtered)
                                if pixel_size > 1000:
                                    # it may be staircase
                                    stair_flag = True

                                contours_ball, hierarchy = cv2.findContours(img_filtered_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                                for c in contours_ball:
                                    if len(c) > 10:
                                        # roundness check
                                        p = cv2.arcLength(c,True)
                                        s = cv2.contourArea(c)
                                        roundness = 4 * np.pi * s/ (p ** 2)
                                        print('ball_roundness:', roundness)
                                        if roundness >= 0.8 and (not stair_flag):
                                            current_level = 4
                                            self.handle_global_data['kick_ball']['floor'] = k
                                            break
                                if current_level is not None:
                                    break
                        else:
                            darkblue_range = self.object_hsv_range['blue H-bar/hole']
                            img_filtered_hole = cv2.inRange(img, darkblue_range[0], 
                                                            darkblue_range[1])
                            pixel_size_hole = cv2.countNonZero(img_filtered_hole)
                            if pixel_size_hole > 500:
                                x = np.where(img_filtered_hole > 1)[0]
                                y = np.where(img_filtered_hole > 1)[1]
                                diff = (y.max()-y.min()) - (x.max()-x.min())
                                if abs(diff) > 50:
                                    # H_bar
                                    current_level = 5
                                    self.handle_global_data['H_bar']['floor'] = k
                                    break
                                else:
                                    current_level = None
                                    break
                            else:
                                # mine or door
                                # img_rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)    
                                # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)   
                                # ret, binary = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
                                door_mine_range = self.object_hsv_range['door/mine']
                                img_filtered_black = cv2.inRange(img, door_mine_range[0], 
                                                                 door_mine_range[1])
                                contours, h = cv2.findContours(img_filtered_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                                if len(contours) != 0:
                                    if len(contours) > 3:
                                        break
                                    area = []
                                    rounds = []
                                    for c in contours:
                                        s = cv2.contourArea(c)
                                        area.append(s)
                                        p = cv2.arcLength(c,False)
                                        r = 4 * np.pi * s/ (p ** 2) if p!=0 else 0
                                        rounds.append(r)
                                        # x_width.append(c[:,:,1].max() - c[:,:,1].min())
                                    # print('black bar width:', x_width)
                                    print('black area:', area)
                                    print('black rounds:', rounds)
                                    if np.array(area).max() < 200:
                                        # mine
                                        current_level = 6
                                        self.handle_global_data['mine']['floor'] = k
                                        break
                                    else:
                                        # get rid of hole by checking if there is the blue edge
                                        darkblue_range = self.object_hsv_range['blue H-bar/hole']
                                        img_filtered_hole = cv2.inRange(img, darkblue_range[0], 
                                                                        darkblue_range[1])
                                        pixel_size_hole = cv2.countNonZero(img_filtered_hole)
                                        if pixel_size_hole>30:
                                            # black hole
                                            break
                                        else:
                                            # door
                                            current_level = 1
                                            self.handle_global_data['door']['floor'] = k
                                            break       
                else:
                    current_level = None        
            else:
                current_level = None
            
        self.current_level_floor = floor_k
        # recognize the next level floor if forseeable
        if self.current_level_floor is not None:
            self.next_level_recognition(img, img_floor)
        else:
            self.next_level_floor = None     
        return current_level

    def next_level_recognition(self, img, img_floor):
        # predict the next level if it is forseeable
        # should based on the current level recognition
        floor_area = np.where(img_floor > 1)
        if len(floor_area[1]) > 0:
            if floor_area[1].min() > 40:
                # recognize next level floor
                for k,v in self.floor_hsv_range.items():
                    # trap and 
                    if k != self.current_level_floor:
                        img_filtered = cv2.inRange(img, v[0], v[1])
                        pixel_size = cv2.countNonZero(img_filtered)
                        if pixel_size > 800:
                            # found the floor for the next level
                            self.next_level_floor = k
                            break
                        else:
                            self.next_level_floor = None            
            else:
                # still mainly in the current level
                self.next_level_floor = 'invisible'
        else:
            self.next_level_floor = None
        return self.next_level_floor
            
    def handle_level_start(self):
        # for the starting level, just going forward
        x_offset = 0
        step_x = 1
        # if found next level floor, then stop
        rec_level = self.levels_recognition()
        if self.next_level_floor not in [None, 'invisible']:
            self.current_level = rec_level
            print('passed start level, go on...^V^')
            print('next level floor is %s' % self.next_level_floor)
            self.passed_level_start = True
        else:
            # assign step_x and x_offset to gait model
            if self.robot.use_self_gait:
                self.robot.walker.param_move['x_move_amp'] = step_x*self.robot.walk_speed
                self.robot.walker.param_move['a_move_amp'] = -x_offset/2
            else:

                self.robot.gait_manager.setXAmplitude(step_x)
                self.robot.gait_manager.setAAmplitude(x_offset)
            self.passed_level_start = False
        return self.passed_level_start
        
    def follow_floor(self, floor):
        img = self.robot.get_img()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # floor = 'white tile'
        binary_floor = cv2.inRange(img_hsv, self.floor_hsv_range[floor][0], 
                                    self.floor_hsv_range[floor][1])
        binary_floor_shadow = cv2.inRange(img_hsv, self.floor_with_shadow_hsv_range[floor][0], 
                                            self.floor_with_shadow_hsv_range[floor][1])
        binary_floor = cv2.bitwise_or(binary_floor, binary_floor_shadow)
        floor_area = np.where(binary_floor > 1)
        if len(floor_area[0]) > 0:
            y_min = floor_area[1].min()
            y_max = np.min([y_min + 30, floor_area[1].max()])
            floor_area = np.where(binary_floor[:,y_min:y_max] > 1)
            if len(floor_area[0]) > 0:
                x = np.mean(floor_area[0])
        else:
            x = 0
            
        x_offset = (binary_floor.shape[0]/2 - x)/binary_floor.shape[0]
        step_x = 0 if abs(x_offset) > 0.3 else (1 - x_offset)
        # assign step_x and x_offset to gait model
        if self.robot.use_self_gait:
            self.robot.walker.param_move['x_move_amp'] = step_x*self.robot.speed
            self.robot.walker.param_move['a_move_amp'] = x_offset
        else:
            self.robot.gait_manager.setXAmplitude(step_x)
            self.robot.gait_manager.setAAmplitude(x_offset)
    
    # def handle_level_door(self):
    #     default_value = 3
    #     if self.handle_global_data['door']['prepared']=='ZeroStage':
    #         self.robot.gait_manager.setXAmplitude(-0.6)
    #         print('退后3步抬高视野')
    #         for i in range(400):
    #             self.robot.walk_step()
    #             self.robot.my_step()
                
    #         self.robot.gait_manager.stop()
    #         self.robot.wait(1500)
    #         contours = []
    #         head_yaw = -50
    #         while self.handle_global_data['door']['counter_len'] != 2:
    #             # scan until found two vertical bar
    #             self.robot.set_head_pitch(np.rad2deg(0.36))
    #             self.robot.set_head_yaw(head_yaw)
    #             self.robot.wait(200)
    #             head_yaw += 5
    #             img = self.robot.get_img()
    #             img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    #             ret, binary_door = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
    #             contours, h = cv2.findContours(binary_door, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    #             self.handle_global_data['door']['counter_len'] = len(contours)
    #             for c in contours:  #排除背景墙上的黑点干扰。
    #                 s = cv2.contourArea(c)
    #                 print('Test，轮廓面积:',s)
    #                 if s <200:
    #                     self.handle_global_data['door']['counter_len'] -= 1
    #             print('scanning: yaw:{}, got contours {}'.format(head_yaw, self.handle_global_data['door']['counter_len']))

    #             if head_yaw>50: #如果被某些奇怪因素误导，然后找不到door，让程序返回。
    #                 self.robot.gait_manager.start()
    #                 return
    #         # found two bar then adjust robot's position
    #         # self.handle_global_data['door']['head_pitch'] = np.rad2deg(self.robot.position_sensors['HeadS'].getValue())
    #         self.handle_global_data['door']['head_yaw'] = np.rad2deg(self.robot.position_sensors['NeckS'].getValue())
    #         print('good head yaw:', self.handle_global_data['door']['head_yaw']+default_value)
    #         if abs(self.handle_global_data['door']['head_yaw']+default_value)>0: #角度比较大
    #             A_Amplitute = ((self.handle_global_data['door']['head_yaw'])+default_value) *0.01
    #             A_Amplitute = self.My_Limit(A_Amplitute,-0.2,0.2)
    #             self.robot.gait_manager.setXAmplitude(0)
    #             self.robot.gait_manager.setAAmplitude(A_Amplitute)
    #             self.robot.gait_manager.start()
    #             extra_loop = 250    #用这个参数控制旋转时间

    #             while extra_loop >=0:
    #                 self.robot.walk_step()
    #                 self.robot.my_step()
    #                 extra_loop -= 1
    #                 if extra_loop< 5:
    #                     print('角度调整完毕')
    #             self.robot.set_head_pitch(np.rad2deg(0))
    #             self.robot.set_head_yaw(np.rad2deg(0))   #预估角度调整结束，头部回正。       
    #             # return  #退出返回再扫描一次。
    #         else:
    #             pass
    #         self.robot.gait_manager.stop()
    #         self.handle_global_data['door']['prepared'] = 'Angle_Prepared'
    #         self.robot.set_head_pitch(np.rad2deg(-0.36))
    #         self.robot.set_head_yaw(np.rad2deg(-0.6))   #侧头
    #         self.robot.set_motor_position('ArmUpperR', np.deg2rad(-0.68))
    #         self.robot.wait(2000)
    #     while (self.handle_global_data['door']['prepared'] == 'Angle_Prepared'):
    #         print('位置调整ing！')
    #         self.robot.wait(1000)
    #         img_hsv = self.robot.get_img(hsv=True)
    #         floor = self.handle_global_data['door']['floor'] 
    #         binary_floor = cv2.inRange(img_hsv, self.floor_hsv_range[floor][0], 
    #                                     self.floor_hsv_range[floor][1])
    #         floor_area = np.where(binary_floor > 1)
    #         x_max = floor_area[0].max()
    #         if x_max - 156 >= 3:   #侧头判断地板位置和机器人位置。
    #             self.robot.StepRight_small_1.play()
    #             continue
    #         elif x_max - 156<-3:
    #             self.robot.StepLeft_small_1.play()
    #             continue

    #         self.robot.set_head_pitch(0)
    #         self.robot.set_head_yaw(0)  #姿态归位
    #         self.robot.wait(100)
    #         self.handle_global_data['door']['prepared'] = 'Position_Prepared'
    #         print('Test，位置调整完毕')
    #         self.robot.gait_manager.start()
    #         return
    #     if self.handle_global_data['door']['prepared'] == 'Position_Prepared':   #开始精调
    #         # self.robot.set_head_pitch(self.handle_global_data['door']['head_pitch'])
    #         img = self.robot.get_img()
    #         img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    #         ret, binary_door = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
    #         contours, h = cv2.findContours(binary_door, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    #         img_p2 = img.copy()
    #         door_y_mean = 0
    #         door_x_mean = 0
            
    #         #* get area if too large means the robot is two closed to the bar then go backword
    #         area = []
    #         for c in contours:
    #             area.append(cv2.contourArea(c))
    #         if max(area) > 500:
    #             step_x = -0.5
    #             x_offset = (np.random.rand(1)[0]-0.5) * 0.1
    #         else:                   
    #             if len(contours) == 2:
    #                 # found two bar
    #                 x1 = contours[0][:,0,1]
    #                 x2 = contours[1][:,0,1]
    #                 x = (x1.mean() + x2.mean())/2
    #                 # for debug
    #                 img_p2 = cv2.line(img_p2, (0,int(x)),(img.shape[1], int(x)), (255,0,0), 1)
    #             else:
    #                 # follow the floor
    #                 img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #                 floor = self.handle_global_data['door']['floor'] 
    #                 binary_floor = cv2.inRange(img_hsv, self.floor_hsv_range[floor][0], 
    #                                             self.floor_hsv_range[floor][1])
    #                 binary_floor_shadow = cv2.inRange(img_hsv, self.floor_with_shadow_hsv_range[floor][0], 
    #                                                 self.floor_with_shadow_hsv_range[floor][1])
    #                 binary_floor = cv2.bitwise_or(binary_floor, binary_floor_shadow)
    #                 floor_area = np.where(binary_floor > 1)
                    
    #                 if len(floor_area[0]) > 0:
    #                     y_min = floor_area[1].min()
    #                     y_max = np.min([y_min + 30, floor_area[1].max()])
    #                     if len(contours) == 1:
    #                         floor_area = np.where(binary_floor[:,y_min:y_max] > 1)
    #                         if len(floor_area[0]) > 0:
    #                             left_right_diff = (binary_floor.shape[0] - floor_area[0].max()) - floor_area[0].min()
    #                             if left_right_diff > 10:
    #                                 # floor is more on the left
    #                                 x_min = 0
    #                                 x_max = contours[0][:,:,1].min()
    #                             elif left_right_diff < -10:
    #                                 # floor is more on the right
    #                                 x_min = contours[0][:,:,1].max()
    #                                 x_max = binary_floor.shape[0]
    #                             else:
    #                                 # then use the mine to define the left/right
    #                                 if contours[0][:,:,1].mean() > binary_floor.shape[0]/2:
    #                                     # mine on the right
    #                                     x_min = 0
    #                                     x_max = contours[0][:,:,1].min()
    #                                 else:
    #                                     # mine on the left
    #                                     x_min = contours[0][:,:,1].max()
    #                                     x_max = binary_floor.shape[0]
    #                         else:
    #                             x_min, x_max = 0, binary_floor.shape[0]
    #                         door_y_mean = contours[0][:,:,0].mean()
    #                         door_x_mean = contours[0][:,:,1].mean()
    #                     else:
    #                         x_min, x_max = 0, binary_floor.shape[0]
    #                         door_y_mean = 0
    #                         door_x_mean = 0
    #                     img_p2 = cv2.rectangle(img_p2, (y_min,x_min),(y_max,x_max), (0,255,0), 1)
    #                     floor_area = np.where(binary_floor[x_min:x_max,y_min:y_max] > 1)
    #                     if len(floor_area[0]) > 0:
    #                         x = np.mean(floor_area[0]) + x_min
    #                     else:
    #                         x = img_gray.shape[0]/2
    #                 else:
    #                     x = img_gray.shape[0]/2
                
    #             x_offset = (img_gray.shape[0]/2 - x)/img_gray.shape[0]
    #             if (door_y_mean > img_gray.shape[1]/2) and (abs(door_x_mean-img_gray.shape[0]/2)<30):
    #                 step_x = -1.0
    #                 print('DANGER',door_x_mean, door_y_mean)
    #             else:
    #                 # step_x = 1.0 - abs(x_offset)*3
    #                 step_x = 1 - abs(x_offset) if abs(x_offset) < 0.15 else 0
    #             self.handle_global_data['door']['forward_len'] += step_x

    #         # assign step_x and x_offset to gait model
    #         if self.robot.use_self_gait:
    #             self.robot.walker.param_move['x_move_amp'] = step_x*self.robot.walk_speed
    #             self.robot.walker.param_move['a_move_amp'] = -x_offset/2
    #         else:
    #             self.robot.gait_manager.setXAmplitude(step_x*0.8)
    #             self.robot.gait_manager.setAAmplitude(x_offset)
    #             if step_x < 0:
    #                 # two narrow space for turn, should move for extra gait loops
    #                 extra_loop = 75 + int(abs(step_x) * 8)
    #                 while extra_loop:
    #                     print('Door: running extra gait loop- %s left'%extra_loop)
    #                     self.robot.walk_step()
    #                     self.robot.my_step()
    #                     extra_loop -= 1
    
    def handle_level_door(self):
        def soccer_walk_old(walk_time, robot = self.robot): #借用soccer关卡中旋转90°的函数。
            while walk_time >1:
                walk_time -= robot.time_step
                robot.gait_manager.step(robot.time_step)
                robot.my_step()
        def soccer_turn(robot = self.robot, turn_angle = 90):
            #angle of a step
            unit_angle = 30
            robot.gait_manager.setXAmplitude(0)
            robot.gait_manager.setYAmplitude(0)
            robot.gait_manager.start()
            A = turn_angle/unit_angle - int(turn_angle/unit_angle)
            if abs(A) > 0.1:
                robot.gait_manager.setAAmplitude(A)
                soccer_walk_old(600)
            
            step = abs(int(turn_angle/unit_angle))
            if step >= 1:
                if turn_angle > 0:
                    robot.gait_manager.setAAmplitude(1)
                else:
                    robot.gait_manager.setAAmplitude(-1)
                soccer_walk_old(step*600)
            robot.gait_manager.stop()
            soccer_walk_old(1200)
        
        self.robot.gait_manager.setXAmplitude(-0.6)
        print('退后3步抬高视野')
        for i in range(150):
            self.robot.walk_step()
            self.robot.my_step()

        self.robot.set_head_pitch(-20)
        #转90°
        soccer_turn(turn_angle = -80)
        self.robot.gait_manager = RobotisOp2GaitManager(self.robot.robot, "config_Y.ini")
        self.robot.gait_manager.start()
        while self.handle_global_data['door']['State'] != 'Completed':
            img_hsv = self.robot.get_img(hsv=True)
            floor = self.handle_global_data['door']['floor'] 
            binary_floor = cv2.inRange(img_hsv, self.floor_hsv_range[floor][0], 
                                        self.floor_hsv_range[floor][1])
            binary_floor_shadow = cv2.inRange(img_hsv, self.floor_with_shadow_hsv_range[floor][0], 
                                            self.floor_with_shadow_hsv_range[floor][1])
            floor_area = cv2.bitwise_or(binary_floor,binary_floor_shadow)
            kernel = np.ones((3, 3), np.uint8)
            floor_area = cv2.morphologyEx(floor_area, cv2.MORPH_ERODE, kernel)
            cv2.imshow('Floor',floor_area)
            cv2.waitKey(1)
            
            floor_area_index = np.where(floor_area > 1)
            
            y_min = floor_area_index[1].min()
            y_max = floor_area_index[1].max()
            x_min = floor_area_index[0].min()
            x_max = floor_area_index[0].max()
            
            print("x:{},{},y:{},{},shape:{}".format(x_min,x_max,y_min, y_max,floor_area.shape))
            side_edge_L = np.where(floor_area[x_min][:] > 1)
            side_edge_R = np.where(floor_area[x_max][:] > 1)
            side_center_L = np.mean(side_edge_L)
            side_center_R = np.mean(side_edge_R)


            print('side_center_L, side_center_R = %s %s'% (side_center_L,side_center_R))
            if side_center_L + side_center_R < 130:
                self.robot.gait_manager.setAAmplitude(0)
                self.robot.gait_manager.setYAmplitude(0)
                self.robot.gait_manager.setXAmplitude(0.5)
                self.robot.gait_manager.start()
                self.slow_walk(1200)
            

            if self.handle_global_data['door']['State'] == 'Adjusting':
                print('调整边线中')
                side_edge_L = np.where(floor_area[x_min][:] > 1)
                side_edge_R = np.where(floor_area[x_max][:] > 1)
                side_center_L = np.mean(side_edge_L)
                side_center_R = np.mean(side_edge_R)
                slope_Zhao = side_center_L - side_center_R    # +10 is an offset angle for the motion requirements.
                dist_Zhao = 86 - np.mean(floor_area[1])
                print ('dist_Zhao,slop_Zhao: %f,%f' %(dist_Zhao,slope_Zhao))
            
                if (abs(slope_Zhao) >15):
                    A_Amplitute = slope_Zhao * 0.02
                    A_Amplitute = self.My_Limit(A_Amplitute,-0.2,0.2)
                    print('Control x,A = %f,%f' %(0,A_Amplitute))
                    
                    self.robot.gait_manager.setXAmplitude(0)
                    self.robot.gait_manager.setYAmplitude(0)
                    self.robot.gait_manager.setAAmplitude(A_Amplitute)
                else:
                    self.handle_global_data['door']['State'] = 'SideWalk'
                    self.robot.set_head_pitch(-15)
            if self.handle_global_data['door']['State'] == 'SideWalk':   #对齐边线了
                
                print('侧步移动中')
                X_diff = 122 - side_center_L - side_center_R
                
                side_edge_L = np.where(floor_area[x_min][:] > 1)
                side_edge_R = np.where(floor_area[x_max][:] > 1)
                side_center_L = np.mean(side_edge_L)
                side_center_R = np.mean(side_edge_R)
                slope_Zhao = side_center_L - side_center_R    # +10 is an offset angle for the motion requirements.
                
                print ('X_diff,slop_Zhao: %f,%f' %(X_diff,slope_Zhao))
                if abs(slope_Zhao) > 3:
                    self.robot.gait_manager.setAAmplitude(0.02 * slope_Zhao)
                else:
                    self.robot.gait_manager.setAAmplitude(0)
                if abs(X_diff) > 3:
                    self.robot.gait_manager.setXAmplitude(self.My_Limit(0.002 * X_diff,-0.02,0.02))
                else:
                    self.robot.gait_manager.setXAmplitude(0)
                    
                self.robot.gait_manager.setYAmplitude(1.0)
                # 找门框
                door_mine_range = self.object_hsv_range['door/mine']
                img_filtered_black = cv2.inRange(img_hsv, door_mine_range[0],
                                                    door_mine_range[1])
                pixel_size_bar = cv2.countNonZero(img_filtered_black)
                print("bar_size = {}".format(pixel_size_bar))
                if (pixel_size_bar) > 500 and (self.handle_global_data['door']['bar_appear'] == 'empty'):
                    self.handle_global_data['door']['bar_appear'] = 'appear'
                elif (pixel_size_bar) < 10 and (self.handle_global_data['door']['bar_appear'] == 'appear'):
                    break
                    
            self.slow_walk(600)     
                    
                    
        soccer_turn(turn_angle = 90)
        self.handle_global_data['door']['State'] = 'Completed'
        print('door completed')
        self.robot.gait_manager = RobotisOp2GaitManager(self.robot.robot, "config_official.ini")
        self.robot.gait_manager.start()
        return 

            
            
            
            

    def handle_level_trap(self):
        self.robot.set_head_pitch(-10)
        img = self.robot.get_img(hsv=True)
        # if found next level floor, use that floor to guide direction
        if self.next_level_floor not in [None, 'invisible']:
            binary_floor = cv2.inRange(img, self.floor_hsv_range[self.next_level_floor][0], 
                                    self.floor_hsv_range[self.next_level_floor][1])
            binary_floor_shadow = cv2.inRange(img, self.floor_with_shadow_hsv_range[self.next_level_floor][0], 
                                            self.floor_with_shadow_hsv_range[self.next_level_floor][1])
            binary_floor = cv2.bitwise_or(binary_floor,binary_floor_shadow)
            floor_area = np.where(binary_floor > 1)
            if len(floor_area[0]) > 0:
                y_min = floor_area[1].min()
                y_max = np.min([y_min + 30, floor_area[1].max()])
                floor_area = np.where(binary_floor[:,y_min:y_max] > 1)
                if len(floor_area[0]) > 0:
                    x = floor_area[0].mean()
                else:
                    x = 0
        else:
            # get green floor
            binary_floor = cv2.inRange(img, self.floor_hsv_range['green trap'][0], 
                                        self.floor_hsv_range['green trap'][1])
            binary_floor_shadow = cv2.inRange(img, self.floor_with_shadow_hsv_range['green trap'][0], 
                                            self.floor_with_shadow_hsv_range['green trap'][1])
            binary_floor = cv2.bitwise_or(binary_floor,binary_floor_shadow)
            # get the trap
            binary_floor_inv = 255 - binary_floor
            # img_gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2GRAY)
            # _, binary_trap_thr = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
            # binary_trap = cv2.bitwise_and(binary_trap_thr, binary_floor_inv)
            contours, h = cv2.findContours(binary_floor_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            # get contours x
            contours_mean_x = []
            trap_on_the_left = None
            if len(contours) > 0:
                for c in contours:
                    contours_mean_x.append(c[:,:,1].mean())
            use_contour = None
            floor_area = np.where(binary_floor > 1)
            if len(floor_area[0]) > 0:
                y_min = floor_area[1].min()
                y_max = np.min([y_min + 30, floor_area[1].max()])
                floor_area = np.where(binary_floor[:,y_min:y_max] > 1)
                if len(floor_area[0]) > 0:
                    left_right_diff = (binary_floor.shape[0] - floor_area[0].max()) - floor_area[0].min()
                    if left_right_diff > 0:
                        # floor is more on the left, trap is on the left
                        use_contour = contours[np.argmin(contours_mean_x)]
                        trap_on_the_left = True
                    else:
                        use_contour = contours[np.argmax(contours_mean_x)]
                        trap_on_the_left = False

            # calculate x_offset and step_x
            if use_contour is not None:
                if use_contour[:,:,1].max() - use_contour[:,:,1].min() >= binary_floor.shape[0] - 5:
                    # if selected contour spans the whole x-axis, then just follow the floor
                    x = np.where(binary_floor>1)[0].mean()
                else:    
                    if trap_on_the_left:
                        if use_contour[:,:,1].mean() < binary_floor_inv.shape[0]/2:
                            # trap on the left
                            rang_min = use_contour[:,:,1].max()
                            rang_max = binary_floor_inv.shape[0]
                        else:
                            # no trap actually
                            rang_min = 0
                            rang_max = 0
                    else:
                        if use_contour[:,:,1].mean() > binary_floor_inv.shape[0]/2:
                            # trap on the right
                            rang_min = 0
                            rang_max = use_contour[:,:,1].min()
                        else:
                            # no trap actually
                            rang_min = 0
                            rang_max = 0
            
                    if (len(floor_area[0])) > 0 and (rang_max > rang_min):
                        floor_area = np.where(binary_floor[rang_min:rang_max,:] > 1)
                        if len(floor_area[0]) > 0:
                            x = np.mean(floor_area[0]) + rang_min
                        else:
                            x = img.shape[0]/2
                    else:
                        # just step forward
                        x = 0
                    
        x_offset = (img.shape[0]/2 - x)/img.shape[0]
        step_x = 0 if abs(x_offset) > 0.2 else (1 - abs(x_offset))
        
        # assign step_x and x_offset to gait model
        if self.robot.use_self_gait:
            self.robot.walker.param_move['x_move_amp'] = step_x*self.robot.speed
            self.robot.walker.param_move['a_move_amp'] = x_offset
        else:
            self.robot.gait_manager.setXAmplitude(step_x)
            self.robot.gait_manager.setAAmplitude(x_offset)
        
    def handle_level_stairs(self):
        # ground color and texture will not change
        print('processing stairs' + str(self.handle_global_data['stairs']['StartFlag']))
        motor_position = self.robot.get_position_sensor_values()
        if (motor_position['HeadS'] > -0.355)|(self.handle_global_data['stairs']['StartFlag']==0): #Head position not down to the limit
            self.handle_global_data['stairs']['Head_Pitch'] -= 0.05
            self.robot.set_head_pitch(np.rad2deg(self.handle_global_data['stairs']['Head_Pitch']))
            motor_position = self.robot.get_position_sensor_values()
            print('head_position%f' %motor_position['HeadS'])

            # self.robot.wait(800)
            img = self.robot.get_img(hsv=True)
            Blue_Stair_1 = self.floor_hsv_range['blue step']
            Blue_Stair_2 = self.floor_with_shadow_hsv_range['blue step']
            img_filtered_1 = cv2.inRange(img, Blue_Stair_1[0], Blue_Stair_1[1])
            img_filtered_2 = cv2.inRange(img, Blue_Stair_2[0], Blue_Stair_2[1])
            img_filtered = cv2.bitwise_or(img_filtered_1,img_filtered_2)

            floor_area = np.where(img_filtered > 1)
            if len(floor_area[0]) > 0:
                x = np.mean(floor_area[0])
                y = np.mean(floor_area[1])
            else:
                x = 0
                y = 0
            print('center (%d,%d)'%(x,y))
            if (y<(img_filtered.shape[1]/2+20)) & (motor_position['HeadS'] > -0.355): # center point still far in the top of the img
                x_offset = (img_filtered.shape[0]/2 - x)/img_filtered.shape[0]
                step_x = 0 if abs(x_offset) > 0.1 else (0.18-abs(x_offset))    
                self.robot.gait_manager.setXAmplitude(step_x)
                self.robot.gait_manager.setAAmplitude(x_offset)
                print('set offsets (%f,%f)'%(x_offset,step_x))
                return  # return is unnecessary, just to remind this will go back to main().
            else: # center point below the middle of the img
                print('center point below the mid')
                self.handle_global_data['stairs']['StartFlag'] = 1
                self.headdown_stop(self.robot)


        while (self.handle_global_data['stairs']['StartFlag'] ==1):
            self.headdown_stop(self.robot)
            self.robot.wait(2000)
            print('Running in the while!')
            img = self.robot.get_img(hsv=True)
            Blue_Stair_1 = self.floor_hsv_range['blue step']
            Blue_Stair_2 = self.floor_with_shadow_hsv_range['blue step']
            img_filtered_1 = cv2.inRange(img, Blue_Stair_1[0], Blue_Stair_1[1])
            img_filtered_2 = cv2.inRange(img, Blue_Stair_2[0], Blue_Stair_2[1])
            img_filtered = cv2.bitwise_or(img_filtered_1,img_filtered_2)
            floor_area = np.where(img_filtered > 1)

            x_min = floor_area[0].min()
            x_max = floor_area[0].max()
            
            array_floor_area = np.array(floor_area)
            right_side_index = np.where(array_floor_area[0][:]==x_max)
            right_side_edge_y = array_floor_area[1][right_side_index]
            mean_y_right = np.mean(right_side_edge_y)

            left_side_index = np.where(array_floor_area[0][:]==x_min)
            left_side_edge_y = array_floor_area[1][left_side_index]
            mean_y_left = np.mean(left_side_edge_y)

            print('left, right = %f,%f'%(mean_y_left,mean_y_right))

            if x_min > 0:
                if mean_y_left - mean_y_right <-15:
                    print('左边太高了')
                    self.robot.gait_manager.setXAmplitude(0.01)
                    self.robot.gait_manager.setAAmplitude(-0.2)
                    self.robot.gait_manager.start()
                    self.slow_walk(600)
                    self.robot.wait(100)
                else:
                    self.robot.play_user_motion_sync(self.robot.StepRight_small)
                continue
            elif x_max < 159:
                if mean_y_left - mean_y_right >15:
                    print('右边太高了')
                    self.robot.gait_manager.setXAmplitude(0.01)
                    self.robot.gait_manager.setAAmplitude(0.2)
                    self.robot.gait_manager.start()
                    self.slow_walk(600)
                    self.robot.wait(100)
                else:
                    self.robot.play_user_motion_sync(self.robot.StepLeft_small)
                continue
            else:
                print('Accurate adjusting until paralleled')
                side_edge_L = np.where(img_filtered[0][:] > 1)
                side_edge_R = np.where(img_filtered[159][:] > 1)
                side_center_L = np.mean(side_edge_L)
                side_center_R = np.mean(side_edge_R)
                slope_Zhao = side_center_L+10 - side_center_R    # +8 is an offset angle for the motion requirements.
                dist_Zhao = 65 - np.mean(floor_area[1])
                print ('dist_Zhao,slop_Zhao: %f,%f' %(dist_Zhao,slope_Zhao))
            
                if (abs(slope_Zhao) >4)|(abs(dist_Zhao)>3.5):
                    A_Amplitute = slope_Zhao * 0.015
                    A_Amplitute = self.My_Limit(A_Amplitute,-0.2,0.2)
                    x_Amplitute = dist_Zhao*0.05
                    x_Amplitute = self.My_Limit(x_Amplitute,-0.3,0.3)
                    # x_Amplitute = 0 if A_Amplitute>0.1 else x_Amplitute
                    print('Control x,A = %f,%f' %(x_Amplitute,A_Amplitute))
                    
                    self.robot.gait_manager.setXAmplitude(x_Amplitute)
                    self.robot.gait_manager.setAAmplitude(A_Amplitute)
                    self.robot.gait_manager.start()
                    self.slow_walk(600)
                    self.robot.wait(100)
                    # self.robot.wait(1800)
                    continue    #change to return for debug
                else:
                    self.robot.wait(500)
                    self.robot.gait_manager.stop()
                    self.robot.wait(500)
                    while(self.handle_global_data['stairs']['UpStair_cnt'] < 5):    # the number is somehow related to 3 times upstair and 2 times downstair.
                        self.handle_global_data['stairs']['UpStair_cnt'] +=1
                        if self.handle_global_data['stairs']['UpStair_cnt'] < 4:
                            self.robot.play_user_motion_sync(self.robot.ClimbStair)
                            self.robot.wait(200)
                            print('Climbing Stairs cnt:%d' % self.handle_global_data['stairs']['UpStair_cnt'])
                        else:
                            # adjust position at peak
                            # if self.handle_global_data['stairs']['UpStair_cnt'] == 4:
                            self.robot.ready_to_walk()
                            self.robot.wait(200)
                            self.robot.gait_manager.setAAmplitude(0.2)
                            self.slow_walk(620)
                            self.robot.gait_manager.stop()
                            self.robot.wait(2000)
                            # self.robot.play_user_motion_sync(self.robot.StepLeft_small)
                            # self.robot.play_user_motion_sync(self.robot.StepLeft_small)
                            self.robot.wait(200)
                            self.robot.play_user_motion_sync(self.robot.DownStair)
                            self.robot.wait(200)
                            print('Going Down Stairs cnt:%d' % self.handle_global_data['stairs']['UpStair_cnt'])
                    self.robot.ready_to_walk()
                    self.robot.wait(200)
                    self.robot.gait_manager.setAAmplitude(0.2)
                    self.robot.gait_manager.setXAmplitude(0)
                    self.robot.gait_manager.setYAmplitude(0)
                    self.slow_walk(600)
                    self.robot.gait_manager.setXAmplitude(1.0)
                    self.robot.gait_manager.setYAmplitude(0)
                    self.robot.gait_manager.setAAmplitude(0)
                    # self.robot.gait_manager.stop()
                    # self.robot.wait(2000)
                    self.Clear_Level_Flags_Stair()
                    print('Stair finished!')
                    # self.robot.wait(2000)
                    # self.robot.ready_to_walk()
                    return  # finish this level if finished
        
    def handle_level_kick_ball(self):

        # variable define
        ball = {
            'found': False,
            'x': -1,
            'y': -1,
            'x_min': -1,
            'x_max': -1,
            'y_min': -1,
            'y_max': -1,
            'x_central': -1,
            'y_central': -1,
            'x_img_len': -1,
            'size':-1,
            'roundness':-1,
            'perimeter':-1,
            'wld_x':-1,
            'wld_y':-1,
            'wld_z':-1
        }
        
        hole = {
            'found': False,
            'x': -1,
            'y': -1,
            'x_min': -1,
            'x_max': -1,
            'y_min': -1,
            'y_max': -1,
            'x_central': -1,
            'y_central': -1,
            'x_img_len': -1,
            'diameter':-1,
            'wld_x':-1,
            'wld_y':-1,
            'wld_z':-1
        }

        # func define
        def eulerAnglesToRotationMatrix(theta):
            R_x = np.array([[1,         0,                  0                   ],
                            [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                            [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                            ])

            R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                            [0,                     1,      0                   ],
                            [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                            ])
                        
            R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                            [math.sin(theta[2]),    math.cos(theta[2]),     0],
                            [0,                     0,                      1]
                            ])

            R = np.dot(R_z, np.dot( R_y, R_x ))

            return R

        def update_data():
            nonlocal ball, hole
            hole['found'] = False
            ball['found'] = False
            
            img_ori = self.robot.get_img()
            img = cv2.cvtColor(img_ori, cv2.COLOR_RGB2HSV)
            hole_hsv = self.object_hsv_range['hole']
            # ball_hsv = self.object_hsv_range['ball']
            ball_hsv = [(105,116,44),(120,238,176)]
            img_filtered_hole = cv2.inRange(img, hole_hsv[0], hole_hsv[1])
            img_filtered_ball = cv2.inRange(img, ball_hsv[0], ball_hsv[1])
            # get the centre of hole
            contours_hole, hierarchy = cv2.findContours(img_filtered_hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            contours_ball, hierarchy = cv2.findContours(img_filtered_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            motor_position = self.robot.get_position_sensor_values()
            camera_pitch = -motor_position['HeadS'] + np.deg2rad(60)
            camera_yaw = motor_position['NeckS']
            self.handle_global_data['kick_ball']['head_pitch'] = np.rad2deg(motor_position['HeadS'])
            self.handle_global_data['kick_ball']['head_yaw'] = np.rad2deg(motor_position['NeckS'])

            #webots world coordinate
            R_cw = eulerAnglesToRotationMatrix([camera_pitch, camera_yaw, 0])
            camera_roll = np.pi
            R_cwpi = eulerAnglesToRotationMatrix([0, 0, camera_roll])
            Trans_cw = [0, 0.35, 0.125]
            
            if True:
                # cv2.namedWindow('img_filtered_hole', cv2.WINDOW_NORMAL)
                # cv2.namedWindow('img_filtered_ball', cv2.WINDOW_NORMAL)
                # cv2.namedWindow('img_mark', cv2.WINDOW_NORMAL)

                img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)
                
                img_specifiedBall=cv2.bitwise_and(img_ori,img_ori,mask=img_filtered_ball)
                img_specifiedHole=cv2.bitwise_and(img_ori,img_ori,mask=img_filtered_hole)
                img_ori = cv2.drawContours(img_ori, contours_ball, -1, (0,0,255), 2)
                img_ori = cv2.drawContours(img_ori, contours_hole, -1, (0,255,0), 2)
                img_ori = np.swapaxes(img_ori, 1, 0)
                img_specifiedBall = np.swapaxes(img_specifiedBall, 1, 0)
                img_specifiedHole = np.swapaxes(img_specifiedHole, 1, 0)
                # cv2.imshow('img_filtered_hole', img_specifiedHole)
                # cv2.imshow('img_filtered_ball', img_specifiedBall)
                # cv2.imshow('img_mark', img_ori)
                # cv2.waitKey(1)

            if len(contours_ball) >= 1:
                for contours in contours_ball:
                    if len(contours) > 10:
                        ball['x'] = np.array(contours)[:,0,1]
                        ball['y'] = np.array(contours)[:,0,0]
                        ball['x_central'] = ball['x'].mean()
                        ball['y_central'] = ball['y'].mean()
                        ball['x_min'] = ball['x'].min()
                        ball['y_min'] = ball['y'].min()
                        ball['x_max'] = ball['x'].max()
                        ball['y_max'] = ball['y'].max()
                        ball['x_img_len'] = ball['x_max'] - ball['x_min']
                        # ball['size'] = cv2.countNonZero(img_filtered_ball[ball['x_min']:ball['x_max'],ball['y_min']:ball['y_max']])
                        ball['size'] = cv2.contourArea(contours)
                        ball['perimeter'] = cv2.arcLength(contours,True)
                        if ball['perimeter'] != 0:
                            ball['roundness'] = 4 * np.pi * ball['size']/ (ball['perimeter'] ** 2)
                        # print('ball_size \t%d, \tlen_con_ball \t%d roundness \t%1.3f' % (ball['size'], len(contours_ball), ball['roundness']))
                        # Check ball roundness 
                        if ball['roundness'] > 0.80:
                            ball['found'] = True
                            # print('ball_size \t%d, \tlen_con_ball \t%d roundness \t%1.3f \tx%d \ty%d ' % (ball['size'], len(contours_ball), ball['roundness'], ball['x_central'], ball['y_central']))
                            ball_world_size = 0.04
                            # ball_cam_z = self.robot.fx * ball_world_size / ball_x_img_len
                            # ball_cam_x = ball_cam_z * (ball['x_central'] - self.robot.camera_width/2) / self.robot.fx
                            # ball_cam_y = ball_cam_z * (ball['y_central'] - self.robot.camera_height/2) / self.robot.fy
                            ball_cam_z = self.robot.fx * ball_world_size / ball['perimeter'] * np.pi
                            ball_cam_x = ball_cam_z * (ball['x_central'] - self.robot.camera_width/2) / self.robot.fx
                            ball_cam_y = ball_cam_z * (ball['y_central'] - self.robot.camera_height/2) / self.robot.fy
                            # print('zc\t{:.2f} \txc \t{:.2f} \tyc \t{:.2f}'.format(ball_cam_z, ball_cam_x, ball_cam_y),end = '')
                            
                            ball_w = np.dot([ball_cam_x,ball_cam_y,ball_cam_z],R_cw)
                            ball_w = np.dot(ball_w,R_cwpi)
                            ball_w = ball_w+ Trans_cw
                            #trans form robot world coordinate(direction same with webots world coordinate) to robot control coordinate
                            ball['wld_x'] = ball_w[2]
                            ball['wld_y'] = ball_w[0]
                            ball['wld_z'] = ball_w[1]
                            
                            # print('\tzw\t{:.2f} \txw \t{:.2f} \tyw \t{:.2f}'.format(ball_w[2], ball_w[0], ball_w[1]))
                            
                            break

                    # self.handle_global_data['kick_ball']['ball_img_size'] = self.handle_global_data['kick_ball']['ball_img_size']*3/4 + ball['x_img_len']/4
            else:
                pass

            if len(contours_hole) >= 1:
                for contours in contours_hole:
                    if len(contours) > 10:
                        hole['found'] = True
                        hole['x'] = np.array(contours)[:,0,1]
                        hole['y'] = np.array(contours)[:,0,0]
                        hole['x_central'] = hole['x'].mean()
                        hole['y_central'] = hole['y'].mean()
                        hole['x_min'] = hole['x'].min()
                        hole['y_min'] = hole['y'].min()
                        hole['x_max'] = hole['x'].max()
                        hole['y_max'] = hole['y'].max()
                        hole['x_img_len'] = hole['x_max'] - hole['x_min']

                        if ball['found'] == True:
                            # assume hole_zc = ball_ac, calculate the hole position in the cam coordinate 
                            hole_cam_z = ball_cam_z
                            hole_cam_x = hole_cam_z * (hole['x_central'] - self.robot.camera_width/2) / self.robot.fx
                            hole_cam_y = hole_cam_z * (hole['y_central'] - self.robot.camera_height/2) / self.robot.fy
                            
                            hole_w = np.dot([hole_cam_x,hole_cam_y,hole_cam_z],R_cw)
                            hole_w = np.dot(hole_w,R_cwpi)
                            hole_w = hole_w+ Trans_cw
                            # hole_w[x,y,z] #webots coordinates
                            # this point above, the hole central and the camera is collinear, and the hole['wld_y'] should be 0
                            # hole['wld_z'] = -Trans_cw[1]*(hole_w[2] - Trans_cw[2])/(hole_w[1] - Trans_cw[1]) + Trans_cw[2]
                            # hole['wld_x'] = -Trans_cw[1]*(hole_w[0] - Trans_cw[0])/(hole_w[1] - Trans_cw[1]) + Trans_cw[0]
                            hole['wld_x'] = -Trans_cw[1]*(hole_w[2] - Trans_cw[2])/(hole_w[1] - Trans_cw[1]) + Trans_cw[2]
                            hole['wld_y'] = -Trans_cw[1]*(hole_w[0] - Trans_cw[0])/(hole_w[1] - Trans_cw[1]) + Trans_cw[0]

                        # hole_y, hole_x = int(np.array(contours).mean(axis=0)[0][0]), int(np.array(contours).mean(axis=0)[0][1])
                        self.handle_global_data['kick_ball']['hole_img_size'] = self.handle_global_data['kick_ball']['hole_img_size']*3/4 + hole['x_img_len']/4
                        break
            
            print('hole\t{:.2f} \t{:.2f} \tball\t{:.2f} \t{:.2f}'.format(hole['wld_x'], hole['wld_y'], ball['wld_x'], ball['wld_y']))

        def soccer_walk_old(walk_time, robot = self.robot):
            while walk_time >1:
                walk_time -= robot.time_step
                robot.gait_manager.step(robot.time_step)
                robot.my_step()

        def head_scan(HeadPosition = 0.2, NeckPosition = 0, HeadAmp = 0.8, NeckAmp = 1.4):
            Step = 9
            HeadAngle = list(np.arange((HeadPosition - HeadAmp/2), (HeadPosition + HeadAmp/2)+0.0001, HeadAmp/(Step-1)))
            NeckAngle = list(np.arange((NeckPosition - NeckAmp/2), (NeckPosition + NeckAmp/2)+0.0001, NeckAmp/(Step-1)))
            HeadAngle = HeadAngle[int(Step/2)::-1] + HeadAngle[1::] + HeadAngle[-2:-(int(Step/2)+2):-1] 
            NeckAngle = NeckAngle + NeckAngle[-2::-1]
            
            # Check if overflow
            if (HeadPosition + HeadAmp/2) > 0.94 or (HeadPosition - HeadAmp/2) < -0.36:
                pass
            if (NeckPosition + NeckAmp/2) > 1.81 or (NeckPosition - NeckAmp/2) < -1.81:
                # self.NeckMotor.setPosition(-1.80)
                pass

            for i in range((Step*2-1)):
                
                
                self.robot.wait(200)
                # NeckPosition = self.NeckSensor.getValue()
                # HeadPosition = self.HeadSensor.getValue()
                update_data()

                if ball['found'] == True and hole['found'] == True:
                    # look to the ball and hole, and then break
                    con_head_pitch_p = -0.1
                    con_head_yaw_p = -0.2
                    for i in range(3):
                        target_head_x = (ball['x_central'] + hole['x_central'])/2
                        target_head_y = (ball['y_central'] + hole['y_central'])/2
                        self.handle_global_data['kick_ball']['head_pitch'] = self.handle_global_data['kick_ball']['head_pitch'] + con_head_pitch_p *(target_head_y - self.robot.camera_height/2)
                        self.handle_global_data['kick_ball']['head_yaw'] = self.handle_global_data['kick_ball']['head_yaw'] + con_head_yaw_p * (target_head_x - self.robot.camera_width/2)
                        self.robot.set_head_pitch(self.handle_global_data['kick_ball']['head_pitch'])
                        self.robot.set_head_yaw(self.handle_global_data['kick_ball']['head_yaw'])
                        self.robot.wait(80)
                        update_data()
                    break

                self.handle_global_data['kick_ball']['head_pitch'] = np.rad2deg(HeadAngle[i])
                self.handle_global_data['kick_ball']['head_yaw'] = np.rad2deg(NeckAngle[i])
                self.robot.set_head_pitch(self.handle_global_data['kick_ball']['head_pitch'])
                self.robot.set_head_yaw(self.handle_global_data['kick_ball']['head_yaw'])
        # walk forward, length unit: m
        def soccer_walk(robot = self.robot, walk_len = 0.5):
            # length of a step
            unit_len = 0.043
            step = int(walk_len/unit_len)

            robot.gait_manager.setXAmplitude(1)
            robot.gait_manager.setYAmplitude(0)
            robot.gait_manager.setAAmplitude(0)
            robot.gait_manager.start()

            soccer_walk_old(step*600)

            robot.gait_manager.stop()

            soccer_walk_old(1200)

        # turn_angle unit: degree
        def soccer_turn(robot = self.robot, turn_angle = 90):
            #angle of a step
            unit_angle = 30
            robot.gait_manager.setXAmplitude(0)
            robot.gait_manager.setYAmplitude(0)
            robot.gait_manager.start()
            A = turn_angle/unit_angle - int(turn_angle/unit_angle)
            if abs(A) > 0.1:
                robot.gait_manager.setAAmplitude(A)

                soccer_walk_old(600)
            
            step = abs(int(turn_angle/unit_angle))
            if step >= 1:
                if turn_angle > 0:
                    robot.gait_manager.setAAmplitude(1)
                else:
                    robot.gait_manager.setAAmplitude(-1)

                soccer_walk_old(step*600)

            robot.gait_manager.stop()
            soccer_walk_old(1200)

        def lock_head2soccer():
            if ball['found'] == True and hole['found'] == True:
                    # look to the ball and hole, and then break
                    con_head_pitch_p = -0.3
                    con_head_yaw_p = -0.2
                    
                    target_head_x = (ball['x_central'] + hole['x_central'])/2
                    target_head_y = (ball['y_central'] + hole['y_central'])/2 + 5
                    self.handle_global_data['kick_ball']['head_pitch'] = self.handle_global_data['kick_ball']['head_pitch'] + con_head_pitch_p *(target_head_y - self.robot.camera_height/2)
                    self.handle_global_data['kick_ball']['head_yaw'] = self.handle_global_data['kick_ball']['head_yaw'] + con_head_yaw_p * (target_head_x - self.robot.camera_width/2)
                    if ball['y_central'] > 110:
                        self.handle_global_data['kick_ball']['head_pitch'] -= 3
                    self.robot.set_head_pitch(self.handle_global_data['kick_ball']['head_pitch'])
                    self.robot.set_head_yaw(self.handle_global_data['kick_ball']['head_yaw'])
                    
        def found_failBack2main():
            print('Exit kick_ball with {}'.format(self.handle_global_data['kick_ball']['status']))
            self.handle_global_data['kick_ball']['status'] = 'firstCall'
            self.robot.gait_manager.setXAmplitude(0.5)
            self.robot.gait_manager.setYAmplitude(0)
            self.robot.gait_manager.setAAmplitude(0)
            self.robot.gait_manager.start()
                
        #######################func end###############################
        if self.handle_global_data['kick_ball']['status'] == 'firstCall':
            self.handle_global_data['kick_ball']['head_ctrl'] = True
            self.handle_global_data['kick_ball']['walk_ctrl'] = False
            self.robot.gait_manager.stop()
            soccer_walk_old(1200)
            head_scan() # scan for ball and hole
            if ball['found'] == True and hole['found'] == True:
                self.handle_global_data['kick_ball']['status'] = 'approaching1'
                self.robot.gait_manager.start()
            else:
                self.handle_global_data['kick_ball']['head_pitch'] = np.rad2deg(0)
                self.handle_global_data['kick_ball']['head_yaw'] = np.rad2deg(0)
                self.robot.set_head_pitch(self.handle_global_data['kick_ball']['head_pitch'])
                self.robot.set_head_yaw(self.handle_global_data['kick_ball']['head_yaw'])
                self.robot.gait_manager.setXAmplitude(-0.5)
                self.robot.gait_manager.setYAmplitude(0)
                self.robot.gait_manager.setAAmplitude(0)
                self.robot.gait_manager.start()
                soccer_walk_old(600)
                return 1 
                # not found ball and hole, need to forward or adjust position, return to main func
                # Or rescan accroding the ball central

        update_data() #update the ball and hole data from images

        if self.handle_global_data['kick_ball']['isInPlanedPath'] == True:
            pass
            if self.handle_global_data['kick_ball']['head_ctrl'] == True:
                pass
            if self.handle_global_data['kick_ball']['walk_ctrl'] == True:
                pass

        if self.handle_global_data['kick_ball']['status'] == 'approaching1':
            if ball['found'] == True and hole['found'] == True:
                # target_x = ball['wld_x'] + (ball['wld_x'] - hole['wld_x'])/(1*1.414)
                # target_y = ball['wld_y'] + (ball['wld_y'] - hole['wld_y'])/(1*1.414)
                target_x = ball['wld_x'] + (ball['wld_x'] - hole['wld_x'])*0.6
                target_y = ball['wld_y'] + (ball['wld_y'] - hole['wld_y'])*0.6
                target_ang = np.rad2deg(math.atan(target_y/target_x))
                target_len = ((target_x ** 2) + (target_y ** 2)) ** 0.5
                dis  = (((ball['wld_x'] - hole['wld_x']) ** 2) + ((ball['wld_y'] - hole['wld_y']) ** 2)) ** 0.5
                
                
                if abs(dis- 0.28) < 0.06 and target_len < 0.23:
                        self.handle_global_data['kick_ball']['status'] = 'adjustKickPosition'
                        print('change to adjustKickPosition')
                # if calculate error is too big
                elif abs(dis- 0.28) < 0.04:
                    soccer_turn(turn_angle = target_ang)
                    soccer_walk(walk_len= target_len*4/5)
                    ang_ball2hole = np.rad2deg(math.atan( ((ball['wld_y'] + hole['wld_y'])/2 - target_y)  /((ball['wld_x'] + hole['wld_x'])/2 - target_x) ))
                    soccer_turn(turn_angle = (ang_ball2hole - target_ang)*2/3)
                    if (ang_ball2hole - target_ang) > 0:
                        self.handle_global_data['kick_ball']['end_turn_dir'] = -1
                    else:
                        self.handle_global_data['kick_ball']['end_turn_dir'] = 1
                    print('\ttarget\t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}'.format(target_x, target_y, target_ang, target_len, dis,ang_ball2hole))
                    head_scan()
                else:
                    soccer_turn(turn_angle = target_ang)
                    soccer_walk(walk_len= target_len*4/5)
                    ang_ball2hole = np.rad2deg(math.atan( ((ball['wld_y'] + hole['wld_y'])/2 - target_y*4/5)  /((ball['wld_x'] + hole['wld_x'])/2 - target_x*4/5) ))
                    soccer_turn(turn_angle = (ang_ball2hole - target_ang)*2/3)
                    if (ang_ball2hole - target_ang) > 0:
                        self.handle_global_data['kick_ball']['end_turn_dir'] = -1
                    else:
                        self.handle_global_data['kick_ball']['end_turn_dir'] = 1
                    print('\ttarget\t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}'.format(target_x, target_y, target_ang, target_len, dis,ang_ball2hole))
                    head_scan()

                # self.robot.gait_manager.start()
                # lock_head2soccer()
                # if calculate error is too big
                # if abs(dis- 0.28) > 0.06:
                #     pass
            elif ball['found'] == True:
                found_failBack2main()
                return 0
            elif hole['found'] == True:
                found_failBack2main()
                return 0
            else:
                found_failBack2main()
                return 0
            
        if self.handle_global_data['kick_ball']['status'] == 'adjustKickPosition':
            self.robot.gait_manager = RobotisOp2GaitManager(self.robot.robot, "config_Y.ini")
            while True:
                if ball['found'] == True and hole['found'] == True:
                    ang_ball2hole = np.rad2deg(math.atan((ball['wld_y'] - hole['wld_y'])/(ball['wld_x'] - hole['wld_x'])))
                    dis  = (((ball['wld_x'] - hole['wld_x']) ** 2) + ((ball['wld_y'] - hole['wld_y']) ** 2)) ** 0.5
                    X_diff = ball['wld_x'] - 0.13
                    Y_diff = ball['wld_y'] - (-0.07)
                    ang_diff = ang_ball2hole - 5
                    print('\tkickposition\t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}'.format(ang_ball2hole, dis, X_diff, Y_diff))
                    # if calculate error is too big
                    if abs(dis- 0.28) < 0.06:
                        if abs(ang_diff) <= 1.5 and abs(X_diff) <= 0.02 and abs(Y_diff) <= 0.01:
                            self.handle_global_data['kick_ball']['status'] = 'kick'
                            print('ready to kick ball')
                            self.robot.gait_manager.setXAmplitude(0.5)
                            self.robot.gait_manager.setYAmplitude(0)
                            self.robot.gait_manager.setAAmplitude(0)
                            self.robot.gait_manager.start()
                            soccer_walk_old(600)
                            self.robot.gait_manager.stop()
                            soccer_walk_old(600)
                            # kick
                            self.robot.motion_manager.playPage(12)
                            soccer_walk_old(600)
                            self.gait_manager = RobotisOp2GaitManager(self.robot.robot, "config_official.ini")
                            self.gait_manager.setBalanceEnable(True)
                            if self.handle_global_data['kick_ball']['end_turn_dir'] > 0:
                                soccer_turn(turn_angle = (self.handle_global_data['kick_ball']['end_turn_dir'] * 20))
                            else:
                                soccer_turn(turn_angle = (self.handle_global_data['kick_ball']['end_turn_dir'] * 60))
                            # soccer_walk(robot = self.robot, walk_len = 0.3)
                            self.robot.gait_manager.setXAmplitude(1)
                            self.robot.gait_manager.setYAmplitude(0)
                            self.robot.gait_manager.setAAmplitude(0)
                            # self.robot.ready_to_walk()
                            self.robot.gait_manager.start()
                            self.handle_global_data['kick_ball']['status'] = 'kickBallCplt'
                            # soccer_walk(walk_len= 0.043)
                            return
                        else:
                            self.robot.gait_manager.setXAmplitude(0)
                            self.robot.gait_manager.setYAmplitude(0)
                            self.robot.gait_manager.setAAmplitude(0)
                            self.robot.set_head_pitch(np.rad2deg(-0.2))
                            self.robot.set_head_yaw(0)
                            if abs(ang_diff) > 1.5:
                                self.robot.gait_manager.setAAmplitude(0.01 * ang_diff)

                            if abs(X_diff) > 0.02:
                                self.robot.gait_manager.setXAmplitude(10 * X_diff)
                            
                            if abs(Y_diff) > 0.01:
                                self.robot.gait_manager.setYAmplitude(30 * Y_diff)
                            if abs(Y_diff) > 0.04:
                                self.robot.gait_manager.setYAmplitude(10 * Y_diff)
                            lock_head2soccer()
                            self.robot.gait_manager.start()
                            soccer_walk_old(600)
                            self.robot.gait_manager.stop()
                            soccer_walk_old(1200)
                    else:
                        soccer_turn(turn_angle = self.handle_global_data['kick_ball']['head_yaw'])
                        self.robot.set_head_yaw(0)
                    self.robot.set_head_yaw(0)
                    update_data()
                else:
                    self.robot.gait_manager.setXAmplitude(-0.5)
                    self.robot.gait_manager.setYAmplitude(0)
                    self.robot.gait_manager.setAAmplitude(0)
                    self.robot.gait_manager.start()
                    soccer_walk_old(600)
                    self.robot.gait_manager.stop()
                    soccer_walk_old(1200)
                    head_scan()
        else:
            self.robot.gait_manager.setXAmplitude(1)
            self.robot.gait_manager.setYAmplitude(0)
            self.robot.gait_manager.setAAmplitude(0)
            self.robot.gait_manager.start()
        return 0
        
    def handle_level_H_bar(self):
        print('processing H_bar')
        if self.handle_global_data['H_bar']['StartFlag'] == 'finished':
            print('H-bar level already passed')
            return

        motor_position = self.robot.get_position_sensor_values()
        if self.handle_global_data['H_bar']['StartFlag']==0:
            #Prepare
            self.robot.gait_manager.start()
            self.handle_global_data['H_bar']['Head_Pitch'] -= 0.02
            self.robot.set_head_pitch(np.rad2deg(self.handle_global_data['H_bar']['Head_Pitch']))
            motor_position = self.robot.get_position_sensor_values()
            print('head_position%f' %motor_position['HeadS'])
            img = self.robot.get_img(hsv=True)

            H_bar1 = self.object_hsv_range['blue H-bar/hole']
            H_bar2 = self.object_with_shadow_hsv_range['blue H-bar/hole']
            img_filtered_1 = cv2.inRange(img, H_bar1[0], H_bar1[1])
            img_filtered_2 = cv2.inRange(img, H_bar2[0], H_bar2[1])
            img_filtered = cv2.bitwise_or(img_filtered_1,img_filtered_2)

            floor_area = np.where(img_filtered > 1)   
            if len(floor_area[0]) > 0:
                x = np.mean(floor_area[0])
                y = np.mean(floor_area[1])
            else:
                x = 0
                y = 0
            print('center (%d,%d)'%(x,y))
            if (y<(img_filtered.shape[1]/2+15)): # center point still far in the top of the img
                x_offset = (img_filtered.shape[0]/2 - x)/img_filtered.shape[0]
                step_x = 0 if abs(x_offset) > 0.1 else (0.2-abs(x_offset)) 
                self.robot.gait_manager.setAAmplitude(x_offset)
                print('set offsets (%f,%f)'%(x_offset,step_x))
                return  
            else: # center point below the middle of the img
                print('center point below the mid')
                self.robot.gait_manager.stop()
                self.headdown_stop(self.robot)
                self.handle_global_data['H_bar']['StartFlag'] =1
        #Head position already down to the limit

        while (self.handle_global_data['H_bar']['StartFlag'] == 1) :  # Histogram control for accurate parallelizing
            print('running in the while')
            self.headdown_stop(self.robot)
            self.robot.wait(200)
            img = self.robot.get_img(hsv=True)
            H_bar1 = self.object_hsv_range['blue H-bar/hole']
            H_bar2 = self.object_with_shadow_hsv_range['blue H-bar/hole']
            img_filtered_1 = cv2.inRange(img, H_bar1[0], H_bar1[1])
            img_filtered_2 = cv2.inRange(img, H_bar2[0], H_bar2[1])
            img_filtered = cv2.bitwise_or(img_filtered_1,img_filtered_2)
            floor_area = np.where(img_filtered > 1)
            x_min = floor_area[0].min()
            x_max = floor_area[0].max()

            if x_min > 0:
                self.robot.play_user_motion_sync(self.robot.StepRight_small)
                continue
            elif x_max < 159:
                self.robot.play_user_motion_sync(self.robot.StepLeft_small)
                continue
            else:
                side_edge_L = np.where(img_filtered[0][:] > 1)
                side_edge_R = np.where(img_filtered[159][:] > 1)
                side_center_L = np.mean(side_edge_L)
                side_center_R = np.mean(side_edge_R)
                slope_Zhao = side_center_L - side_center_R    # +10 is an offset angle for the motion requirements.
                dist_Zhao = 84 - np.mean(floor_area[1])
                print ('dist_Zhao,slop_Zhao: %f,%f' %(dist_Zhao,slope_Zhao))
            
                if (abs(slope_Zhao) >7)|(abs(dist_Zhao)>2):
                    A_Amplitute = slope_Zhao * 0.02
                    A_Amplitute = self.My_Limit(A_Amplitute,-0.2,0.2)
                    x_Amplitute = dist_Zhao*0.05
                    x_Amplitute = self.My_Limit(x_Amplitute,-0.2,0.2)
                    # x_Amplitute = 0 if A_Amplitute>0.1 else x_Amplitute
                    print('Control x,A = %f,%f' %(x_Amplitute,A_Amplitute))
                    
                    self.robot.gait_manager.setXAmplitude(x_Amplitute)
                    self.robot.gait_manager.setAAmplitude(A_Amplitute)
                    self.robot.gait_manager.start()
                    self.slow_walk(1200)
                    self.robot.wait(1800)
                    continue    #change to return for debug
                else:
                    self.handle_global_data['H_bar']['StartFlag'] = 2   #角度对齐，下一步确认是否在中间位置
                    continue
        self.robot.set_head_yaw(np.rad2deg(-0.7))
        print('头右偏，看两边')
        while (self.handle_global_data['H_bar']['StartFlag'] == 2) :  #确认机器人是否在横板中间。
            self.robot.wait(200)
            img = self.robot.get_img(hsv=True)
            H_bar1 = self.object_hsv_range['blue H-bar/hole']
            H_bar2 = self.object_with_shadow_hsv_range['blue H-bar/hole']
            img_filtered_1 = cv2.inRange(img, H_bar1[0], H_bar1[1])
            img_filtered_2 = cv2.inRange(img, H_bar2[0], H_bar2[1])
            img_filtered = cv2.bitwise_or(img_filtered_1,img_filtered_2)
            floor_area = np.where(img_filtered > 1)
            # x_min = floor_area[0].min()
            x_max = floor_area[0].max()

            if x_max - 155 > 3:   #侧头判断横板位置和机器人位置。
                self.robot.play_user_motion_sync(self.robot.StepRight_small)
                continue
            elif x_max - 155<-3:
                self.robot.play_user_motion_sync(self.robot.StepLeft_small)
                continue
            else:   #位置在中间，可以开始翻越
                print('Start Climbing!')
                self.robot.gait_manager.stop()
                self.robot.wait(2000)
                self.robot.play_user_motion_sync(self.robot.ClimbObstacle)
                self.robot.wait(200)
                self.robot.motion_manager.playPage(9, False)
                while self.robot.motion_manager.isMotionPlaying():
                    self.robot.motion_manager.step(self.robot.time_step)
                    self.robot.my_step()
                self.robot.wait(1000)
                self.robot.motion_manager.playPage(11, False)
                while self.robot.motion_manager.isMotionPlaying():
                    self.robot.motion_manager.step(self.robot.time_step)
                    self.robot.my_step()
                self.robot.wait(200)
                self.robot.ready_to_walk()
                self.handle_global_data['H_bar']['StartFlag'] = 'finished'
                return    
 
    def handle_level_mine(self, floor=None):
        self.robot.set_head_pitch(-20)
        img = self.robot.get_img()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
        ret, binary_mine = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
        contours, h = cv2.findContours(binary_mine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        mine_at_centre = False
        floor_position = "c"
        # modulate the contours for guidance
        if len(contours) >= 2:
            # found more than two mines, use the closet two
            use_contour = sorted(contours, key=lambda x: np.mean(x[:,:,0]))[-2:]
        elif len(contours) == 1:
            use_contour = contours
        else:
            use_contour = []
        # re-judge if the two mines can be used to guide the path
        if len(use_contour) == 2:
            # found two, then judge if these two mines can be used for guidance
            y1 = contours[0][:,0,0].mean()
            y2 = contours[1][:,0,0].mean()
            if y1-y2 > 30:
                # too much difference in y-dimension, take as one mine
                use_contour = [contours[0]]
            elif y1-y2 < -30:
                use_contour = [contours[1]]
            else:
                # take as two mine
                use_contour = contours
        
        mine_y_mean = 0
        mine_x_mean = 0
        # use the selected contours for guidance
        if len(use_contour) == 2:
            x1 = use_contour[0][:,0,1]
            x2 = use_contour[1][:,0,1]
            x = (x1.mean() + x2.mean())/2
            x_offset = (img_gray.shape[0]/2 - x)/img_gray.shape[0]
        else:
            # follow the floor
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # floor = 'brick'
            if floor is None:
                floor = self.handle_global_data['mine']['floor'] 
            binary_floor = cv2.inRange(img_hsv, self.floor_hsv_range[floor][0], 
                                        self.floor_hsv_range[floor][1])
            binary_floor_shadow = cv2.inRange(img_hsv, self.floor_with_shadow_hsv_range[floor][0], 
                                              self.floor_with_shadow_hsv_range[floor][1])
            binary_floor = cv2.bitwise_or(binary_floor, binary_floor_shadow)
            
            floor_area = np.where(binary_floor > 1)
            
            if len(floor_area[0]) > 0:
                y_min = floor_area[1].min()
                y_max = np.min([y_min + 30, floor_area[1].max()])
                if len(use_contour) == 1:
                    # if found mine, avoid the mine by shifting x-position
                    # thus, first should know if the floor is more on the left or right
                    floor_area = np.where(binary_floor[:,y_min:y_max] > 1)
                    # num_zero_left = floor_area[]
                    if len(floor_area[0]) > 0:
                        left_right_diff = (binary_floor.shape[0] - floor_area[0].max()) - floor_area[0].min()
                        print("left_right_diff:", left_right_diff)
                        if left_right_diff > 50:
                            # floor is more on the left
                            x_min = 0
                            x_max = use_contour[0][:,:,1].min()
                            floor_position = "left"
                        elif left_right_diff < -50:
                            # floor is more on the right
                            x_min = use_contour[0][:,:,1].max()
                            x_max = binary_floor.shape[0]
                            floor_position = "right"
                        else:
                            # then use the mine to define the left/right
                            if use_contour[0][:,:,1].mean() - binary_floor.shape[0]/2 > 20:
                                # mine on the right
                                x_min = 0
                                x_max = use_contour[0][:,:,1].min()
                            elif use_contour[0][:,:,1].mean() - binary_floor.shape[0]/2 < - 20:
                                # mine on the left
                                x_min = use_contour[0][:,:,1].max()
                                x_max = binary_floor.shape[0]
                            else:
                                # mine almost at the centre
                                mine_at_centre = True
                                x_min, x_max = 0, binary_floor.shape[0]
                    else:
                        x_min, x_max = 0, binary_floor.shape[0]
                    
                    mine_y_mean = contours[0][:,:,0].mean()
                    mine_x_mean = contours[0][:,:,1].mean()
                else:
                    x_min, x_max = 0, binary_floor.shape[0]
                    mine_y_mean = 0
                    mine_x_mean = 0
                floor_area = np.where(binary_floor[x_min:x_max,y_min:y_max] > 1)
                if len(floor_area[0]) > 0:
                    x = np.mean(floor_area[0]) + x_min
                else:
                    x = img_gray.shape[0]/2
            else:
                x = 0
        # special process for mine_at_centre
        if mine_at_centre:
            x_offset = np.sign(use_contour[0][:,:,1].mean() - binary_floor.shape[0]/2) * 1.0
            step_x = -1.0
        elif (mine_y_mean > img_gray.shape[1]/2) and mine_at_centre:
            # mine too closed
            x_offset = (img_gray.shape[0]/2 - x)/img_gray.shape[0]
            step_x = -1.0
            print('DANGER',mine_x_mean,mine_y_mean)
        else:
            x_offset = (img_gray.shape[0]/2 - x)/img_gray.shape[0] * 2.0
            step_x = 1 - abs(x_offset) if abs(x_offset) < 0.3 else 0

        # assign step_x and x_offset to gait model
        if self.robot.use_self_gait:
            self.robot.walker.param_move['x_move_amp'] = step_x*self.robot.speed
            self.robot.walker.param_move['a_move_amp'] = x_offset
        else:
            self.robot.gait_manager.setXAmplitude(step_x)
            self.robot.gait_manager.setAAmplitude(x_offset)
            if step_x <= 0:
                self.robot.gait_manager.setXAmplitude(-0.1)
                # two narrow space for turn, should move for extra gait loops
                extra_loop = 75 + int(abs(step_x) * 75)
                if floor_position == "left":
                    rand_offset = np.random.rand(1)[0] - 0.2
                elif floor_position == "right":
                     rand_offset = -np.random.rand(1)[0] + 0.2
                else:
                    rand_offset = np.random.rand(1)[0] - 0.5
                self.robot.gait_manager.setAAmplitude(x_offset*0.6 + 0.2*rand_offset)
                
                    
                while extra_loop:
                    print('running extra gait loop- %s left'%extra_loop)
                    self.robot.walk_step()
                    self.robot.my_step()
                    extra_loop -= 1
    
    def handle_level_bridge(self):
        # h_p = 30 - self.handle_global_data['bridge']['forward_len']/2
        # self.robot.set_head_pitch(h_p)
        img = self.robot.get_img(hsv=True)
        # if found next level floor, use that floor to guide direction
        if self.next_level_floor not in [None, 'invisible']:
            binary_floor = cv2.inRange(img, self.floor_hsv_range[self.next_level_floor][0], 
                                    self.floor_hsv_range[self.next_level_floor][1])
            binary_floor_shadow = cv2.inRange(img, self.floor_with_shadow_hsv_range[self.next_level_floor][0], 
                                            self.floor_with_shadow_hsv_range[self.next_level_floor][1])
            binary_floor = cv2.bitwise_or(binary_floor,binary_floor_shadow)
            floor_area = np.where(binary_floor > 1)
            if len(floor_area[0]) > 0:
                y_min = floor_area[1].min()
                y_max = np.min([y_min + 30, floor_area[1].max()])
                floor_area = np.where(binary_floor[:,y_min:y_max] > 1)
                if len(floor_area[0]) > 0:
                    x = floor_area[0].mean()
                else:
                    x = 0
        else:
            # follow the floor
            binary_floor = cv2.inRange(img, 
                                    self.floor_hsv_range['green bridge/step'][0],
                                    self.floor_hsv_range['green bridge/step'][1])
            binary_floor_shadow = cv2.inRange(img, 
                                            self.floor_with_shadow_hsv_range['green bridge/step'][0],
                                            self.floor_with_shadow_hsv_range['green bridge/step'][1])
            binary_floor = cv2.bitwise_or(binary_floor, binary_floor_shadow)
            floor_area = np.where(binary_floor > 1)
            

            if len(floor_area[0]) > 0:
                y_min = floor_area[1].min()
                y_max = np.min([y_min + 40, floor_area[1].max()])
                if y_max - y_min < 5:
                    print('passed level {} ^V^'.format(self.level_names[7]))
                    return True
                floor_area = np.where(binary_floor[:,y_min:y_max] > 1)
                x = np.mean(floor_area[0])

            else:
                # no floor found... turn around as the last resort
                x = 0

            
        x_offset = (img.shape[0]/2 - x)/img.shape[0]
        step_x = 1 - abs(x_offset) if abs(x_offset) < 0.3 else 0
        
        # assign step_x and x_offset to gait model
        if self.robot.use_self_gait:
            self.robot.walker.param_move['x_move_amp'] = step_x*self.robot.speed
            self.robot.walker.param_move['a_move_amp'] = x_offset
        else:
            self.robot.gait_manager.setXAmplitude(step_x)
            self.robot.gait_manager.setAAmplitude(x_offset)
        
    def handle_level_end(self):
        img = self.robot.get_img(hsv=True)
        if self.handle_global_data['end']['phase'] == 0:
            # phase1 - follow floor while headed down, until all the end_floor are in the view
            # set head pitch
            # self.robot.set_head_pitch(self.handle_global_data['end']['head_pitch'])
            self.robot.set_head_pitch(0)
            img_floor_filtered1 = cv2.inRange(img,
                                         self.floor_with_shadow_hsv_range['end_floor'][0],
                                         self.floor_with_shadow_hsv_range['end_floor'][1])
            img_floor_filtered2 = cv2.inRange(img,
                                                self.floor_hsv_range['end_floor'][0],
                                                self.floor_hsv_range['end_floor'][1])
            img_floor_filtered = cv2.bitwise_or(img_floor_filtered1, img_floor_filtered2)
            
            img_floor_filtered = cv2.morphologyEx(img_floor_filtered, cv2.MORPH_CLOSE,
                                                  np.ones([5,5], np.uint8))
            floor_area = np.where(img_floor_filtered > 1)
            x = img.shape[0]/2
            if len(floor_area[0]) > 0:
                y_min = floor_area[1].min()
                y_max = np.min([y_min + 40, floor_area[1].max()])
                x_centre = int(img.shape[0]/2)
                floor_area = np.where(img_floor_filtered[:,y_min:y_max] > 1)
                if len(floor_area[0]) > 0:
                    x = np.mean(floor_area[0])
                else:
                    x = 0
                floor_area = np.where(img_floor_filtered[x_centre-10:x_centre+10, :] > 1)
                if len(floor_area[0]) > 0:  
                    y_max = int(np.max(floor_area[1]))
                    y_min = int(np.min(floor_area[1]))
                    floor_area = np.where(img_floor_filtered > 1)
                    if len(floor_area[0]) > 0:
                        x_mean = int(np.mean(floor_area[0]))
                    else:
                        x_mean = 0
                else:
                    y_max,y_min,x_mean = 0,0,0
                
                floor_area = np.where(img_floor_filtered1 > 1)
                if len(floor_area[0]) > 0:
                    y_shadow = int(np.mean(np.where(img_floor_filtered1 > 1)[1]))
                else:
                    y_shadow = 0
                
                if (y_max >= img.shape[1]-10) and (y_min <= img.shape[0]+10) and (abs(x_mean-x_centre)<2) and (y_shadow>=img.shape[1]-50):
                    # get to the right position
                    self.handle_global_data['end']['phase'] = 1
            x_offset = (img.shape[0]/2 - x)/img.shape[0]
            step_x = 0.5
            self.handle_global_data['end']['head_pitch'] = np.max([-20,
                                                                   self.handle_global_data['end']['head_pitch']-1]) 
            # assign step_x and x_offset to gait model
            if self.robot.use_self_gait:
                self.robot.walker.param_move['x_move_amp'] = step_x*self.robot.speed
                self.robot.walker.param_move['a_move_amp'] = x_offset
            else:
                self.robot.gait_manager.setXAmplitude(step_x)
                self.robot.gait_manager.setAAmplitude(x_offset)
        elif self.handle_global_data['end']['phase'] == 1:
            self.robot.set_head_pitch(np.rad2deg(0.3))
            self.robot.gait_manager.stop()
            img_bar_filtered = cv2.inRange(img, 
                                   self.object_hsv_range['yellow bar'][0], 
                                   self.object_hsv_range['yellow bar'][1])
            bar_size = cv2.countNonZero(img_bar_filtered)
            # phase2 - wait until the bar open
            if (self.handle_global_data['end']['GateFlag'] < 5):
                print('wait for the gate open:%d', bar_size)
                if bar_size > 500:
                    self.handle_global_data['end']['GateFlag'] += 1 
            else:
                #Flag > n frames, already closed once
                if bar_size < 200:
                    self.robot.gait_manager.setXAmplitude(1.0)
                    self.robot.gait_manager.start()
                    self.passed_level_end = True
                else:
                    print('cannot detect gate open:%d', bar_size)
    
    def manual_control_process(self):
        self.robot.keyboard_process()
    



                



