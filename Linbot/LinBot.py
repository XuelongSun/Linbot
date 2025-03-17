from threading import Thread
import sys

import numpy as np

from Darwin import DarwinOP
from levels_handle import LevelsHandle
from controller import Motion

# create instance of Darwin robot
robot = DarwinOP(name="LinBot")

# create instance of LevelHandle, pass the robot instance to it, 
# thus levels_handle can get access to all the resources of the robot
levels_handle = LevelsHandle(robot)
    
# define the frame rate of the visual processing
visual_framerate = 10  # frame/s
visual_interval = int(1000 / (robot.time_step * visual_framerate)) # loop/frame
# visual_interval = 75

# manual control
manual_control = False

if __name__ == "__main__":
    # initialize the robot
    robot.initialize()
    robot.RegisterMotions()

    robot.use_self_gait = False
    # main loop
    print("LinBot Run!")
    # ready to walk
    robot.ready_to_walk()

    while True:
        if manual_control:
            levels_handle.manual_control_process()
        else:
        # visual processing
            if robot.loop_count % visual_interval == 0:
                # start level
                # levels_handle.passed_level_start = True
                if not levels_handle.passed_level_start:
                    levels_handle.handle_level_start()
                else:
                    # recognize levels
                    current_level = levels_handle.levels_recognition()
                    
                    # # handle current levels according to the results of levels recognition
                    levels_handle.current_level = current_level
                    if current_level is not None:
                        
                        # print('Got level {}'.format(levels_handle.level_names[levels_handle.current_level]))
                        # current_level = 8
                        current_level_name = levels_handle.level_names[current_level]
                        if current_level_name != 'end':
                            # call the cooresponding function 
                            eval('levels_handle.handle_level_' + current_level_name + '()')
                        else:
                            # end level
                            if not levels_handle.passed_level_end:
                                levels_handle.handle_level_end()
                    else:
                        if levels_handle.current_level_floor is None:
                            if levels_handle.passed_level_end:
                                # task completed!
                                print('WOW, LinBot, you are briiiiiliant!')
                                robot.gait_manager.stop()
                                robot.motion_manager.playPage(1)
                                robot.wait(100)
                                robot.motion_manager.playPage(24)
                                robot.wait(200)
                                break
                            else:
                                # strayed
                                robot.gait_manager.setXAmplitude(-1.0)
                                robot.gait_manager.setAAmplitude(0)
                                extra_loop = 75
                                while extra_loop > 0:
                                    robot.walk_step()
                                    robot.my_step()
                                    extra_loop -= 1
                        elif levels_handle.next_level_floor is None:
                            if levels_handle.passed_level_end:
                                # task completed!
                                print('WOW, LinBot, you are briiiiiliant!')
                                robot.gait_manager.stop()
                                robot.motion_manager.playPage(1)
                                robot.wait(100)
                                robot.motion_manager.playPage(24)
                                robot.wait(200)
                                break
                            else:
                                robot.set_head_pitch(0)
                                # should turn left
                                robot.gait_manager.setXAmplitude(0.0)
                                robot.gait_manager.setAAmplitude(0.5)
                                extra_loop = 75 
                                while extra_loop:
                                    robot.walk_step()
                                    robot.my_step()
                                    extra_loop -= 1
                                print('arrived corner, turn left!')
                        else:
                            robot.set_head_pitch(-10)
                            # should follow floor
                            if levels_handle.next_level_floor != 'invisible':
                                levels_handle.follow_floor(levels_handle.next_level_floor)
                                # print('following next standard floor')
                            else:
                                levels_handle.follow_floor(levels_handle.current_level_floor)
                print("{}th: level_{}:{}, floor:{},{}".format(robot.loop_count,
                                                              levels_handle.current_level,
                                                              levels_handle.level_names[levels_handle.current_level],
                                                              levels_handle.current_level_floor, 
                                                              levels_handle.next_level_floor))
        
        # check if fallen
        fallen = robot.check_if_fallen()
        if fallen == 'Fallen face up':
            robot.motion_manager.playPage(10)
            robot.wait(200)
            robot.ready_to_walk()

        elif fallen == 'Fallen face down':
            robot.motion_manager.playPage(11)
            robot.wait(200)
            robot.ready_to_walk()
        
        # else continue walking:
        robot.walk_step()
        
        # robot step, update sensors and webots environment, etc.
        robot.my_step()
    
    