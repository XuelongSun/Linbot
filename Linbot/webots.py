import os,sys
# add webots controller directory fro external Webots controller
# (or add a PYTHONPATH enviornment value to the system)
webots_path = os.path.join("D:/",'softwares','Webots','lib','controller','python39')
if webots_path not in sys.path:
    sys.path.append(webots_path)

from controller import Robot

class WebotsRunner:
    def __init__(self, name):
        self.name = name
        self.robot = Robot()
        # get the time step of the current world.
        self.time_step = int(self.robot.getBasicTimeStep())
        print(self.time_step)
        # sensors and actuators
        self.sensors = {}
        self.actuators = {}
        # keyboard
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)
        
        # timer
        self.loop_count = 0
        
    def get_sensor(self, device_name):
        sensors = {}
        for name in device_name:
            if not name in self.sensors.keys():
                try:
                    sensors.update({name: self.robot.getDevice(name)})
                    self.sensors.update({name: self.robot.getDevice(name)})
                    self.sensors[name].enable(self.time_step)
                except:
                    print('cannot get device, please check the name {}'.format(name))
            else:
                print('sensor already exists.')
        return sensors
    
    def get_actuator(self, device_name):
        actuators = {}
        for name in device_name:
            if not name in self.sensors.keys():
                try:
                    actuators.update({name: self.robot.getDevice(name)})
                    self.actuators.update({name: self.robot.getDevice(name)})
                except:
                    print("cannot get device, please check the name '{}'".format(name))
            else:
                print('actuator already exists.')
        return actuators
                
    def my_step(self, func=None, fargs=None):
        if self.robot.step(self.time_step) == -1:
            return 1
        else:
            if func is not None:
                func(*fargs)
            self.loop_count += 1
            return 0
    
    def wait(self, ms):
        start_time = self.robot.getTime()
        s = ms / 1000.0
        while s + start_time >= self.robot.getTime():
            self.my_step()
           
    def initialize(self):
        raise NotImplementedError
