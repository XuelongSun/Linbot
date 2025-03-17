import configparser
import numpy as np

class Walking:
    def __init__(self, period=600, DSP_ratio=0.1, 
                 step_fb_ratio=0.28,
                 x_move_amp=0, y_move_amp=0, y_swap_amp=5.0, 
                 z_move_amp=25.0, z_swap_amp=5.0, a_move_amp=0, 
                 config_filename='config.ini'):
        
        self.config = configparser.ConfigParser()
        self.config.read(config_filename)
        if self.config.sections == []:
            print('no valid config file found.')
       
        self.param_time = {'period_time':period, 
                           'DSP_ratio':DSP_ratio,
                           'SSP_time':0, 
                           'SSP_t_start_l':0,'SSP_t_end_l':0,
                           'SSP_t_start_r':0,'SSP_t_end_r':0,
                           'phase_time1':0, 'phase_time2':0, 'phase_time3':0,
                           'x_swap_period_time':0, 'x_move_period_time':0,
                           'y_swap_period_time':0, 'y_move_period_time':0,
                           'z_swap_period_time':0, 'z_move_period_time':0,
                           'a_move_period_time':0
                          }
        
        self.param_move = {'step_fb_ratio':step_fb_ratio,
                           'x_move_amp':x_move_amp, 'x_swap_amp':0,
                           'x_move_amp_shift':0, 'x_swap_amp_shift':0,
                           'x_move_phase_shift':np.pi/2, 'x_swap_phase_shift':np.pi,
                           'y_move_amp':y_move_amp/2, 'y_swap_amp':y_swap_amp + y_move_amp*0.04,
                           'y_move_amp_shift':0, 'y_swap_amp_shift':0,
                           'y_move_phase_shift':np.pi/2, 'y_swap_phase_shift':0,
                           'z_move_amp':z_move_amp/2, 'z_swap_amp':z_swap_amp,
                           'z_move_amp_shift':0, 'z_swap_amp_shift':0,
                           'z_move_phase_shift':np.pi/2, 'z_swap_phase_shift':np.pi*3/2,
                           'a_move_aim_on':False,
                           'a_move_amp':a_move_amp, 'a_move_amp_shift':0, 'a_move_phase_shift':np.pi/2,
                           }

        # kinematics
        self.kinematics = {'leg_length':219.5, 'thigh_length':93.0, 
                           'calf_length':93.0, 'ankle_length':33.5}
        
        # periodic timer and loop counter
        self.timer = 0
        self.period_counter = 0
        
        # simulation time interval
        self.dt = 8 # ms
        
        self.loop_number = int(self.param_time['period_time']/self.dt)
        
        # current phase, could be 0,1,2,3
        self.current_phase = 0
        
        # whether apply balance
        self.balance_enable = True
        
        # control
        self.ctrl_running = False
        self.real_running = False
        
        # store data for analysis
        self.end_points = dict.fromkeys(['xl','xr','yl','yr','zl','zr','cl','cr'])
        for k in self.end_points.keys():
            self.end_points[k] = []
        self.z_ep = {'l':[],'r':[]}
        self.motor_angles = dict.fromkeys(['hip-yaw','hip-roll','hip-pitch',
                                           'knee-pitch',
                                           'ankle-pitch','ankle-roll'])
        for k in self.motor_angles.keys():
            self.motor_angles[k] = []
        self.data = []
    
    def sin(self, t, period, period_shift, mag, mag_shift):
        return mag * np.sin(2*np.pi*t/period - period_shift) + mag_shift
    
    def transform_3d(self, point, theta):
        matrix = np.mat(np.zeros([4,4]))
        # homogenerous transform matrix
        matrix[0,0] = np.cos(theta[2])*np.cos(theta[1])
        matrix[0,1] = np.cos(theta[2])*np.sin(theta[1])*np.sin(theta[0]) - np.sin(theta[2])*np.sin(theta[0])
        matrix[0,2] = np.cos(theta[2])*np.sin(theta[1])*np.cos(theta[0]) + np.sin(theta[2])*np.sin(theta[0])
        matrix[0,3] = point[0]
        
        matrix[1,0] = np.sin(theta[2])*np.cos(theta[1])
        matrix[1,1] = np.sin(theta[2])*np.sin(theta[1])*np.sin(theta[0]) + np.cos(theta[2])*np.cos(theta[0])
        matrix[1,2] = np.sin(theta[2])*np.sin(theta[1])*np.cos(theta[0]) - np.cos(theta[2])*np.sin(theta[0])
        matrix[1,3] = point[1]
        
        matrix[2,0] = -np.sin(theta[1])
        matrix[2,1] = np.cos(theta[1])*np.sin(theta[0])
        matrix[2,2] = np.cos(theta[1])*np.cos(theta[0])
        matrix[2,3] = point[2]
        
        matrix[3,3] = 1
        
        return matrix
    
    def computer_ik(self, x, y, z, a=0, b=0, c=0):
        # HIP_YAW HIP_ROLL HIP_PITCH KNEE_PITCH ANKLE_PITCH ANKLE_ROLL
        angles = [0]*6
        
        # transform the endpoint by rotation
        transform_ad = self.transform_3d([x,y,z-self.kinematics['leg_length']], [a,b,c])
        
        x_ = x + transform_ad[0,2] * self.kinematics['ankle_length']
        y_ = y + transform_ad[1,2] * self.kinematics['ankle_length']
        z_ = z - self.kinematics['leg_length'] + transform_ad[2,2] * self.kinematics['ankle_length']
        
        # get knee
        length = np.sqrt(x_*x_ + y_*y_ + z_*z_)
        self.data.append([self.timer + 600*self.period_counter, length, x_, y_, z_, transform_ad[3,2]])
        temp = length*length - self.kinematics['thigh_length']**2 - self.kinematics['calf_length']**2
        _acos = np.arccos(temp / (2*self.kinematics['thigh_length']*self.kinematics['calf_length']))
        if np.isnan(_acos):
            print('*t={} ({}th loop): IK failed when calculate [knee pitch]'.format(self.timer, self.period_counter))
            return False
        angles[3] = _acos
        
        # get ankle roll
        if np.linalg.det(transform_ad) == 0:
            print('*t={} ({}th loop): IK failed when calculate [ankle roll]'.format(self.timer, self.period_counter))
            return False
        transform_da = transform_ad.I
        _k = np.sqrt(transform_da[1,3]**2 + transform_da[2,3]**2)
        _l = np.sqrt(transform_da[1,3]**2 + (transform_da[2,3]-self.kinematics['ankle_length'])**2)
        _m = (_k**2 - _l**2 - self.kinematics['ankle_length']**2) / (2 * _l * self.kinematics['ankle_length'])
        _m = np.clip(_m, -1.0, 1.0)
        _acos = np.arccos(_m)
        if _acos is np.nan:
            print('*t={} ({}th loop): IK failed when calculate [ankle roll]'.format(self.timer, self.period_counter))
            return False
        angles[5] = _acos if y < 0 else -_acos
        
        # get hip yaw
        transform_cd = self.transform_3d([0,0,-self.kinematics['ankle_length']],
                                         [angles[5], 0, 0])
        if np.linalg.det(transform_cd) == 0:
            print('*t={} ({}th loop): IK failed when calculate [hip yaw]'.format(self.timer, self.period_counter))
            return False
        transform_dc = transform_cd.I
        transform_ac = transform_ad * transform_dc
        _atan = np.arctan2(-transform_ac[0,1], transform_ac[1,1])
        if np.isinf(_atan):
            print('*t={} ({}th loop): IK failed when calculate [hip yaw]'.format(self.timer, self.period_counter))
            return False
        angles[0] = _atan
        
        # get hip roll 
        _atan = np.arctan2(transform_ac[2,1], 
                           -transform_ac[0,1]*np.sin(angles[0]) + transform_ac[1,1]*np.cos(angles[0]))
        if np.isinf(_atan):
            print('*t={} ({}th loop): IK failed when calculate [hip roll]'.format(self.timer, self.period_counter))
            return False
        angles[1] = _atan
        
        # get hip pitch and ankle pitch
        _atan = np.arctan2(transform_ac[0,2]*np.cos(angles[0]) + transform_ac[1,2]*np.sin(angles[0]),
                           transform_ac[0,0]*np.cos(angles[0]) + transform_ac[1,0]*np.sin(angles[0]))
        if np.isinf(_atan):
            print('*t={} ({}th loop): IK failed when calculate [hip pitch, ankle pitch]'.format(self.timer, self.period_counter))
            return False
        _theta = _atan
        _k = np.sin(angles[3]) * self.kinematics['calf_length']
        _l = -self.kinematics['thigh_length'] - np.cos(angles[3]) * self.kinematics['calf_length']
        _m = np.cos(angles[0]) * x_ + np.sin(angles[0]) * y_
        _n = np.cos(angles[1]) * z_ + np.sin(angles[0]) * np.sin(angles[1]) * x_ - np.cos(angles[0]) * np.sin(angles[1]) * y_
        _s = (_k * _n + _l * _m) / (_k *_k + _l * _l)
        _c = (_n - _k * _s) / _l
        _atan = np.arctan2(_s, _c)
        if np.isinf(_atan):
            print('*t={} ({}th loop): IK failed when calculate [hip pitch, ankle pitch]'.format(self.timer, self.period_counter))
            return False
        angles[2] = _atan
        angles[4] = _theta - angles[3] - angles[2]
        
        return angles
        # self.ik_actuator.ee = [x, y, z-self.kinematics['leg_length']]
        # return self.ik_actuator.angles
    
    def update_param_time(self):
        period = self.param_time['period_time']
        dsp_r = self.param_time['DSP_ratio']
        ssp_r = 1 - dsp_r
        
        self.param_time['x_swap_period_time'] = period / 2.0
        self.param_time['x_move_period_time'] = period * ssp_r
        self.param_time['y_swap_period_time'] = period
        self.param_time['y_move_period_time'] = period * ssp_r
        self.param_time['z_swap_period_time'] = period / 2.0
        self.param_time['z_move_period_time'] = period * ssp_r / 2.0
        self.param_time['a_move_period_time'] = period * ssp_r
        
        self.param_time['SSP_time'] = period * ssp_r
        self.param_time['SSP_t_start_l'] = (1 - ssp_r) * period / 4.0
        self.param_time['SSP_t_end_l'] = (1 + ssp_r) * period / 4.0
        self.param_time['SSP_t_start_r'] = (3 - ssp_r) * period / 4.0
        self.param_time['SSP_t_end_r'] = (3 + ssp_r) * period / 4.0
        
        self.param_time['phase_time1'] = (self.param_time['SSP_t_start_l'] + self.param_time['SSP_t_end_l'])/2.0
        self.param_time['phase_time2'] = (self.param_time['SSP_t_start_r'] + self.param_time['SSP_t_end_l'])/2.0
        self.param_time['phase_time3'] = (self.param_time['SSP_t_start_r'] + self.param_time['SSP_t_end_r'])/2.0
    
    def update_param_move(self):
        # forward and back move
        self.param_move['x_swap_amp'] = self.param_move['x_move_amp'] * self.param_move['step_fb_ratio']
        
        # right and left
        self.param_move['y_move_amp_shift'] = abs(self.param_move['y_move_amp'])
        
        # up and down
        self.param_move['z_move_amp_shift'] = self.param_move['z_move_amp'] / 2.0
        self.param_move['z_swap_amp_shift'] = self.param_move['z_swap_amp'] / 2.0
        
        # turning direction
        if not self.param_move['a_move_aim_on']:
            # self.param_move['a_move_amp'] = np.deg2rad(self.param_move['a_move_amp']) / 2.0
            self.param_move['a_move_amp_shift'] = abs(self.param_move['a_move_amp'])
        else:
            self.param_move['a_move_amp'] = -np.deg2rad(self.param_move['a_move_amp']) / 2.0
            self.param_move['a_move_amp_shift'] = -abs(self.param_move['a_move_amp'])

    def computer_endpoint_move(self):
        if self.timer <= self.param_time['SSP_t_start_l']:
            t = self.param_time['SSP_t_start_l']
            t_offset = self.param_time['SSP_t_start_l']
            tzl = self.param_time['SSP_t_start_l']
            tzl_offset = self.param_time['SSP_t_start_l']
            tzr = self.param_time['SSP_t_start_r']
            tzr_offset = self.param_time['SSP_t_start_r']
            p_offset = 0
            
            pelvis_offset_l = 0
            pelvis_offset_r = 0
        elif self.timer <= self.param_time['SSP_t_end_l']:
            t = self.timer
            t_offset = self.param_time['SSP_t_start_l']
            tzl = self.timer
            tzl_offset = self.param_time['SSP_t_start_l']
            tzr = self.param_time['SSP_t_start_r']
            tzr_offset = self.param_time['SSP_t_start_r']
            p_offset = 0
            
            pelvis_offset_l = self.sin(t, self.param_time['z_move_period_time'], 
                                       self.param_move['z_move_phase_shift']+2*np.pi/self.param_time['z_move_period_time']*t_offset,
                                       self.config.getfloat('Walking','pelvis_offset') * 0.175, 
                                       self.config.getfloat('Walking','pelvis_offset') * 0.175)
            pelvis_offset_r = self.sin(t, self.param_time['z_move_period_time'], 
                                       self.param_move['z_move_phase_shift']+2*np.pi/self.param_time['z_move_period_time']*t_offset,
                                       -self.config.getfloat('Walking','pelvis_offset') * 0.5, 
                                       -self.config.getfloat('Walking','pelvis_offset') * 0.5)
        elif self.timer <= self.param_time['SSP_t_start_r']:
            t = self.param_time['SSP_t_end_l']
            t_offset = self.param_time['SSP_t_start_l']
            tzl = self.param_time['SSP_t_end_l']
            tzl_offset = self.param_time['SSP_t_start_l']
            tzr = self.param_time['SSP_t_start_r']
            tzr_offset = self.param_time['SSP_t_start_r']
            p_offset = 0
            pelvis_offset_l = 0
            pelvis_offset_r = 0
        elif self.timer <= self.param_time['SSP_t_end_r']:
            t = self.timer
            t_offset = self.param_time['SSP_t_start_r']
            tzl = self.param_time['SSP_t_end_l']
            tzl_offset = self.param_time['SSP_t_start_l']
            tzr = self.timer
            tzr_offset = self.param_time['SSP_t_start_r']
            p_offset = np.pi
            
            pelvis_offset_l = self.sin(t, self.param_time['z_move_period_time'], 
                                       self.param_move['z_move_phase_shift']+2*np.pi/self.param_time['z_move_period_time']*t_offset,
                                       self.config.getfloat('Walking','pelvis_offset') * 0.5, 
                                       self.config.getfloat('Walking','pelvis_offset') * 0.5)
            pelvis_offset_r = self.sin(t, self.param_time['z_move_period_time'], 
                                       self.param_move['z_move_phase_shift']+2*np.pi/self.param_time['z_move_period_time']*t_offset,
                                       -self.config.getfloat('Walking','pelvis_offset') * 0.175, 
                                       -self.config.getfloat('Walking','pelvis_offset') * 0.175)
        else:
            t = self.param_time['SSP_t_end_r']
            t_offset = self.param_time['SSP_t_start_r']
            tzl = self.param_time['SSP_t_end_l']
            tzl_offset = self.param_time['SSP_t_start_l']
            tzr = self.param_time['SSP_t_end_r']
            tzr_offset = self.param_time['SSP_t_start_r']
            p_offset = np.pi
            
            pelvis_offset_l = 0
            pelvis_offset_r = 0

        move_ep = []
        cal_str = "self.sin({:f}, self.param_time['{}_move_period_time'],"
        cal_str += "self.param_move['{}_move_phase_shift'] + 2*np.pi/self.param_time['{}_move_period_time']*{} + {},"
        cal_str += "{}self.param_move['{}_move_amp'],"
        cal_str += "{}self.param_move['{}_move_amp_shift'])"
        for d in ['l', 'r']:
            for p in ['x','y','z','a']:
                s = '' if d == 'l' else '-'
                # for z, no sign change
                if p == 'z':
                    s = ''
                    if d == 'l':
                        eval('move_ep.append(' + cal_str.format(tzl,p,p,p,tzl_offset,0,s,p,s,p) + ')')
                        # print(cal_str.format(tzl,p,p,p,tzl_offset,0,s,p,s,p))
                    else:
                        eval('move_ep.append(' + cal_str.format(tzr,p,p,p,tzr_offset,0,s,p,s,p) + ')')
                        # print(cal_str.format(tzr,p,p,p,tzr_offset,0,s,p,s,p))
                else:
                    eval('move_ep.append(' + cal_str.format(t,p,p,p,t_offset,p_offset,s,p,s,p) + ')')
        
        move_ep.append(pelvis_offset_l)
        move_ep.append(pelvis_offset_r)
        
        # compute arm swing
        if self.param_move['x_move_amp'] == 0:
            arm_angle = 0.0 
        else:
            arm_angle = self.sin(t, self.param_time['period_time'], np.pi*1.5,
                                 -self.param_move['x_move_amp'] * self.config.getfloat('Walking',
                                                                                       'arm_swing_gain'),
                                 0.0)
        move_ep.append(arm_angle)
        move_ep.append(-arm_angle)
        
        return tuple(move_ep)
    
    def computer_endpoint_total(self):
        # # phase-0: initialization
        # if self.timer == 0:
        #     self.update_param_time()
        #     self.current_phase = 0
        # # phase-1: 
        # elif (self.timer >= self.param_time['phase_time1'] - self.dt/2) and (self.timer <= self.param_time['phase_time1'] + self.dt/2):
        #     self.update_param_move()
        #     print(self.param_move['a_move_amp'])
        #     self.current_phase = 1
        # # phase-2:
        # elif (self.timer >= self.param_time['phase_time2'] - self.dt/2) and (self.timer <= self.param_time['phase_time2'] + self.dt/2):
        #     self.update_param_time()
        #     # ?
        #     self.timer = self.param_time['phase_time2']
        #     self.current_phase = 2
        # elif (self.timer >= self.param_time['phase_time3'] - self.dt/2) and (self.timer <= self.param_time['phase_time3'] + self.dt/2):
        #     self.update_param_move()
        #     print(self.param_move['a_move_amp'])
        #     self.current_phase = 3
                
        end_points = {}
            
        x_swap = self.sin(self.timer, 
                              self.param_time['x_swap_period_time'], 
                              self.param_move['x_swap_phase_shift'], 
                              self.param_move['x_swap_amp'],
                              self.param_move['x_swap_amp_shift'])
        y_swap = self.sin(self.timer, 
                            self.param_time['y_swap_period_time'], 
                            self.param_move['y_swap_phase_shift'], 
                            self.param_move['y_swap_amp'],
                            self.param_move['y_swap_amp_shift'])
        z_swap = self.sin(self.timer, 
                            self.param_time['z_swap_period_time'], 
                            self.param_move['z_swap_phase_shift'], 
                            self.param_move['z_swap_amp'],
                            self.param_move['z_swap_amp_shift'])        
        a_swap, b_swap, c_swap = 0, 0, 0                 
        # 2. compute move
        x_move_l, y_move_l, z_move_l, c_move_l, x_move_r, y_move_r, z_move_r, c_move_r, pel_l, pel_r, arm_l, arm_r = self.computer_endpoint_move() 
        # no pitch and roll
        a_move_l, b_move_l, a_move_r, b_move_r = 0,0,0,0
        
        # 3.summation to get total endpoint
        for d in ['l','r']:
            for p in ['x','y','z','a','b','c']:
                offset = self.config.getfloat('Walking', p + '_offset')
                if p in ['y', 'a', 'c']:
                    offset /= 2.0
                eval("end_points.update({'%s':%s_swap + %s_move_%s + %f})"%(p+d,p,p,d,offset))
        
        hip_pitch_offset =  self.config.getfloat('Walking','hip_pitch_offset')*np.pi/180
        
        return end_points, pel_l*np.pi/180, pel_r*np.pi/180, hip_pitch_offset, arm_l, arm_r
        
    def balance_control(self, gyro_data):
        balance_offset = [0] * 4
        if self.balance_enable:
            balance_offset[0] = gyro_data[0] * self.config.getfloat('Walking','balance_hip_roll_gain')*np.pi/180
            balance_offset[1] = -gyro_data[1] * self.config.getfloat('Walking','balance_knee_gain')*np.pi/180
            balance_offset[2] = -gyro_data[1] * self.config.getfloat('Walking','balance_ankle_pitch_gain')*np.pi/180
            balance_offset[3] = -gyro_data[0] * self.config.getfloat('Walking','balance_ankle_roll_gain')*np.pi/180
        return balance_offset
        
    def process(self, max_period=np.inf):
        while self.period_counter < max_period:
            # phase-0: initialization
            if self.timer == 0:
                self.update_param_time()
                self.current_phase = 0
            # phase-1: 
            elif (self.timer >= self.param_time['phase_time1'] - self.dt/2) and (self.timer <= self.param_time['phase_time1'] + self.dt/2):
                self.update_param_move()
                self.current_phase = 1
            # phase-2:
            elif (self.timer >= self.param_time['phase_time2'] - self.dt/2) and (self.timer <= self.param_time['phase_time2'] + self.dt/2):
                self.update_param_time()
                # ?
                self.timer = self.param_time['phase_time2']
                self.current_phase = 2
            elif (self.timer >= self.param_time['phase_time3'] - self.dt/2) and (self.timer <= self.param_time['phase_time3'] + self.dt/2):
                self.update_param_move()
                self.current_phase = 3
            
            # calculate position of the endpoint for each leg
            # 1. compute swap
            x_swap = self.sin(self.timer, 
                              self.param_time['x_swap_period_time'], 
                              self.param_move['x_swap_phase_shift'], 
                              self.param_move['x_swap_amp'],
                              self.param_move['x_swap_amp_shift'])
            y_swap = self.sin(self.timer, 
                              self.param_time['y_swap_period_time'], 
                              self.param_move['y_swap_phase_shift'], 
                              self.param_move['y_swap_amp'],
                              self.param_move['y_swap_amp_shift'])
            z_swap = self.sin(self.timer, 
                              self.param_time['z_swap_period_time'], 
                              self.param_move['z_swap_phase_shift'], 
                              self.param_move['z_swap_amp'],
                              self.param_move['z_swap_amp_shift'])        
            a_swap, b_swap, c_swap = 0, 0, 0                 
            # 2. compute move
            x_move_l, y_move_l, z_move_l, c_move_l, x_move_r, y_move_r, z_move_r, c_move_r = self.computer_endpoint_move() 
            # no pitch and roll
            a_move_l, b_move_l, a_move_r, b_move_r = 0,0,0,0
            
            # 3.summation to get total endpoint
            for d in ['l','r']:
                self.z_ep[d].append(eval('z_move_{}'.format(d)))
                for p in ['x','y','z','c']:
                    if p in ['x','y','z']:
                        offset = self.config.getfloat('Walking', p + '_offset')
                        self.end_points[p+d].append(eval('{}_swap + {}_move_{} + offset'.format(p,p,d)))
                    else:
                        self.end_points[p+d].append(eval('{}_swap + {}_move_{}'.format(p,p,d)))
            
            # 4.compute inverse kinematics (IK)
            angles_l = self.computer_ik(self.end_points['xl'][-1],
                                        self.end_points['yl'][-1],
                                        self.end_points['zl'][-1])
            if angles_l:
                for i, (k,v) in enumerate(self.motor_angles.items()):
                    v.append(angles_l[i])
                # angles_r = self.computer_ik(self.end_points['xr'][-1],
                #                           self.end_points['yr'][-1],
                #                           self.end_points['zr'][-1])
                
            # update timer
            self.timer += self.dt
            if self.timer >= self.param_time['period_time']:
                self.timer = 0
                self.period_counter += 1

if __name__ == '__main__':
    walker = Walking()
    walker.param_move['x_move_amp'] = 30.0
    walker.update_param_move()
    walker.update_param_time()
    print(walker.param_time)
    print(walker.param_move)
    
    walker.timer = 8
    print(walker.computer_endpoint())
    
    # walker.process(max_period=4)