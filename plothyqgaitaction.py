#matplotlib notebook
import numpy as np
import json
import sqlite3
from enum import Enum, auto
from matplotlib import pyplot as plt
"/home/siwflhc/anaconda3/envs/arm_gpu/bin/python /home/siwflhc/spinningupmodified/plothyq.py"  #run_13_01_2022__08_37_02
class ArmState(Enum):
    Reached = auto()
    InProgress = auto()
    ApproachJointLimits = auto()
    Collision = auto()
    Timeout = auto()
    Undefined = auto()


def adapt_np_array(arr: np.ndarray):
    return json.dumps(arr.tolist())


def convert_np_array(text):
    return np.array(json.loads(text))


def adapt_arm_state(state: ArmState):
    return str(state.name)


def convert_arm_state(state):
    key = state.decode('utf-8')
    return ArmState[key]
#run_25_08_2020__16_46_22

table_name = 'run_17_03_2022__10_48_48' #run_07_09_2020__11_33_10
""" create a database connection to a SQLite database """
conn = None
try:
    conn = sqlite3.connect('minicheetah_balance.db', detect_types=sqlite3.PARSE_DECLTYPES)  #modify place
    print(f'Using sqlite3, version: {sqlite3.sqlite_version}')
except sqlite3.Error as e:
    print(e)

sqlite3.register_adapter(np.ndarray, adapt_np_array)
sqlite3.register_converter("np_array", convert_np_array)
sqlite3.register_adapter(ArmState, adapt_arm_state)
sqlite3.register_converter("armstate", convert_arm_state)


cur = conn.cursor()
sql_statement = \
    ''' 
    select avg(reward),avg(cum_reward),avg(current_x_vel),avg(current_y_vel),avg(z_height_difference_reward),avg(ang_vel),avg(RPY_factor),avg(penalty_joint_limit),avg(penalty_fallen),id,episode_num,avg(penalty_z_speed),avg(friction_penalty),avg(action_penalty_LF),avg(action_penalty_RF),avg(action_penalty_LH),avg(action_penalty_RH)
    from run_18_03_2022__10_06_59
    	Group By episode_num
    '''  #condition is put at where not in "inprogree" or stpe is more or what...   from the saved table  
    #WHERE arm_state == 'Collision' or arm_state == 'ApproachJointLimits'

cur.execute(sql_statement)
data = cur.fetchall()
data2 = [*zip(*data)]
reward = np.array(data2[0])
cum_reward = np.array(data2[1])
x=np.array(data2[2])
y=np.array(data2[3])
z_reward=np.array(data2[4])
ang_vel=np.array(data2[5])
rpy=np.array(data2[6])
jointlimit=np.array(data2[7])
fallen=np.array(data2[8])
id2=np.array(data2[9])
episode=np.array(data2[10])
penalty_z_speed=np.array(data2[11])
friction_penalty=np.array(data2[12])
action_penalty_LF=np.array(data2[13])
action_penalty_RF=np.array(data2[14])
action_penalty_LH=np.array(data2[15])
action_penalty_RH=np.array(data2[16])
# z_height_diff= np.array(data2[2])
# current_z  = np.array(data2[3])
# xreward=np.array(data2[6])
# yreward=np.array(data2[7])
# average_x_different=np.array(data2[17])
# average_y_different=np.array(data2[18])
# action_penalty=np.array(data2[19])


# cur = conn.cursor()
# sql_statement = \
#     ''' 
#     select avg(reward),avg(cum_reward),avg(current_x_vel),avg(current_y_vel),avg(z_height_difference_reward),avg(ang_vel),avg(RPY_factor),avg(penalty_joint_limit),avg(penalty_fallen),id,episode_num,avg(penalty_z_speed),avg(friction_penalty),avg(action_penalty_LF),avg(action_penalty_RF),avg(action_penalty_LH),avg(action_penalty_RH)
#     from run_11_03_2022__12_05_44
#     	Group By episode_num
#     '''  #condition is put at where not in "inprogree" or stpe is more or what...   from the saved table  
#     #WHERE arm_state == 'Collision' or arm_state == 'ApproachJointLimits'


# cur.execute(sql_statement)
# data = cur.fetchall()
# data2 = [*zip(*data)]
# reward = np.array(data2[0])
# cum_reward = np.array(data2[1])
# x=np.array(data2[2])
# y=np.array(data2[3])
# z_reward=np.array(data2[4])
# ang_vel=np.array(data2[5])
# rpy=np.array(data2[6])
# jointlimit=np.array(data2[7])
# fallen=np.array(data2[8])
# id2=np.array(data2[9])
# episode=np.array(data2[10])
# penalty_z_speed=np.array(data2[11])
# friction_penalty=np.array(data2[12])
# action_penalty_LF=np.array(data2[13])
# action_penalty_RF=np.array(data2[14])
# action_penalty_LH=np.array(data2[15])
# action_penalty_RH=np.array(data2[16])




plt.figure(1)
plt.plot(episode,reward,'.',label = "reward")
plt.legend()
plt.title('average_gait_reward_per_train_ep') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig1.jpg', bbox_inches='tight', dpi=700)


plt.figure(2)
plt.plot(episode,cum_reward,'.',label = "cum_reward")
plt.legend()
plt.title('average_cum_reward_per_train_ep') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig2.jpg', bbox_inches='tight', dpi=700)

# plt.figure(3)
# plt.plot(episode,z_height_diff,'.',label = "z_height_diff")
# plt.plot(episode,current_z,'.',label = "current_z")
# plt.legend()
# plt.title('average_z_changes_per_train_ep') 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# plt.savefig('fig3.jpg', bbox_inches='tight', dpi=700)

plt.figure(3)
plt.plot(episode,x,'.',label = "xspeed")
plt.plot(episode,y,'.',label = "yspeed")
plt.legend()
plt.title('current_x_y_speed_per_train_ep') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig3.jpg', bbox_inches='tight', dpi=700)


# plt.figure(42)
# plt.plot(episode,average_x_different,'.',label = "xspeed")
# plt.plot(episode,average_y_different,'.',label = "yspeed")
# plt.legend()
# plt.title('average_x_y_different_per_train_ep') 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# plt.savefig('fig4_2.jpg', bbox_inches='tight', dpi=700)

# plt.figure(5)
# plt.plot(episode,xreward,'.',label = "xreward")
# plt.plot(episode,yreward,'.',label = "yreward")
# plt.plot(episode,z_reward,'.',label = "zreward")
# plt.legend()
# plt.title('average_x_y_z_reward_per_train_ep') 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# plt.savefig('fig5.jpg', bbox_inches='tight', dpi=700)

plt.figure(4)
plt.plot(episode,ang_vel,'.',label = "ang_vel")
plt.plot(episode,rpy,'.',label = "rpy")
plt.legend()
plt.title('average_rpy&angvel_per_train_ep') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig4.jpg', bbox_inches='tight', dpi=700)

plt.figure(5)
plt.plot(episode,jointlimit,'.',label = "penalty_joint_limit")
plt.plot(episode,fallen,'.',label = "penalty_fallen")
#plt.plot(penalty_z_speed,'.',label = "penalty_z_speed")
plt.plot(episode,friction_penalty,'.',label = "friction_penalty")
plt.legend()
plt.title('average_penalty_per_train_ep') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig5.jpg', bbox_inches='tight', dpi=700)

plt.figure(6)
plt.plot(episode,action_penalty_LF,'.',label = "aaction_penalty_LF")
plt.plot(episode,action_penalty_RF,'.',label = "action_penalty_RF")
plt.plot(episode,action_penalty_LH,'.',label = "action_penalty_LH")
plt.plot(episode,action_penalty_RH,'.',label = "action_penalty_RH")
plt.legend()
plt.title('penalty_gaits_per_train_ep') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig6.jpg', bbox_inches='tight', dpi=500)




cur = conn.cursor() #run_15_02_2022__08_43_52
sql_statement = \
    ''' 
    select reward,cum_reward,current_x_vel,current_y_vel,z_height_difference_reward,ang_vel,RPY_factor,penalty_joint_limit,penalty_fallen,id,episode_num,penalty_z_speed,friction_penalty,action_penalty_LF,action_penalty_RF,action_penalty_LH,action_penalty_RH,step_num,X_speed
    from run_18_03_2022__10_06_59
    ''' 
#where id >= 1001 AND id <=3000;
cur.execute(sql_statement)
data = cur.fetchall()
data2 = [*zip(*data)]
reward = np.array(data2[0])
cum_reward = np.array(data2[1])
x=np.array(data2[2])
y=np.array(data2[3])
z_reward=np.array(data2[4])
ang_vel=np.array(data2[5])
rpy=np.array(data2[6])
jointlimit=np.array(data2[7])
fallen=np.array(data2[8])
id2=np.array(data2[9])
episode=np.array(data2[10])
penalty_z_speed=np.array(data2[11])
friction_penalty=np.array(data2[12])
action_penalty_LF=np.array(data2[13])
action_penalty_RF=np.array(data2[14])
action_penalty_LH=np.array(data2[15])
action_penalty_RH=np.array(data2[16])
step_num=np.array(data2[17])
X_speed=np.array(data2[18])





# cur = conn.cursor()
# sql_statement = \
#     ''' 
#     select reward,cum_reward,z_height_difference,current_z,average_x_vel,average_y_vel,reward_x_speed,reward_y_speed,z_height_difference_reward,ang_vel,RPY_factor,penalty_joint_limit,penalty_fallen,id,episode_num
#     from run_02_11_2021__16_35_06
#     ''' 
 
# cur.execute(sql_statement)
# data = cur.fetchall()
# data2 = [*zip(*data)]
# reward = np.append(reward ,np.array(data2[0]))
# cum_reward =np.append(cum_reward,np.array(data2[1]))
# z_height_diff= np.append(z_height_diff,np.array(data2[2]))
# current_z  = np.append(current_z,np.array(data2[3]))
# x=np.append(x,np.array(data2[4]))
# y=np.append(y,np.array(data2[5]))
# xreward=np.append(xreward,np.array(data2[6]))
# yreward=np.append(yreward,np.array(data2[7]))
# z_reward=np.append(z_reward,np.array(data2[8]))
# ang_vel=np.append(ang_vel,np.array(data2[9]))
# rpy=np.append(rpy,np.array(data2[10]))
# jointlimit=np.append(jointlimit,np.array(data2[11]))
# fallen=np.append(fallen,np.array(data2[12]))
# id2=np.append(id2,np.array(data2[13]))
# episode=np.append(episode,np.array(data2[14]))


plt.figure(11)
plt.plot(id2,reward,'.',label = "gait_reward")
plt.legend()
plt.title('reward_per_train_ts') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig11.jpg', bbox_inches='tight', dpi=500)

plt.figure(12)
plt.plot(id2,cum_reward,'.',label = "cum_reward")
plt.legend()
plt.title('cum_reward_per_train_ts') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig12.jpg', bbox_inches='tight', dpi=500)

# plt.figure(13)
# plt.plot(id2,z_height_diff,'.',label = "z_height_diff")
# plt.plot(id2,current_z,'.',label = "current_z")
# plt.legend()
# plt.title('z_changes_per_train_ts') 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# plt.savefig('fig13.jpg', bbox_inches='tight', dpi=500)

plt.figure(13)
plt.plot(id2,x,'.',label = "xspeed")
plt.plot(id2,y,'.',label = "yspeed")
plt.legend()
plt.title('x_y_speed_per_train_ts') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig13.jpg', bbox_inches='tight', dpi=500)


# plt.figure(18)
# plt.plot(id2,average_x_different,'.',label = "xspeed")
# plt.plot(id2,average_y_different,'.',label = "yspeed")
# plt.legend()
# plt.title('average_x&y_different_per_train_ts') 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# plt.savefig('fig18.jpg', bbox_inches='tight', dpi=500)

# plt.figure(15)
# plt.plot(id2,xreward,'.',label = "xreward")
# plt.plot(id2,yreward,'.',label = "yreward")
# plt.plot(id2,z_reward,'.',label = "zreward")
# plt.legend()
# plt.title('x_y_z_reward_per_train_ts') 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# plt.savefig('fig15.jpg', bbox_inches='tight', dpi=500)

plt.figure(14)
plt.plot(id2,ang_vel,'.',label = "ang_vel")
plt.plot(id2,rpy,'.',label = "rpy")
plt.legend()
plt.title('rpy&angvel_per_train_ts') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig14.jpg', bbox_inches='tight', dpi=500)

plt.figure(15)
plt.plot(id2,jointlimit,'.',label = "penalty_joint_limit")
plt.plot(id2,fallen,'.',label = "penalty_fallen")
plt.plot(id2,friction_penalty,'.',label = "friction_penalty")
plt.legend()
plt.title('penalty_per_train_ts') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig15.jpg', bbox_inches='tight', dpi=500)

plt.figure(16)
plt.plot(id2,action_penalty_LF,'.',label = "aaction_penalty_LF")
plt.plot(id2,action_penalty_RF,'.',label = "action_penalty_RF")
plt.plot(id2,action_penalty_LH,'.',label = "action_penalty_LH")
plt.plot(id2,action_penalty_RH,'.',label = "action_penalty_RH")
plt.legend()
plt.title('penalty_gaits_train_ts') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig16.jpg', bbox_inches='tight', dpi=500)



plt.figure(17)
plt.plot(id2,X_speed,'.',label = "X_speed")
plt.legend()
plt.title('X_speed_per_train_ts') 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('fig17.jpg', bbox_inches='tight', dpi=500)


# thresholdbefore900 = threshold2.copy()
# thresholdafter900 = threshold2.copy()

# for i in range (len(step_num)): 
#     if step_num[i-1] >=900:
#         thresholdbefore900[i-1]=0
#     elif step_num[i-1] <900:
#         thresholdafter900[i-1]=0

# idbefore900=id2.copy()
# idafter900=id2.copy()

# for i in range (len(threshold2)): 
#     if thresholdbefore900[i-1] ==0:
#         idbefore900[i-1]=0
#     elif thresholdafter900[i-1] ==0:
#         idafter900[i-1]=0

# plt.figure(18)
# plt.plot(idbefore900,thresholdbefore900,'g.',label = "before900")
# plt.plot(idafter900,thresholdafter900,'b.',label = "after900")
# plt.legend()
# plt.title('average_penalty_per_train_ts') 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# plt.savefig('fig18.jpg', bbox_inches='tight', dpi=500)



# plt.show() 
