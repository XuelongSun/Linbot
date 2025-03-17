更新说明
v2.1.2
DEF ball Robot - Physics - mass : 1→0.055
减轻了球的质量，使机器人踢球时更容易保持平衡。
Reduce the mass of the ball, to make it easier for robot to keep balance.


v2.1.1
修复了一个当踢球关卡出现在第二个拐角处时，球与球洞方向错误的bug。
Fix a bug which causes wrong direction when kick ball part placed at the second turning.
移除了踢球关卡中的金属反光效果，使其更容易被识别。
Adjust the appearance of the ball to make it easier to recognize.

v2.1
替换了2.0版本控制器中的旧函数，以适配新的R2021a版本的webots。
Replaced the old functions in the 2.0 version controller to adapt to the new R2021A version of WEBOTS.


注意
推荐使用python3.7版本，其它版本加载环境时可能会出现ImportError。在webots-工具-首选项-Python command中指向python3.7的路径。

Attention
Python 3.7 is recommended. Other versions may have an Import error when loading the environment.In WEBOTS - Tools - Preferences - Python Command, point to the path to Python 3.7.


调整赛道的方法
切换DEF Reset_Ruler Robot的controller：
Change the controller of DEF Reset_Ruler Robot to custom the map:

controller"Rst_Ruler_random"：
随机生成赛道，包含计时记分功能的裁判系统。每次重新仿真时会刷新赛道，并输出block、block_direc、block_type1、block_type11四项赛道生成的变量值。
Random map. It will print block, block_direc, block_type1, block_type11 variables each time you reset simulation.

controller"Rst_Ruler1"：
生成一个固定赛道，包含计时记分功能的裁判系统，重新仿真时不会变更，以便参赛者训练测试。替换Rst_Ruler1.py开头的MAP_block等四项全局变量即可改变生成的关卡顺序。
Generate a fixed map. You can custom the map by modifying four global variables of Rst_Ruler1.py: MAP_block, MAP_block_direc, MAP_block_type1, MAP_block_type11. We recommend you using the variables printed by random map controller to modify these four global variables.