from pathlib import Path
from functools import reduce

WORK_DIR = Path.cwd().parent

# setting.py
FACTOR = 0.9  # bw的影响因子为0.9，时延的影响因子为0.1

METHOD = 'dijkstra'  # 计算最短路径所使用的方法

DISCOVERY_PERIOD = 10  # 发现网络结构的周期（s）

MONITOR_PERIOD = 5  # 流量监控的周期，bw

DELAY_PERIOD = 1.3  # 检测周期， delay

SCHEDULE_PERIOD = 5  # 最短转发网络感知期

PRINT_SHOW = False  # 是否显示print

INIT_TIME = 30  # 等待初始化

PRINT_NUM_OF_LINE = 8  # 一行打印8个值

LOGGER = True  # 是否保存日志

LINKS_INFO = WORK_DIR  / "mininet_wifi/save_links_info/links_info.xml"  # 链路信息的xml文件路径

WEIGHT = 'bw'

# 初始化源节点和目的节点
src_node_list = [3]
dst_node_list = [15, 16, 17, 18]
# src_node_ip_list = ['192.168.0.1']
# dst_node_ip_list = ['192.168.0.15', '192.168.0.16', '192.168.0.17', '192.168.0.18']
# src_node_ip_list = [f'192.168.0.{node}' for node in src_node_list]
# dst_node_ip_list = [f'192.168.0.{node}' for node in dst_node_list]

# 有线
src_node_ip_list = [f'10.0.0.{node}' for node in src_node_list]
dst_node_ip_list = [f'10.0.0.{node}' for node in dst_node_list]

# print(src_node_ip_list)
# print(dst_node_ip_list)

# 服务器ip和port
# server_ip = '10.0.6.101'
server_ip = '10.33.32.140'
data_server_ip = server_ip  # 前后的名字不统一，这里直接用赋值了

# 动态流表更新时间
flow_hard_timeout = 5
flow_idle_timeout = 20

server_port = 5000
data_server_port =server_port

# 数据服务器路径
server_state_path = '/api/state'
server_action_path = '/api/action'
server_request_path = '/api/request'

# 路径文件夹路径
path_file_path = './savepath/path_table.json'  # 源代码network_structure.py中写死

# 动作文件路径
dst_pro_file_path = './dst_node_pro_file/dst_node_pro.json'  # anycast_tcp_client.py和network_shortest_path.py中写死

# 流表更新间隔时间
flow_update_interval = 0.1

# 是否路由控制(一定要记得修改)
is_routing = False



# finish_time_file = WORK_DIR / "mininet/finish_time.json"
