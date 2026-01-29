import setting
import os
import time

from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER
from ryu.lib import hub
from ryu.lib.packet import packet, arp, ipv4, ethernet
from ryu.ofproto import ofproto_v1_3
from pathlib import Path
import csv
import asyncio
import networkx as nx
from utils import HubSendGraphData, json_default
import eventlet
eventlet.monkey_patch()
import requests
from multiprocessing import Process
from utils import thread_save_dict
import json

# # 解决too many files 的问题
# import resource
# resource.setrlimit(resource.RLIMIT_NOFILE,(2048,2048))

class NetworkShortestPath(app_manager.RyuApp):
    """
    测量链路的最短路径
    """
    OFP_VERSION = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *_args, **_kwargs):
        super(NetworkShortestPath, self).__init__(*_args, **_kwargs)
        self.name = "network_shortest_path"
        self.now_time = time.strftime("%H:%M:%S")  # 获取当前时间
        self.network_structure = lookup_service_brick("discovery")
        self.network_monitor = lookup_service_brick("monitor")
        self.network_delay = lookup_service_brick("detector")
        self.weight_dir = ''
        self.pickle_dir = ''
        self.count = 0  # 保存文件的序号（名字）
        self.client = HubSendGraphData(server_ip=setting.server_ip, server_port=setting.server_port,
                                    state_path=setting.server_state_path, request_path=setting.server_request_path,
                                    action_path=setting.server_action_path)
        self.effective_time_flow_table = {}  # {(dp_id,in_port):(last_flow_time,last_data_time,out_port)}
        # 创建客户端，以供状态信息发送
        # print('==>network_shortest_path:',time.ctime())
        self.shortest_thread = hub.spawn(self.super_scheduler)

        
    def super_scheduler(self):
        """
            总调用线程，
            self.discovery.scheduler() 网络探测
            self.monitor.scheduler()  网络带宽, loss监测
            self.detector.scheduler()  时延检测
            self.create_weight_graph()  刷新图权重
            self.save_links_weight(self.count)  保存图信息
        """
        # hub.sleep(setting.DISCOVERY_PERIOD)
        while True:
            hub.sleep(setting.SCHEDULE_PERIOD)
            # self.network_structure.scheduler()
            # self.network_monitor.scheduler()
            # self.network_delay.scheduler()
            # self.create_weight_graph()
            
            if self.count == 0:
                # self.create_weight_graph()
                self.create_dir()
            self.count += 1
            self.save_links_weight(self.count)


    def create_dir(self):
        """
        创建保存权重的文件夹
        """
        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.weight_dir = './weight/' + now_time
        Path.mkdir(Path(self.weight_dir), exist_ok=True, parents=True)
        self.csv_dir = './csv' + '/' + now_time
        Path.mkdir(Path(self.csv_dir), exist_ok=True, parents=True)
        self.pickle_dir = './pickle' + '/' + now_time
        Path.mkdir(Path(self.pickle_dir), exist_ok=True, parents=True)

    
    def save_links_weight(self, count):
        """
            保存graph信息, csv和pkl
        """
        name = f"{count}-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.save_txt_graph(name)
        # self.save_csv_graph(name)
        self.save_pickle_graph(name)

    def save_txt_graph(self, name):
        """
            保存图信息的txt文件
            ./weight/now_time/name.txt
        """
        with open(self.weight_dir + "/" + name + '.txt', 'w+', newline='') as f:
            _graph = self.network_structure.graph.copy()
            for key in list(_graph.edges(data=True)):
                # key[2]['distance'] = self.network_structure.ap_distance[(key[0], key[1])]
                f.write(str(key) + '\n')
       
    def save_csv_graph(self, name):
        """
            保存图信息的csv文件
            ./weight/now_time/name.csv
        """
        with open(self.csv_dir + "/" + name + '.csv', 'w+', newline='') as f:
            f_csv = csv.writer(f)
            _graph = self.network_structure.graph.copy()
            f_csv.writerows(list(_graph.edges(data=True)))

    def save_pickle_graph(self, name):
        """
            保存图信息的pkl文件
            ./pkl/now_time/name.pkl
        """
        _path = self.pickle_dir / Path(name + '.pkl')
        _graph = self.network_structure.graph.copy()
        # 发送图数据给服务器
        # asyncio.run(self.client.send_graph_data(_graph))  # python3.7+ 后的版本
        hub.spawn(self.client.send_graph_data,_graph)

        # loop = asyncio.new_event_loop()
        # loop.run_until_complete(self.client.send_graph_data(_graph))
        nx.write_gpickle(_graph, _path)


    def create_weight_graph(self):
        """
        通过monitor 和 detector 测量的带宽和时延, 设置graph的权重
        """
        # print(" update the bw and delay values ")
        # print("shortest--->=====>", self.network_monitor.port_free_bandwidth)
        self.network_monitor.create_bandwidth_graph()  # 调用monitor更新链路的带宽
        self.network_monitor.create_loss_graph()  # 调用monitor更新链路的时延
        self.network_delay.create_delay_graph()  # 调用delay的方法更新链路的时延

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
        处理数据包发过来的数据
        """
        # print("shortest------>PacketIn")
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)
        # print("arp_pkt------>", arp_pkt)
        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        # print("ipv4_pkt------>", ipv4_pkt)


        if isinstance(ipv4_pkt, ipv4.ipv4):
            # print("shortest--->=====> IPv4 processing")
            if len(pkt.get_protocols(ethernet.ethernet)):
                eth_type = pkt.get_protocols(ethernet.ethernet)[0].ethertype
                # print("shortest-----------caleulate shortest path----------------")
                # 根据路径下发流表
                # print('flow_id', (ipv4_pkt.src, ipv4_pkt.dst, ipv4_pkt.proto))
                self.calculate_shortest_paths(msg, eth_type, ipv4_pkt.src, ipv4_pkt.dst)

    def calculate_shortest_paths(self, msg, eth_type, src_ip, dst_ip):
        """
        根据解析出的消息计算最短路径
        """
        datapath = msg.datapath
        in_port = msg.match['in_port']

        # 1、找出位置
        dpid = self.network_structure.apid_dict.get(datapath.id)
        src_dst_switches = self.get_switches(dpid, in_port, src_ip, dst_ip)
        # print("src_dst_switches ", src_dst_switches)
        # TODO
        last_packet_time = self.effective_time_flow_table.get((dpid,in_port,src_ip,dst_ip),
                                                            (-setting.flow_update_interval,
                                                             -setting.flow_hard_timeout,None))[0] # 获取上一次流表更新的默认值
        last_flow_time = self.effective_time_flow_table.get((dpid,in_port,src_ip,dst_ip),
                                                            (-setting.flow_update_interval,
                                                             -setting.flow_hard_timeout,None))[1] # 获取上一次流表更新的默认值
        # print('network_shortest_path.py', self.effective_time_flow_table)
        # print((dpid,in_port,src_ip,dst_ip))
        # print('之前的时间', last_packet_time)
        # print('时间差：', time.time()-last_packet_time)

        if time.time()-last_packet_time < setting.flow_update_interval \
                and time.time()-last_flow_time < setting.flow_hard_timeout:   # 如果时间间隔低于设置的时间
            # 延迟一段时间后下发数据
            # hub.sleep(setting.flow_update_interval)
            print('network_shortest_path.py',"===========================>")
            out_port = self.effective_time_flow_table.get((dpid, in_port, src_ip, dst_ip),
                                                            (-setting.flow_update_interval,
                                                             -setting.flow_hard_timeout, None))[2] # 获取上一次流表更新的默认值
            if out_port == None:
                hub.sleep(setting.flow_update_interval)
            else:
                pass
            first_dp = self.network_monitor.datapaths_table[dpid]
            buffer_id = msg.buffer_id
            data = msg.data
            while not out_port:
                hub.sleep(setting.flow_update_interval)
                print('network_shortest_path.py',self.effective_time_flow_table)
                out_port = self.effective_time_flow_table.get((dpid, in_port, src_ip, dst_ip),
                                                              (-setting.flow_update_interval,
                                                               -setting.flow_hard_timeout, None))[2]  # 获取上一次流表更新的默认值
            last_flow_time = self.effective_time_flow_table.get((dpid, in_port, src_ip, dst_ip),
                                                              (-setting.flow_update_interval,
                                                               -setting.flow_hard_timeout, None))[1]
            self._build_packet_out(first_dp, buffer_id, in_port, out_port, data)
            self.effective_time_flow_table[(dpid, in_port, src_ip, dst_ip)] = (time.time(), last_flow_time, out_port)

        elif src_dst_switches:
            src_switch, dst_switch = src_dst_switches
            if src_switch:
                # 1、判断源节点和目的节点是否在任播的范围，若在，查询http动作列表
                # print(f'从节点{src_switch}发送到{dst_switch}')
                # print(f'源节点_ip:{src_ip}，目的节点_ip:{dst_ip}')
                if (src_ip in setting.src_node_ip_list and dst_ip in setting.dst_node_ip_list)\
                        or (src_ip in setting.dst_node_ip_list and dst_ip in setting.src_node_ip_list):
                    # try:
                    response = requests.get(
                        f'http://{setting.data_server_ip}:{setting.data_server_port}/api/action',
                        timeout=5
                    )

                    if response and setting.is_routing:
                        sample_data = response.json()
                        if sample_data:
                            # print(f'获取到的动作数据:{sample_data}')
                            print('下层路径',sample_data['low_act_dict'])
                            path = sample_data['low_act_dict'].get(f'({int(src_ip.split(".")[-1])}, {int(dst_ip.split(".")[-1])})')
                            # print(path)
                            if path and src_switch in path:  # 有一种情况，源节点不在路径中，此时需要用最短路径
                                # print(f'path[0]:{path[0]},logict:{src_switch == path[0]}')
                                # if src_switch == path[0]:  # 控制首个流表下发--硬性删除流表
                                #     self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data,
                                #                       is_control=True,is_first=True)
                                # else:
                                #     self.logger.info("control=====>[PATH] [%s <----> %s]: %s" % (src_ip, dst_ip, path))
                                #     # 3、下发流表
                                #
                                #     self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data,is_control=True)
                                self.logger.info("shortest=====>[PATH] [%s <----> %s]: %s,%s" % (src_ip, dst_ip, path,time.time()))
                                # 3、下发流表
                                self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data,is_control=True)
                                save_process = Process(target=thread_save_dict,
                                                       args=('./save_action/action.json',sample_data))  # TODO
                                save_process.start()
                            else:
                                path = self.calculate_path(src_switch, dst_switch)
                                self.logger.info("shortest=====>[PATH] [%s <----> %s]: %s,%s" % (src_ip, dst_ip, path,time.time()))
                                # 3、下发流表
                                self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data,
                                                  is_control=True)
                        else:
                            path = self.calculate_path(src_switch, dst_switch)
                            self.logger.info("shortest=====>[PATH] [%s <----> %s]: %s,%s" % (src_ip, dst_ip, path,time.time()))
                            # 3、下发流表
                            self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data,
                                              is_control=True)
                    else:
                        path = self.calculate_path(src_switch, dst_switch)
                        self.logger.info("shortest=====>[PATH] [%s <----> %s]: %s,%s" % (src_ip, dst_ip, path,time.time()))
                        # 3、下发流表
                        self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data,is_control=True)

                        if response:  # 保存动作
                            sample_data = response.json()
                            save_process = Process(target=thread_save_dict,
                                                   args=('./save_action/action.json', sample_data))  # TODO
                            save_process.start()


                    # except Exception as e:
                    #     print(f'连接错误：{e}')
                    #     # 2、计算最短路径
                    #     path = self.calculate_path(src_switch, dst_switch)  # TODO
                    #     self.logger.info("shortest=====>[PATH] [%s <----> %s]: %s" % (src_ip, dst_ip, path))
                    #     # 3、下发流表
                    #     # if path[0] == int(src_ip.split(".")[-1]):  # 控制首个流表下发--硬性删除流表
                    #     #     self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data,
                    #     #                       is_control=True, is_first=True)
                    #     # else:
                    #     self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data,is_control=True)
                else:
                    path = self.calculate_path(src_switch, dst_switch)  # TODO
                    self.logger.info("shortest=====>[PATH] [%s <----> %s]: %s,%s" % (src_ip, dst_ip, path, time.time()))
                    # 3、下发流表
                    self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data)
                # 更新最短路径列表并保存文件
                self.network_structure.shortest_path_table[(src_switch, dst_switch)] = path

                # print(type(self.network_structure.shortest_path_table))
                # print(self.network_structure.shortest_path_table)
                # request = requests.post(
                #     f'http://{setting.data_server_ip}:5001/api/path_table',
                #     json=json_default(self.network_structure.shortest_path_table),
                #     headers={'Content-Type': 'application/json'}
                # )

                save_process = Process(target=thread_save_dict,
                                       args=('./savepath/path_table.json', self.network_structure.shortest_path_table))  # TODO
                save_process.start()

        else:
            self.logger.info(f"shortest----> src_dst_switches, 135：{src_dst_switches}")

    def get_switches(self, dpid, in_port, src_ip, dst_ip):
        """
        根据src_ip求得dpid
        """
        src_switch = dpid
        dst_switch = list()

        src_location = self.network_structure.get_host_ip_location(src_ip)  # (dpid, in_port)
        if in_port in self.network_structure.not_use_ports[dpid]:
            # print(f"shortest------>src_location == (dpid, in_port): {src_location} == {(dpid, in_port)}", )
            if (dpid, in_port) == src_location:
                src_switch = src_location[0]
            else:
                return None

        dst_location = self.network_structure.get_host_ip_location(dst_ip)  # XXX 
        if dst_location:
    
            dst_switch = dst_location[0]
        return src_switch, dst_switch


    def get_port(self, dst_ip):
        """
        根据目的ip获得出去的端口
        """
        for key in self.network_structure.access_table.keys():  # {(dpid, in_port): (src_ip, src_mac)}
            if dst_ip == self.network_structure.access_table[key][0]:
                dst_port = key[1]
                return dst_port
        return None

    def get_port_pair(self, src_dpid, dst_dpid):
        """
        根据源dpid和目的dpid获得src.port_no, dst.port_no
        """
        if (src_dpid, dst_dpid) in self.network_structure.link_port_table:
            return self.network_structure.link_port_table[(src_dpid, dst_dpid)]  # {(src.dpid, dst.dpid): (src.port_no, dst.port_no)}
        elif (dst_dpid, src_dpid) in self.network_structure.link_port_table:
            return self.network_structure.link_port_table[
                (dst_dpid, src_dpid)][::-1]  # {(src.dpid, dst.dpid): (src.port_no, dst.port_no)}
        else:
            print("shortest--->dpid: %s -> dpid: %s is not in links", (src_dpid, dst_dpid))
            print("shortest--->link_port_table", self.network_structure.link_port_table)
            return None
    
    def calculate_path(self, src_dpid, dst_dpid):
        """
        计算最短路径
        """
        self.network_structure.calculate_shortest_paths(src_dpid, dst_dpid, setting.WEIGHT)
        shortest_path = self.network_structure.shortest_path_table[(src_dpid, dst_dpid)]  # {(src.dpid, dst.dpid): [path]}源交换机到目的交换机的最短路径
        return shortest_path

    def install_flow(self, path, eth_type, src_ip, dst_ip, in_port, buffer_id, data=None, is_control=False):
        """
        多种情况需要考虑, 根据条件判断走哪一个端口
        """
        if path is None or len(path) == 0:
            print("shortest--->Path Error")
            return
        else:
            first_dp = self.network_monitor.datapaths_table[path[0]]
            if is_control:
                print(f'{src_ip}和{dst_ip}之间为动态流表')

            if len(path) > 2:
                # print("shortest--->len(path) > 2")
                for i in range(1, len(path) - 1):
                    port_pair = self.get_port_pair(path[i - 1], path[i])
                    port_pair_next = self.get_port_pair(path[i], path[i + 1])
                    # print("shortest--->len(path) > 2 port_pair, port_pair_next", port_pair, port_pair_next)

                    if port_pair and port_pair_next:
                        # TODO: 为什么port_pair[1]是src_port?  同一个交换机的不同口, 见图
                        src_port, dst_port = port_pair[1], port_pair_next[0]  # 同一个交换机的不同口, 见图
                        datapath = self.network_monitor.datapaths_table[path[i]]

                        # 交换机A ----->（src_port）交换机B（dst_port） -----> 交换机C
                        # 下发正向流表   （同一个交换机里，正向，从src_port进，dst_port出）
                        # 新加流表第一项流表控制（促使这里的流表最先消失，后面流表可以空转消失）

                        self.send_flow_mod(datapath, eth_type, src_ip, dst_ip, src_port, dst_port,
                                               is_control=is_control)

                        # 交换机A <-----（src_port）交换机B（dst_port） <----- 交换机C
                        # 下发反向流表   （同一个交换机里，正向，从dst_port进，src_port出）
                        self.send_flow_mod(datapath, eth_type, dst_ip, src_ip, dst_port, src_port,
                                               is_control=is_control)
                    else:
                        print(f"shortestERROR--->len(path) > 2 "
                              f"path_0, path_1, port_pair: {path[i - 1], path[i], port_pair}, "
                              f"path_1, path_2, next_port_pair: {path[i], path[i + 1], port_pair_next}")
                        return

            if len(path) > 1:
                # TODO: 大于2，就满足大于1， 所以会运行里面的给第一个和最后一个swicth下flow
                # print("shortest--->install_flow: len(path) == 2")  
                port_pair = self.get_port_pair(path[-2], path[-1])

                if port_pair is None:
                    print("shortest--->port not found")
                    return

                src_port = port_pair[1]
                dst_port = self.get_port(dst_ip)
                if dst_port is None:
                    print("shortest--->Last port is not found")
                    return

                last_dp = self.network_monitor.datapaths_table[path[-1]]
                # TODO: 下发最后一个交换机的流表
                self.send_flow_mod(last_dp, eth_type, src_ip, dst_ip, src_port, dst_port, is_control=is_control)
                if path[-1] in setting.src_node_list:
                    self.send_flow_mod(last_dp, eth_type, dst_ip, src_ip, dst_port, src_port, is_control=is_control,is_first=True)
                else:
                    self.send_flow_mod(last_dp, eth_type, dst_ip, src_ip, dst_port, src_port,is_control=is_control)

                port_pair = self.get_port_pair(path[0], path[1])
                if port_pair is None:
                    print("shortest--->port not found in -2 switch")
                    return

                out_port = port_pair[0]
                # TODO: 发送第一个交换机流表
                if path[0] in setting.src_node_list:
                    self.send_flow_mod(first_dp, eth_type, src_ip, dst_ip, in_port, out_port,is_control=is_control,is_first=True)
                else:
                    self.send_flow_mod(first_dp, eth_type, src_ip, dst_ip, in_port, out_port, is_control=is_control)
                self.send_flow_mod(first_dp, eth_type, dst_ip, src_ip, out_port, in_port,is_control=is_control)  # 作用是什么？
                # 第一个交换机数据处理
                self.effective_time_flow_table[(path[0], in_port, src_ip, dst_ip)] = (time.time(), time.time(), out_port)
                self._build_packet_out(first_dp, buffer_id, in_port, out_port, data)
            else:
                out_port = self.get_port(dst_ip)
                if out_port is None:
                    print("shortest--->out_port is None in same dp")
                    return
                self.send_flow_mod(first_dp, eth_type, src_ip, dst_ip, in_port, out_port,is_control=is_control)
                self.send_flow_mod(first_dp, eth_type, dst_ip, src_ip, out_port, in_port,is_control=is_control)
                # 第一个交换机数据处理
                self.effective_time_flow_table[(path[0], in_port, src_ip, dst_ip)] = (time.time(), time.time(), out_port)
                self._build_packet_out(first_dp, buffer_id, in_port, out_port, data)

    def send_flow_mod(self, datapath, eth_type, src_ip, dst_ip, src_port, dst_port,is_control=False,is_first=False):
        """
        开始下发流表
        """
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(dst_port)]
        match = parser.OFPMatch(in_port=src_port, eth_type=eth_type, ipv4_src=src_ip, ipv4_dst=dst_ip)
        self.add_flow(datapath, 1, match, actions,is_control=is_control,is_first=is_first)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None,is_control=False,is_first=False):
        """
        安装下发流表项
        """
        ofproto = datapath.ofproto
        ofp_parser = datapath.ofproto_parser

        inst = [ofp_parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if is_control:
            mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority, idle_timeout=setting.flow_idle_timeout,
                                        match=match, instructions=inst)  # 非首条流表进行软性消失
            # mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority,
            #                             match=match, instructions=inst)  # 非首条流表进行软性消失
            if is_first:
                mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority, hard_timeout=setting.flow_hard_timeout,
                                            match=match, instructions=inst)    # 首条流表30秒硬性消失
                # mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority,
                #                             match=match, instructions=inst)    # 首条流表30秒硬性消失
                print(f'控制条件下，首个流表项安装成功')
            print(f'控制条件下，下发动态流表')


        elif buffer_id:
            mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority, idle_timeout=15,
                                        hard_timeout=60, match=match, instructions=inst)
            print(f'buffer_id条件下，下发动态流表')
        else:
            priority = 2
            mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority,
                                        match=match, instructions=inst)
        datapath.send_msg(mod)

    def _build_packet_out(self, datapath, buffer_id, src_port, dst_port, data):
        """
        构造输出的包
        """
        actions = []  # 动作指令集
        if dst_port:
            actions.append(datapath.ofproto_parser.OFPActionOutput(dst_port))

        msg_data = None
        if buffer_id == datapath.ofproto.OFP_NO_BUFFER:
            if data is None:
                return None
            msg_data = data

        # send packet out msg  to datapath
        out = datapath.ofproto_parser.OFPPacketOut(datapath=datapath, buffer_id=buffer_id,
                                                   in_port=src_port, actions=actions, data=msg_data)
        if out:
            datapath.send_msg(out)

