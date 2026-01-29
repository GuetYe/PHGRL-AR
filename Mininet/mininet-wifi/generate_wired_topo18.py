#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/7/13 17:30
@File:generate_wired_topo.py
@Desc: 生成有线拓扑
"""
from utils import parse_xml_data,generate_switch_port,remove_finish_file,get_mininet_device,run_ip_add_default,write_pingall_time,create_topo_links_info_xml
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import RemoteController
from mininet.util import dumpNodeConnections
from mininet.cli import CLI
import time

# 利用数据生成topo
class Graph2Topo(Topo):
    def __init__(self, graph, host_switch_list):
        super(Graph2Topo, self).__init__()
        self.node_idx = graph.nodes
        self.edges_pairs = graph.edges
        self.host_port = 9

        _bw = 6  # 带宽
        _d = 0.1  # 时延
        _l = 0   # 丢失
        _distance = 1  # 距离，“1”表示有线,相当于跳数

        special_setting = {(3,6), (3,7), (6,3), (7,3)}
        s_bw = 10  # 带宽


        # 添加交换机
        switches = {}
        for s in self.node_idx:
            switches.setdefault(s,self.addSwitch(f's{s}'))
            print('添加交换机')

        # 添加链路  需要交换机和相应的端口,端口的分配可以通过图的度来生成。
        # 获取端口
        switch_port_dict = generate_switch_port(graph)
        links_info = {}  # 用于存储链路信息
        for l in self.edges_pairs:
            port1 = switch_port_dict[l[0]].pop(0) + 1  # 这里加1是因为端口号是从1开始编码的
            port2 = switch_port_dict[l[1]].pop(0) + 1
            if l in special_setting:
                self.addLink(switches[l[0]], switches[l[1]], port1=port1,port2=port2,bw=s_bw,delay=_d,loss=_l)
                links_info.setdefault(l, {"port1": port1, "port2": port2, "free_bw": s_bw, "delay": _d, "loss": _l,
                                          "used_bw": 0, "pkt_err": 0, "pkt_drop": 0, "distance": _distance})
            else:
                self.addLink(switches[l[0]], switches[l[1]], port1=port1,port2=port2,bw=_bw,delay=_d,loss=_l)
                links_info.setdefault(l,{"port1":port1,"port2":port2,"free_bw":_bw,"delay":_d,"loss":_l,"used_bw": 0, "pkt_err": 0, "pkt_drop": 0, "distance": _distance})
        create_topo_links_info_xml(links_info_xml_path,links_info)

        # 添加host
        host_switch = []  # 主机交换机列表
        for host_id, switch_id in host_switch_list:
            if switch_id in host_switch:
                self.host_port += 1
            else:
                host_switch.append(switch_id)
            _h = self.addHost(f'h{host_id}',ip=f'10.0.0.{host_id}',mac=f'00.00.00.0{host_id}' if len(str(host_id))==1 else f'00.00.00.{host_id}')
            self.addLink(_h, f's{switch_id}',port1=0,port2=self.host_port,bw=2*s_bw)

# 启动topo
def main(topo,finish_file,host_switch_list):
    print('===删除旧文件')
    remove_finish_file(finish_file)

    # 创建网络
    net = Mininet(topo=topo,link=TCLink,controller=RemoteController,waitConnected=True,build=False)
    c0 = net.addController(name='c0',ip='127.0.0.1',port=6633)

    net.build()  # 创建
    net.start()  # 启动

    print('获取主机服务列表')
    hosts = get_mininet_device(net,host_switch_list,device='h')

    print("===转存主机连接")
    dumpNodeConnections(net.hosts)

    print("===等待ryu初始化")
    time.sleep(5)

    # 添加网关ip
    run_ip_add_default(hosts)

    # ping
    net.pingAll()

    write_pingall_time(finish_file)

    CLI(net)
    net.stop()


if __name__ == '__main__':
    # 拓扑文件名
    # topo_file = './topologies/wired_topo_test.xml'
    topo_file = './topologies/wired_topo_18.xml'
    finish_file = './finish_time.json'
    links_info_xml_path = './save_links_info/links_info.xml'
    # 分析xml文件，获取数据
    graph, nodes_num, edges_num, host_switch_list = parse_xml_data(topo_file)

    print(nodes_num, edges_num, host_switch_list)

    topo = Graph2Topo(graph, host_switch_list)

    main(topo, finish_file, host_switch_list)
