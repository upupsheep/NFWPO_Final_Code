# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
from numpy import random
from collections import deque
import heapq
import time
import json
import os
import sys
import inspect
import math 
#import tensorflow as tf
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common import sender_obs, config
from common.simple_arg_parse import arg_or_default
inf_bw = 5000
MAX_CWND = 5000
MIN_CWND = 4
OB_TIME = 1
Utilization_rate=500
send_test_line=[1,250,500,750]
send_test_num=[1000,1000,1000,1000]
REWARD_CONTROL = 4
sender2_TCP = False
MAX_RATE = 1000
MIN_RATE = 1
packet_return_size = 1000
REWARD_SCALE = 0.1
Change_rate_step = 200
Sender0_mode = 3 #0:RL 1:TCP 2:ACP 3:Utilization
MAX_STEPS = 1000
step_control = 0
EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'
EVENT_TYPE_RETURN = 'R'
EVENT_TYPE_FIN = 'F'
BYTES_PER_PACKET = 1500
LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0
USE_LATENCY_NOISE = False
MAX_LATENCY_NOISE = 1.001
USE_CWND = False
alpha=0.95
ACP_step_size=1
prob_test_line=[1,100,200,300]
prob_test_num=[10,80,20,90]
def debug_pack(pack):
    print(pack.packet_ID)
    print(pack.content)
    print(pack.spawn_time)
    print(pack.now_link)
    print(pack.event_type)
    print(pack.path)
    print(pack.path_num)
    print(pack.home_path)
class Packet():
    def __init__(self, p_ID,pa_ID,spawn,path_num,path,con=[]):
        self.packet_ID=p_ID
        self.path_ID=pa_ID
        self.content=con
        self.spawn_time=spawn
        self.now_link=[0,0]
        self.event_type=EVENT_TYPE_SEND
        self.path=path
        self.path_num=0
        self.home_path=[]
        self.link1_bw=0
        self.link2_bw=0
        self.utilization=0
        self.utilization0=0
    def reset(self):
        self.packet_ID=0
        self.content=[]
        self.spawn_time=0
        self.now_link=[0,0]
        self.event_type=EVENT_TYPE_SEND
        self.path=[]
        self.home_path=[]
    
class Link():
    def __init__(self, link_id , bandwidth, delay, queue_size, loss_rate,neibor,start_point,end_point):
        self.bw = float(bandwidth)
        self.dl = delay
        self.neibor = neibor
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.return_queue_delay = 0.0
        self.link_id=link_id
        self.queue_delay_update_time = 0.0
        self.return_queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw
        self.min_bw, self.max_bw = (100, 500)
        self.min_lat, self.max_lat = (0.05, 0.5)
        self.min_queue, self.max_queue = (0, 8)
        self.min_loss, self.max_loss = (0.0, 0.05)
        self.end_point=end_point
        self.start_point=start_point
        self.return_queue_delay_update_time = 0.0
        self.max_queue_in_link = queue_size
        self.now_queue_in_link = 0
        self.queue_link = deque([])
        self.max_queue_in_link_return = queue_size
        self.now_queue_in_link_return = 0
        self.queue_link_return = deque([])
    
    def update_packet_queue(self, dq, t, pack_in_link):
        #input()
        while t > 0:   
            if dq :
                tmp = dq.popleft()
                if t < tmp:
                    tmp = tmp - t
                    dq.appendleft(tmp)
                    t = 0
                else:
                    t = t - tmp
                    pack_in_link -= 1
                    assert pack_in_link >= 0
            else:
                return dq, t, pack_in_link
        
        return dq, t, pack_in_link
    def get_cur_queue_delay(self, event_time,event_type):
        if event_type == EVENT_TYPE_RETURN:
            queue_delta_time =  event_time - self.return_queue_delay_update_time
            self.return_queue_delay_update_time = event_time
            self.queue_link_return, queue_delta_time, self.now_queue_in_link_return = self.update_packet_queue(self.queue_link_return, queue_delta_time, self.now_queue_in_link_return)    
            return sum(self.queue_link_return)
        else:
            queue_delta_time =  event_time - self.queue_delay_update_time
            #print(self.link_id,":",event_time, "-", self.queue_delay_update_time, "=", queue_delta_time)
            self.queue_delay_update_time = event_time
            self.queue_link, queue_delta_time, self.now_queue_in_link = self.update_packet_queue(self.queue_link, queue_delta_time, self.now_queue_in_link)
            return sum(self.queue_link)

    def packet_enters_link(self, event_time,event_type):
        if event_type==EVENT_TYPE_RETURN:
            self.get_cur_queue_delay(event_time,event_type)
            self.now_queue_in_link_return += 1
            self.queue_link_return.append(random.exponential(1 / self.bw)/packet_return_size)
            return True
        else:
            self.get_cur_queue_delay(event_time,event_type)
            if (random.random() < self.lr):
                return False
            if self.now_queue_in_link == self.max_queue_in_link:
                #print("\tDrop!")
                return False
            self.now_queue_in_link += 1
            #print("\tNow queue = %d" % self.now_queue_in_link)
            self.queue_link.append(random.exponential(1 / self.bw))
            return True  
    
    

    def get_cur_latency(self, event_time,event_type):
        if event_type==EVENT_TYPE_RETURN :
            return 0.001*self.dl + self.get_cur_queue_delay(event_time,event_type)
        else:
        
            return self.dl + self.get_cur_queue_delay(event_time,event_type)

    
    def state_change(self):
        if (random.random() < (1.0/3)):
            self.bw_change()
        elif  (random.random() < (2.0/3)):
            self.dl_change()
        else:
            self.lr_change()
    def bw_change(self):
        delta =random.uniform(-0.5,0.5)
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.bw=self.bw * (1.0 + delta)
        else:
            self.bw=self.bw / (1.0 - delta)
        if self.bw>self.max_bw:
            self.bw=self.max_bw
        elif self.bw<self.min_bw:
            self.bw=self.min_bw
    def dl_change(self):
        delta =random.uniform(-0.5,0.5)
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.dl=self.dl * (1.0 + delta)
        else:
            self.dl=self.dl / (1.0 - delta)
        if self.dl>self.max_lat:
            self.dl=self.max_lat
        elif self.dl<self.min_lat:
            self.dl=self.min_lat
    def lr_change(self):
        delta =random.uniform(-0.5,0.5)
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.lr=self.lr * (1.0 + delta)
        else:
            self.lr=self.lr / (1.0 - delta)
        if self.lr>self.max_loss:
            self.lr=self.max_loss
        elif self.lr<self.min_loss:
            self.lr=self.min_loss
    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

class Network():
    
    def __init__(self, senders, links,num):
        #print(len(senders))
        self.q = []
        self.aoi=[]
        self.count=0
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.packet_counter = num
        #print(self.packet_counter)
        self.queue_initial_packets()
        #print(self.packet_counter)
    def queue_initial_packets(self):
        for num,sender in enumerate(self.senders):
            sender.register_network(self)
            sender.reset_obs()
        #\print(self.packet_counter)
        noi=random.uniform(-0.5,0.5)
        pack=Packet(self.packet_counter,0,0,0,[0, 17, 22, 27, 30, 14],"test0")
        heapq.heappush(self.q, (0,1,noi, self.senders[0], EVENT_TYPE_SEND, 0, 0.0,False,0, pack)) 
        self.packet_counter+=1
    def reset(self):
        self.cur_time = 0.0
        self.q = []
        self.packet_counter
        #print(len(self.links))
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        #print(self.packet_counter)
        self.queue_initial_packets()
        #print(self.packet_counter)
    def get_cur_time(self):
        return self.cur_time

    def run_for_dur(self, dur):
        mes_show=False
        recv_total = 0
        drop_total = np.zeros((3,1))
        recv_total0 = 0
        recv_total1 = 0
        recv_total2 = 0
        recv_total3 = 0
        recv_total4 = 0
        recv_total5 = 0
        recv_totals = np.zeros((3,5))
        t = np.zeros((4,3))
        p = np.zeros((3,3))
        #print(dur)
        end_time = self.cur_time + dur
        for sender in self.senders:
            sender.reset_obs()
        #print(self.cur_time , end_time)
        tmpreward = np.zeros(2)
        tmp_time = self.cur_time
        total_pack_num = 0
        ca = 0
        #senders.reset_temp_and_delta_time(self.cur_time)
        #while total_pack_num < 100:
        aoi_array=[]     
       
        p5=0
        for i in range(3):
            t[i][0] = self.senders[i].p0
            t[i][1] = self.senders[i].p1
            t[i][2] = self.senders[i].p2
            for j in range(3):
                p[i][j] = (t[i][j] == 0) * 1e5 
        ###speed up
        """
        for i in range(3):
            for j in range(3):
                if (t[i][j]) > 50:
                    drop_total[i] += (t[i][j] - 50)
                    t[i][j] = 50
        """
        #print("t",t)
        while self.cur_time < end_time:
            #print(self.cur_time,end_time)
            #print(heapq)
            #            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False)) 
            #input()
            #print(self.q)
            event_time, event, noise, sender, event_type, next_hop, cur_latency, dropped, acp_tmp, pack = heapq.heappop(self.q)

            pack.now_link[0]=next_hop

            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = pack.path_num
            #print(new_next_hop)
            new_latency = cur_latency
            new_dropped = dropped
            push_new_event = False
            #debug_pack(pack)
            #input()
            #print("flight : ",sender.get_bytes_in_flight())
            if event_type == EVENT_TYPE_ACK:
                #print("pack.path_num == len(pack.path)-1:",pack.path_num,len(pack.path)-2)
                if pack.path_num == len(pack.path)-1:
                    link_latency = self.links[pack.path[new_next_hop]].get_cur_latency(self.cur_time,EVENT_TYPE_RETURN)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    new_event_type = EVENT_TYPE_RETURN
                    push_new_event = True
                    if new_dropped == False:
                        new_dropped = not self.links[pack.path[new_next_hop]].packet_enters_link(self.cur_time,EVENT_TYPE_RETURN)
                    pack.home_path.append(self.links[pack.path[new_next_hop]].link_id)
                    pack.home_path.append(link_latency)
                else :
                    new_next_hop = pack.path_num + 1
                    #print(new_next_hop)
                    #print("yo",pack.path[new_next_hop])
                    link_latency = self.links[pack.path[new_next_hop]].get_cur_latency(self.cur_time,event_type)
                    #print(link_latency)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
                    if new_dropped == False:
                        new_dropped = not self.links[pack.path[new_next_hop]].packet_enters_link(self.cur_time,event_type)
                    pack.home_path.append(self.links[pack.path[new_next_hop]].link_id)
                    pack.home_path.append(link_latency)
            if event_type == EVENT_TYPE_RETURN:
                if pack.path_num == 0:
                    if dropped:
                        self.senders[sender.sender_ID].on_packet_lost()
                        total_pack_num +=1
                        drop_total[pack.path_ID] += 1
                    else:
                        """
                        print(pack.home_path)
                        print(cur_latency)
                        input()
                        """
                        if pack.path_ID==0:
                           recv_total0 += 1 
                        elif pack.path_ID==1:
                           recv_total1 += 1
                        elif pack.path_ID==2:
                           recv_total2 += 1 
                        elif pack.path_ID==3:
                           recv_total3 += 1 
                        elif pack.path_ID==4:
                           recv_total4 += 1 
                        recv_total += 1
                        self.senders[sender.sender_ID].on_packet_acked(cur_latency)
                        total_pack_num +=1
                        recv_totals[sender.sender_ID][pack.path_ID] +=1
                        #sender.set_cur_time_to_start_time()
                        #print("Packet acked at time %f" % self.cur_time)
                    #mes_show=False
                else:
                    new_next_hop = next_hop - 1
                    link_latency = self.links[pack.path[next_hop]].get_cur_latency(self.cur_time,event_type)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
                    if new_dropped == False:
                        new_dropped = not self.links[pack.path[next_hop]].packet_enters_link(self.cur_time,event_type)
                    pack.home_path.append(self.links[pack.path[new_next_hop]].link_id)
                    pack.home_path.append(link_latency)
            if event_type == EVENT_TYPE_SEND:
                if pack.path_num == 0:      
                    
                    if sender.can_send_packet():
                        #print("Packet sent at time %f" % self.cur_time)
                        self.senders[sender.sender_ID].on_packet_sent()
                        push_new_event = True
                        noi=random.uniform(-0.5,0.5)
                        """
                        if sender.sender_ID==0:
                            heapq.heappush(self.q, (self.cur_time +1.0 / sender.rate,1,noi, sender, EVENT_TYPE_SEND, 0, 0.0,False,acp_tmp, Packet(self.packet_counter,self.cur_time,0,[0,1,2],"send1"))) 
                        else:
                            heapq.heappush(self.q, (self.cur_time +1.0 / sender.rate,1,noi, sender, EVENT_TYPE_SEND, 0, 0.0,False,acp_tmp, Packet(self.packet_counter,self.cur_time,0,[3,1,4],"send2")))  
                        """
                        ran_tmp=0
                        ran_tmpp=p[0][0]
                        for i in range(3):
                            for j in range(3):
                                if ran_tmpp>p[i][j] and t[i][j]>0:
                                    ran_tmpp=p[i][j]
                                    ran_tmp=i*3+j
                        if ran_tmpp>p5 and t[0][0]<0:
                            ran_tmpp=p5
                            ran_tmp=9
                        topath_ID=ran_tmp%3
                        tosend_ID=ran_tmp//3
                        if tosend_ID <= 2:
                            sender_ = self.senders[tosend_ID]
                        if ran_tmp == 0 and t[tosend_ID][topath_ID]>0 :
                            t[tosend_ID][topath_ID]-=1
                            pack=Packet(self.packet_counter, topath_ID, self.cur_time, 0, [0, 16, 19, 21, 24, 14], "test0")
                            p[tosend_ID][topath_ID] += 1/sender_.p0
                        elif ran_tmp == 1 and t[tosend_ID][topath_ID]>0:
                            t[tosend_ID][topath_ID]-=1
                            pack=Packet(self.packet_counter, topath_ID, self.cur_time, 0, [0, 16, 19, 21, 23, 25, 27, 30, 14], "test1")
                            p[tosend_ID][topath_ID] += 1/sender_.p1
                        elif ran_tmp == 2 and t[tosend_ID][topath_ID]>0:
                            t[tosend_ID][topath_ID]-=1
                            pack=Packet(self.packet_counter, topath_ID, self.cur_time, 0, [0, 17, 22, 27, 30, 14], "test2")
                            p[tosend_ID][topath_ID] += 1/sender_.p2
                        elif ran_tmp == 3 and t[tosend_ID][topath_ID]>0:
                            t[tosend_ID][topath_ID]-=1
                            pack=Packet(self.packet_counter, topath_ID, self.cur_time, 0, [0, 16, 19, 21, 23, 25, 10], "test3")
                            p[tosend_ID][topath_ID] += 1/sender_.p0
                        elif ran_tmp == 4 and t[tosend_ID][topath_ID]>0:
                            t[tosend_ID][topath_ID]-=1
                            pack=Packet(self.packet_counter, topath_ID, self.cur_time, 0, [0, 16, 18, 20, 22, 10], "test4")
                            p[tosend_ID][topath_ID] += 1/sender_.p1
                        elif ran_tmp == 5 and t[tosend_ID][topath_ID]>0:
                            t[tosend_ID][topath_ID]-=1
                            pack=Packet(self.packet_counter, topath_ID, self.cur_time, 0, [0, 17, 22, 10], "test5")
                            p[tosend_ID][topath_ID] += 1/sender_.p2
                        elif ran_tmp == 6 and t[tosend_ID][topath_ID]>0:
                            t[tosend_ID][topath_ID]-=1
                            pack=Packet(self.packet_counter, topath_ID, self.cur_time, 0, [7, 23, 26, 28, 29, 14], "test6")
                            p[tosend_ID][topath_ID] += 1/sender_.p0
                        elif ran_tmp == 7 and t[tosend_ID][topath_ID]>0:
                            t[tosend_ID][topath_ID]-=1
                            pack=Packet(self.packet_counter, topath_ID, self.cur_time, 0, [7, 24, 14], "test7")
                            p[tosend_ID][topath_ID] += 1/sender_.p1
                        elif ran_tmp == 8 and t[tosend_ID][topath_ID]>0:
                            t[tosend_ID][topath_ID]-=1
                            pack=Packet(self.packet_counter, topath_ID, self.cur_time, 0, [7, 23, 25, 27, 30, 14], "test8")
                            p[tosend_ID][topath_ID] += 1/sender_.p2
                        
                        else:
                            push_new_event = False
                            p5 += 1/sender.p5
            
                        self.packet_counter+=1
                        pack_ = Packet(self.packet_counter,0,self.cur_time,0,[0,0,0,0],"send %d"% ran_tmp)
                        heapq.heappush(self.q, (self.cur_time +  sender.arrival_time[sender.now_packet], 1, noi, sender, EVENT_TYPE_SEND, 0, 0.0, False, acp_tmp, pack_)) 
                        sender.now_packet = sender.now_packet + 1
                        sender = sender_
                    else:
                        push_new_event = True
    
                    new_event_type = EVENT_TYPE_ACK
                    new_next_hop = next_hop 
                    
                    link_latency = self.links[pack.path[next_hop]].get_cur_latency(self.cur_time,event_type)
                    #print(link_latency)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)

                    if push_new_event:
                        new_latency += link_latency
                        new_event_time += link_latency
                        new_dropped = not self.links[pack.path[next_hop]].packet_enters_link(self.cur_time,event_type)
                    pack.home_path.append(self.links[pack.path[new_next_hop]].link_id)
                    pack.home_path.append(link_latency)
            pack.path_num=new_next_hop
            noi=random.uniform(-0.5,0.5)
            if new_dropped:
                self.senders[0].on_packet_lost()
                total_pack_num +=1
                drop_total[pack.path_ID] += 1
                push_new_event = False
            if push_new_event:
                heapq.heappush(self.q, (new_event_time,1,noi, sender, new_event_type, new_next_hop, new_latency, new_dropped,acp_tmp,pack))
                #print(new_event_time)
            #if push_new_event and event_type==EVENT_TYPE_SEND:
                #print("Packet sent at time %f" % self.cur_time)
            pack.now_link[1]=new_next_hop
            #input()
            if(pack.packet_ID<20000 and mes_show):
                if event_type == EVENT_TYPE_SEND:
                    print(pack.packet_ID,":",pack.content,"is",event_type," ",new_dropped,"from -1 to",pack.path[pack.now_link[1]]," in",self.cur_time)
                elif event_type == EVENT_TYPE_ACK and new_event_type == EVENT_TYPE_RETURN:
                    print(pack.packet_ID,":",pack.content,"is",new_event_type," ",new_dropped,"from des to",pack.path[pack.now_link[1]]," in",self.cur_time)
                elif next_hop == 0 and new_event_type == EVENT_TYPE_RETURN:
                    print(pack.packet_ID,":",pack.content,"is",new_event_type," ",new_dropped,"from",pack.path[pack.now_link[0]],"to start in",self.cur_time)
                else:
                    print(pack.packet_ID,":",pack.content,"is",event_type," ",new_dropped,"from",pack.path[pack.now_link[0]],"to",pack.path[pack.now_link[1]]," in",self.cur_time)
                #print(new_event_type)
            #print("aoi_cur_time",sender.aoi_cur_time)
            #print(self.links[1].queue_delay)
            
        #print(recv_total0," ",recv_total1," ",recv_total2," ",recv_total3," ",recv_total4," ",drop_total)
        self.senders[0].t0=recv_total0
        self.senders[0].t1=recv_total1
        self.senders[0].t2=recv_total2
        self.senders[0].t3=recv_total3
        self.senders[0].t4=recv_total4
        sender_mi = self.senders[0].get_run_data()
        throughput = sender_mi.get("recv rate") / (8 * BYTES_PER_PACKET)
        latency = sender_mi.get("avg latency")
        sender_mi1 = self.senders[1].get_run_data()
        throughput1 = sender_mi1.get("recv rate") / (8 * BYTES_PER_PACKET)
        latency1 = sender_mi1.get("avg latency")
        sender_mi2 = self.senders[2].get_run_data()
        throughput2 = sender_mi2.get("recv rate") / (8 * BYTES_PER_PACKET)
        latency2 = sender_mi2.get("avg latency")
        loss = sender_mi.get("loss ratio")
        bw_cutoff = self.links[0].bw * 0.8
        lat_cutoff = 2.0 * self.links[0].dl * 1.5
        loss_cutoff = 2.0 * self.links[0].lr * 1.5
        #print("thpt %f, bw %f" % (throughput, bw_cutoff))
        #reward = 0 if (loss > 0.1 or throughput < bw_cutoff or latency > lat_cutoff or loss > loss_cutoff) else 1 #
        """
        filename=('aoi/aoi'+str(self.count)+'.json')
        self.count+=1

        with open(filename, 'w') as f:
            json.dump(self.aoi, f, indent=4)
        self.aoi=[]
        """
        # Super high throughput
        #reward = REWARD_SCALE * (20.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Very high thpt
        reward_flow1 = 0
        reward_flow2 = 0
        reward_flow3 = 0
        drop_total_ = sum(drop_total)

        
        reward_flow1 = math.log(max(throughput, math.exp(-17.5))) - 1.5 * math.log(max(latency, math.exp(-5))) - 0.5 * math.log(max(drop_total[0], 1))
        reward_flow2 = math.log(max(throughput1, math.exp(-17.5))) - 1.5 * math.log(max(latency1, math.exp(-5))) - 0.5 * math.log(max(drop_total[1], 1))
        reward_flow3 = math.log(max(throughput2, math.exp(-17.5))) - 1.5 * math.log(max(latency2, math.exp(-5))) - 0.5 * math.log(max(drop_total[2], 1))
        if REWARD_CONTROL == 0 :
            reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)
        elif REWARD_CONTROL == 1 :
            reward = tmpreward[0]
        elif REWARD_CONTROL == 2 :
            reward = tmpreward[0]
        elif REWARD_CONTROL == 3 :
            if self.cur_time-tmp_time==0:
                reward=0
            else:
                reward = tmpreward[0]/(self.cur_time-tmp_time)
        elif REWARD_CONTROL == 4 :
            reward = reward_flow1 + reward_flow2 + reward_flow3 
        
        # High thpt
        #reward = REWARD_SCALE * (5.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Low latency
        #reward = REWARD_SCALE * (2.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        #if reward > 857:
        #print("Reward = %f, thpt = %f, lat = %f, loss = %f" % (reward, throughput, latency, loss))
        
        #reward = (throughput / RATE_OBS_SCALE) * np.exp(-1 * (LATENCY_PENALTY * latency / LAT_OBS_SCALE + LOSS_PENALTY * loss))
        return reward, recv_total0, recv_total1, recv_total2, recv_total3, recv_total4, drop_total_[0]

class Sender():
     #[Sender(random.uniform(0.3, 1.5) * bw, [self.links[0], self.links[1]], 0, self.features, history_len=self.history_len)]
    def __init__(self, sender_id,rate,start,dest, features, cwnd=25, history_len=10):
        self.id = Sender._get_next_id()
        self.sender_ID=sender_id
        self.starting_rate = rate
        self.arrival_time =  1/(random.poisson(lam=rate, size=10000)+1)
        self.now_packet = 0
        self.rate = Utilization_rate
        #print(self.rate)
        self.pa_queue = 0
        self.rate0 = rate
        self.rate1 = rate
        self.rate2 = rate
        self.rate3 = rate
        self.p0 = 1/6
        self.p1 = 1/6
        self.p2 = 1/6
        self.p3 = 1/6
        self.p4 = 1/6
        self.p5 = 1/6
        self.t0 = 0
        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.t4 = 0
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.aoi_cur_time = 0
        self.aoi_delta_time = 0
        self.aoi_temp_time = 0
        self.aoi_start_time = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        
        
        self.sample_time = []
        self.net = None
        self.start=start
        #self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd
        self.link1_bw=0
        self.link2_bw=0
        self.utilization= []
        self.utilization.append(0.5)
        self.utilization0= []
        self.utilization0.append(0.5)
        self.backlog_=0
        self.delta_=0
        self.backlog_sum=0
        self.delta_sum=0
        self.acp_time=0
        self.T=0.1
        self.Bt_b=0
        self.Bt_n=0
        self.Dt_b=0
        self.Dt_n=0
        self.B=[]
        self.D=[]
        self.flag=0
        self.gamma=0
        self.Z_bar=0.1
        self.tZ=0
        self.RTT_bar=0
        self.step_size=ACP_step_size
        self.bk_star=0.25
    _next_id = 1
    def reset_temp_and_delta_time(self,curu_time):
        self.aoi_delta_time=0
        self.aoi_temp_time = curu_time
    def get_aoi_reward(self):
    
        #print(self.aoi_cur_time,self.aoi_delta_time)
        return self.aoi_cur_time*self.aoi_delta_time+self.aoi_delta_time*self.aoi_delta_time/2
    def update_temp_time(self,curu_time):
        self.aoi_temp_time = curu_time
    def set_detla_time(self,curu_time):
        self.aoi_delta_time = curu_time - self.aoi_temp_time
    def set_cur_time_to_start_time(self):
        self.aoi_cur_time = self.aoi_start_time
        self.aoi_start_time=0
    def set_start_time(self):
        self.aoi_start_time=0
    def update_cur_and_start_time(self):
        self.aoi_cur_time = self.aoi_cur_time + self.aoi_delta_time
        self.aoi_start_time = self.aoi_start_time + self.aoi_delta_time
    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result
    def get_bytes_in_flight(self):
        return self.bytes_in_flight
    def apply_rate_delta(self, delta):
        #print(delta)

        delta *= (config.DELTA_SCALE)
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))
        
    def apply_cwnd_delta(self, delta):
        delta *= config.DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if Sender0_mode == 1:
            #print("test")
            #print(self.bytes_in_flight,"    ",self.cwnd)
            #input()
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        elif self.sender_ID==1 and sender2_TCP == True:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET
        self.backlog_ += 1
    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)

        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET
        self.backlog_ -= 1
    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET
        self.backlog_ -= 1
        
    def set_rate(self, new_rate):
        #print(cou)
        self.rate = new_rate
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE
        #self.rate = MAX_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        #print(smi)
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()
        
        obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            packet_size=BYTES_PER_PACKET,
            link1_bw=self.link1_bw,
            link2_bw=self.link2_bw,
            utilization=self.utilization,
            utilization0=self.utilization0,
            aoi=self.aoi_cur_time
        )
###############################################ACP#################
    def acp_update(self,rtt,event_time):
        self.RTT_bar=alpha*rtt+(1-alpha)*self.RTT_bar
        delta_z=event_time-self.tZ
        self.Z_bar=alpha*delta_z+(1-alpha)*self.Z_bar
        self.tZ=event_time
        self.T=min(self.RTT_bar,self.Z_bar)*10
    def INC(self):
        self.bk_star=self.step_size
        #print("inc")
    def DEC(self):
        self.bk_star=-self.step_size
        #print("dec")
    def MDEC(self,gamma):
        self.bk_star=-(1-1/(2<<gamma))*self.bytes_in_flight/8
        #print("mdec")
###############################################ACP#################       
    def reset_obs(self):
        #print(self.sender_ID)
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()
        self.utilization = []
        self.utilization.append(0.5)
        self.utilization0 = []
        self.utilization0.append(0.5)
    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

class SimulatedNetworkEnv(gym.Env):
    def __init__(self,
                 history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                    default="send rate,"
                          + "recv rate,"
                          + "avg latency,"
                          + "loss ratio"
                          )):
        self.viewer = None
        self.rand = None
        self.reward_array=[]
        self.min_bw, self.max_bw = (100, 500)
        self.min_lat, self.max_lat = (0.05, 0.5)
        self.min_queue, self.max_queue = (0, 400)
        self.min_loss, self.max_loss = (0.0, 0.0)
        self.history_len = history_len
        print("History length: %d" % history_len)
        self.features = features.split(",")
        print("Features: %s" % str(self.features))
        self.packet_temp = 0
        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links,self.packet_temp)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.max_steps = MAX_STEPS
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None
        self.last_time = 0
        #self.action_space = spaces.Box(np.array([-1e2, -1e2, -1e2, -1e2, -1e2]), np.array([1e2, 1e2, 1e2, 1e2, 1e2]), dtype=np.float32)
        self.action_space = spaces.Box(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2]), dtype=np.float32)
        #print(self.action_space)
        self.observation_space = None
        use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0
        self.reward_sum1 = 0.0
        self.reward_ewma1 = 0.0
        self.event_record = {"Events":[]}
        self.episodes_run = -1
        self.step_fir = True
        
    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        #print(sender_obs)
        return sender_obs

    def step(self, actions):

        if Sender0_mode == 0:
            action = actions
            self.senders[0].apply_rate_delta(action)
        elif Sender0_mode == 1:
            self.senders[0].rate = MAX_RATE
        elif Sender0_mode == 3:
            #self.senders[0].rate = Utilization_rate
            """
            tem_p0=math.exp(actions[0])
            tem_p1=math.exp(actions[1])
            tem_p2=math.exp(actions[2])
            tem_p3=math.exp(actions[3])
            tem_p4=math.exp(actions[4])
            sum_tem=tem_p0+tem_p1+tem_p2+tem_p3+tem_p4
            self.senders[0].p0=tem_p0/sum_tem
            self.senders[0].p1=tem_p1/sum_tem
            self.senders[0].p2=tem_p2/sum_tem
            self.senders[0].p3=tem_p3/sum_tem
            self.senders[0].p4=tem_p4/sum_tem
            """

            self.senders[0].p0 = actions[0]
            self.senders[0].p1 = actions[1]
            self.senders[0].p2 = actions[2]
            self.senders[1].p0 = actions[3]
            self.senders[1].p1 = actions[4]
            self.senders[1].p2 = actions[5]
            self.senders[2].p0 = actions[6]
            self.senders[2].p1 = actions[7]
            self.senders[2].p2 = actions[8]
            rate_sum = 0
            for i in range(3):
                for j in range(3):
                    rate_sum += max(0, actions[i * 3 +j])
            self.senders[i].set_rate(rate_sum)
            
            #print(self.senders[0].p0," ",self.senders[0].p1,"",self.senders[0].p2)
        else:
            action = actions
            self.senders[0].apply_rate_delta(action[0])
        for sender in self.senders:
            self.senders[sender.sender_ID].arrival_time = 1/(random.poisson(lam=sender.rate, size=10000)+1)
            self.senders[sender.sender_ID].now_packet = 0

        #print("Running for %fs" % self.run_dur)
        
        reward, recv_total0, recv_total1, recv_total2, recv_total3, recv_total4, drop_total = self.net.run_for_dur(self.run_dur)
        if (self.steps_taken % 250 == 0):
            print("Step:",self.steps_taken,":",recv_total0," ",recv_total1," ",recv_total2," ",recv_total3," ",recv_total4," ",drop_total)
        for sender in self.senders:
            sender.record_run()
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()

        reward1 = drop_total
        sender_mi = self.senders[0].get_run_data()

        event = {}
        event["Name"] = "Step"
        event["Time"] = self.steps_taken
        event["Reward"] = reward
        event["Reward1"] = reward1

        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        event["Link1 bandwidth"]=self.links[1].bw
        event["Link2 bandwidth"]=self.links[2].bw
        
        event["p0"]=self.senders[0].p0
        event["p1"]=self.senders[0].p1
        event["p2"]=self.senders[0].p2
        event["p3"]=self.senders[0].p3
        event["p4"]=self.senders[0].p4
        event["t0"]=self.senders[0].t0 / 1.0
        event["t1"]=self.senders[0].t1 / 1.0
        event["t2"]=self.senders[0].t2 / 1.0
        event["t3"]=self.senders[0].t3 / 1.0
        event["t4"]=self.senders[0].t4 / 1.0
        event["Send Rate1"] = sender_mi.get("send rate")
        event["Throughput1"] = sender_mi.get("recv rate")
        event["Utilization"]= sender_mi.get("avg utilization")
        event["Utilization0"]= sender_mi.get("avg utilization0")
        event["Aoi"]=sender_mi.get("aoi")
        #event["Cwnd"] = sender_mi.cwnd
        #event["Cwnd Used"] = sender_mi.cwnd_used
        self.event_record["Events"].append(event)
        #sender_obs=np.zeros(30)
        if event["Latency"] > 0.0:
            self.run_dur = 1 * sender_mi.get("avg latency")

        self.run_dur=OB_TIME
        #print("Sender obs: %s" % sender_obs)
        sender_obs=np.array(sender_obs)
        should_stop = False
        #print("reward_sum:",self.reward_sum,"  reward:", reward)
        self.reward_sum += reward
        self.reward_sum1 += reward1
        """
        if self.steps_taken%10==0:
            print("steps_taken",self.steps_taken)
        """
        return sender_obs, reward, (self.steps_taken >= self.max_steps or should_stop), {}#,123

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()
    def update_reward_array(self,reward_):
        self.reward_array.append(reward_)
    def create_new_links_and_senders(self):
        bw1   = 50
        bw0   = inf_bw
        lat   = 0.05
        queue =  500
        loss  = 0
        #print("queue:",queue)
        #bw    = 200
        #lat   = 0.03
        #queue = 5
        #loss  = 0.00
        self.node_link0 = Link(0,bw0, lat, queue, loss,['MA', 0],True,False)
        self.node_link1 = Link(1,bw0, lat, queue, loss,['NY', 0],False,True)
        self.node_link2 = Link(2,bw0, lat, queue, loss,['NJ', 1],False,True)
        self.node_link3 = Link(3,bw0, lat, queue, loss,['MD', 2],False,True)
        self.node_link4 = Link(4,bw0, lat, queue, loss,['PA', 3],False,True)
        self.node_link5 = Link(5,bw0, lat, queue, loss,['MI', 3],False,True)
        self.node_link6 = Link(6,bw0, lat, queue, loss,['GA', 4],False,True)
        self.node_link7 = Link(7,bw0, lat, queue, loss,['IL', 5],False,True)
        self.node_link8 = Link(8,bw0, lat, queue, loss,['IL', 6],False,True)
        self.node_link9 = Link(9,bw0, lat, queue, loss,['NE', 6],False,True)
        self.node_link10 = Link(10,bw0, lat, queue, loss,['TX', 7],False,True)
        self.node_link11 = Link(11,bw0, lat, queue, loss,['CO', 8],False,True)
        self.node_link12 = Link(12,bw0, lat, queue, loss,['UT', 8],False,True)
        self.node_link13 = Link(13,bw0, lat, queue, loss,['WA', 9],False,True)
        self.node_link14 = Link(14,bw0, lat, queue, loss,['CA', 10],False,True)
        self.node_link15 = Link(15,bw0, lat, queue, loss,['CA', 11],False,True)
        self.link0 = Link(16,bw1, lat, queue, loss,[0, 1],False,False)
        self.link1 = Link(17,bw1, lat, queue, loss,[0, 4],False,False)
        self.link2 = Link(18,bw1, lat, queue, loss,[1, 2],False,False)
        self.link3 = Link(19,bw1, lat, queue, loss,[1, 3],False,False)
        self.link4 = Link(20,bw1, lat, queue, loss,[2, 4],False,False)
        self.link5 = Link(21,bw1, lat, queue, loss,[3, 5],False,False)
        self.link6 = Link(22,bw1, lat, queue, loss,[4, 7],False,False)
        self.link7 = Link(23,bw1, lat, queue, loss,[5, 6],False,False)
        self.link8 = Link(24,bw1, lat, queue, loss,[5, 10],False,False)
        self.link9 = Link(25,bw1, lat, queue, loss,[6, 7],False,False)
        self.link10 = Link(26,bw1, lat, queue, loss,[6, 8],False,False)
        self.link11 = Link(27,bw1, lat, queue, loss,[7, 11],False,False)
        self.link12 = Link(28,bw1, lat, queue, loss,[8, 9],False,False)
        self.link13 = Link(29,bw1, lat, queue, loss,[9, 10],False,False)
        self.link14 = Link(30,bw1, lat, queue, loss,[10, 11],False,False)
        self.links = [self.node_link0, self.node_link1, self.node_link2, self.node_link3, self.node_link4, self.node_link5, self.node_link6, self.node_link7, \
                      self.node_link8, self.node_link9, self.node_link10, self.node_link11, self.node_link12, self.node_link13, self.node_link14, self.node_link15, \
                      self.link0, self.link1, self.link2, self.link3, self.link4, self.link5, self.link6, self.link7, self.link8, self.link9, self.link10, self.link11, \
                      self.link12, self.link13, self.link14]
        #print(self.links )
        self.senders = [Sender(0,1000, self.links[0], 0, self.features, history_len=self.history_len), Sender(1,1000, self.links[0], 0, self.features, history_len=self.history_len), Sender(2,1000, self.links[0], 0, self.features, history_len=self.history_len)]
        self.run_dur = 3 * lat

    def reset(self):
        self.packet_temp=self.net.packet_counter
        self.steps_taken = 0
        if step_control==0 or self.step_fir==False:
            self.step_fir = True
            #print(self.senders.rate)
            self.net.reset()
            self.create_new_links_and_senders()

            self.net = Network(self.senders, self.links,self.packet_temp)
        
        else:   
            self.run_dur = 3 * random.uniform(self.min_lat, self.max_lat)
            #self.net = Network(self.senders, self.links,self.packet_temp)
        
        self.episodes_run += 1

        #self.dump_events_to_file("prob_best.json" ,0)
        self.event_record = {"Events":[]}
        self.net.run_for_dur(self.run_dur)
        self.net.run_for_dur(self.run_dur)
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        self.reward_ewma1 *= 0.99
        self.reward_ewma1 += 0.01 * self.reward_sum1

        self.reward_sum = 0.0
        self.reward_sum1 = 0.0
        self.update_reward_array(self.reward_ewma)
        #self.dump_events_to_file("1101ver1dur05aoi_log_reward.json",1)
        return self._get_all_sender_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dump_events_to_file(self, filename,index):
        if index==0:
            with open(filename, 'w') as f:
                json.dump(self.event_record, f, indent=5)
        else:
            with open(filename, 'w') as f:
                json.dump(self.reward_array, f, indent=5)
register(id='PccNs_uti_NSFnet_multiV2-v0', entry_point='NSFnet_multiV2:SimulatedNetworkEnv')
#env = SimulatedNetworkEnv()
#env.step([1.0])
