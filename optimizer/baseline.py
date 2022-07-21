import sys
import os
from pathlib import Path
import networkx as nx
import time
import numpy as np
from collections import OrderedDict
import pickle
import pandas as pd
import random
import copy
import csv

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))

from PyProM.src.data.Eventlog import Eventlog
from object.instance import Instance
from object.resource import Resource

from prediction.model import net
from optimizer.Ioptimizer import AbstractOptimizer



class BaseOptimizer(AbstractOptimizer):
	def __init__(self, *args, **kwargs):
		super(BaseOptimizer, self).__init__(*args, **kwargs)
		self.w_comp_time = list()

	def prepare_real(self, test_path, org_log_path, modelNextActivity, modelNextTimestamp, checkpointDir, modelArchitecture):
		"""prepare experiment on the real log

		Keyword arguments:
		test_path -- path to the test log
		org_log_path -- path to the entire log
		"""

		# load prediction model
		model_next_act = self.load_model(checkpointDir, modelNextActivity, modelArchitecture)
		model_next_time = self.load_model(checkpointDir, modelNextTimestamp, modelArchitecture)

		# set prediction model
		Instance.set_model_next_act(model_next_act)
		Instance.set_model_next_time(model_next_time)

		# load eventlog
		eventlog = self.load_real_data(path=org_log_path)

		# load test log
		test_log = self.load_real_data(path=test_path)
		self.num_cases = len(set(test_log['CASE_ID']))
		self.avg_weight = test_log['weight'].mean()

		#no act-res matrix
		self.act_res_mat = None

		# initialize instance set
		instance_set = self.initialize_real_instance(test_log)

		#initialize resource set
		resource_set = self.initialize_real_resource(test_log)

		#Set attributes of instance -> to be used to gernerate input for prediction
		self.set_basic_info(eventlog)

		return resource_set, instance_set



	def update_object(self, ongoing_instance, resource_set, t):
		"""create the bipartite graph with the prediction results

		Keyword arguments:
		ongoing_instance -- ongoing instance set
		resource_set -- all resources for resource allocation
		t -- current time
		"""
		# if resource is free, set the status to 'True'
		for j in resource_set:
			if j.get_next_actual_ts() == t:
				j.set_status(True)

		# if resource is free, set the status to 'True'
		for i in ongoing_instance:
			if i.get_next_actual_ts() == t:
				cur_actual_act = i.get_cur_actual_act()
				if cur_actual_act != False:
					self.queue[cur_actual_act] -= 1
				i.set_status(True)
			elif i.get_next_actual_ts() < t:
				i.update_weight()

		# generate bipartite graph
		ready_instance = [x for x in ongoing_instance if x.get_status()==True]
		ready_resource = [x for x in resource_set if x.get_status()==True]
		G = nx.DiGraph()
		for i in ready_instance:
			actual_act = i.get_next_actual_act()
			for j in ready_resource:
				if actual_act in j.get_skills():
					G.add_edge('s',i, capacity=1)
					G.add_edge(j,'t',capacity=1)
					weight = i.get_weight()
					cost = weight * (-1)
					G.add_edge(i,j,weight=cost,capacity=1)
		return G


	def execute_plan(self, ongoing_instance, resource_set, M, t):
		"""execute the resource allocation and update the situation accordingly.

		Keyword arguments:
		ongoing_instance -- ongoing instance set
		resource_set -- all resources for resource allocation
		M -- optimal schedule
		t -- current time
		"""

		ready_instance = [x for x in ongoing_instance if x.get_status()==True]
		ready_resource = [x for x in resource_set if x.get_status()==True]
		if M!=False:
			for i in M:
				if i in ready_instance:
					for j, val in M[i].items():
						# check if there is a flow
						if val==1 and j in ready_resource:
							cur_pred_dur, cur_time_uncertainty = i.predict_next_time(self.queue, context=True, pred_act=i.get_next_actual_act(), resource=j.get_name())
							i.set_pred_act_dur(j, cur_pred_dur, cur_time_uncertainty)
							i.update_actuals(t, j, self.mode, self.act_res_mat,self.queue)

							j.set_next_pred_ts(i.get_next_pred_ts())
							j.set_next_ts_uncertainty(i.get_next_ts_uncertainty(j))
							j.set_next_actual_ts(i.get_next_actual_ts())
							j.set_status(False)

							# update contextual information
							cur_actual_act = i.get_cur_actual_act()
							if cur_actual_act != False:
								self.queue[cur_actual_act] += 1

							i.clear_pred_act_dur()
							# to implement FIFO rule
							i.reset_weight()


	def update_completes(self, completes, ongoing_instance, t):
		"""check if instance finishes its operation

		Keyword arguments:
		completes -- set of complete instances
		ongoing_instance -- ongoing instance set
		t -- current time
		"""
		for i in ongoing_instance:
			finished = i.check_finished(t)
			if finished==True:
				cur_actual_act = i.get_cur_actual_act()
				self.queue[cur_actual_act] -= 1
				i.set_weighted_comp()
				ongoing_instance.remove(i)
				completes.append(i)
				self.w_comp_time.append(i.get_weighted_comp())
				"""
				with open("./exp_result/exp_7.txt", "a") as f:
					f.write("{}-{}: start at {}, end at {}, weighted_comp = {} \n".format(i.get_name(), i.get_weight(), i.release_time, i.get_next_actual_ts(), i.get_weighted_comp()))
				"""
		return completes


	def main(self, test_path, mode, date, exp_name, modelNextActivity, modelNextTimestamp, checkpointDir, modelArchitecture, verboose=False, **kwargs):
		time1 = time.time()
		t=0
		#initialize
		ongoing_instance = list()
		completes = list()
		self.mode = mode
		self.date = date

		if mode=='test':
			if "res_info_path" in kwargs:
				res_info_path = kwargs['res_info_path']
			else:
				raise AttributeError("Resource Information is required")
			resource_set, instance_set = self.prepare_test(test_path, res_info_path, modelNextActivity, modelNextTimestamp, checkpointDir, modelArchitecture)

		elif mode == 'real':
			if 'org_log_path' in kwargs:
				org_log_path = kwargs['org_log_path']
			else:
				raise AttributeError("no org_log_path given.")
			resource_set, instance_set = self.prepare_real(test_path, org_log_path, modelNextActivity, modelNextTimestamp, checkpointDir, modelArchitecture)
			if verboose:
				print("num resource:{}".format(len(resource_set)))

		else:
			raise AttributeError('Optimization mode should be given.')


		while len(instance_set) != len(completes):
			if verboose:
				print("{} begins".format(t))
			#ongoing instance를 추가
			ongoing_instance = self.update_ongoing_instances(instance_set, ongoing_instance, t)
			#print('current ongoing instance: {}'.format(len(ongoing_instance)))
			G = self.update_object(ongoing_instance, resource_set,t)
			#print('current cand instance and resource: {}, {}'.format(cand_instance, cand_resource))
			M = self.update_plan(G,t)
			#print('current matching: {}'.format(M))
			self.execute_plan(ongoing_instance, resource_set, M, t)
			completes = self.update_completes(completes, ongoing_instance, t)
			if verboose:
				print('current completes: {}'.format(len(completes)))
			t+=1
		time2 = time.time()

		total_weighted_sum = sum(self.w_comp_time)
		total_computation_time = (time2-time1)

		print("total weighted sum: {}".format(total_weighted_sum))
		print('suggested algorithm took {:.1f} s'.format(total_computation_time))
		
		summaryFile = open("./exp_result/{}.txt".format(exp_name), 'a', newline='')
		writer = csv.writer(summaryFile, delimiter=';')
		writer.writerow(["Data", "Cases", "Avg_weight", "Total_weighted_sum", "Total_computation_time"])

		if self.mode=='real':
			writer.writerow([test_path, self.num_cases, self.avg_weight, total_weighted_sum, total_computation_time])
		else:
			writer.writerow([test_path, self.num_cases, self.avg_weight, total_weighted_sum, total_computation_time])
