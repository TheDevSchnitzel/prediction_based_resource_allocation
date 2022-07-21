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

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

class SuggestedOptimizer(AbstractOptimizer):
	def __init__(self, *args, **kwargs):
		super(SuggestedOptimizer, self).__init__(*args, **kwargs)

	def prepare_real(self, test_path, org_log_path, modelNextActivity, modelNextTimestamp, checkpointDir, estimationDir, modelArchitecture):
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

		# (CHANGED)
		# load estimation model
		est_next_time = self.load_model(estimationDir, modelNextTimestamp, modelArchitecture)

		# set prediction model
		Instance.set_est_next_time(est_next_time)

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


	#@timing
	def update_object(self, ongoing_instance, resource_set, t):
		"""create the bipartite graph with the prediction results

		Keyword arguments:
		ongoing_instance -- ongoing instance set
		resource_set -- all resources for resource allocation
		t -- current time
		"""
		G = nx.DiGraph()
		# if resource is free, set the status to 'True'
		for j in resource_set:
			if j.get_next_actual_ts() <= t:
				j.set_status(True)

		# if instance is free, set the status to 'True'
		"""
		for i in ongoing_instance:
			if i.get_next_actual_ts() <= t:
				i.set_status(True)
		"""

		# if resource is free, set the status to 'True'
		for i in ongoing_instance:
			# if instance finishes current operation,
			if i.get_next_actual_ts() == t:
				# set the status to 'True'
				i.set_status(True)

				# update contextual information
				cur_actual_act = i.get_cur_actual_act()
				if cur_actual_act != False:
					self.queue[cur_actual_act] -= 1

				if self.exp_name != 'exp_2':
					# if it has just been released or the next act. prediction was wrong, update the processing time prediction
					if i.first or i.get_next_actual_act() != i.get_next_pred_act():
						i.clear_pred_act_dur()
						for j in resource_set:
							if i.get_next_actual_act() in j.get_skills():
								next_pred_dur, next_time_uncertainty = i.predict_next_time(self.queue, context=True, pred_act=i.get_next_actual_act(), resource=j.get_name())
								# set prediction uncertainty to 0 since it is ready for the next act.
								i.set_next_act_uncertainty(0)
								i.set_pred_act_dur(j, next_pred_dur, 0)
					else:
						#set prediction uncertainty to 0 since it is ready for the next act.
						i.set_next_act_uncertainty(0)
				else:
					# if it has just been released or the next act. prediction was wrong, update the processing time prediction
					if i.first or i.get_next_actual_act() != i.get_next_pred_act():
						i.clear_pred_act_dur()
						for j in resource_set:
							if i.get_next_actual_act() in j.get_skills():
								next_pred_dur, next_time_uncertainty = int(self.act_res_mat[i.get_next_actual_act()][j.get_name()]), 0
								# give noise
								if np.random.uniform(0,1) < 0.5:
									next_pred_dur += self.precision * next_pred_dur
								else:
									next_pred_dur -= self.precision * next_pred_dur
								next_pred_dur = round(next_pred_dur)
								if next_pred_dur == 0:
									next_pred_dur = 1

								i.set_next_act_uncertainty(0)
								i.set_pred_act_dur(j, next_pred_dur, 0)
					else:
						#set prediction uncertainty to 0 since it is ready for the next act.
						i.set_next_act_uncertainty(0)

			# if instance is under operation and the next act. prediction uncertainty is above the threshold, we do not allocate resources for it
			elif i.get_next_actual_ts() > t:
				if i.get_next_act_uncertainty() > self.act_uncertainty:
					continue

			for j in i.get_pred_act_dur_dict().keys():
				# if the processing time prediction uncertainty is above the threshold, we do not include the edge.
				if i.get_next_actual_ts() > t:
					if i.get_next_ts_uncertainty(j) > self.ts_uncertainty and j.get_next_ts_uncertainty() > self.ts_uncertainty:
						continue
				# generate bipartite graph
				G.add_edge('s',i, capacity=1)
				G.add_edge(j,'t',capacity=1)
				weight = i.get_weight()
				pred_dur = i.get_pred_act_dur(j)
				pred_dur += max([i.get_next_pred_ts()-t, j.get_next_pred_ts()-t, 0])
				cost = int(pred_dur / weight * 10)
				G.add_edge(i,j,weight=cost,capacity=1, pred_dur=pred_dur)

		return G

	

	def modify_plan(self, G, M, t):
		"""if some instances can be handled within the waiting time for best-matched instance, handle the instance who has the maximum weight.
		(We don't use it at the moment)

		Keyword arguments:
		G -- bipartite graph
		t -- current time
		"""
		if M!=False:
			for i, _ in M.items():
				if isinstance(i, Instance)==False:
					continue
				# if some instances can be handled within the waiting time for best-matched instance, handle the instance who has the maximum weight.
				temp_dict = dict()
				for j, val in M[i].items():
					if val==1:
						remaining = i.get_next_actual_ts()-t
						if remaining <= 0:
							break
						in_edges_to_j = G.in_edges([j], data=True)
						for source, dest, data in in_edges_to_j:
							if source.get_status()==True:
								if data['pred_dur'] <= remaining:
									#Also, we should check whether source is already assigned.
									assigned = False
									for r, val in M[source].items():
										if val == 1:
											assigned = True
									if assigned == False:
										temp_dict[source] = source.get_weight()

				if len(temp_dict)!=0:
					new_instance = max(temp_dict, key=temp_dict.get)
					M[i][j] = 0
					M[new_instance][j] = 1
					print("Match changed: from {} to {}, {}".format(i,new_instance, j.get_name()))
		return M


	#@timing
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
						if val==1 and M[j]['t']==1:
							if j in ready_resource:
								# if both instance and resource are ready for resource allocation,
								# update the current situation regarding the instance
								# (CHANGED)
								i.update_actuals(t, j, self.mode, self.act_res_mat,self.queue)

								# update the info. for the resource
								j.set_next_pred_ts(i.get_next_pred_ts())
								j.set_next_ts_uncertainty(i.get_next_ts_uncertainty(j))
								j.set_next_actual_ts(i.get_next_actual_ts())
								j.set_status(False)

								# update contextual information
								cur_actual_act = i.get_cur_actual_act()
								if cur_actual_act != False:
									self.queue[cur_actual_act] += 1

								if self.exp_name != 'exp_2':
									next_pred_act, next_act_uncertainty = i.predict_next_act(self.queue, context=True)
									i.set_next_pred_act(next_pred_act)
									i.set_next_act_uncertainty(next_act_uncertainty)
								else:

									if np.random.uniform(0,1) > self.precision:
										next_pred_act, next_act_uncertainty = i.get_next_actual_act(), 0
									else:
										activities = copy.deepcopy(self.activities)
										activities.remove(i.get_next_actual_act())
										next_pred_act, next_act_uncertainty = random.choice(activities), 0

									i.set_next_pred_act(next_pred_act)
									i.set_next_act_uncertainty(next_act_uncertainty)

								# clear dict for processing time and predict the processing time for the next activity
								i.clear_pred_act_dur()
								for k in resource_set:
									if next_pred_act in k.get_skills():
										if self.exp_name != 'exp_2':
											next_pred_dur, next_time_uncertainty = i.predict_next_time(self.queue, context=True, pred_act=next_pred_act, resource=k.get_name())
										else:
											# give noise
											next_pred_dur, next_time_uncertainty = int(self.act_res_mat[next_pred_act][k.get_name()]), 0
											if np.random.uniform(0,1) < 0.5:
												next_pred_dur += self.precision * next_pred_dur
											else:
												next_pred_dur -= self.precision * next_pred_dur
											next_pred_dur = round(next_pred_dur)
											if next_pred_dur <= 0:
												next_pred_dur = 1
										i.set_pred_act_dur(k, next_pred_dur, next_time_uncertainty)


	#@timing
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
				# update the contextual information
				cur_actual_act = i.get_cur_actual_act()
				self.queue[cur_actual_act] -= 1

				# compute the total weighted completion time and computation time
				i.set_weighted_comp()
				ongoing_instance.remove(i)
				completes.append(i)
				self.w_comp_time.append(i.get_weighted_comp())
				self.pred_time += i.get_pred_time_list()
				"""
				with open("./exp_result/exp_6.txt", "a") as f:
					f.write("{}-{}: start at {}, end at {}, weighted_comp = {} \n".format(i.get_name(), i.get_weight(), i.release_time, i.get_next_actual_ts(), i.get_weighted_comp()))
				"""
		return completes

	def main(self, test_path, mode, alpha, beta, precision, date, exp_name, modelNextActivity, modelNextTimestamp, checkpointDir, estimationDir, modelArchitecture, verboose=False, **kwargs):
		time1 = time.time()
		t=0
		#initialize
		ongoing_instance = list()
		completes = list()
		self.exp_name = exp_name
		self.act_uncertainty=alpha
		self.ts_uncertainty=beta
		self.precision = precision
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
			resource_set, instance_set = self.prepare_real(test_path, org_log_path, modelNextActivity, modelNextTimestamp, checkpointDir, estimationDir, modelArchitecture)
			if verboose:
				print("num resource:{}".format(len(resource_set)))

		else:
			raise AttributeError('Optimization mode should be given.')

		while len(instance_set) != len(completes):
			if verboose:
				print("{} begins".format(t))
			#Add ongoing instance
			ongoing_instance = self.update_ongoing_instances(instance_set, ongoing_instance, t)
			#print('current ongoing instance: {}'.format(len(ongoing_instance)))

			G = self.update_object(ongoing_instance, resource_set, t)
			#print("{} updated object".format(t))

			M = self.update_plan(G,t)
			#print("{} updated plan".format(t))

			#M = self.modify_plan(G, M,t)
			#print("{} modified plan".format(t))

			self.execute_plan(ongoing_instance, resource_set, M, t)
			#print("{} executed plan".format(t))

			completes = self.update_completes(completes, ongoing_instance, t)
			if verboose:
				print('current completes: {}'.format(len(completes)))

			# for log generation
			for i in ongoing_instance:
				cost_dict = dict()
				for j in i.get_pred_act_dur_dict().keys():
					weight = i.get_weight()
					#j.set_duration_dict(i,pred_dur)
					pred_dur = i.get_pred_act_dur(j)
					pred_dur += max([i.get_next_pred_ts()-t, j.get_next_pred_ts()-t, 0])
					cost = int(pred_dur / weight * 10)
					cost_dict[j] = cost
				if verboose:
					print("ongoing {} - status: {}, next: {}, cost: {}".format(i.get_name(),i.get_status(), i.get_next_actual_act(),cost_dict))

			t+=1
			if t > 2500:
				print("STOP")
				break
		time2 = time.time()

		total_weighted_sum = sum(self.w_comp_time)
		total_pred_time = sum(self.pred_time)
		total_computation_time = (time2-time1)
		total_opti_time = total_computation_time - total_pred_time

		print("total weighted sum: {}".format(total_weighted_sum))
		print('suggested algorithm took {:.1f} s'.format(total_computation_time))
		print("total time for predictions: {:.1f} s".format(total_pred_time))
		print("total time for optimizations: {:.1f} s".format(total_opti_time))
		
		summaryFile = open("./exp_result/{}.txt".format(exp_name), 'a', newline='')
		writer = csv.writer(summaryFile, delimiter=';')
		writer.writerow(["Data", "Cases", "Avg_weight", "Total_weighted_sum", "Total_computation_time", "total_pred_time", "total_opti_time","alpha", "beta", "precision" ])

		if self.mode=='real':
			writer.writerow([test_path, self.num_cases, self.avg_weight, total_weighted_sum, total_computation_time, total_pred_time,total_opti_time, alpha, beta, self.precision*100])
		else:
			writer.writerow([test_path, self.num_cases, self.avg_weight, total_weighted_sum, total_computation_time, total_pred_time,total_opti_time ,alpha, beta, self.precision*100])
