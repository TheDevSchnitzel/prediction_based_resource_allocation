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

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))

from PyProM.src.data.Eventlog import Eventlog
from object.instance import Instance
from object.resource import Resource

from prediction.model import net


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

class AbstractOptimizer(object):
	def __init__(self, *args, **kwargs):
		super(AbstractOptimizer, self).__init__(*args, **kwargs)
		self.w_comp_time = list()
		self.pred_time = list()
		self.act_res_mat = None

	def read_act_res_mat(self, path="./sample_data/new_resource_0806_1.csv"):
		"""Read activity-resource matrix which specifies the processing time

		Keyword arguments:
		path -- file path
		"""
		act_res_mat = pd.read_csv(path)
		act_res_mat['Resource'] = 'Resource'+act_res_mat['Resource'].astype('str')
		act_res_mat = act_res_mat.set_index('Resource')
		act_res_mat = act_res_mat.to_dict()
		return act_res_mat

	def load_data(self,path):
		"""Load eventlog

		Keyword arguments:
		path -- file path
		"""
		eventlog = Eventlog.from_txt(path, sep=',')
		eventlog = eventlog.assign_caseid('CASE_ID')
		eventlog = eventlog.assign_activity('Activity')
		eventlog = eventlog.assign_resource('Resource')
		self.activities = list(set(eventlog['Activity']))
		return eventlog

	def load_real_data(self,path):
		"""Load real-life log (Requires modification according to the schema)

		Keyword arguments:
		path -- file path
		"""
		eventlog = Eventlog.from_txt(path, sep=',')
		eventlog = eventlog.assign_caseid('CASE_ID')
		eventlog = eventlog.assign_activity('Activity')
		eventlog['Resource'] = eventlog['Resource'].astype(int)
		eventlog = eventlog.assign_resource('Resource')
		eventlog = eventlog.assign_timestamp(name='StartTimestamp', new_name='StartTimestamp', _format = '%Y.%m.%d %H:%M:%S', errors='raise')

		def to_minute(x):
			t = x.time()
			minutes = t.hour * 60 + t.minute
			return minutes

		eventlog['Start'] = eventlog['StartTimestamp'].apply(to_minute)
		return eventlog

	def initialize_real_instance(self, eventlog):
		"""Initialize real instance
		Difference between test and real instance
		1. Real - using date info.
		2. Real - release time is set to the appearing time of an instance

		Keyword arguments:
		eventlog -- test log
		"""
		instance_set = list()
		activity_trace = eventlog.get_event_trace(workers=4, value='Activity')
		resource_trace = eventlog.get_event_trace(4,'Resource')
		date_trace = eventlog.get_event_trace(workers=4, value='StartDate')
		time_trace = eventlog.get_event_trace(workers=4, value='Start')
		dur_trace = eventlog.get_event_trace(workers=4, value='Duration')
		weight_trace = eventlog.get_event_trace(workers=4, value='weight')

		for case in date_trace:
			for j, time in enumerate(date_trace[case]):
				if time >= self.date:
					initial_index =j-1
					release_time = time_trace[case][j]
					break
			weight = min(weight_trace[case])
			instance = Instance(name=case, weight=weight, release_time=release_time, act_sequence=activity_trace[case], res_sequence=resource_trace[case],dur_sequence=dur_trace[case], initial_index=initial_index)
			instance_set.append(instance)

		return instance_set

	def initialize_test_instance(self, eventlog):
		"""Initialize test instance

		Keyword arguments:
		eventlog -- test log
		"""
		instance_set = list()
		activity_trace = eventlog.get_event_trace(workers=4, value='Activity')
		resource_trace = eventlog.get_event_trace(4,'Resource')
		time_trace = eventlog.get_event_trace(workers=4, value='Start')
		dur_trace = eventlog.get_event_trace(workers=4, value='Duration')
		weight_trace = eventlog.get_event_trace(workers=4, value='weight')
		for case in activity_trace:
			release_time = min(time_trace[case])
			#release_time = 0
			weight = min(weight_trace[case])
			instance = Instance(name=case, weight=weight, release_time=release_time, act_sequence=activity_trace[case], res_sequence=resource_trace[case],dur_sequence=dur_trace[case])
			instance_set.append(instance)
		return instance_set

	def initialize_real_resource(self, test_log):
		"""Initialize real instance
		No difference at the moment

		Keyword arguments:
		test_log -- test log
		"""
		resource_set = list()
		resource_list = sorted(list(test_log.get_resources()))
		for res in resource_list:
			act_list = list(test_log.loc[test_log['Resource']==res,'Activity'].unique())
			resource = Resource(res, act_list)
			resource_set.append(resource)
		return resource_set

	def initialize_test_resource(self, eventlog):
		"""Initialize test resource

		Keyword arguments:
		eventlog -- test log
		"""
		resource_set = list()
		resource_list = sorted(list(eventlog.get_resources()))
		for res in resource_list:
			act_list = list(eventlog.loc[eventlog['Resource']==res,'Activity'].unique())
			resource = Resource(res, act_list)
			resource_set.append(resource)
		return resource_set

	def set_basic_info(self, eventlog):
		"""set basic info. for instances

		Keyword arguments:
		eventlog -- test log
		"""

		# To be aligned with the entire log, we load the information generated from entire log
		if self.mode == 'test':
			with open('./prediction/checkpoints/traininglog_0806_1.csv_activities.pkl', 'rb') as f:
				activities = pickle.load(f)
			with open('./prediction/checkpoints/traininglog_0806_1.csv_resources.pkl', 'rb') as f:
				resources = pickle.load(f)
		else:
			with open('./prediction/checkpoints/modi_BPI_2012_dropna_filter_act.csv_activities.pkl', 'rb') as f:
				activities = pickle.load(f)
			with open('./prediction/checkpoints/modi_BPI_2012_dropna_filter_act.csv_resources.pkl', 'rb') as f:
				resources = pickle.load(f)
		act_char_to_int = dict((str(c), i) for i, c in enumerate(activities))
		act_int_to_char = dict((i, str(c)) for i, c in enumerate(activities))
		res_char_to_int = dict((str(c), i) for i, c in enumerate(resources))
		res_int_to_char = dict((i, str(c)) for i, c in enumerate(resources))

		# for contextual information
		self.queue = OrderedDict()
		for act in activities:
			if act != '!':
				self.queue[act] = 0

		# maxlen information
		activity_trace = eventlog.get_event_trace(4,'Activity')
		trace_len = [len(x) for x in activity_trace.values()]
		maxlen = max(trace_len)

		# set info.
		Instance.set_activity_list(activities)
		Instance.set_resource_list(resources)
		Instance.set_act_char_to_int(act_char_to_int)
		Instance.set_act_int_to_char(act_int_to_char)
		Instance.set_res_char_to_int(res_char_to_int)
		Instance.set_res_int_to_char(res_int_to_char)
		Instance.set_maxlen(maxlen)

	def load_model(self, checkpoint_dir, model_name, modelArchitecture):
		"""load prediction model

		Keyword arguments:
		checkpoint_dir -- directory path
		model_name -- decide which model to load
		"""
		model = net(modelArchitecture)
		model.load(checkpoint_dir, model_name)
		return model

	def prepare_test(self, test_path, res_info_path,  modelNextActivity, modelNextTimestamp, checkpointDir, modelArchitecture):
		"""prepare experiment on the artificial log

		Keyword arguments:
		test_path -- path to the test log
		res_info_path -- path to the activity-resource processing time
		"""
		# load prediction model
		model_next_act = self.load_model(checkpointDir, modelNextActivity, modelArchitecture)
		model_next_time = self.load_model(checkpointDir, modelNextTimestamp, modelArchitecture)

		# set prediction model
		Instance.set_model_next_act(model_next_act)
		Instance.set_model_next_time(model_next_time)

		# load log
		test_log = self.load_data(path=test_path)
		self.num_cases = len(set(test_log['CASE_ID']))
		self.avg_weight = test_log['weight'].mean()

		#initialize resource set
		resource_set = self.initialize_test_resource(test_log)

		#create act-res matrix
		self.act_res_mat = self.read_act_res_mat(res_info_path)

		# initialize instance set
		instance_set = self.initialize_test_instance(test_log)

		#Set attributes of instance -> to be used to gernerate input for prediction
		self.set_basic_info(test_log)

		return resource_set, instance_set

    #@timing
	def update_ongoing_instances(self, instance_set, ongoing_instance, t):
		"""include released instances to the ongoing instance set

		Keyword arguments:
		instance_set -- all instances for resource allocation
		ongoing_instance -- ongoing instance set
		t -- current time
		"""
		for i in instance_set:
			if i.get_release_time() == t:
				ongoing_instance.append(i)
		return ongoing_instance


    #@timing
	def update_plan(self, G, t):
		"""solve the min-cost max-flow algorithm to find an optimal schedule

		Keyword arguments:
		G -- bipartite graph
		t -- current time
		"""
		nodes=G.nodes()
		if len(nodes)!=0:
			M = nx.max_flow_min_cost(G, 's', 't')
		else:
			M=False
		return M



