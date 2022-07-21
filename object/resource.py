import sys
import os
from pathlib import Path

class Resource(object):
	def __init__(self, name, skills, *args, **kwargs):
		super(Resource, self).__init__()
		self.name = name
		self.skills = skills
		self.next_pred_ts=0
		self.next_actual_ts = 0
		self.next_ts_uncertainty = 1
		self.status=True
		self.dur_dict = dict()

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name

	def get_name(self):
		return self.name

	def get_skills(self):
		return self.skills

	def get_next_pred_ts(self):
		return self.next_pred_ts

	def get_next_ts_uncertainty(self):
		return self.next_ts_uncertainty

	def get_next_actual_ts(self):
		return self.next_actual_ts

	def set_next_actual_ts(self, next_actual_ts):
		self.next_actual_ts = next_actual_ts

	def set_next_pred_ts(self, next_pred_ts):
		self.next_pred_ts = next_pred_ts

	def set_next_ts_uncertainty(self, conf):
		self.next_ts_uncertainty = conf

	def set_status(self, status):
		self.status = status

	def get_status(self):
		return self.status



