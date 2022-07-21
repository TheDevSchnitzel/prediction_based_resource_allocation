from optimizer.baseline import BaseOptimizer

import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='test', type=str)
	parser.add_argument('--test_path', default='./result/testlog_0121_1_100.csv', type=str)
	parser.add_argument('--date', default='2012-03-01', type=str)
	parser.add_argument('--exp_name', default='exp', type=str)

	parser.add_argument('--modelName', default='', type=str)
	parser.add_argument('--modelArch', default='LSTM', choices=["LSTM", "Performer"], type=str)
	parser.add_argument('--modelNextActivitySuffix', default='next_activity', type=str)
	parser.add_argument('--modelNextTimestampSuffix', default='next_timestamp', type=str)

	parser.add_argument('--checkpointDir', default='./prediction/checkpoints/', type=str)
	parser.add_argument('--estimationDir', default='./prediction/estimation/', type=str)

	
	parser.add_argument('--res_info_path', default="./sample_data/artificial/new_resource_0806_1.csv", type=str)
	parser.add_argument('--org_log_path', default='./sample_data/real/modi_BPI_2012_dropna_filter_act.csv', type=str)
	args = parser.parse_args()

	Opt = BaseOptimizer()

	if args.mode == 'test':
		"""Experiment on an artificial event log"""

		Opt.main(test_path=args.test_path
			, mode=args.mode
			, res_info_path=args.res_info_path
			, date=args.date
			, exp_name=args.exp_name
			, modelNextActivity=args.modelName+args.modelNextActivitySuffix
			, modelNextTimestamp=args.modelName+args.modelNextTimestampSuffix
			, checkpointDir=args.checkpointDir
			, modelArchitecture = args.modelArch)

	else:
		"""Experiment on an real-life event log"""
		Opt.main(org_log_path = args.org_log_path
			, test_path = args.test_path
			, mode=args.mode
			, date=args.date
			, exp_name=args.exp_name
			, modelNextActivity=args.modelName+args.modelNextActivitySuffix
			, modelNextTimestamp=args.modelName+args.modelNextTimestampSuffix
			, checkpointDir=args.checkpointDir
			, modelArchitecture = args.modelArch)