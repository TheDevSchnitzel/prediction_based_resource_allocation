import config
from feature_generator import FeatureGenerator

import pickle
from datetime import datetime
from model import net

accuracy_values = list()
accuracy_sum = 0.0
accuracy_value = 0.0
precision_values = list()
precision_sum = 0.0
precision_value = 0.0
recall_values = list()
recall_sum = 0.0
recall_value = 0.0
f1_values = list()
f1_sum = 0.0
f1_value = 0.0
training_time_seconds = list()

args = ""

if __name__ == '__main__':
	args = config.load()

	level = args.inter_case_level
	#filename = req['name']

	filename = args.data_dir + args.data_set
	model_name = args.modelName + args.task

	contextual_info = args.contextual_info
	if args.task == 'next_activity':
		loss = 'categorical_crossentropy'
		regression = False
	elif args.task == 'next_timestamp':
		loss = 'mae'
		regression = True

	batch_size = args.batch_size_train
	num_folds = args.num_folds

    # load data
	FG = FeatureGenerator()
	df = FG.create_initial_log(filename)



	#split train and test
	#train_df, test_df = FG.train_test_split(df, 0.7, 0.3)
	train_df = df
	test_df = train_df
	#create train
	train_df = FG.order_csv_time(train_df)
	train_df = FG.queue_level(train_df)
	train_df.to_csv('./training_data.csv')
	state_list = FG.get_states(train_df)
	train_X, train_Y_Event, train_Y_Time = FG.one_hot_encode_history(train_df, args.checkpoint_dir+args.data_set)


	if contextual_info:
		train_context_X = FG.generate_context_feature(train_df,state_list)
		model = net(args.modelArch)
		print("train_X.shape: " + str(train_X.shape))
		print("train_context_X.shape: " + str(train_context_X.shape))
		if regression:
			model.train(train_X, train_context_X, train_Y_Time, regression, loss, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir)
		else:
			model.train(train_X, train_context_X, train_Y_Event, regression, loss, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir)
	else:
		model_name += '_no_context_'
		train_context_X = None
		model = net(args.modelArch)
		if regression:
			model.train(train_X, train_context_X, train_Y_Time, regression, loss, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir, context=contextual_info)
		else:
			model.train(train_X, train_context_X, train_Y_Event, regression, loss, batch_size=batch_size, num_folds=num_folds, model_name=model_name, checkpoint_dir=args.checkpoint_dir, context=contextual_info)
	"""
	test_df = FG.order_csv_time(test_df)
	test_df = FG.queue_level(test_df)
	test_state_list = FG.get_states(test_df)

	test_X, test_Y_Event, test_Y_Time = FG.one_hot_encode_history(test_df)
	test_context_X = FG.generate_context_feature(test_df,test_state_list)
	"""
	test_X, test_Y_Event, test_Y_Time = train_X, train_Y_Event, train_Y_Time
	test_context_X = train_context_X
	test_X = test_X[500]
	test_context_X = test_context_X[500]
	test_Y_Event = test_Y_Event[500]
	#MC_pred, MC_uncertainty = model.predict(test_X, test_context_X, test_Y_Event)
