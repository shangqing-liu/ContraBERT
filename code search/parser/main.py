import os
from scripts.mutation import extract_ast, mutation
import json
import gzip
import numpy as np
import random
import re
import pickle
import argparse
from sklearn.model_selection import train_test_split
from scripts.mutation import save_jsonl_gz


def main(args):
	set_seed()
	all_ast_dict = dict()
	if not os.path.exists('./data/all_ast_dict.pkl'):
		for language in args.language:
			data_dir = os.path.join(args.data_folder, language, 'final/jsonl/train')
			files = os.listdir(data_dir)
			samples = []
			for file in files:
				samples.extend(load_jsonl_gz(os.path.join(data_dir, file)))
			ast_list = extract_ast(samples, args, language)
			if language not in all_ast_dict.keys():
				all_ast_dict[language] = ast_list
			else:
				all_ast_dict[language].extend(ast_list)
		if args.build_vocab:
			for key in all_ast_dict.keys():
				variable_vocab = set()
				func_name_vocab = set()
				for ast in all_ast_dict[key]:
					variable_vocab, func_name_vocab = build_vocab(ast, variable_vocab, func_name_vocab)
				write_vocabs(list(variable_vocab), './data/vocabs/' + key + '_variable_vocabs.json')
				write_vocabs(list(func_name_vocab), './data/vocabs/' + key + '_func_name_vocabs.json')
				print('Language:%s vocab build finished, %d variable names, %d func_names' % (key, len(list(variable_vocab)), len(
					list(func_name_vocab))))
		with open('./data/all_ast_dict.pkl', 'wb') as f:
			pickle.dump(all_ast_dict, f)
	else:
		with open('./data/all_ast_dict.pkl', 'rb') as f:
			all_ast_dict = pickle.load(f)
	mutation(all_ast_dict, args)


def camel_case_split(identifier):
	matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
	return [m.group(0) for m in matches]


def set_seed(seed=2020):
	random.seed(seed)
	np.random.seed(seed)


def build_vocab(ast, variable_vocab, func_name_vocab):
	if len(ast.metadata['variables']) > 0:
		for variable in ast.metadata['variables']:
			if ast.language == 'php':
				var = variable[1].replace("$", "").strip()
			else:
				var = variable[1]
			variable_vocab.add(var)
	if len(ast.metadata['func_name']) > 0:
		func_name_vocab.add(ast.metadata['func_name'][0][1])
	else:
		# print(type(ast))
		pass
	return variable_vocab, func_name_vocab


def write_vocabs(vocab, file_path):
	norm_words = []
	for word in vocab:
		if '_' in word:
			subwords = word.split('_')
			for subword in subwords:
				words = camel_case_split(subword)
				norm_words.extend(words)
		else:
			words = camel_case_split(word)
			norm_words.extend(words)
	with open(file_path, 'w') as f:
		json.dump(list(set(norm_words)), f)


def load_jsonl_gz(file_name):
	instances = []
	with gzip.GzipFile(file_name, 'r') as f:
		lines = list(f)
	for line in lines:
		instance = json.loads(line)
		instances.append(instance)
	return instances


def build_dataset(args):
	for language in args.language:
		instances = load_jsonl_gz('./data/' + language + '_mutation_data.jsonl.gz')
		train_set, dev_set = train_test_split(instances, test_size=0.01)
		save_jsonl_gz('/mnt/data/shangqing1/ContraBERT/data/pretrain_data/codesearchnet/train/' + language + '_train.jsonl.gz', train_set)
		save_jsonl_gz('/mnt/data/shangqing1/ContraBERT/data/pretrain_data/codesearchnet/dev/' + language + '_dev.jsonl.gz', dev_set)
		print('Language %s has %d samples in train' % (language, len(train_set)))
		print('Language %s has %d samples in dev' % (language, len(dev_set)))


def test():
	# languages = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
	languages = ['javascript']
	for lan in languages:
		instances = load_jsonl_gz('/mnt/data/shangqing1/ContraBERT/parser/data/' + lan + '_mutation_data.jsonl.gz')
		for instance in instances:
			mutated_funcs = []
			mutated_docstrings = []
			for mutated_type in instance['mutated_funcs'].keys():
				if len(instance['mutated_funcs'][mutated_type]) > 0:
					for ele in instance['mutated_funcs'][mutated_type]:
						mutated_funcs.append(ele['tokens'])
				else:
					continue
			for mutated_type in instance['mutated_string'].keys():
				if len(instance['mutated_string'][mutated_type]) > 0:
					for ele in instance['mutated_string'][mutated_type]:
						mutated_docstrings.append(ele)
				else:
					continue
			if len(mutated_funcs) == 0 or len(mutated_docstrings) == 0:
				print('----------')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Generate Mutations for Functions')
	parser.add_argument('-f', '--data_folder', help='data folder', default='./data')
	parser.add_argument("-l", '--language', nargs='+', help='program language', default=['php'])
	# parser.add_argument("-l", '--language', nargs='+', help='program language', default=['go', 'java', 'javascript', 'php',
	# 																					 'python', 'ruby'])
	parser.add_argument('-m', '--mutation_path', help='mutation path', default='./mutations')
	parser.add_argument('-n', '--n_times', help='generate mutations in n times of original functions', default=1)
	parser.add_argument('-v', '--build_vocab', help='build_vocabs', action='store_true')
	args = parser.parse_args()
	main(args)
	build_dataset(args)
	test()
