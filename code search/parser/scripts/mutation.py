import random
import numpy as np
from scripts.ast import goTree, jsTree, pythonTree, javaTree, phpTree, rubyTree
from tree_sitter import Language, Parser
import json
import gzip
from tqdm import tqdm
import multiprocessing
import codecs

Language.build_library(
  # Store the library in the `build` directory
  './languages/my-languages.so',

  # Include one or more languages
  [
	'./languages/tree-sitter-go',
	'./languages/tree-sitter-javascript',
	'./languages/tree-sitter-python',
	'./languages/tree-sitter-java',
	'./languages/tree-sitter-ruby',
	'./languages/tree-sitter-php'
  ]
)

GO_LANGUAGE = Language('./languages/my-languages.so', 'go')
JS_LANGUAGE = Language('./languages/my-languages.so', 'javascript')
PY_LANGUAGE = Language('./languages/my-languages.so', 'python')
JAVA_LANGUAGE = Language('./languages/my-languages.so', 'java')
RUBY_LANGUAGE = Language('./languages/my-languages.so', 'ruby')
PHP_LANGUAGE = Language('./languages/my-languages.so', 'php')


def load_json(file_name):
	instances = []
	with open(file_name, 'r') as f:
		instances = json.load(f)
	return instances


def write_mutation(file, content):
	with open(file, 'w') as f:
		f.write(content)


def get_new_name(name, names, new_names):
	return new_names[names.index(name)]


def rename_in_content(locations, names, new_names, content):
	tmp = ''
	ll = []
	for location in locations:
		ll.extend(location)
	# print(ll)

	same_lines = []
	locations_in_same_line = []
	for l in ll:
		line_start, char_start, line_end, char_end = l
		if line_start != line_end:
			print("identifier Location Error.")
			return
		if line_start not in same_lines:
			same_lines.append(line_start)
			locations_in_same_line.append([l])
		else:
			locations_in_same_line[same_lines.index(line_start)].append(l)

	lines = content.split("\n")
	for i, line in enumerate(lines):
		if i in same_lines:
			ls = locations_in_same_line[same_lines.index(i)]
			ls.sort()
			if len(ls) ==1:
				_, start, _, end = ls[0]
				new_name = get_new_name(line[int(start):int(end)], names, new_names)
				newline = line[:int(start)]+new_name+line[int(end):]
			elif len(ls) == 2:
				_, start_front, _, end_front = ls[0]
				_, start_after, _, end_after = ls[1]
				newline = line[:int(start_front)]+get_new_name(line[int(start_front):int(end_front)], names, new_names)+line[int(end_front):int(start_after)]+get_new_name(line[int(start_after):int(end_after)], names, new_names)+line[int(end_after):]
			else:
				newline = ''
				for j, l in enumerate(ls):
					_, start, _, end = l
					new_name = get_new_name(line[int(start):int(end)], names, new_names)
					if j == 0:
						newline_ = line[:int(start)]+new_name
						newline = newline + newline_
					elif j == len(ls)-1:
						_, start_front, _, end_front = ls[j-1]
						newline_ = line[int(end_front):int(start)] + new_name + line[int(end):]
						newline = newline + newline_
					else:
						_, start_front, _, end_front = ls[j-1]
						newline_ = line[int(end_front):int(start)] + new_name
						newline = newline + newline_

		else:
			newline = line
		tmp = tmp + newline if i ==len(lines)-1 else  tmp + newline + '\n'
		# if i == len(lines)-1:
		# 	tmp = tmp + newline
		# else:
		# 	tmp = tmp + newline + '\n'
	return tmp


def mutation_by_rename(ast, mutation_path, vocab, n_times):
	mutation_contents = []
	identifiers = ast.metadata['variables']
	if len(identifiers) == 0:
		return mutation_contents
	content = ast.content
	language = ast.language
	filename = ast.filename

	names = []
	locations = []
	for location, identifier in identifiers:
		# print(location, identifier)
		if identifier not in names:
			names.append(identifier)
			locations.append([location])
		else:
			locations[names.index(identifier)].append(location)
	for i in range(int(n_times)):
		n = random.randint(0, len(names)-1)
		names_vocab = random.sample(vocab, len(names)-n)
		if language == 'php':
			names_vocab = ["$"+str(i) for i in names_vocab]	
		new_names = names[:n] + names_vocab
		mutation_content = rename_in_content(locations, names, new_names, content)
		# items = filename.split(".")
		# file = os.path.join(mutation_path, items[0]+'-rn-'+str(i)+'.'+items[-1])
		# write_mutation(file,mutation_content)
		if language == 'php':
			mutation_content = mutation_content.replace('<?php', '').replace('?>', '')
		tokenized_tokens = tokenize_code(ast, mutation_content)
		mutation_contents.append({'code': mutation_content, 'tokens': tokenized_tokens})
	return mutation_contents


def mutation_by_rename_funcName(ast, mutation_path, func_name_vocab, n_times):
	mutation_contents = []
	if len(ast.metadata['func_name']) == 0:
		return mutation_contents
	location, func_name = ast.metadata['func_name'][0]
	content = ast.content
	language = ast.language
	filename = ast.filename

	line_start, char_start, line_end, char_end = location
	lines = content.split("\n")
	items = filename.split(".")
	for i in range(int(n_times)):
		new_func = random.choice(func_name_vocab)
		mutation_content = ''
		for j, line in enumerate(lines):
			if j == int(line_start):
				mutation_content = mutation_content + line[:int(char_start)] + new_func + line[char_end:] + '\n'
			else:
				mutation_content = mutation_content + line + '\n'
		# file = os.path.join(mutation_path, items[0]+'-func-'+str(i)+'.'+items[-1])
		# write_mutation(file,mutation_content)
		if language == 'php':
			mutation_content = mutation_content.replace('<?php', '').replace('?>', '')
		tokenized_tokens = tokenize_code(ast, mutation_content)
		mutation_contents.append({'code': mutation_content, 'tokens': tokenized_tokens})
	return mutation_contents


def mutation_by_sampling(ast, mutation_path, n_times):
	mutation_contents = []
	content = ast.content
	language = ast.language
	filename = ast.filename


	lines = content.split("\n")
	items = filename.split(".")
	for i in range(n_times):
		# number = random.randint(1,len(lines)-1)
		not_null_lines = [line for i, line in enumerate(lines) if i != 0 and line.strip() != "" and  line.strip() != "{" and line.strip() != "}"]
		number = 2 if len(not_null_lines) > 3 else len(not_null_lines)-1
		random_lines = random.sample(not_null_lines, number)
		mutation_content = ''
		for line in lines:
			if line not in random_lines:
				mutation_content = mutation_content + line + '\n'
		# file = os.path.join(mutation_path, items[0]+'-sp-'+str(i)+'.'+items[-1])
		# write_mutation(file,mutation_content)
		if language == 'php':
			mutation_content = mutation_content.replace('<?php', '').replace('?>', '')
		tokenized_tokens = tokenize_code(ast, mutation_content)
		mutation_contents.append({'code': mutation_content, 'tokens': tokenized_tokens})
	return mutation_contents


def mutation_by_insert_dead_code(ast, mutation_path, vocab, n_times):
	mutation_contents = []
	assignments = ast.metadata['assignments']
	identifiers = ast.metadata['variables']
	content = ast.content
	language = ast.language
	filename = ast.filename

	# n_times = n_times if n_times<len(assignments) else len(assignments)
	if assignments == []:
		return []
	lines = content.split("\n")
	items = filename.split(".")
	for i in range(n_times):
		assignment = random.choice(assignments)
		location, code = assignment
		line_number, _, _, _ = location

		identifier_names = []
		for identifier in identifiers:
			lc, ir = identifier
			ln, _, _, _ = lc
			if ln == line_number:
				identifier_names.append(ir)

		if identifier_names==[]:
			continue

		mutation_content = ''
		for j, line in enumerate(lines):
			if j == line_number:
				mutation_assignment = line
				for identifier_name in identifier_names:
					# print(mutation_assignment,identifier_name)
					new_name = random.choice(vocab) if random.choice(vocab) != identifier_name else random.choice(vocab)
					if language == 'php':
						mutation_assignment = line.replace(identifier_name, "$"+new_name)
					else:
						mutation_assignment = line.replace(identifier_name, new_name)
					# print(mutation_assignment)
				mutation_content = mutation_content + mutation_assignment + '\n' + line + '\n' if random.randint(0,1) else mutation_content + line + '\n' + mutation_assignment + '\n'
			else:
				mutation_content = mutation_content + line + '\n'
		# print(tokenize_code(ast,mutation_content))
		# file = os.path.join(mutation_path, items[0]+'-in-'+str(i)+'.'+items[-1])
		# write_mutation(file,mutation_content)
		if language == 'php':
			mutation_content = mutation_content.replace('<?php', '').replace('?>', '')
		tokenized_tokens = tokenize_code(ast, mutation_content)
		mutation_contents.append({'code': mutation_content, 'tokens': tokenized_tokens})
	return mutation_contents


def mutation_by_reorder(ast, mutation_path, vocab, n_times):
	mutation_contents = []
	content = ast.content
	language = ast.language
	filename = ast.filename

	lines = content.split("\n")
	items = filename.split(".")
	for i in range(n_times):
		two_assignments = random.sample(lines[1:], 2)
		mutation_content = ''
		for line in lines:
			if line in two_assignments:
				newline = two_assignments[1] if two_assignments[0] == line else two_assignments[0]
				mutation_content = mutation_content + newline + '\n'
			else:
				mutation_content = mutation_content + line + '\n'
		# file = os.path.join(mutation_path, items[0]+'-ro-'+str(i)+'.'+items[-1])
		# write_mutation(file,mutation_content)
		if language == 'php':
			mutation_content = mutation_content.replace('<?php', '').replace('?>', '')
		tokenized_tokens = tokenize_code(ast, mutation_content)
		mutation_contents.append({'code': mutation_content, 'tokens': tokenized_tokens})
	return mutation_contents


def mutation_by_delete_docstring(ast, mutation_path, n_times):
	mutation_contents = []
	for i in range(n_times):
		tokens = ast.doc_tokens.copy()
		if len(tokens) <= 1:
			mutation_contents.append(tokens)
		else:
			index = random.randrange(0, len(tokens) - 1)
			copy_tokens = tokens[:index] + tokens[index+1:]
			mutation_contents.append(copy_tokens)
	return mutation_contents


def mutation_by_switch_docstring(ast, mutation_path, n_times):
	mutation_contents = []
	for i in range(n_times):
		tokens = ast.doc_tokens.copy()
		if len(tokens) >= 3:
			while True:
				first = random.randrange(0, len(tokens) - 1)
				second = random.randrange(0, len(tokens) - 1)
				if first != second:
					tmp = tokens[second]
					tokens[second] = tokens[first]
					tokens[first] = tmp
					break
			mutation_contents.append(tokens)
		else:
			mutation_contents.append(tokens)
	return mutation_contents


def mutation_by_copy_docstring(ast, mutation_path, n_times):
	mutation_contents = []
	for i in range(n_times):
		tokens = ast.doc_tokens.copy()
		if len(tokens) <= 1:
			index = 0
			copy_tokens = tokens[:index] + [tokens[index]] + tokens[index:]
			mutation_contents.append(copy_tokens)
		else:
			index = random.randrange(0, len(tokens) - 1)
			copy_tokens = tokens[:index] + [tokens[index]] + tokens[index:]
			mutation_contents.append(copy_tokens)
	return mutation_contents


def mutation(all_ast_dict, args):
	for key in all_ast_dict.keys():
		if key in args.language:
			ast_list = all_ast_dict[key]
			variables_vocab = load_json('./data/vocabs/' + key + '_variable_vocabs.json')
			func_name_vocab = load_json('./data/vocabs/' + key + '_func_name_vocabs.json')
			mutation_results, failed_count = parallel_mutation_ast(ast_list, mutation_single_ast, key,
																   (args, variables_vocab, func_name_vocab))
			save_jsonl_gz('./data/' + key + '_mutation_data.jsonl.gz', mutation_results)


def mutation_single_ast(ast, args, variables_vocab, func_name_vocab):
	try:
		rename_var_names = mutation_by_rename(ast, args.mutation_path, variables_vocab, args.n_times)
		rename_func_names = mutation_by_rename_funcName(ast, args.mutation_path, func_name_vocab, args.n_times)
		sample_funcs = mutation_by_sampling(ast, args.mutation_path, args.n_times)
		insert_funcs = mutation_by_insert_dead_code(ast, args.mutation_path, variables_vocab, args.n_times)
		reorder_funcs = mutation_by_reorder(ast, args.mutation_path, variables_vocab, args.n_times)
		delete_token_docstrings = mutation_by_delete_docstring(ast, args.mutation_path, args.n_times)
		switch_token_docstrings = mutation_by_switch_docstring(ast, args.mutation_path, args.n_times)
		copy_token_docstring = mutation_by_copy_docstring(ast, args.mutation_path, args.n_times)
		return {'code': ast.content, 'code_tokens': ast.code_tokens, 'doc_tokens':  ast.doc_tokens,
				'mutated_funcs': {'rename_var_names': rename_var_names, 'rename_func_names': rename_func_names,
				'sample_funcs': sample_funcs, 'insert_funcs': insert_funcs, 'reorder_funcs': reorder_funcs},
				'mutated_string': {'delete_token_docstrings': delete_token_docstrings, 'switch_token_docstrings': switch_token_docstrings,
				'copy_token_docstring': copy_token_docstring}}
	except:
		return None


def extract_ast(samples, args, cur_lan):
	keywords = load_json("./data/vocabs/keywords.json")
	ast_list, failed_count = parallel_extract_ast(samples, extract_single_ast, cur_lan, (keywords,))
	print('Finish AST Process %s, %d samples failed' % (cur_lan, failed_count))
	return ast_list


def extract_single_ast(sample, keywords):
	ast = None
	parser = Parser()
	if sample['language'] == 'go':
		parser.set_language(GO_LANGUAGE)
		ast = goTree(sample, parser, keywords)
	elif sample['language'] == 'java':
		parser.set_language(JAVA_LANGUAGE)
		ast = javaTree(sample, parser, keywords)
	elif sample['language'] == 'python':
		parser.set_language(PY_LANGUAGE)
		ast = pythonTree(sample, parser, keywords)
	elif sample['language'] == 'javascript':
		parser.set_language(JS_LANGUAGE)
		ast = jsTree(sample, parser, keywords)
	elif sample['language'] == 'ruby':
		parser.set_language(RUBY_LANGUAGE)
		ast = rubyTree(sample, parser, keywords)
	elif sample['language'] == 'php':
		parser.set_language(PHP_LANGUAGE)
		ast = phpTree(sample, parser, keywords)
	if len(ast.metadata['variables']) == 0 and len(ast.metadata['func_name']) == 0:
		return None
	return ast


def parallel_extract_ast(array, single_instance_process, cur_lan, args=(), n_cores=None):
	failed_count = 0
	filter_results = []
	if n_cores == 1:
		results = [single_instance_process(x, *args) for x in tqdm(array)]
		for result in results:
			if result is not None:
				filter_results.append(result)
			else:
				failed_count += 1
		return filter_results, failed_count
	with tqdm(total=len(array), desc="Extract AST Process Language %s" % cur_lan) as pbar:
		def update(*args):
			pbar.update()
		if n_cores is None:
			n_cores = multiprocessing.cpu_count()
		with multiprocessing.Pool(processes=n_cores) as pool:
			jobs = [
				pool.apply_async(single_instance_process, (x, *args), callback=update) for x in array
			]
			for job in jobs:
				if job.get() is not None:
					filter_results.append(job.get())
				else:
					failed_count += 1
		return filter_results, failed_count


def parallel_mutation_ast(array, single_instance_process, cur_lan, args=(), n_cores=None):
	failed_count = 0
	mutation_results = []
	if n_cores == 1:
		results = [single_instance_process(x, *args) for x in tqdm(array)]
		for result in results:
			if result is not None:
				mutation_results.append(result)
			else:
				failed_count += 1
		return mutation_results, failed_count
	with tqdm(total=len(array), desc="Mutation Process Language %s" % cur_lan) as pbar:
		def update(*args):
			pbar.update()
		if n_cores is None:
			n_cores = multiprocessing.cpu_count()
		with multiprocessing.Pool(processes=n_cores) as pool:
			jobs = [
				pool.apply_async(single_instance_process, (x, *args), callback=update) for x in array
			]
			for job in jobs:
				if job.get() is not None:
					mutation_results.append(job.get())
				else:
					failed_count += 1
		return mutation_results, failed_count


def tokenize_code(ast, content):
	language = ast.language
	parser = Parser()
	if language == 'go':
		parser.set_language(GO_LANGUAGE)
	elif language == 'java':
		parser.set_language(JAVA_LANGUAGE)
	elif language == 'python':
		parser.set_language(PY_LANGUAGE)
	elif language == 'javascript':
		parser.set_language(JS_LANGUAGE)
	elif language == 'ruby':
		parser.set_language(RUBY_LANGUAGE)
	elif language == 'php':
		parser.set_language(PHP_LANGUAGE)
	else:
		print("Language Error.")
		return None
	if language == 'php':
		content = "<?php" + content + "?>"
		body = bytes(content, "utf8")
	else:
		body = bytes(content, "utf8")
	tree = parser.parse(body)
	leaf_nodes = []
	traverse_leaf_nodes(tree.root_node, leaf_nodes)
	tokens = [match_token(i, content) for i in leaf_nodes]
	if language == 'php':
		strips_tokens = tokens[1:len(tokens) - 1]
		tokens = strips_tokens
	return tokens


def traverse_leaf_nodes(node, results):
	if not node.children:
		return results.append(node)
	for n in node.children:
		traverse_leaf_nodes(n, results)


def match_token(node, blob):
	lines = blob.split('\n')
	line_start = node.start_point[0]
	line_end = node.end_point[0]
	char_start = node.start_point[1]
	char_end = node.end_point[1]
	if line_start != line_end:
		return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
	else:
		return lines[line_start][char_start:char_end]


def save_jsonl_gz(filename, data):
	with gzip.GzipFile(filename, 'w') as out_file:
		writer = codecs.getwriter('utf-8')
		for element in data:
			writer(out_file).write(json.dumps(element))
			writer(out_file).write('\n')

