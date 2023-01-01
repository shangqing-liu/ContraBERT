class javaTree:
	def __init__(self, sample, parser, keywords):
		blob = sample['code']
		self.content = blob
		self.doc_tokens = sample['docstring_tokens']
		self.keywords = keywords
		self.metadata = self.get_function_metadata(blob, parser)
		self.language = 'java'
		self.filename = sample['path'].split("/")[-1]
		self.code_tokens = sample['code_tokens']

	def get_function_metadata(self, blob, parser):
		metadata = {
			'func_name': [],
			'variables': [],
			'assignments': [],
			'if_statements': [],
			'for_statements': [],
			'while_statements': [],
			'declarations': []
		}
		if len(blob.split('\n')) > 500 or len(blob.split()) > 1000:
			return metadata
		body = bytes(blob, "utf8")
		tree = parser.parse(body)
		root_node = tree.root_node
		nodes = []
		traverse_type(root_node, nodes)
		index = 0
		for n in nodes:
			_tuple = match_from_span(n, blob)
			if n.type in ['identifier'] and n.parent.type in ['assignment_expression', 'variable_declarator', 'argument_list', 'formal_parameter']:
				if index == 0:
					metadata['func_name'].append(_tuple)
					index += 1
				else:
					metadata['variables'].append(_tuple)
			elif n.type in ['identifier'] and _tuple[1] not in self.keywords:
				metadata['variables'].append(_tuple)
			elif n.type in ['assignment_expression', 'assignment', 'augmented_assignment', 'for_in_clause',
							'assignment_expression',
							'operator_assignment', 'assignment_statement', 'augmented_assignment_expression',
							'assignment_pattern']:
				metadata['assignments'].append(_tuple)
			# elif n.type in ['if_statement']:
			# 	metadata['if_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['while_statement']:
			# 	metadata['while_statements'].append(_tuple)
		# print(blob)
		# print(metadata)
		# print('----------------')
		return metadata


class pythonTree:
	def __init__(self, sample, parser, keywords):
		blob = sample['code']
		self.content = blob
		self.doc_tokens = sample['docstring_tokens']
		self.keywords = keywords
		self.metadata = self.get_function_metadata(blob, parser)
		self.language = 'python'
		self.filename = sample['path'].split("/")[-1]
		self.code_tokens = sample['code_tokens']

	def get_function_metadata(self, blob, parser):
		metadata = {
			'func_name': [],
			'variables': [],
			'assignments': [],
			'if_statements': [],
			'for_statements': [],
			'while_statements': [],
			'declarations': []
		}
		if len(blob.split('\n')) > 500 or len(blob.split()) > 1000:
			return metadata
		body = bytes(blob, "utf8")
		tree = parser.parse(body)
		root_node = tree.root_node
		nodes = []
		traverse_type(root_node, nodes)
		for n in nodes:
			_tuple = match_from_span(n, blob)
			if n.type in ['identifier'] and n.prev_sibling is not None and n.prev_sibling.type in ['def']:
				metadata['func_name'].append(_tuple)
			elif n.type in ['identifier'] and _tuple[1] not in self.keywords:
				metadata['variables'].append(_tuple)
			elif n.type in ['assignment_expression', 'assignment', 'augmented_assignment', 'for_in_clause',
							'assignment_expression',
							'operator_assignment', 'assignment_statement', 'augmented_assignment_expression',
							'assignment_pattern']:
				metadata['assignments'].append(_tuple)
			# elif n.type in ['if_statement']:
			# 	metadata['if_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['while_statement']:
			# 	metadata['while_statements'].append(_tuple)
		# if len(metadata['func_name']) == 0:
		# 	print(blob)
		# 	print(metadata)
		# 	print('----------------')
		return metadata


class goTree:
	def __init__(self, sample, parser, keywords):
		blob = sample['code']
		self.content = blob
		self.doc_tokens = sample['docstring_tokens']
		self.keywords = keywords
		self.metadata = self.get_function_metadata(blob, parser)
		self.language = 'go'
		self.filename = sample['path'].split("/")[-1]
		self.code_tokens = sample['code_tokens']

	def get_function_metadata(self, blob, parser):
		metadata = {
			'func_name': [],
			'variables': [],
			'assignments': [],
			'if_statements': [],
			'for_statements': [],
			'while_statements': [],
			'declarations': []
		}
		if len(blob.split('\n')) > 500 or len(blob.split()) > 1000:
			return metadata
		body = bytes(blob, "utf8")
		tree = parser.parse(body)
		root_node = tree.root_node
		nodes = []
		traverse_type(root_node, nodes)
		for n in nodes:
			_tuple = match_from_span(n, blob)
			if n.type in ['field_identifier'] and n.parent.type in ['method_declaration']:
				metadata['func_name'].append(_tuple)
			elif n.type in ['identifier'] and n.parent.type in ['function_declaration']:
				metadata['func_name'].append(_tuple)
			elif n.type in ['identifier'] and _tuple[1] not in self.keywords:
				metadata['variables'].append(_tuple)
			elif n.type in ['assignment_expression', 'assignment', 'augmented_assignment', 'for_in_clause',
							'assignment_expression',
							'operator_assignment', 'assignment_statement', 'augmented_assignment_expression',
							'assignment_pattern']:
				metadata['assignments'].append(_tuple)
		# 	elif n.type in ['if_statement']:
		# 		metadata['if_statements'].append(_tuple)
		# 	elif n.type in ['for_statement']:
		# 		metadata['for_statements'].append(_tuple)
		# 	elif n.type in ['for_statement']:
		# 		metadata['for_statements'].append(_tuple)
		# 	elif n.type in ['while_statement']:
		# 		metadata['while_statements'].append(_tuple)
		# print(blob)
		# print(metadata)
		# print('----------------')
		return metadata


class jsTree:
	def __init__(self, sample, parser, keywords):
		blob = sample['code']
		self.content = blob
		self.doc_tokens = sample['docstring_tokens']
		self.keywords = keywords
		self.metadata = self.get_function_metadata(blob, parser)
		self.language = 'javascript'
		self.filename = sample['path'].split("/")[-1]
		self.code_tokens = sample['code_tokens']

	def get_function_metadata(self, blob, parser):
		metadata = {
			'func_name': [],
			'variables': [],
			'assignments': [],
			'if_statements': [],
			'for_statements': [],
			'while_statements': [],
			'declarations': []
		}
		if len(blob.split('\n')) > 500 or len(blob.split()) > 1000:
			return metadata
		body = bytes(blob, "utf8")
		tree = parser.parse(body)
		root_node = tree.root_node
		nodes = []
		traverse_type(root_node, nodes)
		for n in nodes:
			_tuple = match_from_span(n, blob)
			if n.type in ['identifier']:
				if n.prev_sibling is not None and n.prev_sibling.type in ['function']:
					metadata['func_name'].append(_tuple)
				elif _tuple[1] not in self.keywords:
					metadata['variables'].append(_tuple)
				else:
					continue
					
			elif n.type in ['assignment_expression', 'assignment', 'augmented_assignment', 'for_in_clause',
							'assignment_expression',
							'operator_assignment', 'assignment_statement', 'augmented_assignment_expression',
							'assignment_pattern']:
				metadata['assignments'].append(_tuple)
			# elif n.type in ['if_statement']:
			# 	metadata['if_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['while_statement']:
			# 	metadata['while_statements'].append(_tuple)
		# print(blob)
		# print(metadata)
		# print('----------------')
		return metadata


class phpTree:
	def __init__(self, sample, parser, keywords):
		# blob = "<?php"+' '.join(sample['code_tokens'])+"?>"
		blob = "<?php"+sample['code']+"?>"
		body = bytes(blob, "utf8")
		tree = parser.parse(body)
		self.content = blob
		self.doc_tokens = sample['docstring_tokens']
		self.keywords = keywords
		self.metadata = self.get_function_metadata(tree.root_node, blob)
		self.language = 'php'
		self.filename = sample['path'].split("/")[-1]
		self.code_tokens = sample['code_tokens']

	def get_function_metadata(self, root_node, blob):
		metadata = {
			'func_name': [],
			'variables': [],
			'assignments': [],
			'if_statements': [],
			'for_statements': [],
			'while_statements': [],
			'declarations': []
		}
		if len(blob.split('\n')) > 500 or len(blob.split()) > 1000:
			return metadata
		nodes = []
		traverse_type(root_node, nodes)
		for n in nodes:
			_tuple = match_from_span(n, blob)
			if n.type in ['name']:
				if n.prev_sibling is not None and n.prev_sibling.type in ['function']:
					metadata['func_name'].append(_tuple)
			elif n.type in ['variable_name']:
				if 'this' not in _tuple[1] and _tuple[1] not in self.keywords:
					metadata['variables'].append(_tuple)
			elif n.type in ['assignment_expression', 'assignment', 'augmented_assignment', 'for_in_clause',
							'assignment_expression',
							'operator_assignment', 'assignment_statement', 'augmented_assignment_expression',
							'assignment_pattern']:
				metadata['assignments'].append(_tuple)
			# elif n.type in ['if_statement']:
			# 	metadata['if_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['while_statement']:
			# 	metadata['while_statements'].append(_tuple)
		# if len(metadata['func_name']) == 0:
		# 	print(blob)
		# 	print(metadata)
		# 	print('----------------')
		return metadata


class rubyTree:
	def __init__(self, sample, parser, keywords):
		blob = sample['code']
		body = bytes(blob, "utf8")
		tree = parser.parse(body)
		self.content = blob
		self.doc_tokens = sample['docstring_tokens']
		self.keywords = keywords
		self.metadata = self.get_function_metadata(tree.root_node, blob)
		self.language = 'ruby'
		self.filename = sample['path'].split("/")[-1]
		self.code_tokens = sample['code_tokens']

	def get_function_metadata(self, root_node, blob):
		metadata = {
			'func_name': [],
			'variables': [],
			'assignments': [],
			'if_statements': [],
			'for_statements': [],
			'while_statements': [],
			'declarations': []
		}
		if len(blob.split('\n')) > 500 or len(blob.split()) > 1000:
			return metadata
		nodes = []
		traverse_type(root_node, nodes)
		for n in nodes:
			_tuple = match_from_span(n, blob)
			if n.type in ['identifier'] and n.prev_sibling is not None and n.prev_sibling.type in ['def']:
				metadata['func_name'].append(_tuple)
			elif n.type in ['identifier'] and _tuple[1] not in self.keywords:
				if n.parent.type in ['call']:
					continue
				else:
					metadata['variables'].append(_tuple)
			elif n.type in ['assignment_expression', 'assignment', 'augmented_assignment', 'for_in_clause',
							'assignment_expression',
							'operator_assignment', 'assignment_statement', 'augmented_assignment_expression',
							'assignment_pattern']:
				metadata['assignments'].append(_tuple)
			# elif n.type in ['if_statement']:
			# 	metadata['if_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['for_statement']:
			# 	metadata['for_statements'].append(_tuple)
			# elif n.type in ['while_statement']:
			# 	metadata['while_statements'].append(_tuple)
		# if len(metadata['func_name']) == 0:
		# 	print(blob)
		# 	print(metadata)
		# 	print('----------------')
		return metadata


def traverse_type(node, results):
	if node.type:
		results.append(node)
	if not node.children:
		return
	for n in node.children:
		traverse_type(n, results)


def match_from_span(node, blob):
	lines = blob.split('\n')
	line_start = node.start_point[0]
	line_end = node.end_point[0]
	char_start = node.start_point[1]
	char_end = node.end_point[1]
	if line_start != line_end:
		return (line_start, char_start, line_end, char_end), '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
	else:
		return (line_start, char_start, line_end, char_end), lines[line_start][char_start:char_end]