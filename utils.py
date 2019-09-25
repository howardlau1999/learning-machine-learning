def beam_search(scores, max_beam_size):
	paths = [[]]
	beam_scores = scores[0].new([0]) if isinstance(scores, (tuple, list)) else scores.new([0])
	for score in scores:
		beam_size = score.size(0)
		if beam_size > max_beam_size: beam_size = max_beam_size
		top_score, top_index = score.topk(beam_size)
		cur_beam_size = beam_scores.size(0)
		top_choice = beam_scores.unsqueeze(1) + top_score.unsqueeze(0) # cur_beam_size x beam_size
		beam_scores, top_choice_index = top_choice.view(-1).topk(beam_size)
		top_index = top_index.tolist()
		top_choice_index = top_choice_index.tolist()
		paths = [paths[index // beam_size] + [top_index[index % beam_size]] for index in top_choice_index]
	return paths, beam_scores

def reduce_beam(beam, beam_size, key):
	beam.sort(key=key, reverse=True)
	return beam[:beam_size]

# clip norm

def clip_norm(data, max_norm, norm_type=2, eps=1e-6):
	norm = []
	norm_type = float(norm_type)
	if norm_type == float("inf"):
		norm = [None if d is None else d.abs().max().item() for d in data]
	else:
		norm = [None if d is None else d.norm(norm_type).item() for d in data]
	if max_norm is not None:
		max_norm = float(max_norm)
		for i, d in enumerate(data):
			if norm[i] is None: continue
			clip_coef = max_norm / (norm[i] + eps)
			if clip_coef < 1:
				d.mul_(clip_coef)
	return norm

def clip_grad_norm(parameters, max_grad_norm, norm_type=2, eps=1e-6):
	data = [p.grad.data for p in parameters if p.grad is not None]
	return clip_norm(data, max_grad_norm, norm_type, eps)

def clip_weight_norm(parameters, max_weight_norm, norm_type=2, eps=1e-6):
	data = [p.data for p in parameters if p.grad is not None]
	return clip_norm(data, max_weight_norm, norm_type, eps)

def weight_init(m):
	import torch.nn.init as init
	'''
	Usage:
		model = Model()
		model.apply(weight_init)
	'''
	if isinstance(m, nn.Conv1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.BatchNorm1d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm2d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm3d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.Linear):
		init.xavier_normal_(m.weight.data)
		init.normal_(m.bias.data)
	elif isinstance(m, nn.Bilinear):
		init.xavier_normal_(m.weight.data)
		init.normal_(m.bias.data)
		if m.weight.data.size(1) == m.weight.data.size(2):
			scatter_index = torch.arange(m.weight.data.size(1))
			scatter_index = scatter_index.unsqueeze_(0).unsqueeze_(2).expand_as(m.weight.data)
			m.weight.data.scatter_(1, scatter_index, 1.0)
	elif isinstance(m, nn.LSTM):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.LSTMCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRU):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRUCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
