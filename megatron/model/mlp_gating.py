from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

import copy


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
	# gates has shape of SE
	num_tokens = gates.shape[0]
	num_experts = gates.shape[1]
	# to(torch.int64) works around a bug in torch.onnx.export:
	# it should cast k to int64 when converting torch.topk but it doesn't.
	capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
	if capacity < min_capacity:
		capacity = min_capacity.to(torch.int64)
	return capacity


@torch.jit.script
def _top_idx(source, k):
	return torch.topk(source, k=k, dim=0)[1]


class Experts(torch.nn.Module):
	def __init__(self, expert, num_local_experts=1):
		super(Experts, self).__init__()

		self.yyh_local_experts = torch.nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])

	def forward(self, inputs, inputs_weight, top_idx):
		# inputs: (s, m), inputs_weight: (s, e)
		expert_output = torch.zeros_like(inputs)
		out_non_zero_ratio = None
		for e_idx, expert in enumerate(self.yyh_local_experts):
			token_idx = top_idx[:, e_idx]  # (capacity)
			these_tokens = inputs[token_idx]  # (capacity, dim)

			out = expert(these_tokens)

			if type(out) is tuple:
				if out_non_zero_ratio is None:
					out_non_zero_ratio = out[2]
				else:
					out_non_zero_ratio += out[2]

				out = out[0]  # Ignore the bias term for now

			expert_output[token_idx] += out * inputs_weight[:, e_idx][token_idx].unsqueeze(-1).type_as(inputs)

		return expert_output, out_non_zero_ratio / len(self.yyh_local_experts)


def main_thresholdGating(logits: Tensor, capacity_factor: float, min_capacity: int, k: int, threshold: float) -> Tuple[
	Tensor, Tensor, Tensor, Tensor]:
	# everything is in fp32 in this function
	gates = F.softmax(logits, dim=1)

	top1_p, _ = torch.max(gates, dim=1)

	### devloping ###
	# one_expert_token_num = (top1_p > threshold).sum()
	#################

	capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

	# Create a mask for 1st's expert per token
	indices1_s = torch.argmax(gates, dim=1)
	num_experts = int(gates.shape[1])
	mask1 = F.one_hot(indices1_s, num_classes=num_experts)

	# gating decisions (no use)
	exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

	# threshold
	# gates_sorted, gates_indices = torch.sort(gates, dim=-1, descending=True)
	# explore_top_k_num = max(min(int(2 * k), num_experts), 1)
	explore_top_k_num = num_experts
	gates_sorted, gates_indices = torch.topk(gates, dim=-1, k=explore_top_k_num, largest=True, sorted=True)

	cum_sorted_gates = torch.cumsum(gates_sorted, dim=-1)
	chosen_flag = (cum_sorted_gates - gates_sorted) < threshold  # (token num, explore_top_k_num)
	chosen_flag[:, 0] = True  # at least select one expert
	whole_chosen_indices = chosen_flag * (
			gates_indices + 1)  # (token num, explore_top_k_num) \in (0, expert_num + 1), 0 means not choose

	# get masks and capacity locations
	explore_top_k_num = whole_chosen_indices.sum(dim=0).ne(0).sum()
	whole_chosen_indices = whole_chosen_indices[:, :explore_top_k_num]  # (token num, explore_top_k_num)

	each_token_want_num = (whole_chosen_indices > 0).sum(dim=1)
	avg_want_num = each_token_want_num.sum() / each_token_want_num.shape[0]

	scatter_importance = torch.arange(explore_top_k_num, 0, -1, device=whole_chosen_indices.device).expand(
		whole_chosen_indices.shape)  # (token num, explore_top_k_num)
	tensor_all_mask = torch.zeros((whole_chosen_indices.shape[0], num_experts + 1), device=whole_chosen_indices.device,
								  dtype=whole_chosen_indices.dtype).scatter_(1, whole_chosen_indices,
																			 scatter_importance)[:,
					  1:]  # (token num, expert_num)
	token_num, expert_num = tensor_all_mask.shape

	# random token selection (ignore position)
	expert_received_num = (tensor_all_mask > 0).sum(dim=0)
	receive_ratio = expert_received_num * 100 / token_num

	top_idx = _top_idx(tensor_all_mask, capacity)  # (capacity, expert num)
	new_mask1 = tensor_all_mask * torch.zeros_like(tensor_all_mask).scatter_(0, top_idx, 1)
	tensor_all_mask = (new_mask1 > 0).int()
	# -------------------------------------------

	# tensor_all_locations = torch.cumsum(tensor_all_mask, dim=0) - 1 # (token_num, expert_num)

	expert_received_num = tensor_all_mask.sum(dim=0)
	expert_not_full_ratio = (expert_received_num < capacity).sum() / expert_num

	# tensor_all_locations = tensor_all_locations * tensor_all_mask # (token_num, expert_num)

	# Compute l_aux  !!!!!!!!!!!!!!!!!!!!!!!!?????????????????????
	me = torch.mean(gates, dim=0)

	# ce = torch.sum(mask1.float(), dim=0)
	# for i in range(1, dynamic_k):
	#     ce += torch.sum(all_mask[i].float(), dim=0)
	# ce /= ce.sum()

	# ce = expert_received_num / expert_received_num.sum()

	ce = torch.mean(mask1.float(), dim=0)

	l_aux = torch.sum(me * ce) * num_experts
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!?????????????????????

	token_chosen_num = tensor_all_mask.sum(dim=-1)  # (token_num)
	token_not_full_ratio = (token_chosen_num < each_token_want_num).sum() / token_num

	all_valid_chosen_num = token_chosen_num.sum()
	avg_valid_chosen_num = all_valid_chosen_num / token_num

	tensor_all_mask_float = tensor_all_mask.float()
	tensor_all_gates = gates * tensor_all_mask_float  # (s, e)

	# all_locations_sc = _one_hot_to_float(tensor_all_locations, capacity) # (s, e, c)
	# combine_weights = all_locations_sc * tensor_all_gates.unsqueeze(-1) # (s, e, c)

	combine_weights = tensor_all_gates
	# dispatch_mask = combine_weights.bool() # no used actually
	dispatch_mask = None  # no used actually

	gate_info = {
		"top1_p": top1_p,
		"chosen_num": avg_valid_chosen_num,
		"token_not_full_ratio": token_not_full_ratio,
		"expert_not_full_ratio": expert_not_full_ratio,
		"want_num": avg_want_num,
		"receive_ratio": receive_ratio
	}
	return l_aux, combine_weights, dispatch_mask, exp_counts, gate_info, top_idx
