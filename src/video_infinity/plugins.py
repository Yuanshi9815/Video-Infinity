import torch
import torch.distributed as dist
import math

def my_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, token_num_scale=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    base_scale_factor = 1 / math.sqrt(query.size(-1)) * (scale if scale is not None else 1.)
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype).to(query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask.to(query.dtype).to(query.device)
    
    no_mask_count = torch.where(attn_bias < -100, 0, 1).sum(1)
    biased_scale_factor = torch.log(no_mask_count) / torch.log(torch.tensor(16)) if token_num_scale else 1.
    scale_factor = (base_scale_factor * biased_scale_factor).unsqueeze(-1) if token_num_scale else base_scale_factor
    attn_weight = query @ key.transpose(-2, -1) 
    attn_weight *= scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class ModulePlugin:
    def __init__(self, module, module_id, global_state=None):
        self.module = module
        self.module_id = module_id
        self.global_state = global_state
        self.enable = True
        self.implement_forward()

    @property
    def is_log_node(self):
        return self.global_state.get('dist_controller').rank == 0 and self.module_id[1] == 0

    @property
    def t(self):
        return self.global_state.get('timestep')
    
    @property
    def p(self):
        return self.t / 1000

    def implement_forward(self):
        module = self.module
        if not hasattr(module, "old_forward"):
            module.old_forward = module.forward
        self.new_forward = self.get_new_forward()
        def forward(*args, **kwargs):
            self.update_config() # update config
            return self.new_forward(*args, **kwargs) if self.enable else self.old_forward(*args, **kwargs)
        module.forward = forward

    def set_enable(self, enable=True):
        self.enable = enable
        
    def get_new_forward(self):
        raise NotImplementedError
    
    def update_config(self, config:dict=None):
        if config is None:
            config = self.global_state.get('plugin_configs', {}).get(self.module_id[0], {})
        for key, value in config.items():
            setattr(self, key, value)


class GroupNormPlugin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)

    def get_new_forward(self):
        module = self.module
    
        def new_forward(x):
            shape = x.shape
            N, C, G = shape[0], shape[1], module.num_groups
            assert C % G == 0

            x = x.reshape(N, G, -1)
            
            mean = x.mean(-1, keepdim=True)
            dist.all_reduce(mean)
            mean = mean / dist.get_world_size()
            var = ((x - mean) ** 2).mean(-1, keepdim=True) 
            dist.all_reduce(var)
            var = var / dist.get_world_size()

            x = (x - mean) / (var + module.eps).sqrt()
            x = x.view(shape)


            new_shape = [1 for _ in shape]
            new_shape[1] = -1

            return x * module.weight.view(new_shape) + module.bias.view(new_shape)

        return new_forward

class ConvLayerPlugin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)
        self.padding = 4
        self.rank = dist.get_rank()
        self.adj_groups = self.global_state.get('dist_controller').adj_groups

    def pad_context(self, h, padding=None):
        padding = self.padding if padding is None else padding
        share_to_left = h[:, :, :padding].contiguous()
        share_to_right = h[:, :, -padding:].contiguous()
        if self.rank % 2:
            # 1. the rank is odd, pad the left first 
            if self.rank:
                # not the first rank, have left context
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
            # 2. then pad the right
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
        else:
            # 1. the rank is even, pad the right first
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
            # 2. then pad the left
            if self.rank:
                # not the first rank, have left context
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
        torch.cuda.synchronize()
        h_with_context = torch.cat([left_context, h, right_context], dim=2)
        return h_with_context

    def get_new_forward(self):
        module = self.module
        def new_forward(hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
            hidden_states = (
                hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
            )

            identity = hidden_states

            hidden_states = self.pad_context(hidden_states)
            hidden_states = module.conv1(hidden_states)
            hidden_states = module.conv2(hidden_states)
            hidden_states = module.conv3(hidden_states)
            hidden_states = module.conv4(hidden_states)
            hidden_states = hidden_states[:, :, self.padding:-self.padding]


            hidden_states = identity + hidden_states

            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
                (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
            )
            return hidden_states

        return new_forward
    

class AttentionPlugin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)
        self.padding = 24
        self.top_k = 16
        self.top_k_chunk_size = 24
        self.attn_scale = 1.
        self.token_num_scale = False
        self.rank = dist.get_rank()
        self.adj_groups = self.global_state.get('dist_controller').adj_groups
        self.world_size = self.global_state.get('dist_controller').world_size
        self.dynamic_scale = False

    def pad_context(self, h, padding=None):
        padding = self.padding if padding is None else padding

        share_to_left = h[:, :padding].contiguous()
        share_to_right = h[:, -padding:].contiguous()
        if self.rank % 2:
            # 1. the rank is odd, pad the left first 
            if self.rank:
                # not the first rank, have left context
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
            # 2. then pad the right
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
        else:
            # 1. the rank is even, pad the right first
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
            # 2. then pad the left
            if self.rank:
                # not the first rank, have left context
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
        torch.cuda.synchronize()

        h_with_context = torch.cat([left_context, h, right_context], dim=1)
        return h_with_context, padding
    
    def get_topk(self, q, k, v, top_k=None):
        # h = [N, F, C]
        top_k = self.top_k if top_k is None else top_k
        share_num = int(max(top_k // self.world_size, 0))

        stride = max(q.shape[1] // share_num, 1) if share_num else 1000000

        topk_indices = torch.arange(0, q.shape[1], stride, device=q.device)

        k_to_share, v_to_share =  k[:, topk_indices], v[:, topk_indices]

        gather_k = [torch.zeros_like(k_to_share) for _ in range(self.world_size)]
        gather_v = [torch.zeros_like(v_to_share) for _ in range(self.world_size)]

        dist.all_gather(gather_k, k_to_share)
        dist.all_gather(gather_v, v_to_share)

        gather_k = torch.cat(gather_k, dim=1)[:, :top_k]
        gather_v = torch.cat(gather_v, dim=1)[:, :top_k]

        return gather_k, gather_v

    def gather_context(self, h):
        self.temporal_n = h.shape[1]
        stack_list = [torch.zeros_like(h) for _ in range(self.world_size)]
        dist.all_gather(stack_list, h)
        return torch.cat(stack_list, dim=1)

    def get_new_forward(self):
        module = self.module
        def new_forward(x, encoder_hidden_states=None, attention_mask=None):
            context=encoder_hidden_states

            temporal_n = x.shape[1]
            q = module.to_q(x)
            
            context = x if context is None else context
            k, v = module.to_k(context), module.to_v(context)
            b, _, _ = q.shape
            q, k, v = map(
                lambda t: t.unsqueeze(3).reshape(b, t.shape[1], module.heads, -1).permute(0, 2, 1, 3).reshape(b*module.heads, t.shape[1], -1),
                (q, k, v),
            )

            global_k, global_v = self.get_topk(q, k, v)
            num_global = global_k.shape[1]

            padded_k, _ = self.pad_context(k)
            padded_v, padding = self.pad_context(v)

            padded_k = torch.cat([padded_k, global_k], dim=1)
            padded_v = torch.cat([padded_v, global_v], dim=1)

            # if self.is_log_node:
            #     print("Total KV num:", padding*2 + global_k.shape[1], "Global KV num:", num_global, "Padding:", padding)

            attn_mask = torch.ones(temporal_n, temporal_n + 2*padding + num_global, dtype=q.dtype).to(q.device)
            for i in range(temporal_n):
                attn_mask[i, 0: max(0, i)] = float('-inf')
                attn_mask[i, min(temporal_n+2*padding, i+1+2*padding): temporal_n+2*padding] = float('-inf')
                
            if self.dynamic_scale and self.local_phase is not None and self.global_phase is not None:
                if self.t < self.local_phase['t']:
                    attn_mask[:, temporal_n+2*padding:] += self.local_phase['global_biase']
                    attn_mask[:, :temporal_n+2*padding] += self.local_phase['local_biase']
                if self.t >= self.global_phase['t']:
                    attn_mask[:, temporal_n+2*padding:] += self.global_phase['global_biase']
                    attn_mask[:, :temporal_n+2*padding] += self.global_phase['local_biase']
            out = my_attention(
                q, padded_k, padded_v,
                attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
                scale=self.attn_scale,
                token_num_scale=self.token_num_scale
            )


            out = (
                out.unsqueeze(0).reshape(b, module.heads, out.shape[1], -1).permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], -1)
            )

            # linear proj
            hidden_states = module.to_out[0](out)
            hidden_states = module.to_out[1](hidden_states)
            
            return hidden_states

        return new_forward
    

class Conv3DPligin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)
        self.padding = 1
        self.rank = dist.get_rank()
        self.adj_groups = self.global_state.get('dist_controller').adj_groups

    def pad_context(self, h):
        padding = self.padding
        share_to_left = h[:, :, :padding].contiguous()
        share_to_right = h[:, :, -padding:].contiguous()
        if self.rank % 2:
            # 1. the rank is odd, pad the left first 
            if self.rank:
                # not the first rank, have left context
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
            # 2. then pad the right
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
        else:
            # 1. the rank is even, pad the right first
            if self.rank != dist.get_world_size() - 1:
                # not the last rank, have right context
                padding_list = [torch.zeros_like(share_to_right) for _ in range(2)]
                dist.all_gather(padding_list, share_to_right, group=self.adj_groups[self.rank])
                right_context = padding_list[1].to(h.device, non_blocking=True)
            else:
                right_context = torch.zeros_like(share_to_right).to(h.device, non_blocking=True)
            # 2. then pad the left
            if self.rank:
                # not the first rank, have left context
                padding_list = [torch.zeros_like(share_to_left) for _ in range(2)]
                dist.all_gather(padding_list, share_to_left, group=self.adj_groups[self.rank-1])
                left_context = padding_list[0].to(h.device, non_blocking=True)
            else:
                left_context = torch.zeros_like(share_to_left).to(h.device, non_blocking=True)
        torch.cuda.synchronize()
        h_with_context = torch.cat([left_context, h, right_context], dim=2)
        return h_with_context

    def get_new_forward(self):
        module = self.module
        def new_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.pad_context(hidden_states)
            hidden_states = module.old_forward(hidden_states)[:,:,self.padding:-self.padding]
            return hidden_states

        return new_forward

class UNetPlugin(ModulePlugin):
    def __init__(self, module, module_id, global_state=None):
        super().__init__(module, module_id, global_state)

    def get_new_forward(self):
        module = self.module
    
        def new_forward(*args, **kwargs):
            self.global_state.set('timestep', args[1].item())
            return module.old_forward(*args, **kwargs)

        return new_forward