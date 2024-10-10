import math
from collections import namedtuple

import torch
import torch.nn.functional as F


class StreampromptUpdate(object):
    def __init__(self, config):
        super().__init__()
        Args = namedtuple('Args', ['mem_size'])
        params = Args(mem_size=config.mem_size)
        self.params = params
        # self.update_size = config.update_size
        self.curr_step=0

    def update(self, buffer, x, y, **kwargs):
        batch_size = x.size(0)

        # add whatever still fits in the buffer
        place_left = max(0, buffer.buffer_img.size(0) - buffer.current_index)
        if place_left:
            offset = min(place_left, batch_size)
            buffer.buffer_img[buffer.current_index: buffer.current_index + offset].data.copy_(x[:offset])
            buffer.buffer_label[buffer.current_index: buffer.current_index + offset].data.copy_(y[:offset])

            buffer.current_index += offset
            buffer.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                filled_idx = list(range(buffer.current_index - offset, buffer.current_index, ))
                return filled_idx


        #TODO: the buffer tracker will have bug when the mem size can't be divided by batch size
        x, y = x[place_left:], y[place_left:]
        with torch.no_grad():
            batch_size = x.size(0)
            buffer_img = torch.cat((x, buffer.buffer_img), dim=0)
            emds = buffer.model.feat.patch_embed(buffer_img.detach())

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, buffer.n_seen_so_far).long()
        valid_indices = (indices < buffer.buffer_img.size(0)).long()

        num_samples = len(valid_indices.nonzero().squeeze(-1))
        num_samples = min(10, max(1, num_samples))
        data_sim = self.data_selection(emds[:batch_size], buffer.model)
        buffer_sim = self.data_selection(emds[batch_size:], buffer.model)

        buffer.n_seen_so_far += batch_size

        sorted_indices = torch.argsort(data_sim, descending=True)
        ranks = torch.arange(1, len(data_sim) + 1, dtype=torch.float32)
        data_prob = 1 / ranks
        data_prob = 1 - data_prob / data_prob.sum()
        data_prob[sorted_indices] = data_prob.clone()

        sorted_indices = torch.argsort(buffer_sim, descending=True)
        ranks = torch.arange(1, len(buffer_sim) + 1, dtype=torch.float32)
        buffer_prob = 1 / ranks
        buffer_prob = 1 - buffer_prob / buffer_prob.sum()
        buffer_prob[sorted_indices] = buffer_prob.clone()

        idx_buffer = torch.multinomial(buffer_prob, num_samples=num_samples, replacement=False)
        idx_new_data = torch.multinomial(data_prob, num_samples=num_samples, replacement=False)


        if idx_buffer.numel() == 0:
            return []

        assert idx_buffer.max() < buffer.buffer_img.size(0)
        assert idx_buffer.max() < buffer.buffer_label.size(0)
        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}
        buffer.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())].cuda()
        buffer.buffer_label[list(idx_map.keys())] = y[list(idx_map.values())].cuda()
        return list(idx_map.keys())


    def data_selection(self, data_tensor, model):
        # extract the prompt
        p = []
        k = []
        a = []
        # select learned prompt
        pt = int(model.prompt.e_pool_size / model.prompt.n_tasks)
        s = int(model.prompt.task_count * pt)
        f = int((model.prompt.task_count + 1) * pt)
        # for name, param in model.named_parameters():
        for name, param in model.prompt.named_parameters():
            if 'e_p' in name:
                p.append(param[:f].detach().clone())

        with torch.no_grad():
            # prompt_tensor = torch.cat(p, dim=0)
            prompt_tensor = torch.stack(p, dim=0).sum(2)
            # flatten prompt_tensor so that each prompt is a vector
            prompt_tensor = prompt_tensor.view(-1, 768)  # (pool_size * p_length, dim)

            # normalised along the last dimension
            prompt_norm = F.normalize(prompt_tensor, p=2, dim=-1)
            data_norm = F.normalize(data_tensor, p=2, dim=-1)
            # use einsum to calculate the cosine similarity
            cos_sim = torch.matmul(data_norm, prompt_norm.transpose(0, 1).unsqueeze(0))  # (B, N, pool_size * p_length)
            avg_similarities = cos_sim.mean(dim=[1, 2])  # (B,)

        torch.cuda.empty_cache()

        return avg_similarities