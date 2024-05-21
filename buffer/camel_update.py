import torch
import torch.nn.functional as F
from collections import namedtuple
from ddsketch import DDSketch
import numpy as np
from dataselections.camel import camel_selection
from models.zoo import load_different_vit


class CamelUpdate(object):
    def __init__(self, config):
        super().__init__()
        Args = namedtuple('Args', ['mem_size'])
        params = Args(mem_size=config.mem_size)
        self.params = params
        self.model = load_different_vit(name='vit').cuda()

    def update(self, buffer, x, y, **kwargs):
        batch_size = x.size(0)
        if self.params.mem_size >= batch_size:
            place_left = self.params.mem_size - buffer.current_index
            if place_left >= batch_size:
                buffer.buffer_img[buffer.current_index: buffer.current_index + batch_size].data.copy_(x[:])
                buffer.buffer_label[buffer.current_index: buffer.current_index + batch_size].data.copy_(y[:])
                buffer.current_index += batch_size
            else:
                # Merge step
                buffer_img = torch.cat((x.cuda(), buffer.buffer_img[:buffer.current_index].cuda()), dim=0)
                buffer_label = torch.cat((y.cuda(), buffer.buffer_label[:buffer.current_index].cuda()), dim=0)
                # Reduce step
                selected_idx = camel_selection(examples=buffer_img, labels=buffer_label, model=self.model, size=self.params.mem_size)
                buffer.buffer_img = buffer_img[selected_idx]
                buffer.buffer_label = buffer_label[selected_idx]
                buffer.current_index = self.params.mem_size

        # the buffer is full
        # the buffer size is smaller than the batch size at the beginning
        if self.params.mem_size < batch_size:
            if buffer.current_index < self.params.mem_size:
                buffer_img = x
                buffer_label = y
            else:
                # Merge step
                buffer_img = torch.cat((x.cuda(), buffer.buffer_img), dim=0)
                buffer_label = torch.cat((y.cuda(), buffer.buffer_label), dim=0)

            # Reduce step
            selected_idx = camel_selection(buffer_img.cuda(), buffer_label.cuda(), self.model, self.params.mem_size)
            buffer.buffer_img = buffer_img[selected_idx]
            buffer.buffer_label = buffer_label[selected_idx]
            buffer.current_index = self.params.mem_size

        buffer.n_seen_so_far += batch_size


    def quantile_sketch(self, data):
        orig_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            features, _, _ = self.model(data.float().cuda(), train=False)
            features = F.normalize(features, p=2, dim=-1)
        self.model.train(orig_mode)
        data_flattened = features.view(-1).cpu().numpy()
        sketch = DDSketch()

        for value in data_flattened:
            sketch.add(value)

        quantiles = [0.25, 0.5, 0.75]
        buckets = [sketch.get_quantile_value(q) for q in quantiles]
        buckets = [float('-inf')] + buckets + [float('inf')]

        bucket_indices = np.digitize(data_flattened, buckets)
        bucket_means = np.zeros(len(buckets) - 1)
        for i in range(1, len(buckets)):
            bucket_data = data_flattened[bucket_indices == i]
            bucket_means[i - 1] = np.mean(bucket_data) if bucket_data.size > 0 else 0

        q_bits = 8
        min_mean = min(bucket_means)
        max_mean = max(bucket_means)
        encoded_means = [(int((mean - min_mean) / (max_mean - min_mean) * (2 ** q_bits - 1))) for mean in bucket_means]

        bucket_indices = np.digitize(data.view(-1).cpu().numpy(), bins=buckets) - 1
        reconstructed_data = np.array([encoded_means[index] for index in bucket_indices])
        return torch.tensor(reconstructed_data, dtype=torch.int8).view(data.shape).cpu()