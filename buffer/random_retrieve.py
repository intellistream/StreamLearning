from buffer.buffer_utils import random_retrieve


class Random_retrieve(object):
    def __init__(self, config):
        super().__init__()
        self.num_retrieve = config.eps_mem_batch

    def retrieve(self, buffer, **kwargs):
        return random_retrieve(buffer, self.num_retrieve)