import torch

class RandomSubSits(object):
    def __init__(self, t_len, generator=None):
        self.t_len = t_len
        self.generator = generator

    def __call__(self, sits):
        sits_len = sits['imgs'].shape[0]
        start_idx = torch.randint(0, sits_len-self.t_len+1, (), generator=self.generator)
        sits['imgs'] = sits['imgs'][start_idx:start_idx+self.t_len]
        sits['dates'] = sits['dates'][start_idx:start_idx+self.t_len]
        return sits
    
class DeterminedSubSits(object):
    def __init__(self, start_idx, t_len):
        self.t_len = t_len
        self.start_idx = start_idx

    def __call__(self, sits):
        sits['imgs'] = sits['imgs'][self.start_idx:self.start_idx+self.t_len]
        sits['dates'] = sits['dates'][self.start_idx:self.start_idx+self.t_len]
        return sits
    
class DeterminedSubSitsFromEnd(object):
    def __init__(self, t_len):
        self.t_len = t_len

    def __call__(self, sits):
        sits['imgs'] = sits['imgs'][-self.t_len:]
        sits['dates'] = sits['dates'][-self.t_len:]
        return sits