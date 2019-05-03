import torch

def collate_fn(data):
    data.sort(key = lambda x:len(x[1]), reverse=True)
    imgs, caps = zip(*data)
    imgs = torch.stack(imgs, 0)
    lens = [len(c) for c in caps] 
    targets = torch.zeros(len(caps), max(lens)).long()
    for i, cap in enumerate(caps):
        end = lens[i]
        targets[i, :end] = cap[:end]
    return imgs, targets, lens
