from torch.utils.tensorboard import SummaryWriter

writter=SummaryWriter()
for i in range(100):
    writter.add_scalar()