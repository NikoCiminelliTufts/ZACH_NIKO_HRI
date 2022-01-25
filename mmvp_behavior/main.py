from options import Options
from model import Model


def train():
    opt = Options().parse()
    print("Model Config: ",opt)

    model = Model(opt)
    model.train()
    # model.load_weight()
    # model.evaluate(epoch=30)

if __name__ == '__main__':
    import torch
    print(torch.cuda.is_available())
    train()
