from platform import architecture
from cyclic_swav import CyclicSwav
import torch
import torch.nn as nn
from leoloader import pascal_loader
from models import FeatureExtractor

from metrics import PredsmIoU




class LinearFinetune(torch.nn.Module):
    def __init__(self, model, num_classes, train_mask_size):
        super(LinearFinetune, self).__init__()
        self.model = model
        self.train_mask_size = train_mask_size
        ## freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        self.finetune_head = nn.Conv2d(384, num_classes, kernel_size=1)

    def forward(self, x, use_head=False):
        x, _ = self.model(x, use_head=use_head)
        x = x.permute(0, 2, 1)
        x = x.view(x.size(0), x.size(1),  28,  28)
        with torch.no_grad():
            x = nn.functional.interpolate(x, size=(self.train_mask_size, self.train_mask_size),
                                                mode='bilinear')
        x = self.finetune_head(x)
        return x


def validate(model, val_loader, epoch):
    model.eval()
    miou =  PredsmIoU(10, 10, involve_bg=True)
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.cuda()
            y = y.cuda()
            gt = y*255
            gt = nn.functional.interpolate(gt.float(), size=(100, 100),
                                                mode='nearest')

            gt = gt.squeeze()
            valid = (gt != 255)
            out = model(x)
            out = torch.argmax(out, dim=1)
            miou.update(gt[valid].flatten(), out[valid].flatten())
    print('Epoch: {}, mIoU: {}'.format(epoch, miou.compute(True, linear_probe=True)[0]))
    model.train()



def main():
    model_path = "dino-s16.pth"
    architecture = "dino-s16"
    mask_size = 100
    # feature_extractor = FeatureExtractor(architecture, model_path, [1024, 1024, 512, 256])  ##  [1024, 1024, 512, 256] unfreeze_layers=["blocks.11", "blocks.10"]
    model = FeatureExtractor(architecture, model_path)  ##  [1024, 1024, 512, 256] unfreeze_layers=["blocks.11", "blocks.10"]
    # model = CyclicSwav(feature_extractor, 200)
    # model.load_state_dict(torch.load("logs/20221220/182822/0.14383187223473465_28.pth"))
    model = LinearFinetune(model, 21, mask_size)
    model.cuda()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    train_loader = pascal_loader(60, "../../dataset/leopascal/VOCSegmentation", "trainaug", 100)
    print("train_loader", len(train_loader))
    val_loader =  pascal_loader(60, "../../dataset/leopascal/VOCSegmentation", "val", 100)
    print("val_loader", len(val_loader))
    for epoch in range(50):
        validate(model, val_loader, epoch)
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            y*=255
            y = nn.functional.interpolate(y.float(), size=(mask_size, mask_size),
                                                mode='nearest')
            out = model(x, use_head=False)
            loss = criterion(out, y.long().squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
        schedular.step()
        torch.save(model.state_dict(), 'linear_finetune.pth')
        validate(model, val_loader, epoch)


if __name__ == '__main__':
    main()

    

    