import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class GoNet(nn.Module):
    '''
    AlexNet Two Stream architecture 

        ________
       |        | conv layers              
       |Previous|------------------>|      Connected Layers
       | frame  |                   |    ___     ___     ___
       |________|                   |   |   |   |   |   |   |   fc4
                                    |   |   |   |   |   |   |    * (left)
                  Convoluted        |-->|fc1|-->|fc2|-->|fc3|--> * (top)
                      Layers        |   |   |   |   |   |   |    * (right)
                                    |   |___|   |___|   |___|    * (bottom)
        ________                    |   (4096)  (4096)  (4096)  (4)
       |        |                   |
       | Current|------------------>|
       | frame  |
       |________|


    '''

    def __init__(self):
        super(GoNet, self).__init__()

        self.cnn = nn.Sequential(
                nn.Conv2d(3, 96, 11, 4), #(b x 96 x 55 x 55)
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(3, 2),       #(b x 96 x 27 x 27)
                nn.Conv2d(96, 256, 5, padding=2),       #(b x 256 x 27 x 27)
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(3, 2),                     #(b x 256 x 13 x 13)
                nn.Conv2d(256, 384, 3, padding=1),      #(b x 384 x 13 x 13)
                nn.ReLU(),
                nn.Conv2d(384, 384, 3, padding=1),      #(b x 384 x 13 x 13)
                nn.ReLU(),
                nn.Conv2d(384, 256, 3, padding=1),      #(b x 256 x 13 x 13)
                nn.ReLU(),
                nn.MaxPool2d(3, 2)                      #(b x 256 x 6 x 6)
                )

        self.fc = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(256 * 6 * 6 * 2, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4)
                )

    
    def forward(self, img1, img2):

        x1, x2 = self.cnn(img1), self.cnn(img2)
        
        x1 = x1.view(img1.size(0), 256 * 6 * 6)
        x2 = x2.view(img2.size(0), 256 * 6 * 6)

        x = torch.cat((x1, x2), 1)
        x = self.fc(x)

        return x 



