import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from point_transformer_pytorch import PointTransformerLayer
from torchmetrics import Accuracy, Precision, Recall

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from vedo import *

ct = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
writer = SummaryWriter('./runs/' + ct)

#'''
class PT(nn.Module):
    def __init__(self):
        super(PT, self).__init__()

        self.pt1 = PointTransformerLayer(
            dim = 3,
            pos_mlp_hidden_dim = 32,
            attn_mlp_hidden_mult = 4,
            num_neighbors = 16 )

        self.pt2 = PointTransformerLayer(
            dim = 3,
            pos_mlp_hidden_dim = 32,
            attn_mlp_hidden_mult = 4,
            num_neighbors = 16 )

        self.pt3 = PointTransformerLayer(
            dim = 3,
            pos_mlp_hidden_dim = 32,
            attn_mlp_hidden_mult = 4,
            num_neighbors = 16 )

        self.lin = nn.Linear(3,2)
        self.sm = nn.Softmax(dim = 2 )

    def forward(self, feats, pos, mask):
        
        h = self.pt1(feats, pos)        
        h = F.sigmoid(h)
        h = self.pt2(h, pos)        
        h = F.sigmoid(h)
        h = self.pt3(h, pos)
        h = F.sigmoid(h)
        h = self.lin(h)
        h = self.sm(h)        
        
        return h


def getBatch(batchSize = 5) : 

    points = []
    labels = []

    for i in range(batchSize) : 
        if np.random.rand() < 0.5 : 
            box = o3d.geometry.TriangleMesh.create_box(width=2*np.random.rand(), height = 2*np.random.rand(), depth = 2*np.random.rand()).translate((4*np.random.rand(), 4*np.random.rand(), 4*np.random.rand()))    
            pts = box.sample_points_poisson_disk(number_of_points=100)        
            points.append(pts.points)
            labels.append(np.ones(len(pts.points)))
        else : 
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius = np.random.rand(), resolution=10).translate((4*np.random.rand(), 4*np.random.rand(), 4*np.random.rand()))
            pts = sphere.sample_points_poisson_disk(number_of_points=100)
            points.append(pts.points)
            labels.append(np.zeros(len(pts.points)))
    
    points = np.vstack(np.array(points))
    labels = np.concatenate(np.array(labels))

    return points, labels


#feats = torch.randn(1, 500, 1)
#pos = torch.randn(1, 1000, 3)
mask = torch.ones(1, 500).bool()

model = PT()

#loading 
opt = torch.optim.Adam(model.parameters(), lr=0.001)
#lossFunc = nn.MSELoss()
lossFunc = nn.CrossEntropyLoss()

#lossFunc =  nn.MSELoss()(state_action_values.float(), expected_state_action_values.float())

for ep in range(100000) : 

    points, labels = getBatch(5)  #not really batch

    feats = torch.FloatTensor(points)  #positions as features !!!
    feats = torch.unsqueeze(feats, dim = 0)

    pos = torch.FloatTensor(points)
    pos = torch.unsqueeze(pos, dim = 0)
    mask = torch.ones(1, 500).bool()

    logits = model(feats, pos, mask = mask).float()
    logits = torch.squeeze(logits)

    labels = torch.tensor(labels)
    
    loss = lossFunc(logits.float(), labels.long())

    writer.add_scalar('main/loss', loss, ep)

    opt.zero_grad()
    loss.backward()
    opt.step()

    #logits = torch.clip(logits, 0, 1).int()
    labels = labels.type(torch.int32)

    accuracy = Accuracy()
    acc = accuracy(logits, labels.type(torch.int32))
    writer.add_scalar('metrics/acc', acc, ep)

    precision = Precision(average='macro', num_classes=2)
    pre = precision(logits, labels.type(torch.int32))
    writer.add_scalar('metrics/precision', pre, ep)

    recall = Recall(average='macro', num_classes=2)
    rec = recall(logits, labels)
    writer.add_scalar('metrics/recall', rec, ep)

    f1 = 2*(pre*rec)/(pre+rec)  #F1Score module wasn't working before
    writer.add_scalar('metrics/f1', f1, ep)

    if ep % 200 == 0 :             

        np.save('logits', logits.detach().numpy())
        np.save('pts', points)
        np.save('labels', labels)

        '''
        labels = labels.detach().numpy()
        pts = points
        labCols = [(1,0,0), (0,1,0)]
        cols = [(201/255,201/255,201/255), (148/255,148/255,148/255), (158/255,143/255,126/255)]

        co0 = [cols[i] for i in labels]
        co1 = [(l[0], l[1], 0) for l in np.clip(logits.detach().numpy(),0,1)]
        co2 = [labCols[np.argmax(i)] for i in np.clip(logits.detach().numpy(), 0, 1)] #shouldn't be argmax
        co3 = [labCols[i] for i in labels]

        #Point Cloud
        s0 = Points(pts, r=7, c = co0)
        s0.pointColors(co0)

        #Prediction
        s1 = Points(pts, r=7, c = co1)
        s1.pointColors(co1)

        #threshold
        s2 = Points(pts, r=7, c = co2)
        s2.pointColors(co2)

        #ground truth
        s3 = Points(pts, r=7, c = co3) #.color(co) #.color("blue")
        s3.pointColors(co3)

        plt1 = Plotter(N=4, axes=1, size = (400,400), offscreen = True)
        plt1.show(s0, "Point cloud", at=0)
        plt1.show(s3, "Ground truth", at=1)
        plt1.show(s1, "Prediction", at=2)
        plt1.show(s2, "Threshold", at=3, interactive=1)

        screenshot(f'results{ep}.png')
        '''
        