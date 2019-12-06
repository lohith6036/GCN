import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CoraFull
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Planetoid
from collections import Counter
from collections import defaultdict
import numpy as np
import argparse
import pickle as pkl
from utils_infomap import get_data_v3



parser = argparse.ArgumentParser(description="Model Name")

parser.add_argument("-model",action="store",dest="model",type=int,default=1)
parser.add_argument("-net",action="store",dest="net",type=int,default=1)
pr = parser.parse_args()


label_ids = defaultdict(list)

if   pr.net == 1:
     print("Data Cora")
     _data = Planetoid(root="./pcora",name="Cora")
elif pr.net == 2:
     print("Data CiteSeer")
     _data = Planetoid(root="./pciteseer",name="Citeseer")
elif pr.net == 3:
     print("Data Pubmed")
     _data = Planetoid(root="./ppubmed",name="Pubmed")
elif pr.net == 4:
     print("Data CoraFull")
     _data = CoraFull("./Corafull")
elif pr.net == 5:
     print("Data Coauthor CS")
     _data = Coauthor("./CS","CS")
elif pr.net == 6:
     print("Data Coauthor Physics")
     _data = Coauthor("./Physics","Physics")
elif pr.net == 7:
     print("Data Amazon Computer")
     _data = Amazon("./Computer","Computers")
elif pr.net == 8:
     print("Data Amazon Photos")
     _data = Amazon("./Photo","Photo")


#_data = Coauthor("./Physics","Physics")
#_data = Coauthor("./CS","CS")

#_data = CoraFull("./Corafull")

#_data = Planetoid(root="./pcora",name="Cora")
#_data = Planetoid(root="./pciteseer",name="Citeseer")
#_data = Planetoid(root="./ppubmed",name="Pubmed")

#_data = Amazon("./Computer","Computers")
#_data = Amazon("./Photo","Photo")

classes = _data.num_classes
print("Number of Features,Number of Classes ",_data.num_features,_data.num_classes,)

print("Class Distribution ",Counter(_data.data.y.numpy()))

labels = _data.data.y.numpy()

#print(labels)

all_labels_data = []

for i in range(len(labels)):
    label_ids[labels[i]].append(i)   # Now we know which label is in which Node ID
    all_labels_data.append(i)


np.random.shuffle(all_labels_data)

train_mask_ids = []
test_mask_ids = []
val_mask_ids = []

# here i do 20 random selection of per labels


'''for _x in label_ids:
    temp = np.random.choice(label_ids[_x],int(0.5*len(label_ids[_x])),replace=False)
    #print("Inside ",_x,Counter(_data.data.y[temp].numpy()),len(temp))
    #id1 = int(len(temp)/2.0)
    temp1 = temp[0:35]
    temp2 = temp[35:]
    train_mask_ids.extend(temp1)
    test_mask_ids.extend(temp2)'''
    #print("Inside ",_x,Counter(_data.data.y[temp1].numpy()),len(temp1))
    #print("Inside ",_x,Counter(_data.data.y[temp2].numpy()),len(temp2))
#print("Trains ",len(train_mask_ids))


# here I do random selection of labels

#train_mask_ids.extend(all_labels_data[0:int(0.1*len(all_labels_data))])
#val_mask_ids.extend(all_labels_data[int(0.2*len(all_labels_data)):int(0.3*len(all_labels_data))])
#test_mask_ids.extend(all_labels_data[int(0.6*len(all_labels_data)):])


#print("Masks ",len(train_mask_ids),len(val_mask_ids),len(test_mask_ids))
   
pkl.dump(_data.data.y,open("label.pkl","wb"))
pkl.dump(_data.data.x,open("feature.pkl","wb"))
pkl.dump(_data.data.edge_index,open("graph.pkl","wb"))
pkl.dump(torch.LongTensor(train_mask_ids),open("train_ids.pkl","wb"))
pkl.dump(torch.LongTensor(val_mask_ids),open("valid_ids.pkl","wb"))
pkl.dump(torch.LongTensor(test_mask_ids),open("test_ids.pkl","wb"))


#print("Train Test Size ",Counter(_data.data.y[train_mask_ids].tolist()),Counter(_data.data.y[test_mask_ids].tolist()),len(train_mask_ids),len(test_mask_ids))



#print("Sanity ",Counter(Train_mask))
#print("Sanity ",Counter(_data.data.y[Train_mask].numpy()))
#print(Counter(_data.data.y))

torch.manual_seed(0)


#dataset = Planetoid(root="/tmp/Cora/",name="Cora")
#dataset = _data.data
#dataset = dataset.data


dataset = _data


print(dataset[0].x.shape)
print(dataset.slices)
print(dataset.num_classes)
print(dataset.data.num_nodes)
print(dataset.data.num_edges)

class LinearLayer(torch.nn.Module):
    def __init__(self,in_feature,out_feature,in_hidden):
        super(LinearLayer,self).__init__()
        self.fc1 = nn.Linear(in_feature,in_hidden)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_hidden,out_feature)

    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)



class GCNmyConv(torch.nn.Module):
    def __init__(self,in_channels,out_channels,num_nodes):
        super(GCNmyConv,self).__init__()
        #self.lin = torch.nn.Linear(in_channels,out_channels)
        #self.num_nodes = num_nodes
    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        edge_index = add_self_loops(edge_index,num_nodes = x.size(0))
        x = self.lin(x)
        return self.propagate(aggr="mean",edge_index=edge_index,x=x)

    def message(self,x_j,edge_index):
        row,col = edge_index
        deg = degree(row,dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row]*deg_inv_sqrt[col]
        return norm.view(-1,1)*x_j

    def update(self,aggr_out):
        return aggr_out


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1  = GCNConv(dataset.num_features,512)
        self.conv2 = GCNConv(512,256)
        self.conv3 = GCNConv(256,128)
        #self.conv2 = SAGEConv(16,dataset.num_classes,3)

    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x,training=self.training)
        x = self.conv2(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x,training=self.training)
        x = self.conv3(x,edge_index)
        x = F.relu(x)
        #x = F.dropout(x,training=self.training)
        #x = self.conv4(x,edge_index)
        return F.log_softmax(x,dim=1)


#torch.cuda.manual_seed_all(1000)
torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
#data  = dataset.to(device)
if pr.model:
    model = Net().to(device)
else:
    model = LinearLayer(dataset.num_features,dataset.num_classes,16).to(device)

# Here I do the selection of labels based on community
train_mask_ids, test_mask_ids , val_mask_ids = get_data_v3()
print("Masks ",len(train_mask_ids),len(val_mask_ids),len(test_mask_ids))
#print(train_mask_ids)

data = dataset.data
data.__setitem__("Train_Mask",torch.Tensor.long(torch.Tensor(train_mask_ids)))
data.__setitem__("Test_Mask",torch.Tensor.long(torch.Tensor(test_mask_ids)))
data.__setitem__("Val_Mask",torch.Tensor.long(torch.Tensor(val_mask_ids)))




data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.05,weight_decay=5e-4)
model.train()
print("Starting Epochs ")
print(data.x.shape)
print(data.edge_index.shape)

model.train()



for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    #print(out.shape)
    #print(out[data.train_mask],data.y[data.train_mask])
    loss = F.nll_loss(out[data.Train_Mask],data.y[data.Train_Mask])
    #print(loss.item())
    loss.backward()
    optimizer.step()

'''for k,v in a.items():
    print(k)
    for com in v:
        for k1,v1 in com.items():
            print(k1)
            print(Counter(labels[v1]))'''

print("class labels distribution in py_geo,--training labels",Counter(labels[train_mask_ids]))
print ("ground truth of test labels:", Counter(labels[test_mask_ids]))
#print ("Test Labels ",Counter(data.y[data.Test_Mask].cpu().numpy()))
#print ("Train Labels ",Counter(data.y[data.Train_Mask].cpu().numpy()))

model.eval()
_,pred = model(data).max(dim=1)
predicted = pred.cpu().numpy()

print("predicted test labels:", Counter(predicted[test_mask_ids]))
_correct1 = pred[data.Test_Mask].eq(data.y[data.Test_Mask]).sum().item()
#_correct2 = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
acc1 = _correct1 / data.Test_Mask.size()[0]
#acc2 = _correct2 / data.val_mask.sum().item()

g_test_labels_ids = labels[test_mask_ids]
p_test_labels_ids = predicted[test_mask_ids]

g_test_labels_dict = defaultdict(list)
p_test_labels_dict = defaultdict(list)

for i in range(len(g_test_labels_ids)):
        g_test_labels_dict[g_test_labels_ids[i]].append(i)

#print(g_test_labels_dict)

for i in range(len(p_test_labels_ids)):
        p_test_labels_dict[p_test_labels_ids[i]].append(i)


true_positive_dict = {}
false_positive_dict = {}
false_negative_dict = {}

for i in range(classes):
    u = set(g_test_labels_dict[i])
    v = set(p_test_labels_dict[i])
    w = u.intersection(v)
    true_positive_dict[i] = list(w)
    negatives = []
    pre_neg = []
    for k,v in g_test_labels_dict.items():
        if k != i :
            for j in v:
                negatives.append(j)
    for k,v in p_test_labels_dict.items():
        if k!=i:
            for j in v:
                pre_neg.append(j)
    neg = set(negatives)
    p_neg = set(pre_neg)
    z = neg.intersection(v)
    y = p_neg.intersection(u)
    false_positive_dict[i] = list(z)
    false_negative_dict[i] = list(y)

#print("true positive counter")
#for k,v in true_positive_dict.items():
        #print(k,len(v))
#print("false positive counter")
#for k,v in false_positive_dict.items():
        #print(k,len(v))
precision = {}
recall = {}
f_score = {}
tp_sum = 0
fp_sum = 0
fn_sum = 0
for i in range(classes):
    tp = len(true_positive_dict[i])
    predic_positive = len(p_test_labels_dict[i])            
    ac_positive = len(g_test_labels_dict[i])
    tp_sum += tp
    fp = len(false_positive_dict[i])
    fp_sum += fp
    fn = len(false_negative_dict[i])
    fn_sum += fn
    s1 = tp + fp
    s2 = tp + fn
    if s1 == 0:
        prec1 = 0.0
    else :
        prec1 = tp/float (s1)
    if s2 == 0:
        rec1 = 0.0
    else:
        rec1 = tp/float (s2)
    fsum = prec1 + rec1
    if fsum == 0.0:
        f_sc = 0.0
    else :
        f_sc = 2*(prec1*rec1)/float(fsum)
    #prec1 = tp/float (s1)
    #rec1 = tp/float (s2)
    '''if predic_positive == 0:
        prec = 0.0
    else:
        prec =  tp/float (predic_positive)'''
    #rec = tp/float (ac_positive)
    precision[i] = prec1
    recall[i] = rec1
    f_score[i] = f_sc
prec_micro = tp_sum/float(tp_sum + fp_sum)
rec_micro = tp_sum/float(tp_sum + fn_sum)
fsc_micro = 2*(prec_micro*rec_micro)/float(prec_micro+rec_micro)
print("micro results--")
print(prec_micro,rec_micro,fsc_micro)
prec_macro = rec_macro = fsc_macro = 0.0
for p,v in precision.items():
    prec_macro += v
for r,v in recall.items():
    rec_macro += v
for f,v in f_score.items():
    fsc_macro += v
prec_macro = prec_macro/float(classes)
rec_macro = rec_macro/float(classes)
fsc_macro = fsc_macro/float(classes)
#print(precision,recall)
print("macro results--")
print(prec_macro,rec_macro,fsc_macro)
print("accuracy--")
print(acc1)
