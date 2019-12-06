import pickle as pk
import numpy as np
import networkx as nx
import scipy as sp
import torch
import infomap
import community
import math
import tensorflow as tf
import collections
import operator
from collections import defaultdict
from collections import OrderedDict
from itertools import permutations
from collections import Counter
from networkx.algorithms import tree

#ind.cora.allx  ind.cora.ally  ind.cora.graph  ind.cora.test.index  ind.cora.tx  ind.cora.ty  ind.cora.x  ind.cora.y

def get_test_index(f):
    f1 = open(f,"r")
    ids = []
    for _f in f1:
        _f = _f.strip("\n\r")
        ids.append(int(_f))
    _t = np.array(ids)
    #_t = np.sort(_t)
    return _t


def convert_adj_to_sparse(adj):
    sparse_adj = sp.sparse.coo_matrix(adj,dtype=np.float32)
    indices = np.vstack((sparse_adj.row,sparse_adj.col))
    values = sparse_adj.data
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = sparse_adj.shape
    return torch.sparse.FloatTensor(i,v,torch.Size(s))



def get_data(cuda="True"):
    end = [".allx",".ally",".tx",".ty",".x",".y",".graph",".test.index"]
    _allx,_ally,_te_x,_te_y,_tr_x,_tr_y,_net,_test = [u[0]+u[1] for u in zip(["ind.cora"]*8,end)]  # Change the network here 
    allx = pk.load(open(_allx,"rb"),encoding="latin1")
    ally = pk.load(open(_ally,"rb"),encoding="latin1")
    te_x = pk.load(open(_te_x,"rb"),encoding="latin1")
    te_y = pk.load(open(_te_y,"rb"),encoding="latin1")
    tr_x = pk.load(open(_tr_x,"rb"),encoding="latin1")
    tr_y = pk.load(open(_tr_y,"rb"),encoding="latin1")
    net = pk.load(open(_net,"rb"),encoding="latin1")
    test_idx = get_test_index(_test)
    test_idx_reorder = np.sort(test_idx)
    #print(tr_x.shape,te_x.shape,test_id.size,allx.shape)
    features = sp.sparse.vstack((allx,te_x)).tolil()  # because without lil format there is computational burden
    print(features.shape)
    features[test_idx,:] = features[test_idx_reorder,:]
    labels = np.vstack((ally,te_y))
    labels[test_idx,:] = labels[test_idx_reorder,:]
    idx_train = range(len(tr_y))
    idx_val = range(len(tr_y), len(tr_y)+500)
    g = nx.from_dict_of_lists(net)
    print("number of nodes,edges ",g.number_of_nodes(),g.number_of_edges())
    adj = nx.to_numpy_array(g,dtype=np.float)
    adj = adj + np.eye(adj.shape[0])   # A = A'
    #print(adj[5,2546],adj[2546,5])
    #adj = sp.sparse.coo_matrix(adj)
    adj = convert_adj_to_sparse(adj)
    adj = normalize_adj(adj)            # make this commented for GraphSage
    features = normalize_features(features)
    #adj = convert_adj_to_sparse(adj)  # this is converted to sparse format due to its immense size
    adj = torch.FloatTensor(adj.to_dense())
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    #print(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(test_idx_reorder)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    #return g,adj,features,labels,idx_train,idx_val,idx_test
    return idx_train,idx_test,idx_val



def get_data_v1(cuda=True):

# Here the data is obtained from pytorch-geometric to eliminate unnecessary shuffling done in Kipf's code
    edge_index = pk.load(open("graph.pkl","rb"))
    row,col = edge_index
    edges = [(int(u),int(v)) for u,v in zip(row.tolist(),col.tolist())]
    g = nx.Graph()
    g.add_edges_from(edges)
    adj = np.zeros((torch.max(edge_index).item()+1,torch.max(edge_index).item()+1))
    for u,v in list(g.edges()):
        adj[u,v] = 1
        adj[v,u] = 1    
    adj = nx.to_numpy_array(g,dtype=np.float)
    adj = adj + np.eye(adj.shape[0])
    adj = sp.sparse.coo_matrix(adj)
    #adj = normalize_adj(adj)
    adj = torch.FloatTensor(adj.todense())
    features = pk.load(open("feature.pkl","rb"))
    features = normalize_features(features.numpy())
    features = torch.FloatTensor(features)
    labels = pk.load(open("label.pkl","rb"))
    idx_train = pk.load(open("train_ids.pkl","rb"))
    idx_val = pk.load(open("valid_ids.pkl","rb"))
    idx_test = pk.load(open("test_ids.pkl","rb"))
    print(len(idx_train),len(idx_test),len(idx_val))
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        #idx_train = idx_train.cuda()
        #idx_val = idx_val.cuda()
        #idx_test = idx_test.cuda()
    #return g,adj,features,labels,idx_train,idx_val,idx_test
    return idx_train,idx_test,idx_val



'''def get_data_v2(cuda=True):

# Here the data is obtained from pytorch-geometric to eliminate unnecessary shuffling done in Kipf's code
    edge_index = pk.load(open("graph.pkl","rb"))
    row,col = edge_index
    edges = [(int(u),int(v)) for u,v in zip(row.tolist(),col.tolist())]
    g = nx.Graph()
    g.add_edges_from(edges)
    print("Graph Read ")
    nodes = nx.nodes(g)
    print ("nodes:",nodes)
    adj = np.zeros((torch.max(edge_index).item()+1,torch.max(edge_index).item()+1))
    for u,v in list(g.edges()):
        adj[u,v] = 1
        adj[v,u] = 1    
    adj = nx.to_numpy_array(g,dtype=np.float)
    adj = adj + np.eye(adj.shape[0])
    adj = sp.sparse.coo_matrix(adj)
    print("Adjacency Made")
    adj = torch.FloatTensor(adj.todense())
    features = pk.load(open("feature.pkl","rb"))
    features = normalize_features(features.numpy())
    features = torch.FloatTensor(features)
    print("Features Normalized ")
    labels = pk.load(open("label.pkl","rb"))
    #idx_train = pk.load(open("train_ids.pkl","rb"))
    #idx_val = pk.load(open("valid_ids.pkl","rb"))
    #idx_test = pk.load(open("test_ids.pkl","rb"))
    info = infomap.Infomap("--two-level --silent")
    for e in list(g.edges()):
        info.addLink(*e)
    info.run()
    c = info.getModules()
    z = defaultdict(list)
    for u in c:
        z[c[u]].append(u)
    c1 = list(z.values())
    print ("communities",len(c1))
    c1_arr = np.array(c1)
    #temp_data = np.array([nx.density(g.subgraph(z)) for z in c1],dtype=np.float)
    temp_data = np.array([len(z) for z in c1],dtype=np.float)
    temp_data_idx = np.argsort(temp_data)
    c1_arr = c1_arr[temp_data_idx][::-1] # descending 
    total = int(0.1 * g.number_of_nodes())
    train_ids = []
    val_ids = []
    test_ids = []
    for z in c1_arr:
        g1 = g.subgraph(z)
        total_node = list(g1.nodes())
        g2 = nx.k_core(g1)
        n1 = list(g2.nodes())
        tar_nodes = np.random.choice(n1,int(0.5*len(n1)),replace=False)
        other_nodes = list(set(total_node).difference(set(tar_nodes)))
        tar_nodes1 = list(set(n1).difference(set(tar_nodes)))
        if len(train_ids) <= total:
            train_ids.extend(tar_nodes)
        test_ids.extend(other_nodes)
        val_ids.extend(tar_nodes1)
    
    
    idx_train = np.array(train_ids)
    idx_val = np.array(val_ids)
    idx_test = np.array(test_ids)
    print("Train Validation Test ",len(idx_train),len(idx_val),len(idx_test))
    print("No of Communities {}".format(len(c1)))
         
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        #idx_train = idx_train.cuda()
        #idx_val = idx_val.cuda()
        #idx_test = idx_test.cuda()
    #return g,adj,features,labels,idx_train,idx_val,idx_test
    return idx_train,idx_test,idx_val'''

def bfs(graph, root): 
    visited, queue = set(), collections.deque([root])
    visited.add(root)
    while queue: 
        vertex = queue.popleft()
        for neighbour in graph[vertex]: 
            if neighbour not in visited: 
                visited.add(neighbour) 
                queue.append(neighbour)
    return visited

def get_data_v3(cuda=True):

# Here the data is obtained from pytorch-geometric to eliminate unnecessary shuffling done in Kipf's code
    edge_index = pk.load(open("graph.pkl","rb"))
    row,col = edge_index
    edges = [(int(u),int(v)) for u,v in zip(row.tolist(),col.tolist())]
    g = nx.Graph()
    g.add_edges_from(edges)
    print("Graph Read ")
    
    nnodes = nx.number_of_nodes(g)
    nodes = nx.nodes(g)
    #print(nodes)
    cr = dict(nx.core_number(g))
    cr_vals = set(v for v in cr.values())
    cr_dict = {}
    for d in cr_vals:
        tmp = []
        for k,v in cr.items():
            if v == d:
                tmp.append(k)
        cr_dict[d] = tmp
    print ("core numbers of original graph", len(cr_vals))

    print("number of nodes--",nnodes)
    cut = int(0.1*nnodes)
    print ("cut value--",cut)
    #print("number of nodes,edges ",g.number_of_nodes(),g.number_of_edges())
    adj = np.zeros((torch.max(edge_index).item()+1,torch.max(edge_index).item()+1))
    for u,v in list(g.edges()):
        adj[u,v] = 1
        adj[v,u] = 1
    adj = nx.to_numpy_array(g,dtype=np.float)
    adj = adj + np.eye(adj.shape[0])
    adj = sp.sparse.coo_matrix(adj)
    print("Adjacency Made")

    adj = torch.FloatTensor(adj.todense())
    features = pk.load(open("feature.pkl","rb"))
    features = normalize_features(features.numpy())
    features = torch.FloatTensor(features)
    print("Features Normalized ")

    labels = pk.load(open("label.pkl","rb"))
    lb = labels.numpy()
    ground_dict = Counter(lb)
    classes = len(ground_dict)
    

    #community detection --Infomap
    info = infomap.Infomap("--two-level --silent -s 8")
    for e in list(g.edges()):
        info.addLink(*e)
    info.run()
    c = info.getModules() #node:community
    z = defaultdict(list) 
    for u in c:
        z[c[u]].append(u) #community:[nodes]
    #print("number of communities detected")
    #print (len(z))
    com_size = {}
    for k,v in z.items():
        com_size[k] = len(v)
    #print(com_size)

    #community detection-- Louvain
    partition = community.best_partition(g) #node:community
    com = defaultdict(list)
    for p in partition:
            com[partition[p]].append(p)
    print("number of communities detected")
    print (len(com))
    
    a = set()
    a_wt = []
    for te in edges:
        u = te[0]
        v = te[1]
        com_u = partition[u]
        com_v = partition[v]
        t = (com_u,com_v)
        a.add(t)
        if com_u > com_v :
            m = (com_v,com_u)
            a_wt.append(m)
        else:
            a_wt.append(t)

    edge_wt = Counter(a_wt)
    #print(edge_wt)

    meta_wt_edge = {}
    #print(len(a))
    meta_nodes = list(com.keys())
    #print (len(meta_nodes))
    per = list(permutations(meta_nodes,2))
    b = set()
    for cc in per:
        b.add(cc)
    meta_edge = a.intersection(b)

    for k,v in edge_wt.items():
        if k in meta_edge:
            meta_wt_edge[k] = v

    #print("meta edges")
    #print(meta_wt_edge)

    meta_net =  nx.Graph()
    meta_net.add_nodes_from(meta_nodes)
    meta_net.add_edges_from(meta_edge)
    print ("meta graph formed")
    
    m_nodes = nx.number_of_nodes(meta_net)
    print ("number of meta nodes",m_nodes)
    
    m_edges = meta_net.number_of_edges()
    print ("number of meta edges",m_edges)

    train_ids = []
    
    edge_set = set(edges)
    for m in meta_nodes:
        coms = com[m]
        perm = set(permutations(coms,2))
        in_edges = edge_set.intersection(perm)
        #print(in_edges)
        in_net = nx.Graph()
        in_net.add_edges_from(in_edges)
        #print(in_net.edges())
        in_clus = nx.clustering(in_net)
        #print("clustering",in_clus)
        h = max(in_clus.items(), key = operator.itemgetter(1))[0]
        train_ids.append(h)
        

    #meta_edgelist = list(meta_net.edges())

    '''cores = dict(nx.core_number(meta_net))

    mst = nx.minimum_spanning_tree(meta_net, algorithm='prim')
    #print("tree edges",mst.edges())
    mst_edgelist = list(sorted(mst.edges()))
    mst_nodes =  list(mst.nodes())
    mst_adj = {}
    for s in mst_nodes:
        mst_l = []
        for e in mst_edgelist:
            if s == e[0] :
                mst_l.append(e[1])
        mst_adj[s] = mst_l

    #print(mst_adj)
    #print(mst_edgelist)
    core_vals = set(v for v in cores.values())
    core_dict = {}
    for d in core_vals:
        tmp = []
        for k,v in cores.items():
            if v == d:
                tmp.append(k)
        core_dict[d] = tmp'''



    #print(core_dict)
    #print ("number of cores in meta network:", len(core_dict))

    '''core_class = {}
    for k,v in core_dict.items():
        cls = []
        for m in v:
            nd = z[m]
            for x in nd:
                cl = lb[x]
                cls.append(cl)
        core_lb = Counter(cls)
        mm =  max(v for k,v in core_lb.items())
        for k1,v1 in core_lb.items():
            if v1 == mm:
                core_class[k]=k1
    print("class information per core--")
    print(core_class)   #The class information/core is printed

    com_class = {}
    for mn in meta_nodes:
        cls = []
        nd = z[mn]
        for x in nd:
            cl = lb[x]
            cls.append(cl)
        com_lb = Counter(cls)
        mm = max(v for k,v in com_lb.items())
        for k1,v1 in com_lb.items():
            if v1 == mm :
                com_class[mn] = k1

    print("class information per community--")
    #print(com_class) #The class information/community is printed

    com_cls = []
    for k,v in com_class.items():
        com_cls.append(v)
    print(Counter(com_cls))

    sorted_core = dict(OrderedDict(sorted(core_dict.items(),reverse=True)))

    reverse_core = dict(OrderedDict(sorted(sorted_core.items())))'''
    
    '''t_n = []
    for v in sorted_core[25]:
        for t in z[v]:
            t_n.append(t)
    t_lb = []
    for t in t_n:
        t_lb.append(lb[t])''' #for checking the class labels distribution in each core

    #build 2nd order network--

    '''meta_info = infomap.Infomap("--two-level --silent -s 8")
    for e in list(meta_net.edges()):
        meta_info.addLink(*e)
    meta_info.run()
    cc = meta_info.getModules() #node:community
    zz = defaultdict(list)
    for u in cc:
        zz[cc[u]].append(u) #community:[nodes]
    print("number of meta communities detected")
    print (len(zz))

    meta_coms = {}
    for k,v in zz.items():
        cls = []
        for b in v:
            lbl = com_class[b]
            cls.append(lbl)
        metacom_lb = Counter(cls)
        meta_coms[k] = metacom_lb

    print("class information of meta communities of 2nd order network--")
    print(meta_coms)

    meta_cr = dict(nx.core_number(meta_net))
    meta_cr_vals = set(v for v in meta_cr.values())
    meta_cr_dict = {}
    for d in meta_cr_vals:
        tmp = []
        for k,v in meta_cr.items():
            if v == d:
                tmp.append(k)
        meta_cr_dict[d] = tmp
    print("cores in 2nd order network--")
    print(meta_cr_dict)

    #Selection of training nodes
    core_window =3

    t_cores = []
    cnt = 0
    for cr,coms in sorted_core.items():
        t_cores.append(cr)
        cnt += 1
        if cnt == core_window:
            break
    print("t_cores--",t_cores)
    #print("t_coms--",len(t_coms))
   
    #build adjacency matrix of edges--
    t_coms = core_dict[7]
    p = len(t_coms)
    rows,cols = (p,p)
    adje = [[0]*cols]*rows
    for me in meta_edgelist:
        u = me[0]
        v = me[1]
        if u in t_coms:
            if v in t_coms:
                #h += 1
                ui = t_coms.index(u)
                vi = t_coms.index(v)
                adje[ui][vi] += 1
    #print(adje)'''
    '''for me in meta_edge:
        u = me[0]
        if u == 5:
            print(me)'''

    '''t_arr = []
    for i in range(core_window):
        t_arr.append(0)

    
    tr_dict = {}
    for cls in range(classes):
        tr_nodes = []
        fl  = 0
        ar = 0
        cnt_cls = int(0.1*(ground_dict[cls]))
        print("cls and count--",cls,cnt_cls)
        while(True):
            for cr in t_cores:
                coms = core_dict[cr]
                j = t_arr[ar]
                cm = coms[j]
                j = (j+1)%len(coms)
                t_arr[ar] = j
                ar += 1
                #cm = int(np.random.choice(coms,1))
                nn = z[cm]
                n = int(np.random.choice(nodes,1))
                l = lb[n]
                if l == cls and n not in tr_nodes:
                    tr_nodes.append(n)
                    if len(tr_nodes) == cnt_cls:
                        fl = 1
                        break
                if ar == core_window:
                    ar = 0
            if fl == 1:
                tr_dict[cls] = tr_nodes
                break



    t_lbls = []
    for k,v in tr_dict.items():
        for t in v:
            lbl = lb[t]
            t_lbls.append(lbl)
            
    print("class level distribution--training labels",Counter(t_lbls))

    train_ids = []
    val_ids = []
    test_ids = []
    test_mask_ids = []
    for k,v in tr_dict.items():
        for t in v:
            train_ids.append(t)
    #for n in nodes2:
        #train_ids.append(n)
    f = 0
    while True:
        if len(train_ids)<cut:
            r = int(np.random.choice(nodes,1,replace = False))
            if r not in train_ids:
                train_ids.append(r)
                if len(train_ids)==cut:
                    f = 1
        if f == 1:
            break
    #print("train ids--",len(train_ids))'''
    #sorted_core = dict(OrderedDict(sorted(core_dict.items(),reverse=True)))
    #print(sorted_core)    

    #c_meta_nodes = sorted_core[7]
    #y = int(np.random.choice(c_meta_nodes,1))

    #train_ids = []
    #train_coms = bfs(mst_adj,y)
    #print(train_coms)

    '''f = 0
    while True:
        for tc in train_coms:
            yy = z[tc]
            x = int(np.random.choice(yy,1))
            train_ids.append(x)
            if len(train_ids) == cut :
                f = 1
                break
        if f == 1:
            break
        else:
            continue'''

    #print(train_ids)

    #train-test nodes choice

    '''for m in meta_nodes:
        f_nodes = z[m]
        x = int(np.random.choice(f_nodes,1,replace=False))
        train_ids.append(x)'''

    val_ids = []
    test_ids = []
    rm_ids = []
        
    for n in nodes:
        if n not in train_ids:
            #if n not in nodes2:
                rm_ids.append(n)
    #print ("test ids--",len(test_ids))

    #val_ids.extend(rm_ids[0:int(0.1*len(nodes))])
    val_ids = np.random.choice(rm_ids,len(train_ids),replace=False)

    r_ids = []
    for n in rm_ids:
        if n not in val_ids:
            r_ids.append(n)
    #val_ids= np.random.choice(test_ids,int(0.1*len(nodes)),replace= False)
    test_ids = np.random.choice(r_ids,1084,replace = False)
    #val_ids = np.random.choice(test_ids,int(0.1*len(nodes)),replace= False)
    #test_mask_ids = np.random.choice(test_ids,1084,replace = False)

    with open("test_labels_infomap.txt",'wb') as fp:
        pk.dump(test_ids,fp)
        
    with open("training_labels_infomap.txt","wb") as fp:
        pk.dump(train_ids,fp)

        
    idx_train = np.array(train_ids)
    idx_val = np.array(val_ids)
    idx_test = np.array(test_ids)
    print("Train Validation Test ",len(idx_train),len(idx_val),len(idx_test))


    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        #idx_train = idx_train.cuda()
        #idx_val = idx_val.cuda()
        #idx_test = idx_test.cuda()
    #return g,adj,features,labels,idx_train,idx_val,idx_test
    return idx_train,idx_test,idx_val


def normalize_adj(m):
    print("In adj Nomalizee")
    rowsum = np.array(m.sum(1))
    rowsum = np.power(rowsum,-1).flatten()
    rowsum[np.isinf(rowsum)] = 0  # error handling
    D = sp.sparse.diags(rowsum)
    adj = D.dot(m).dot(D)     # D^-1*A'D^-1
    #print(adj)
    #print(type(adj))
    #adj = np.array(adj,dtype=np.float32)
    return adj


def normalize_features(m):
    rowsum = np.array(m.sum(1))
    z_idx = np.where(rowsum==0)[0]
    rowsum[z_idx] = 1                    # for cases where all features are zeros 
    rowsum = np.power(rowsum,-1).flatten()
    rowsum[np.isinf(rowsum)] = 0  # error handling
    D = sp.sparse.diags(rowsum)
    F = D.dot(m)     # D^-1*F
    return F


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

