# 그래프마이닝과 GNN의 기초적인 컨셉

"""
SETUP
- NetworkX : 그래프마이닝이나 그래프조작에서 흔히 쓰는 툴
"""
# Import the NetworkX package
import networkx as nx

"""
Graph
- NetworkX provides several classes to store different types of graphs, such as directed and undirected graph
"""
# Create an undirected graph G
G = nx.Graph()
print(G.is_directed()) #그래프 타입 확인

# Create a directed graph H
H = nx.DiGraph()
print(H.is_directed()) #그래프 타입 확인

# Add graph level attribute(그래프 속성 추가)
G.graph["Name"] = "Bar" # 딕셔너리에 데이터를 저장하는 것과 동일
print(G.graph)


"""
Node
- node를 속성과 함께 그래프에 추가하기
"""
# Add one node with node level attributes
G.add_node(0, feature=0, label=0)

# Get attributes of the node 0
node_0_attr = G.nodes[0]
print("Node 0 has the attributes {}".format(node_0_attr))

# 속성과 함께 여러개의 노드 추가
G.add_nodes_from([
  (1, {"feature": 1, "label": 1}),
  (2, {"feature": 2, "label": 2})
]) # add_nodes_from 메서드를 사용하고 인자로 list내부 (node, attrdict)의 튜플 형태로 전달

# for문으로 그래프를 iterable하게 사용할 수 있음.
for node in G.nodes(data=True): # data=True: 속성을 함께 반환함
  print(node)

# 노드의 개수를 얻을 수 있음
num_nodes = G.number_of_nodes()
print("G has {} nodes".format(num_nodes))



"""
Edge
"""
# weight=0.5 속성을 갖도록 노드 0과 1사이의 엣지를 만들어 줌
G.add_edge(0, 1, weight=0.5)

# (0, 1) 엣지의 속성을 얻기
edge_0_1_attr = G.edges[(0, 1)]
print("Edge (0, 1) has the attributes {}".format(edge_0_1_attr))


# 여러 엣지를 속성과 함께 추가하기
G.add_edges_from([
  (1, 2, {"weight": 0.3}),
  (2, 0, {"weight": 0.1})
])

# 노드와 마찬가지로 반복 가능.
for edge in G.edges(): # data=True를 추가하면 속성을 함께 반환
  print(edge)

# 엣지의 개수 반환
num_edges = G.number_of_edges()
print("G has {} edges".format(num_edges))

"""
Visualization
"""
# 그래프 그리기
nx.draw(G, 
        with_labels = True) # label을 함께 그리기


"""
Node Degree and Neighbor
"""
node_id = 1

# 개별 노드에 대한 degree를 확인할 수 있음
print("Node {} has degree {}".format(node_id, G.degree[node_id]))

# 개별 노드에 대한 이웃노드를 for-loop으로 얻을 수 있음.
for neighbor in G.neighbors(node_id):
  print("Node {} has neighbor {}".format(node_id, neighbor))



"""
Other Functionalities
-NetworkX는 그래프를 다루는 많은 유용한 메서드를 제공하는데, 여기서는 대표적으로 PageRank를 사용 
"""
# 새로운 directed graph 만들기, 이때의 각노드는 순차적으로 앞선 index의 노드만을 가리키도록 초기화된다.
num_nodes = 4
G = nx.DiGraph(nx.path_graph(num_nodes))
nx.draw(G, with_labels = True)

# PageRank알고리즘으로부터 score를 얻기
pr = nx.pagerank(G, alpha=0.8)
pr


"""
PyTorch Geometric Tutorial
- PyG는 Pythorch의 확장 라이브러리로 GNN모델을 개발하기 위해서 유용하게 사용된
- 많은 기초적인 layer와 벤치마크 데이터셋을 가지고 있음
"""
import torch
print("PyTorch has version {}".format(torch.__version__))


"""
setup
- 10분 이상 소요
"""
# Install torch geometric
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
!pip install -q torch-geometric


"""
Visualization
- 시각화를 위한 Helper함수 작성
"""
%matplotlib inline
import torch
import networkx as nx
import matplotlib.pyplot as plt

# NX와 Pytorch tensor 모두를 시각화 할 수 있는 함수이다.
def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    # 데이터가 pytorch tensor일 경우
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                       f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
                       f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
                       fontsize=16)
    # 데이터가 NX일 경우
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()



"""
Introduction
- 불규칙한 데이터(이미지나 텍스트가 아닌)를 위한 고전적인 DL을 일반화하고, NN을 이용하여 노드나 엣지와의 관계성을 파악
- PyG에 기반한 기초적인 GNN컨셉 소개
- 논문(Kipf et al.(2017))과 간단한 구조의 데이터인 Zachary's karate club network를 참고하여 탐색
  (이 그래프는 34명의 가라데동아리 회원과 그들의 관계에 대한 데이터) 
"""

"""
Dataset
- PyG를 통해 위의 데이터를 받아옴 (torch_geometric.datasets )
"""
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

#데이터를 받아오면 이들의 정보를 우선적으로 파악.
#데이터가 하나의 그래프로 이루어져 있고, 각 노드는 34 차원의 특성 벡터를 갖고 있음. 
#또한, 그래프는 4개의 클래스를 갖는데 이들은 커뮤니티에 해당함

data = dataset[0]  # Get the first graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


"""
Data
- 각 그래프는 PyG의 그래프를 표현하기 위한 모든 정보를 가진 Data 클래스의 오브젝트임. 
- 데이터 오브젝트는 아래 4가지로 구성
  1) edge_index: 그래프의 연결 관계를 보여준다. 
  2) node_features: 노드의 개별 feature vector를 보여준다.(여기서는 각 노드가 34차원의 feature vector를 갖는다)
  3) node_labels: 각 노드의 label을 가진다.
  4) train_mask: 훈련시 사용할 노드와 사용하지 않을 노드를 분류하기 위한 정보를 가진다.

또한, data 오브젝트는 유용한 함수들을 제공하는데, isolated node, self-loop 등을 확인할 수 있다.
"""
from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

edge_index = data.edge_index
print(edge_index.t())


"""
Edge Index

"""
from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

edge_index = data.edge_index
print(edge_index.t()) #edge index확인

# 각 엣지는 2개의 노드를 가진 튜플로 이뤄지는데, 첫번째는 출발 지점이고 두번째는 도착 노드.
# 이러한 표현은 COO(coordinate) format으로 알려져 있고, sparse matrices를 표현하기 위해서 사용됨. 그리고 이것은 networkx형태로 바꾸어 시각화할 수 있음

from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)

"""
첫번째 GNN모델 만들기 
이 모델의 기본 구성요소로 GCN layer를 사용하고, PyG는 이미 GCNConv로 구현되어있음. 이것은 node feature representation x와 COO graph의 연결 표현인 edge_index를 사용.
GNN의 input은 
1) Graph(G=(V,E))
2) Node(vi ∈ V)
3) Feature Vector(Xi(0)
​
이때 함수 f→V×Rd를 학습하면, 우리의 downstream task에 따라서 다양한 형태의 아웃풋을 얻을 수 있고, 이들을 이용하여 다양한 예측 수행가능.
여기서는 각 노드의 community 분류.
"""
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

model = GCN()
print(model)

"""
1) 모델을 위한 building block을 init메서드에서 정의. 이때 정의되는 각각의 gcn layer는 1-hop의 이웃의 정보를 합치는 것에 해당
 이들을 구성할 때 3-hop의 이웃의 정보를 합치는 것이 목표이므로 3개의 layer를 정의함.
2) GCNConv층은 node feature벡터의 차원을 2로 줄인다 (34 -> 4 -> 4 -> 2)
  각 레이어는 tanh의 비선형 활성화 함수를 사용
3) 1),2) 과정이 끝나면 하나의 선형 변환(torch.nn.Linear)층을 통해서 노드를 4개의 클래스에 해당하는 값으로 매핑핑
4) 결로부터 최종적인 노드 임베딩을 얻음
"""
model = GCN()

_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

visualize(h, color=data.y)
"""
모델의 가중치를 훈련하기 전에도 모델은 그래프의 커뮤니티 구조와 매우 유사한 노드의 임베딩을 생성하는 것을 알 수 있음 
모델의 가중치가 무작위로 완전히 초기화되었고 지금까지 어떠한 훈련도 수행하지 않았지만, 동일한 색상(커뮤니티)의 노드들은 이미 밀접하게 모여 있음
"""

"""
Train
- 모델의 모든 것이 차별화 및 매개 변수화 가능하기 때문에 일부 레이블을 추가하고 모델을 학습하며 임베딩이 어떻게 반응하는지 관찰할 수 이있음. 여기서 우리는 준지도 또는 전이 학습을 사용함. 클래스당 하나의 노드에 대해 간단히 훈련하지만 완전한 입력 그래프 데이터를 사용할 수 있음.
"""
import time

model = GCN()
criterion = torch.nn.CrossEntropyLoss()  # loss를 크로스 엔트로피로 정의.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 옵티마이저를 adam이용.

def train(data):
    optimizer.zero_grad()  # 그래디언트 초기화.
    out, h = model(data.x, data.edge_index)  # single forward.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 훈련 노드를 기반으로 loss 계산.
    loss.backward()  # 그래디언트의 계산.
    optimizer.step()  # 가중치 업데이트.

    accuracy = {}
    # training accuracy 정확도를 4개의 examples를 이용하여 계산
    predicted_classes = torch.argmax(out[data.train_mask], axis=1) # [0.6, 0.2, 0.7, 0.1] -> 2
    target_classes = data.y[data.train_mask]
    accuracy['train'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float())
    
    # validation accuracy를 전체 그래프를 이용하여 계산
    predicted_classes = torch.argmax(out, axis=1)
    target_classes = data.y
    accuracy['val'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float())

    return loss, h, accuracy

for epoch in range(500):
    loss, h, accuracy = train(data)
    # 10epochs마다 임베딩 시각화
    if epoch % 100 == 0:
        visualize(h, color=data.y, epoch=epoch, loss=loss, accuracy=accuracy)
        time.sleep(0.3)

