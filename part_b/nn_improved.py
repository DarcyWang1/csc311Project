import torch
import numpy as np
from torch import optim
import torch.nn.utils.prune
from torch.autograd import Variable

from part_a.item_response import irt, sigmoid
from part_a.utils import load_public_test_csv, load_train_sparse, load_valid_csv
from part_b.utils import load_question_meta, load_student_meta
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    print(np.sum(zero_train_matrix == -1))
    # Fill in the missing entries to 0.
    #zero_train_matrix[train_matrix==0] = -1
    zero_train_matrix[np.isnan(train_matrix)] = 0
    print(np.sum(zero_train_matrix == 0))
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    student_data = load_student_meta(base_path)
    question_data = load_question_meta(base_path)


    return zero_train_matrix, train_matrix, valid_data, test_data,question_data,student_data


class non_connected(torch.nn.Module):
    def __init__(self, questation_type, k):
        super(non_connected, self).__init__()
        t_count, q_count = questation_type.shape
        print(q_count,t_count)
        self.layer_1=torch.nn.Linear(q_count, k)
        self.layer_2 = torch.nn.Linear(k, q_count)
        self.another_layer_1 = torch.nn.Linear(q_count, t_count+k)
        print(torch.Tensor([[1] * q_count] * k).size())
        print(torch.Tensor(questation_type).size())
        a =torch.cat((torch.Tensor([[1]*q_count]*k),torch.Tensor(questation_type)),0)
        torch.nn.utils.prune.custom_from_mask(self.another_layer_1,
                                             name='weight',
                                              mask=a)
        self.another_layer_2 = torch.nn.Linear(t_count+k, q_count)
        #self.layer_3 = torch.nn.Linear(k, t_count)
        #self.layer_4 = torch.nn.Linear(t_count,q_count)
        #torch.nn.utils.prune.custom_from_mask(self.layer_4,
         #                                     name='weight',
          #                                    mask=torch.Tensor(
           #                                       questation_type.T))

    def forward(self, input):
        l1 = torch.nn.Sigmoid()(self.another_layer_1(input))
        #print(self.layer_1.weight)
        #print('l1',l1)
        l2 = torch.nn.Sigmoid()(self.another_layer_2(l1))
        #print('l2',l2)
        #l3 = torch.nn.Sigmoid()(self.layer_3(l2))
        #print('l3',l3)
        #l4 = torch.nn.Sigmoid()(self.layer_4(l3))
        #print('l4',l4)
        #l1= torch.nn.Sigmoid()(self.layer_1(input))
        #l2=torch.nn.Sigmoid()(self.layer_2(l1))
        return l2


    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """

        return torch.norm(self.layer_2.weight, 2) ** 2+\
               torch.norm(self.layer_1.weight, 2) ** 2\
               #+torch.norm(self.layer_3.weight, 2) ** 2 +\
               #+torch.norm(self.layer_4.weight, 2) ** 2


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch,
          global_max_acc):
    epochs, train_losses, valid_accs, valid_losses = [], [], [], []
    max_acc = -1

    # Tell PyTorch you are training the model.
    #model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    vacc = []
    for epoch in range(0, num_epoch):
        model.train()
        print()
        train_loss = 0.
        optimizer.zero_grad()
        inputs = zero_train_data.clone()
        output = model(Variable(zero_train_matrix))
        msk = np.isnan(train_data)
        target = inputs.clone()
        target[msk]=output[msk]
        loss = torch.sum((output - target) ** 2.)
        if lamb and lamb > 0:
            loss += (lamb / 2) * model.get_weight_norm()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        vacc.append(eval(model,zero_train_matrix,1-msk.cpu().detach().numpy(),valid_data)[1])
        print(epoch,loss.item())
        #print(eval(model,zero_train_matrix,1-msk.cpu().detach().numpy(),valid_data))

    return model ,vacc
def eval(m,train,train_msk, validation):

    model.eval()
    t = ((m(train).cpu().detach().numpy() > 0.5) - train.cpu().detach().numpy())
    #print(t)
    vmsk = np.isnan(validation)
    vdata = validation*1
    vdata[vmsk]=0
    v = ((m(torch.Tensor(vdata)).cpu().detach().numpy() > 0.5) - vdata)
    return 1-np.sum(np.abs(t*train_msk))/np.sum(train_msk),1-np.sum(np.abs(v*(1-vmsk)))/np.sum(1-vmsk)










def fix_valdation(valdation):
    matrix = np.full((max(valdation['user_id'])+1,max(valdation['question_id'])+1),np.nan)
    for i in range(len(valdation['user_id'])):
        matrix[valdation['user_id'][i],valdation['question_id'][i]] = valdation['is_correct'][i]
    return matrix



def start_with_irt():
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    lr = 0.01
    iter_num = 20
    bag_size = 0.5
    num_bag = 3
    buffer = np.zeros(shape=sparse_matrix.shape)
    theta, beta, _, __ = irt(sparse_matrix, val_data, lr, iter_num)
    tb = (np.asarray([theta]).T @ (np.asarray([[1]] * beta.shape[0])).T) - \
             (np.asarray([[1]] * theta.shape[0]) @ np.asarray([beta]))
    return (sigmoid(tb)>0.5).astype(int)



if __name__=='__main__':
    zero_train_matrix, train_matrix, valid_data, test_data, question_data, student_data=load_data()
    #s = max(((max((list(student_data.values())), key=lambda l: max(l)))))
    q = np.zeros((max(((max((list(question_data.values())),key=lambda l:max(l)))))+1, max(question_data.keys()) + 1))
    for i in question_data.keys():
        for j in question_data[i]:
            q[j][i] = 1
    #s= np.zeros((s + 1, max(student_data.keys()) + 1))
    #for i in student_data.keys():
     #   for j in student_data[i]:
      #      s[j][i] = 1

    model = non_connected(q,100)
    num_epoch = 700
    m ,v = train(model,0.005,0,train_matrix,zero_train_matrix,(fix_valdation(valid_data)),num_epoch,0)
    print(max(v))
    plt.plot(list(range(num_epoch)),v)
    plt.show()
    #print(torch.autograd.Variable(zero_train_matrix).size())
   # print(model.parameters())
    #m.eval()
    ztm = zero_train_matrix.cpu().detach().numpy()
