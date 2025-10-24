import torch.nn as nn
import torch.nn.functional as F
import torch
import esm
from Bio import SeqIO
import argparse
import os

parser = argparse.ArgumentParser(description="Finetuned ESM-2 model for predicting whether a protein is from bacteria or phages")
parser.add_argument('input', type=str, help="Path of the input protein fasta file")
parser.add_argument('model', type=str, help="Path of the model parameter file")
parser.add_argument('-o', '--output', type=str, default='results.txt', help="Path of the output file, default is ./results.txt")
parser.add_argument('-g', '--GPU', type=str, default='', help="Determine which GPU(s) to use, see README for more information")
parser.add_argument('-b', '--BatchSize', type=int, help='Define the batch size used in the prediction', default=2)
args = parser.parse_args()

if args.GPU == '':
    device = torch.device('cpu')
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ESMORF_binary(nn.Module):
    def __init__(self, n_class, lis = ["32","31","30","29"]): # 调整4层
        super(ESMORF_binary, self).__init__()
        self.esm = esm.pretrained.esm2_t33_650M_UR50D()[0]
        for name, param in self.named_parameters():
            sp = name.split('.')
            if len(set(lis) & set(sp)) != 0:
                param.requires_grad =True
            else:
                param.requires_grad =False
        self.fc1 = nn.Linear(1280, 960)
        self.fc2 = nn.Linear(960, 480)
        self.fc3 = nn.Linear(480, 120)
        self.fc4 = nn.Linear(120, 30)
        self.fc5 = nn.Linear(30, n_class)
    def forward(self, batch_tokens):
        x = self.esm(batch_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        batch_tokens = batch_tokens.unsqueeze(-1)
        x = x.masked_fill(batch_tokens==2, 0)
        x = x.masked_fill(batch_tokens==1, 0)[:, 1:, :]
        num = torch.sum(batch_tokens>2, axis=1)
        x = x.sum(axis=1) / num
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))    
        return self.fc5(x)

esmorf = ESMORF_binary(2)
esmorf = esmorf.to(device)
esmorf.load_state_dict(torch.load(args.model, map_location=device))
if args.GPU != '':
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        esmorf = nn.DataParallel(esmorf)

alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]
batch_converter = alphabet.get_batch_converter()

data = []
for seq_record in SeqIO.parse(args.input, "fasta"):
    data.append((seq_record.id, str(seq_record.seq).replace('J', 'X').rstrip('*')))

total_group = int((len(data)-1) / args.BatchSize) + 1

result = []
for i in range(total_group):
    if i == (total_group-1):
        new_group = data[(i*args.BatchSize):]
    else:
        new_group = data[(i*args.BatchSize):((i+1)*args.BatchSize)]
    inputs = batch_converter(new_group)[2]
    if inputs.shape[1] > 1024:
        inputs = inputs[:, :1024]
    inputs = inputs.to(device)
    predicts = F.softmax(esmorf(inputs),1).cpu().detach().numpy()
    for j in range(len(predicts)):
        if predicts[j, 1] >= 0.5:
            result.append((new_group[j][0], predicts[j, 1], 'Phage'))
        else:
            result.append((new_group[j][0], predicts[j, 1], 'Bacterium'))

with open(args.output, 'w') as f:
    f.write('seq_id\tphage_prob\tpredict_result\n')
    for k in result:
        f.write(f'{k[0]}\t{k[1]}\t{k[2]}\n')
