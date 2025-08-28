import torch
from torch import nn

def weights_init(m):
    for child in m.children():
        if isinstance(child,nn.Linear) or isinstance(child,nn.Conv1d):
            torch.nn.init.xavier_uniform_(child.weight)

class FC_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=4, dropout=0.5):
        super(FC_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.num_classes = num_classes

        self.regressor_fc= nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim ),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            # nn.BatchNorm1d(self.hidden_dim//2),
            nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(self.hidden_dim//2, self.hidden_dim // 2),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(self.hidden_dim, self.hidden_dim//2)
        )

        self.classifer_1 = nn.Linear(self.hidden_dim // 2, self.num_classes)
        self.classifer_2 = nn.Linear(self.hidden_dim // 2, self.num_classes)
        self.classifer_3 = nn.Linear(self.hidden_dim // 2, self.num_classes)
        self.classifer_4 = nn.Linear(self.hidden_dim // 2, self.num_classes)
        self.classifer_5 = nn.Linear(self.hidden_dim // 2, self.num_classes)


    def forward(self, input):
        features = self.regressor_fc(input)
        output_1 = self.classifer_1(features)
        output_2 = self.classifer_2(features)
        output_3 = self.classifer_3(features)
        output_4 = self.classifer_4(features)
        output_5 = self.classifer_5(features)

        return output_1, output_2, output_3, output_4, output_5


class ConvNet1D(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_classes=4, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=5),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=5),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=5),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.flatten = nn.Flatten()

        self.classifer_1 = nn.Linear(hidden_dim, self.num_classes)
        self.classifer_2 = nn.Linear(hidden_dim, self.num_classes)
        self.classifer_3 = nn.Linear(hidden_dim, self.num_classes)
        self.classifer_4 = nn.Linear(hidden_dim, self.num_classes)
        self.classifer_5 = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.adaptive_pool(out)
        out = self.flatten(out) # N, 128

        output_1 = self.classifer_1(out)
        output_2 = self.classifer_2(out)
        output_3 = self.classifer_3(out)
        output_4 = self.classifer_4(out)
        output_5 = self.classifer_5(out)

        return output_1, output_2, output_3, output_4, output_5

class ConvNet1D_stress(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_classes=4, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=5),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=5),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=5),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.flatten = nn.Flatten()

        self.classifer_1 = nn.Linear(hidden_dim, self.num_classes)
        self.classifer_2 = nn.Linear(hidden_dim, self.num_classes)
        self.classifer_3 = nn.Linear(hidden_dim, self.num_classes)
        self.classifer_4 = nn.Linear(hidden_dim, self.num_classes)
        self.classifer_5 = nn.Linear(hidden_dim, self.num_classes)
        self.classifer_6 = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.adaptive_pool(out)
        out = self.flatten(out) # N, 128

        output_1 = self.classifer_1(out)
        output_2 = self.classifer_2(out)
        output_3 = self.classifer_3(out)
        output_4 = self.classifer_4(out)
        output_5 = self.classifer_5(out)
        output_6 = self.classifer_6(out)

        return output_1, output_2, output_3, output_4, output_5, output_6

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=5),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=5),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=5),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):

        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.adaptive_pool(output)
        out = self.flatten(output)

        return out #[batch, hidden_dim]


class Gate(nn.Module):
    def __init__(self, input_dim, input_channel, num_experts):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_dim * input_channel, num_experts)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=2)
        output = torch.softmax(self.fc(x), dim=1)
        return output #[batch, num_experts]


# class Gate(nn.Module):
#     def __init__(self, input_dim, input_channel, num_experts):
#         super(Gate, self).__init__()
#         self.hidden_dim =1
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(input_dim, self.hidden_dim, kernel_size=3,padding=1),
#             # nn.BatchNorm1d(self.hidden_dim),
#             nn.ReLU(),
#             # nn.Dropout(0.5),
#             # nn.MaxPool1d(kernel_size=5),
#         )
#
#         self.fc = nn.Linear(input_channel, num_experts)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = torch.flatten(x, start_dim=1, end_dim=2)
#         output = torch.softmax(self.fc(x), dim=1)
#         return output #[batch, num_experts]


class MMOE(nn.Module):
    def __init__(self, input_dim, input_channel, num_experts, hidden_dim, num_class):
        super(MMOE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate1 = Gate(input_dim, input_channel,  num_experts)
        self.gate2 = Gate(input_dim, input_channel,  num_experts)
        self.gate3 = Gate(input_dim, input_channel,  num_experts)
        self.gate4 = Gate(input_dim, input_channel,  num_experts)
        self.gate5 = Gate(input_dim, input_channel,  num_experts)


        self.task1_head = nn.Linear(hidden_dim, num_class)
        self.task2_head = nn.Linear(hidden_dim, num_class)
        self.task3_head = nn.Linear(hidden_dim, num_class)
        self.task4_head = nn.Linear(hidden_dim, num_class)
        self.task5_head = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # [batch, num_experts, hidder_dim]
        gate1_output = self.gate1(x).unsqueeze(2) # [batch, num_experts, 1]
        gate2_output = self.gate2(x).unsqueeze(2)
        gate3_output = self.gate3(x).unsqueeze(2)
        gate4_output = self.gate4(x).unsqueeze(2)
        gate5_output = self.gate5(x).unsqueeze(2)


        task1_fea = torch.sum(expert_outputs * gate1_output, dim=1) #[batch, hidden_dim]
        task2_fea = torch.sum(expert_outputs * gate2_output, dim=1)
        task3_fea = torch.sum(expert_outputs * gate3_output, dim=1)
        task4_fea = torch.sum(expert_outputs * gate4_output, dim=1)
        task5_fea = torch.sum(expert_outputs * gate5_output, dim=1)

        task1_output = self.task1_head(task1_fea)
        task2_output = self.task2_head(task2_fea)
        task3_output = self.task3_head(task3_fea)
        task4_output = self.task4_head(task4_fea)
        task5_output = self.task5_head(task5_fea)

        return task1_output, task2_output, task3_output, task4_output, task5_output, task1_fea,task2_fea,task3_fea,task4_fea,task5_fea, expert_outputs



class MMOE_stress(nn.Module):
    def __init__(self, input_dim, input_channel, num_experts, hidden_dim, num_class):
        super(MMOE_stress, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate1 = Gate(input_dim, input_channel,  num_experts)
        self.gate2 = Gate(input_dim, input_channel,  num_experts)
        self.gate3 = Gate(input_dim, input_channel,  num_experts)
        self.gate4 = Gate(input_dim, input_channel,  num_experts)
        self.gate5 = Gate(input_dim, input_channel,  num_experts)
        self.gate6 = Gate(input_dim, input_channel, num_experts)


        self.task1_head = nn.Linear(hidden_dim, num_class)
        self.task2_head = nn.Linear(hidden_dim, num_class)
        self.task3_head = nn.Linear(hidden_dim, num_class)
        self.task4_head = nn.Linear(hidden_dim, num_class)
        self.task5_head = nn.Linear(hidden_dim, num_class)
        self.task6_head = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # [batch, num_experts, hidder_dim]
        gate1_output = self.gate1(x).unsqueeze(2) # [batch, num_experts, 1]
        gate2_output = self.gate2(x).unsqueeze(2)
        gate3_output = self.gate3(x).unsqueeze(2)
        gate4_output = self.gate4(x).unsqueeze(2)
        gate5_output = self.gate5(x).unsqueeze(2)
        gate6_output = self.gate6(x).unsqueeze(2)


        task1_fea = torch.sum(expert_outputs * gate1_output, dim=1) #[batch, hidden_dim]
        task2_fea = torch.sum(expert_outputs * gate2_output, dim=1)
        task3_fea = torch.sum(expert_outputs * gate3_output, dim=1)
        task4_fea = torch.sum(expert_outputs * gate4_output, dim=1)
        task5_fea = torch.sum(expert_outputs * gate5_output, dim=1)
        task6_fea = torch.sum(expert_outputs * gate6_output, dim=1)

        task1_output = self.task1_head(task1_fea)
        task2_output = self.task2_head(task2_fea)
        task3_output = self.task3_head(task3_fea)
        task4_output = self.task4_head(task4_fea)
        task5_output = self.task5_head(task5_fea)
        task6_output = self.task6_head(task6_fea)

        return task1_output, task2_output, task3_output, task4_output, task5_output, task6_output, task1_fea,task2_fea,task3_fea,task4_fea,task5_fea, task6_fea,expert_outputs