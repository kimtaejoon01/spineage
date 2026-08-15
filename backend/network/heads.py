import torch
import torch.nn as nn

class Cumulative_Probability_Head(nn.Module):
    def __init__(self, args, dim_in, dim_out):
        super(Cumulative_Probability_Head, self).__init__()
        self.args = args
            
        if args.heads_balance == 0:
            self.hazard_fc = nn.Linear(dim_in, dim_out)
            self.relu = nn.ReLU(inplace=True)
                    
        elif args.heads_balance == 1:
            self.fc1 = nn.Linear(dim_in, 256)
            self.relu1 = nn.ReLU(inplace=True)
            self.bn1 = nn.BatchNorm1d(256)
            
            self.fc2 = nn.Linear(256, dim_out)
            self.relu2 = nn.ReLU(inplace=True)
                    
        elif args.heads_balance == 2:
            self.fc1 = nn.Linear(dim_in, 256)
            self.relu1 = nn.ReLU(inplace=True)
            self.bn1 = nn.BatchNorm1d(256)
            
            self.fc2 = nn.Linear(256, 256)
            self.relu2 = nn.ReLU(inplace=True)
            self.bn2 = nn.BatchNorm1d(256)
            
            self.fc3 = nn.Linear(256, dim_out)
            self.relu3 = nn.ReLU(inplace=True)
        
        else:
            NotImplementedError


        self.base_hazard_fc = nn.Linear(dim_in, 1)
        mask = torch.ones([dim_out, dim_out])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        if self.args.heads_balance == 0:
            hazards = self.hazard_fc(x)
            hazards = self.relu(hazards)
        
        elif self.args.heads_balance == 1:
            hazards = self.fc1(x)

            if self.args.bn_order == 'bn_relu':
                hazards = self.bn1(hazards)
                hazards = self.relu1(hazards)
            elif self.args.bn_order == 'relu_bn':
                hazards = self.relu1(hazards)
                hazards = self.bn1(hazards)
            else:
                NotImplementedError

            hazards = self.fc2(hazards)  
            # hazards = self.relu2(hazards)
            
        elif self.args.heads_balance == 2:
            hazards = self.fc1(x)
            if self.args.bn_order == 'bn_relu':
                hazards = self.bn1(hazards)
                hazards = self.relu1(hazards)
                hazards = self.fc2(hazards)
                hazards = self.bn2(hazards)
                hazards = self.relu2(hazards)
            elif self.args.bn_order == 'relu_bn':
                hazards = self.relu1(hazards)
                hazards = self.bn1(hazards)
                hazards = self.fc2(hazards)
                hazards = self.relu2(hazards)
                hazards = self.bn2(hazards)
            else:
                NotImplementedError
                
            hazards = self.fc3(hazards) 
            # hazards = self.relu3(hazards)        
        else:
            NotImplementedError
                        
        return hazards

    def forward(self, x):
        hazards = self.hazards(x) # (B, max_followup)
        B, T = hazards.size() 
        expanded_hazards = hazards.unsqueeze(-1).expand(
            B, T, T
        ) 
        masked_hazards = (
            expanded_hazards * self.upper_triagular_mask
        )
        base_hazard = self.base_hazard_fc(x) # (B, 1)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard # (B, max_followup)
        return cum_prob

class ProjectionHead(nn.Module):
    def __init__(self, args, dim_in=2048, dim_out=2):
        super().__init__()
        self.args = args

        if args.heads_balance == 0:
            dim_hidden = 4096
        
            self.linear1 = nn.Linear(dim_in, dim_hidden)
            self.bn1 = nn.BatchNorm1d(dim_hidden) 
            self.relu1 = nn.ReLU(True)
            
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
            self.bn2 = nn.BatchNorm1d(dim_hidden)
            self.relu2 = nn.ReLU(True)
            
            self.linear3 = nn.Linear(dim_hidden, dim_out)

        elif args.heads_balance == 1:
            dim_hidden = 256
            
            self.linear1 = nn.Linear(dim_in, dim_hidden)
            self.bn1 = nn.BatchNorm1d(dim_hidden) 
            self.relu1 = nn.ReLU(True)
            
            self.linear2 = nn.Linear(dim_hidden, dim_out)
                    
        elif args.heads_balance == 2:
            dim_hidden = 256
        
            self.linear1 = nn.Linear(dim_in, dim_hidden)
            self.bn1 = nn.BatchNorm1d(dim_hidden) 
            self.relu1 = nn.ReLU(True)
            
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
            self.bn2 = nn.BatchNorm1d(dim_hidden)
            self.relu2 = nn.ReLU(True)
            
            self.linear3 = nn.Linear(dim_hidden, dim_out)
        else:
            NotImplementedError
           
    def forward(self, x):
        x = self.linear1(x).unsqueeze(-1)
        if self.args.bn_order == 'bn_relu':
            x = self.bn1(x).squeeze(-1) 
            x = self.relu1(x)
        elif self.args.bn_order == 'relu_bn':
            x = self.relu1(x)            
            x = self.bn1(x).squeeze(-1) 
        else:
            NotImplementedError

        x = self.linear2(x)
        
        if self.args.heads_balance in [0, 2]:
            if self.args.bn_order == 'bn_relu':
                x = x.unsqueeze(-1) 
                x = self.bn2(x).squeeze(-1) 
                x = self.relu2(x)
                x = self.linear3(x) # (B, 2)
            elif self.args.bn_order == 'relu_bn':
                x = x.unsqueeze(-1) 
                x = self.relu2(x)
                x = self.bn2(x).squeeze(-1) 
                x = self.linear3(x) # (B, 2)
            else:
                NotImplementedError
                            
        return x