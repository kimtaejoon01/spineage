from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from acsmodels.acs_convnext import convnext_tiny
from acsconv.operators import ACSConv
from network.heads import ProjectionHead, Cumulative_Probability_Head

import torchvision.models as models
        
        
class MultiTaskLearner(nn.Module):
    def __init__(self, args):
        super(MultiTaskLearner, self).__init__()
        
        self.args = args
        self.encoder = Encoder(args)
        if args.classifier == 'fc':
            self.decoders = FC(args)
        elif args.classifier == 'gru':
            self.decoders = GRU(args)
            
    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for _, p in self.encoder.named_parameters())

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for _, p in self.decoders.named_parameters())

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        if self.args.dimension == '2D':        
            return self.encoder.backbone.classifier[2].parameters()
        else:
            return self.encoder.backbone.head.parameters()
            
    def forward(self, cls_sample, pred_sample, turn_off=[]):
        features = self.encoder(cls_sample, pred_sample, turn_off)
        return *self.decoders(*features), features
        

class Encoder(nn.Module):
    def __init__(self, args): 
        super(Encoder, self).__init__()
        
        self.args = args
        if self.args.dimension == '2D':
            self.backbone = models.convnext_tiny(pretrained=args.pretrained)
            self.backbone.classifier[2] = nn.Linear(768, args.hidden_dim, bias=True)
        
        elif self.args.dimension == '3D':
            self.backbone = convnext_tiny(pretrained=args.pretrained)
            self.backbone.downsample_layers[0][0] = ACSConv(1, 96, kernel_size=(4, 4, 4), stride=(4, 4, 4), acs_kernel_split=(32, 32, 32))
            self.backbone.head = nn.Linear(768, args.hidden_dim, bias=True)
            
        else:
            AssertionError()
            
    def forward(self, cls_sample, pred_sample, turn_off=[]):
        if 'cls' in turn_off:
            self.args.cls_enabled = False
        if 'pred' in turn_off:
            self.args.pred_enabled = False
        
        if self.args.cls_enabled:
            n = cls_sample['vers_img_tensor'].size()[1] 
            ver_list = []
            
            for i in range(n):
                if self.args.dimension == '2D':
                    feat = self.backbone(cls_sample['vers_img_tensor'][:,i])
                elif self.args.dimension == '3D':
                    feat = self.backbone(cls_sample['vers_img_tensor'][:,i:i+1])
                
                if self.args.classifier == 'fc':
                    ver_list.append(feat)
                elif self.args.classifier == 'gru':
                    ver_list.append(feat[:,None])
                    
            cls_feat = torch.cat(ver_list, 1)
            
        else:
            cls_feat = None
            
        if self.args.pred_enabled:
            n = pred_sample['vers_img_tensor'].size()[1] 
            ver_list = []
            
            for i in range(n):
                if self.args.dimension == '2D':
                    feat = self.backbone(pred_sample['vers_img_tensor'][:,i])
                elif self.args.dimension == '3D':
                    feat = self.backbone(pred_sample['vers_img_tensor'][:,i:i+1])
                
                if self.args.classifier == 'fc':
                    ver_list.append(feat)
                elif self.args.classifier == 'gru':
                    ver_list.append(feat[:,None])
                    
            pred_feat = torch.cat(ver_list, 1)
            
        else:
            pred_feat = None
            
        return cls_feat, pred_feat, turn_off

class FC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if (args.task == 'CLS') : 
            if args.slices_select == 'random' and args.dimension == '2D':
                self.cls_decoder = nn.Sequential(nn.Linear(args.hidden_dim*9, 2))
            else:
                self.cls_decoder = nn.Sequential(nn.Linear(args.hidden_dim*3, 2))
        else: 
            self.cls_decoder = ProjectionHead(args, dim_in=self.args.dim_in, dim_out=2)
        
        self.pred_decoder = Cumulative_Probability_Head(args, dim_in=self.args.dim_in, dim_out=self.args.max_followup)
        
    def forward(self, cls_feat, pred_feat, turn_off):
        
        cls_output_dict = {'logit': None, 'prob': None}
        pred_output_dict = {'logit': None, 'prob': None}
        
        if self.args.cls_enabled:
            cls_output_dict['logit'] = self.cls_decoder(cls_feat) # (batch_size, 2)
            cls_output_dict['prob'] = F.softmax(cls_output_dict['logit'], dim=-1)

        if self.args.pred_enabled:
            pred_output_dict['logit'] = self.pred_decoder(pred_feat) # (batch_size, max_followup)
            pred_output_dict['prob'] = F.sigmoid(pred_output_dict['logit'])
            
        if 'cls' in turn_off:
            self.args.cls_enabled = True        
        if 'pred' in turn_off:
            self.args.pred_enabled = True
            
        return cls_output_dict, pred_output_dict 
    
    
class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()

        input_dim = 256 
        hidden_dim = 256
        layer_dim = 1
        dropout_prob = 0.3
        self.args = args
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=True
        )
        self.cls_fc = ProjectionHead(dim_in=hidden_dim*2, dim_out=2, dim_hidden=hidden_dim)
        self.pred_fc = Cumulative_Probability_Head(hidden_dim*2, dim_out=args.max_followup)

    def forward(self, cls_feat, pred_feat, turn_off):
        
        cls_output_dict = {}
        pred_output_dict = {}
        
        if self.args.cls_enabled:
            cls_out, _ = self.gru(cls_feat)
            cls_out = cls_out[:, -1, :]   
            cls_output_dict['logit'] = self.cls_fc(cls_out) 
        
        if self.args.pred_enabled:
            pred_out, _ = self.gru(pred_feat)
            pred_out = pred_out[:, -1, :]   
            pred_output_dict['logit'] = self.pred_fc(pred_out) 
        
        if 'cls' in turn_off:
            self.args.cls_enabled = True        
        if 'pred' in turn_off:
            self.args.pred_enabled = True 
                    
        return cls_output_dict, pred_output_dict
    
class Clinical_C(nn.Module):
    def __init__(self, args, dim_in=7):
        super(Clinical_C, self).__init__()
        dim_out = args.max_followup
        
        self.fc1 = nn.Linear(dim_in, 32)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(32, dim_out)
        
        self.base_hazard_fc = nn.Linear(dim_in, 1)
        mask = torch.ones([dim_out, dim_out])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        hazards = self.fc1(x)
        hazards = self.relu1(hazards)
        hazards = self.bn1(hazards)
        hazards = self.dropout1(hazards)
        
        hazards = self.fc2(hazards)
        hazards = self.relu2(hazards)
        hazards = self.bn2(hazards)
        hazards = self.dropout2(hazards)
        
        hazards = self.fc3(hazards)
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