import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

class model1(nn.Module):
    def __init__(self, name,
                 coord_x=0, coord_y=1, coord_t=-2, coord_m=-1,
                 emb_dim=4,
                 d_model=512,
                 nhead=8, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        '''
        modelo basado en transformers
        el primer elemento de la secuencia es el punto de referencia con coord_m = 0
        el resto se miden respecto del primer elem salvo coord_m
        '''
        super().__init__()
        self.name = name
        self.x = coord_x
        self.y = coord_y
        self.t = coord_t
        self.m = coord_m
        self.pad_token = torch.tensor([0]*emb_dim).to(device)

        self.emb_dim = emb_dim
        self.d_model = d_model

        # proyectar embeddings a d_model
        if self.d_model != self.emb_dim:
            self.linear_in = nn.Linear(self.emb_dim, d_model)

        self.transformer = nn.Transformer(d_model=self.d_model,
                                          batch_first=True,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        
        self.linear_out = nn.Linear(self.d_model, self.emb_dim)
        self.act = customActivation(self.t, self.m)
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        '''
        input: (batch_size, input_seq_length, emb_dim)
        target: (batch_size, target_seq_length, emb_dim)

        scores: (batch_size, target_seq_length, emb_dim)
        '''
        encoder_output = self.forward_encoder(input)
        output = self.forward_decoder(target, encoder_output)
        return output
    
    def forward_encoder(self, input):
        if self.d_model != self.emb_dim:
                input = self.linear_in(input)
    
        # esto para se mantiene constante
        output = self.transformer.encoder(input, is_causal=True)
        return output
    
    def forward_decoder(self, target, encoder_output):
        if self.d_model != self.emb_dim:
            target = self.linear_in(target)

        output = self.transformer.decoder(target, encoder_output, is_causal=True) # (batch_size, target_seq_length, d_model)
        if self.d_model != self.emb_dim:
            output = self.linear_out(output) # (batch_size, target_seq_length, emb_dim)

        return self.act(output)

    def generate(self, input, max_length=10):
        '''
        input: (input_seq_length, emb_dim)
        target_pred: (generated_seq_length, emb_dim)
        '''
        with torch.no_grad():
            input = input.unsqueeze(0)

            encoder_output = self.forward_encoder(input[:-1, :])
            current_target = input[-1, :]
            print(current_target.shape)

            for _ in range(max_length - 1):
                scores = self.forward_decoder(current_target, encoder_output) # (1, current_target, emb_dim)
                next_event = scores[:, -1, :]
    
                if self.stop_criteria(current_target, next_event):
                    break

                current_target = torch.cat([current_target, next_event], dim=1) # (1, current_target+1, emb_dim)
            return current_target.squeeze(0)
    
    def batch_generate(self, input_batch, max_length=512):
        """
        Lento pero deberia funcionar
        input_batch: (batch_size, input_seq_length, emb_dim)
        returns: list (generated_seq_length, emb_dim) para cada elemento en el batch
        """
        results = []
        for i in range(input_batch.size(0)):
            generated = self.generate(input_batch[i], max_length)
            results.append(generated)
        return results
    
    def loss(self, input, target):
        '''
        input: (batch_size, input_seq_length, emb_dim)
        target: (batch_size, target_seq_length, emb_dim)
        '''
        target_predictor = torch.cat([input[:, -1, :], target[:, :-1, :]], dim=1) # right shifted

        target_predicted = self.forward(input, target_predictor)
        target_predicted = target_predicted.reshape(-1, self.emb_dim) # (batch_size * target_seq_length-1, emb_dim)

        target = target.reshape(-1, self.emb_dim) # (batch_size * target_seq_length-1, emb_dim)

        # Mse
        loss = self.mse(target, target_predicted)
        return loss
    
    def stop_criteria(self, events, next_event):
        '''
        events: (1, seq_len, emb_dim)
        next_event: (1, 1, emb_dim)
        '''
        output = False
        last_event = events[:, -1, :] # (1, 1, emb_dim)
        if last_event[self.m].item() < next_event[self.m].item():
            output = True
        return output
    
class customActivation(nn.Module):
    def __init__(self, coord_t, coord_m):
        super().__init__()
        self.t = coord_t
        self.m = coord_m

    def forward(self, x):
        output = x.clone()

        # las magnitudes estan entre 0 y 10
        output[..., self.m] = torch.sigmoid(output[..., self.m]) * 10
        
        # delta_t positivo
        output[..., self.t] = F.relu(output[..., self.t])
        
        # el resto: identidad
        return output