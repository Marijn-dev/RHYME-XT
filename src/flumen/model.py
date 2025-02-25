import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class CausalFlowModel(nn.Module):

    def __init__(self,
                 state_dim,
                 control_dim,
                 output_dim,
                 control_rnn_size,
                 control_rnn_depth,
                 encoder_size,
                 encoder_depth,
                 decoder_size,
                 decoder_depth,
                 use_POD,
                 use_trunk,
                 use_fourier,
                 trunk_size,
                 POD_modes,
                 use_batch_norm=False):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        
        self.control_rnn_size = control_rnn_size

        self.POD_enabled = use_POD
        self.use_trunk = use_trunk
        self.trunk_enabled = use_trunk
        self.POD_modes = POD_modes
        self.trunk_size = trunk_size
        
        self.fourier_enabled = use_fourier
        assert self.POD_modes <= self.state_dim,  'POD_modes too high'

        if self.POD_enabled:
            self.in_size_encoder,self.out_size_decoder,self.control_dim = self.POD_modes,self.POD_modes,self.POD_modes


        elif self.trunk_enabled:
            if self.fourier_enabled:
                self.in_size_encoder = (state_dim // 2 + 1) * 2
                self.control_dim = (control_dim // 2 + 1) * 2

            else:
                self.in_size_encoder = state_dim

            self.out_size_decoder = self.POD_modes

        # Fourier Net
        elif self.fourier_enabled:
            self.in_size_encoder = (state_dim // 2 + 1) * 2
            self.out_size_decoder = output_dim
            self.control_dim = (control_dim // 2 + 1) * 2

        # regular flow
        else:
            self.in_size_encoder = state_dim
            self.out_size_decoder = output_dim

        self.u_rnn = torch.nn.LSTM(
            input_size=1 + self.control_dim,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=control_rnn_depth,
            dropout=0,
        )

        x_dnn_osz = control_rnn_depth * control_rnn_size

        # Flow encoder
        self.x_dnn = FFNet(in_size=self.in_size_encoder,
                           out_size=x_dnn_osz,
                           hidden_size=encoder_depth *
                           (encoder_size * x_dnn_osz, ), 
                           use_batch_norm=use_batch_norm)

        # Flow decoder
        u_dnn_isz = control_rnn_size
        self.u_dnn = FFNet(in_size=u_dnn_isz,
                           out_size=self.out_size_decoder,
                           hidden_size=decoder_depth *
                           (decoder_size * u_dnn_isz, ),
                           use_batch_norm=use_batch_norm)
        
        # Trunk network
        if self.trunk_enabled:
            self.trunk = TrunkNet(in_size=1,
                            out_size=self.POD_modes,
                            hidden_size=self.trunk_size,
                            use_batch_norm=use_batch_norm)

    def forward(self, x, rnn_input,PHI,locations, deltas):
        if self.fourier_enabled:
            x_fft = torch.fft.rfft(x) 
            x0 = torch.cat([x_fft.real, x_fft.imag],dim=-1) # concatenate real and imag

            unpadded_u, unpacked_lengths = pad_packed_sequence(rnn_input, batch_first=True) # unpack input
            deltas = unpadded_u[:, :, -1:]   
            u = unpadded_u[:, :, :-1]                                         # extract inputs values

            u_fft = torch.fft.rfft(u, dim=-1)  
            u_fft = torch.cat([u_fft.real, u_fft.imag], dim=-1)  
            u_deltas = torch.cat((u_fft, deltas), dim=-1)  
            input = pack_padded_sequence(u_deltas, unpacked_lengths, batch_first=True)
        
        # Project inputs to Flow model
        elif self.POD_enabled: 
            x0 = torch.einsum("ni,bn->bi",PHI[:,:self.POD_modes],x) 

            unpadded_u, unpacked_lengths = pad_packed_sequence(rnn_input, batch_first=True) # unpack input
            deltas = unpadded_u[:, :, -1:]                                    # extract deltas
            u = unpadded_u[:, :, :-1]                                         # extract inputs values
            
            u_projected = torch.einsum('ni,btn->bti',PHI[:,:self.POD_modes],u) # project inputs
            u_deltas = torch.cat((u_projected, deltas), dim=-1)  
            input = pack_padded_sequence(u_deltas, unpacked_lengths, batch_first=True)
        
        else:
            x0 = x
            input = rnn_input

        h0 = self.x_dnn(x0)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(input, (h0, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                           batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h
        output_flow = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :])
        
        # Regular
        if self.POD_enabled == False and self.trunk_enabled == False:
            return output_flow
            

        elif self.POD_enabled:
            ## POD + Trunk
            if self.trunk_enabled:
                trunk_output =  self.trunk(locations.view(-1,1))
                basis_functions = PHI[:,:self.POD_modes] + trunk_output 
                
            ## POD
            else:
                basis_functions = PHI[:,:self.POD_modes]

        ## Trunk
        else:
            basis_functions = self.trunk(locations.view(-1,1))

        output = torch.einsum("ni,bi->bn",basis_functions,output_flow)
        return output

class CausalFlowModelV2(nn.Module):

    def __init__(self,
                 state_dim,
                 control_dim,
                 output_dim,
                 control_rnn_size,
                 control_rnn_depth,
                 encoder_size,
                 encoder_depth,
                 decoder_size,
                 decoder_depth,
                 use_batch_norm=False):
        super(CausalFlowModelV2, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim

        self.control_rnn_size = control_rnn_size

        self.u_rnn = torch.nn.LSTM(
            input_size=1 + control_dim,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=control_rnn_depth,
            dropout=0,
        )

        ### ENCODER ###
        x_dnn_osz = control_rnn_depth * control_rnn_size
        
        # Convolutional encoder
        self.encoder = CONV_Encoder(in_size=state_dim,
                               out_size=x_dnn_osz)
        
        ### DECODER ###
        u_dnn_isz = control_rnn_size
        # Convolutional decoder 
        self.decoder = CONV_Decoder(in_size=u_dnn_isz,
                                    out_size=output_dim)

    def forward(self, x, rnn_input, deltas):
        h0 = self.encoder(x)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                           batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h
       
        output = self.decoder(encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :])
        return output

class FFNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 activation=nn.Tanh,
                 use_batch_norm=False):
        super(FFNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))

        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        self.layers.append(activation())

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))

            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))

            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
    
class TrunkNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 activation=nn.Tanh,
                 use_batch_norm=False):
        super(TrunkNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))
        self.layers.append(activation())

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))
            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

class CONV_Encoder(nn.Module):
    def __init__(self,
                 in_size,
                 out_size):         
        super(CONV_Encoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        # activation and pool layer
        self.activation = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)
        
        # convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(in_channels=16,out_channels=8,kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(in_channels=8,out_channels=4,kernel_size=3,padding=1)
        
        # linear layer to get right dimensionality of hidden states
        self.fc = nn.Linear(in_features=100,out_features=out_size)

    def forward(self, input):
        # reshape input from [1,batch_size,channel] -> [batch_size,1,channel] ??
        input = input.view(-1,1,self.in_size)
        input = self.activation(self.conv1(input))
        input = self.pool(input)
        input = self.activation(self.conv2(input))
        input = self.pool(input)
        input = self.activation(self.conv3(input))
        
        ## omit this one cause more downsizing isnt needed
        # input = self.pool(input)

        # flatten  
        input = input.view(input.size(0),1,-1)
        input = self.fc(input)
        
        # reshape to (batch_size,out_size)
        input = input.view(-1,self.out_size)
        return input

class CONV_Decoder(nn.Module):
    def __init__(self,
                 in_size,
                 out_size):         
        super(CONV_Decoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        # activation and pool layer
        self.activation = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)
        
        # convolutional layers
        # self.TransposeConv1 = nn.ConvTranspose1d(in_channels=4,out_channels=4,kernel_size=2,stride=2,padding=0,output_padding=0)
        self.TransposeConv2 = nn.ConvTranspose1d(in_channels=4,out_channels=8,kernel_size=2,stride=2,padding=0)
        self.TransposeConv3 = nn.ConvTranspose1d(in_channels=8,out_channels=16,kernel_size=2,stride=2,padding=0)
        self.TransposeConv4 = nn.ConvTranspose1d(in_channels=16,out_channels=1,kernel_size=3,stride=1,padding=1)
        
        # linear layer to get right dimensionality of hidden states
        self.fc = nn.Linear(in_features=in_size,out_features=100)
        self.fc2 = nn.Linear(in_features=100,out_features=out_size)

    def forward(self, input):
        # reshape input from [1,batch_size,channel] -> [batch_size,1,channel] 
        input = input.view(-1,1,self.in_size)

        # Linear Layer receiving Ld
        input = self.activation(self.fc(input))

        # reshape to CNN input
        # input = input.view(-1,4,12)
        input = input.view(-1,4,25)

        # ConvTranspose layers
        # input = self.activation(self.TransposeConv1(input))
        
        input = self.activation(self.TransposeConv2(input))
        input = self.activation(self.TransposeConv3(input))
        input = self.activation(self.TransposeConv4(input))

        # reshape for linear layer
        # input = input.view(input.size(0),1,-1)
        # print(input.shape)
        
        # final layer is linear layer
        input = self.fc2(input)

        # reshape for correct output
        input = input.view(-1,self.out_size)

        return input


