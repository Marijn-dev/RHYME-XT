import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

# class CausalFlowModel(nn.Module):

#     def __init__(self,
#                  state_dim,
#                  control_dim,
#                  output_dim,
#                  control_rnn_size,
#                  control_rnn_depth,
#                  encoder_size,
#                  encoder_depth,
#                  decoder_size,
#                  decoder_depth,
#                  use_nonlinear,
#                  IC_encoder_decoder,
#                  regular,
#                  use_conv_encoder,
#                  trunk_size_svd,
#                  trunk_size_extra,
#                  NL_size,
#                  trunk_modes,
#                  trunk_model,
#                  use_batch_norm):
#         super(CausalFlowModel, self).__init__()

#         self.state_dim = state_dim
#         self.control_dim = control_dim
#         self.output_dim = output_dim
#         self.control_rnn_size = control_rnn_size
#         self.trunk_modes = trunk_modes
#         self.trunk_size_svd = trunk_size_svd
#         self.trunk_size_extra = trunk_size_extra
#         self.NL_size = NL_size
#         self.conv_encoder_enabled = use_conv_encoder
#         self.nonlinear_enabled = use_nonlinear 
#         self.IC_encoder_decoder_enabled = IC_encoder_decoder
#         self.regular_enabled = regular
#         self.basis_function_modes = self.trunk_modes  
#         self.out_size_decoder = self.basis_function_modes
#         self.in_size_encoder = self.control_dim = self.basis_function_modes
#         self.control_rnn_depth = control_rnn_depth

#         self.u_rnn = torch.nn.LSTM(
#             input_size=1 + self.control_dim,
#             hidden_size=control_rnn_size,
#             batch_first=True,
#             num_layers=control_rnn_depth,
#             dropout=0,
#         )

#         ### Enforce IC ###
#         if self.IC_encoder_decoder_enabled:
#             assert control_rnn_size > self.trunk_modes, "Control RNN size must be greater than trunk modes"	
#             x_dnn_osz = control_rnn_depth * (control_rnn_size-self.trunk_modes)
#         else: 
#             x_dnn_osz = control_rnn_depth * control_rnn_size
            
#             ### Flow decoder (MLP) ###
#             self.u_dnn = FFNet(in_size=control_rnn_size,
#                             out_size=self.out_size_decoder,
#                             hidden_size=decoder_depth *
#                             (decoder_size * control_rnn_size, ),
#                             use_batch_norm=use_batch_norm)
        
#         ### Flow encoder (CNN) ###
#         if self.conv_encoder_enabled:
#             self.x_dnn = CNN_encoder(in_size=self.in_size_encoder,
#                            out_size=x_dnn_osz,use_batch_norm=use_batch_norm)
            
#         ### Flow encoder (MLP) ###
#         else:
#             self.x_dnn = FFNet(in_size=self.in_size_encoder,
#                            out_size=x_dnn_osz,
#                            hidden_size=encoder_depth *
#                            (encoder_size * x_dnn_osz, ), 
#                            use_batch_norm=use_batch_norm)

#         ### Trunk (MLP) ###
#         self.trunk_svd = trunk_model # Trained on SVD
#         if trunk_modes > state_dim:
#             self.trunk_extra = TrunkNet(in_size=256,out_size=self.trunk_modes-self.state_dim,hidden_size=self.trunk_size_extra,use_batch_norm=False,dropout_prob=0.1)
#         else: 
#             self.trunk_extra = None

#         ### Nonlinear decoder (MLP) ###
#         self.output_NN = FFNet(in_size=trunk_modes,out_size = 1,hidden_size=self.NL_size,use_batch_norm=use_batch_norm)

#     def forward(self, x, rnn_input,locations, deltas):

#         ### Projection ###
#         if self.regular_enabled == False:
#             unpadded_u, unpacked_lengths = pad_packed_sequence(rnn_input, batch_first=True)     # unpack input
#             u = unpadded_u[:, :, :-1]                                                           # extract inputs values
#             if self.trunk_extra is not None:                 
#                 trunk_output_svd = self.trunk_svd(locations.view(-1, 1)) 
#                 trunk_output_extra = self.trunk_extra(locations.view(-1, 1))
#                 trunk_output = torch.cat([trunk_output_svd, trunk_output_extra], dim=1)  
#             else:
#                 trunk_output = self.trunk_svd(locations.view(-1, 1)) 
#             x = torch.einsum("ni,bn->bi",trunk_output[:, :self.basis_function_modes],x) # a(0)
#             u = torch.einsum('ni,btn->bti',trunk_output[:, :self.basis_function_modes],u) # projected inputs
#             u_deltas = torch.cat((u, deltas), dim=-1)          
#             rnn_input = pack_padded_sequence(u_deltas, unpacked_lengths, batch_first=True)      # repack RNN input

        
#         ### Flow encoder ###
#         h0 = self.x_dnn(x)
#         if self.IC_encoder_decoder_enabled:
#             x_repeated = x.repeat(1,self.control_rnn_depth)
#             x_chunks = torch.chunk(x_repeated,chunks=self.control_rnn_depth, dim=1)
#             enc_chunks = torch.chunk(h0, chunks=self.control_rnn_depth, dim=1)
#             h0 = [torch.cat([b_part, a_part], dim=1) for b_part, a_part in zip(x_chunks, enc_chunks)]
#             h0 = torch.stack(h0)
#         else:
#             h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
#         c0 = torch.zeros_like(h0)

#         ### Flow RNN ###
#         rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0,c0))
#         h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
#                                                            batch_first=True)
#         h_shift = torch.roll(h, shifts=1, dims=1)
#         h_shift[:, 0, :] = h0[-1]
#         encoded_controls = (1 - deltas) * h_shift + deltas * h

#         ### Flow decoder ###
#         if self.IC_encoder_decoder_enabled:
#             output_flow = encoded_controls[range(encoded_controls.shape[0]),
#                                              h_lens - 1, :][:,:self.trunk_modes]
                     
#         else:
#             output_flow = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
#                                              h_lens - 1, :])
        
#         ### Nonlinearity ###
#         if self.nonlinear_enabled:
#             output = torch.einsum("ni,bi->bni",trunk_output[:, :self.basis_function_modes],output_flow)
#             batch_size = output.shape[0]
#             output = self.output_NN(output)
#             output = output.view(batch_size,-1)
            
#         ### Inner product ###
#         else: 
#             print(output_flow.shape)
#             output = torch.einsum("ni,bi->bn",trunk_output[:, :self.basis_function_modes],output_flow)
#         return output, trunk_output

class RHYME_XT(nn.Module):

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
                 use_nonlinear,
                 IC_encoder_decoder,
                 trunk_size_svd,
                 trunk_size_extra,
                 NL_size,
                 trunk_modes_svd,
                 trunk_modes_extra,
                 trunk_model,
                 use_batch_norm):
        super(RHYME_XT, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.control_rnn_size = control_rnn_size
        self.trunk_modes = trunk_modes_svd + trunk_modes_extra
        self.trunk_size_svd = trunk_size_svd
        self.trunk_size_extra = trunk_size_extra
        self.NL_size = NL_size
        self.nonlinear_enabled = use_nonlinear 
        self.IC_encoder_decoder_enabled = IC_encoder_decoder
        self.basis_function_modes = self.trunk_modes  
        self.out_size_decoder = self.basis_function_modes
        self.in_size_encoder = self.control_dim = self.basis_function_modes
        self.control_rnn_depth = control_rnn_depth

        ### Flow function RNN ###
        self.u_rnn = torch.nn.LSTM(
            input_size=1 + self.control_dim,
            hidden_size=control_rnn_size,
            batch_first=True,
            num_layers=control_rnn_depth,
            dropout=0,
        )

        ### Flow function encoder and decoder ###
        if self.IC_encoder_decoder_enabled: # Initial Condition option, encoder is simple slicing operation in this option
            assert control_rnn_size > self.trunk_modes, "Control RNN size must be greater than trunk modes"	
            x_dnn_osz = control_rnn_depth * (control_rnn_size-self.trunk_modes)
        else: 
            x_dnn_osz = control_rnn_depth * control_rnn_size
            
            # Regular decoder (MLP)
            self.u_dnn = FFNet(in_size=control_rnn_size,
                            out_size=self.out_size_decoder,
                            hidden_size=decoder_depth *
                            (decoder_size * control_rnn_size, ),
                            use_batch_norm=use_batch_norm)
           
        # Encoder (MLP)
        self.x_dnn = FFNet(in_size=self.in_size_encoder,
                        out_size=x_dnn_osz,
                        hidden_size=encoder_depth *
                        (encoder_size * x_dnn_osz, ), 
                        use_batch_norm=use_batch_norm)

        ### Trunk (MLP) ###
        self.trunk_svd = trunk_model # Trained on SVD
        if trunk_modes_extra > 0:
            self.trunk_extra = TrunkNet(in_size=256,out_size=trunk_modes_extra,hidden_size=self.trunk_size_extra,use_batch_norm=False,dropout_prob=0.1)
        else: 
            self.trunk_extra = None

        ### Nonlinear decoder (MLP) ###
        self.output_NN = FFNet(in_size=self.trunk_modes,out_size = 1,hidden_size=self.NL_size,use_batch_norm=use_batch_norm)

    def forward(self, u0, f, locations_output, deltas,locations_input=None):

        ### Trunk outputs used to project inputs and outputs ###
        if locations_input is None:
            locations_input = locations_output

        if self.trunk_extra is not None:                 
            trunk_input_svd = self.trunk_svd(locations_input.view(-1, 1)) 
            trunk_input_extra = self.trunk_extra(locations_input.view(-1, 1))
            trunk_input = torch.cat([trunk_input_svd, trunk_input_extra], dim=1)  

            trunk_output_svd = self.trunk_svd(locations_output.view(-1, 1)) 
            trunk_output_extra = self.trunk_extra(locations_output.view(-1, 1))
            trunk_output = torch.cat([trunk_output_svd, trunk_output_extra], dim=1)  
        else:
            trunk_input = self.trunk_svd(locations_input.view(-1, 1)) 
            trunk_output = self.trunk_svd(locations_output.view(-1, 1))

        ### Project initial condition ###
        a0 = torch.einsum("ki,bk->bi",trunk_input[:, :self.basis_function_modes],u0) # u0 -> a0
        
        ### Pass projected initial condition through Encoder ###
        h0 = self.x_dnn(a0)
        if self.IC_encoder_decoder_enabled: # 
            a0_repeated = a0.repeat(1,self.control_rnn_depth)
            a0_chunks = torch.chunk(a0_repeated,chunks=self.control_rnn_depth, dim=1)
            a0_enc_chunks = torch.chunk(h0, chunks=self.control_rnn_depth, dim=1)
            h0 = [torch.cat([b_part, a_part], dim=1) for b_part, a_part in zip(a0_chunks, a0_enc_chunks)]
            h0 = torch.stack(h0)
        else:
            h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        ### Project forcing function ###
        unpadded_f, unpacked_lengths = pad_packed_sequence(f, batch_first=True)     # unpack input
        unpadded_f_without_deltas = unpadded_f[:, :, :-1] 
        I = torch.einsum('ki,btk->bti',trunk_input[:, :self.basis_function_modes],unpadded_f_without_deltas) # projection
        I_deltas = torch.cat((I, deltas), dim=-1)          
        rnn_input = pack_padded_sequence(I_deltas, unpacked_lengths, batch_first=True)      # repack input

        ### Pass projected forcing function through RNN ###
        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0,c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                           batch_first=True)
        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]
        encoded_controls = (1 - deltas) * h_shift + deltas * h

        ### Pass output of RNN through Decoder ###
        if self.IC_encoder_decoder_enabled:
            output_flow = encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :][:,:self.trunk_modes]
                     
        else:
            output_flow = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :])
        
        ### Calculate final output ###
        if self.nonlinear_enabled:  # Nonlinear layer
            output = torch.einsum("ni,bi->bni",trunk_output[:, :self.basis_function_modes],output_flow)
            batch_size = output.shape[0]
            output = self.output_NN(output)
            output = output.view(batch_size,-1)
            return output, trunk_output

        else:                       # Inner product layer
            output = torch.einsum("ni,bi->bn",trunk_output[:, :self.basis_function_modes],output_flow)
            return output, trunk_output 

### MLP ###
class FFNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 use_batch_norm,
                 activation=nn.Tanh):
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
            self.layers.append(nn.Dropout(0.1))  # Dropout layer

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
                 use_batch_norm,
                 dropout_prob=0.0,
                 activation=nn.Tanh):
        super(TrunkNet, self).__init__()

        ### Fourier feature mapping to generate more features ###
        B = torch.normal(mean=0.0, std=0.5, size=(128, 1))
        self.register_buffer("B", B)  

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))
        self.layers.append(activation())
        
        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        if dropout_prob > 0:
            self.layers.append(nn.Dropout(dropout_prob))

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))
            self.layers.append(activation())
            # self.layers.append(nn.Dropout(0.01))  # Dropout layer

            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))
            if dropout_prob > 0:
                self.layers.append(nn.Dropout(dropout_prob))


        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        input_proj = 2 * torch.pi * input @ self.B.T
        input = torch.cat([torch.sin(input_proj), torch.cos(input_proj)], dim=-1)
        for layer in self.layers:
            input = layer(input)
        return input



