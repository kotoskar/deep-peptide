'''
The CRF state space models. Many parameters are hardcoded due to the complexity of the CRF configuration.
'''
import torch
import torch.nn as nn
from .multi_tag_crf import CRF
from .lstm_cnn import LSTMCNN

class CustomTransformerWrapper(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        from transformers import AutoModelForMaskedLM
        
        transformer = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True)
        self.input_dim = input_dim
        self.hidden_dim = 960
        self.encoder = transformer.transformer.blocks

        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, output_dim) if output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] - input embeddings
        Returns:
            [batch_size, seq_len, output_dim or hidden_dim]
        """
        x = self.input_proj(x)  # Project to 960
        for block in self.encoder:
            x, _ = block(x)
        x = self.output_proj(x)
        return x


class CRFBaseModel(nn.Module):
    '''Extend this model by defining a feature_extractor.'''
    def __init__(
        self,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61 # total number of states in the state space model
        ) -> None:


        super().__init__()
        self.max_len = 50
        self.min_len = 5
        self.feature_extractor = None
        self.features_to_emissions = nn.Linear(64, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(self.max_len, self.min_len, n_branches=2 if num_labels==3 else 1)
        self.allowed_transitions = allowed_transitions
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)

    @staticmethod
    def get_crf_constraints(max_len: int = 60, min_len: int = 5, n_branches: int = 1):
        '''Build the peptide state space model.
        Each peptide starts as state 1 and goes through 2, 3.
        From 3, it can either go to 4 or skip ahead to any other state up to 59.
        From 59, go to 60. 
        This enforces a minimum peptide length of 5. Each peptide is forced to end in 60,
        so this state can learn peptide end properties.
        '''
        allowed_starts = [0,1]
        allowed_ends = [0, max_len]

        allowed_state_transitions = []
        allowed_state_transitions.append((0,0)) # None to None
        allowed_state_transitions.append((0,1)) # None to Peptide_0
        allowed_state_transitions.append((max_len,1)) # Peptide_50 to Peptide_0, no need to have 1 AA gap
        allowed_state_transitions.append((max_len,0)) # Peptide_50 (peptide end position) to None

        for i in range(1, max_len): 
            to_next = (i, i+1)
            allowed_state_transitions.append(to_next)

            if i >min_len-1: #make skip forward connections
                skip_to_i = (min_len-2,i) #3
                allowed_state_transitions.append(skip_to_i) 

        allowed_state_transitions.append((max_len-1,max_len)) # peptide end position -1 to peptide end position
        # logic of this state space model is that the end state is the same for all peptides, regardless their length.

        # branch 1 + no state: 0-50
        # branch 2: 51-101
        if n_branches == 2:
            start = 1 + max_len
            end = 2*max_len

            allowed_starts.append(start)
            allowed_ends.append(end)
            allowed_state_transitions.append((0,start))
            allowed_state_transitions.append((end,start)) 
            allowed_state_transitions.append((end,0))

            # can go directly from end of peptide to start of propeptide and vice versa.
            allowed_state_transitions.append((end,1))
            allowed_state_transitions.append((start-1, start))

            for i in range(start, end): 
                to_next = (i, i+1)
                allowed_state_transitions.append(to_next)

                if i >min_len-1: #make skip forward connections
                    skip_to_i = (start+min_len-3, i)#((min_len-2,i))
                    allowed_state_transitions.append(skip_to_i) 

        return allowed_state_transitions, allowed_starts, allowed_ends

    def _debug_crf(self, targets):
        '''Check label sequences for incompatibilities with the defined state grammar.'''
        for i in range(targets.shape[0]):

            for j in range(1, targets.shape[1]):
                l = int(targets[i,j].item())
                l_prev = int(targets[i,j-1].item())

                if (l_prev, l) not in self.allowed_transitions:
                    print(f'Found invalid transition from {l_prev} to {l}.')


    
    def _repeat_emissions(self, emissions):
        '''Turn a (batch_size, seq_len, 2) tensor into (batch_size, seq_len, num_states) by repeating the emissions at position 1.'''

        if emissions.shape[-1] == 2:
            emissions_out = torch.zeros(emissions.shape[0], emissions.shape[1], self.num_states, dtype=emissions.dtype, device=emissions.device)    
            emissions_out[:,:,0] = emissions[:,:,0]
            emissions_out[:,:, 1:(self.max_len+1)] = emissions[:,:,1].unsqueeze(-1)
        elif emissions.shape[-1] == 3:
            emissions_out = torch.zeros(emissions.shape[0], emissions.shape[1], self.num_states, dtype=emissions.dtype, device=emissions.device)
            emissions_out[:,:,0] = emissions[:,:,0]
            emissions_out[:,:, 1:] = emissions[:,:,1].unsqueeze(-1)
            emissions_out[:,:, (self.max_len+1):] = emissions[:,:,2].unsqueeze(-1)
        else:
            raise NotImplementedError()
        
        return emissions_out


    def forward(
        self,
        embeddings,
        mask,
        targets=None,
        skip_marginals: bool = False,
        top_k: int = 1,
        decode: bool = True,          # <-- НОВОЕ: можно выключить decode в train
        return_probs: bool = True,    # <-- НОВОЕ: можно не считать probs вообще
    ):
        # embeddings: [B, C, L] (как у тебя в LSTMCNN)
        # mask:       [B, L] (0/1)
        mask = mask.bool()

        if isinstance(self.feature_extractor, CustomTransformerWrapper):
            # предполагаем, что wrapper ждёт [B, L, C]
            feats_in = embeddings.transpose(1, 2)  # [B, L, C]
            features = self.feature_extractor(feats_in)  # [B, L, F]
        else:
            # LSTMCNN ждёт [B, C, L]
            features = self.feature_extractor(embeddings, mask)  # [B, L, F]

        emissions = self.features_to_emissions(features)        # [B, L, num_labels]
        emissions = self._repeat_emissions(emissions)           # [B, L, num_states]

        loss = None
        if targets is not None:
            # targets ожидаются [B, L] long
            targets = targets.long()
            loss = self.crf(emissions=emissions, tags=targets, mask=mask, reduction="mean") * -1
            if loss.item() > 10000:
                self._debug_crf(targets)

        viterbi_paths = None
        path_probs = None
        if decode:
            viterbi_paths, path_probs = self.crf.decode(emissions=emissions, mask=mask, top_k=top_k)

        probs = None
        if return_probs:
            # Если skip_marginals=True, используем softmax как суррогат (как у тебя было)
            probs = (
                self.crf.compute_marginal_probabilities(emissions, mask)
                if not skip_marginals
                else torch.softmax(emissions, dim=-1)
            )

        if targets is not None:
            return probs, viterbi_paths, loss
        else:
            return probs, viterbi_paths, path_probs


    @staticmethod
    def _esm_embed(sequence:str, device: torch.device, repr_layers: int=33) -> torch.Tensor:


        from esm import pretrained
        esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
        batch_converter = esm_alphabet.get_batch_converter()
        esm_model.to(device)


        data = [
            ("protein1", sequence),
        ]
        labels, strs, toks = batch_converter(data)

        repr_layers_list = [
            (i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in range(repr_layers)
        ]

        out = None

        toks = toks.to(device)

        minibatch_max_length = toks.size(1)

        tokens_list = []
        end = 0
        while end <= minibatch_max_length:
            start = end
            end = start + 1022
            if end <= minibatch_max_length:
                # we are not on the last one, so make this shorter
                end = end - 300
            tokens = esm_model(toks[:, start:end], repr_layers=repr_layers_list, return_contacts=False)["representations"][repr_layers - 1]
            tokens_list.append(tokens)

        out = torch.cat(tokens_list, dim=1).cpu()

        # set nan to zeros
        out[out!=out] = 0.0

        res = out.transpose(0,1)[1:-1] 
        seq_embedding = res[:,0]

        return seq_embedding

    def predict_from_sequence(self, sequence: str, top_k: int = 5):
        self.eval()
        with torch.no_grad():
            device =  next(self.parameters()).device
            embedding = self._esm_embed(sequence, device)
            embedding = torch.unsqueeze(embedding.permute(1,0), 0)
            mask = torch.unsqueeze(torch.ones(embedding.shape[2]),0)
        

            pos_probs, pos_preds, path_probs = self(embedding, mask, top_k=top_k)

            return pos_probs.squeeze(), pos_preds[0], path_probs[0]

    @staticmethod
    def _make_tag_bitmap(length, start, end, start_state=1, min_len=5, max_len=50):
        '''Make a multi-tag bitmap for the given peptide positions where all other positions are flexible.'''
        with torch.no_grad():
            label = torch.zeros((length, max_len*2+1))

            peptide_length = end-start +1 # inclusive.
            peptide_label = torch.concat(
                [ 
                torch.arange(start_state, start_state+min_len-2),#np.arange(1, 4), # from start to first position with skip connections
                # (end_state -1) - (peptide_length - min_len)
                torch.arange((start_state+max_len-2 - (peptide_length - min_len)), start_state+max_len) #np.arange( 59-(peptide_length-5) ,61) 
                ]
            )

            # set the positions in the matrix to true.
            label[torch.arange(start-1,end), peptide_label] = 1
            label[:start-1,:] = 1
            label[end:, :] = 1

        return label

    def predict_peptide_probability(self, sequence:str, start: int, stop: int):
        '''Computes probability of a peptide given all possible paths.'''
        self.eval()
        with torch.no_grad():
            device =  next(self.parameters()).device
            embedding = self._esm_embed(sequence, device)
            embedding = torch.unsqueeze(embedding.permute(1,0), 0)
            mask = torch.unsqueeze(torch.ones(embedding.shape[2]),0)
        
            features = self.feature_extractor(embedding, mask) # (batch_size, seq_len, feature_dim)
            emissions = self.features_to_emissions(features) # (batch_size, seq_len, num_labels)
            emissions = self._repeat_emissions(emissions) # (batch_size, seq_len, num_states)

            targets = self._make_tag_bitmap(len(sequence), start, stop, start_state=1)
            targets = torch.unsqueeze(targets,0)
            llh_pep= self.crf(emissions = emissions, tag_bitmap=targets.long(), mask = mask.bool(), reduction='none')
            
            targets = self._make_tag_bitmap(len(sequence), start, stop, start_state=51)
            targets = torch.unsqueeze(targets,0)
            llh_pro= self.crf(emissions = emissions, tag_bitmap=targets.long(), mask = mask.bool(), reduction='none')

            return torch.exp(llh_pep[0]).item(), torch.exp(llh_pro[0]).item()


class EmbeddingProjector(nn.Module):
    """Small trainable adapter that projects precomputed residue embeddings before the feature extractor.

    Expects embeddings shaped [B, C, L] (channel-first), and returns [B, C_proj, L].
    """
    def __init__(self, input_size: int, proj_size: int = 512, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_size, proj_size)
        self.act = nn.GELU()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # [B, C, L] -> [B, L, C]
        x = embeddings.transpose(1, 2)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = self.act(x)
        x = self.dropout(x)
        # [B, L, C_proj] -> [B, C_proj, L]
        return x.transpose(1, 2)


class MultiScaleConv1DBlock(nn.Module):
    """Residual multi-scale Conv1d block for channel-first residue features [B, C, L]."""

    def __init__(
        self,
        channels: int,
        kernels=(3, 7, 15),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2)
            for k in self.kernels
        ])
        self.merge = nn.Conv1d(channels * len(self.kernels), channels, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        outs = [self.act(conv(x)) for conv in self.branches]
        y = torch.cat(outs, dim=1)     # [B, C*num_scales, L]
        y = self.merge(y)              # [B, C, L]
        y = self.act(y)
        y = self.dropout(y)
        y = y + x                      # residual
        y = self.out_ln(y.transpose(1, 2)).transpose(1, 2)
        return y


class MultiScaleProjectedLSTMCNN(nn.Module):
    """EmbeddingProjector -> MultiScaleConv1DBlock -> existing LSTMCNN backbone."""
    def __init__(
        self,
        input_size: int,
        proj_size: int = 256,
        dropout_projector: float = 0.2,
        multiscale_kernels=(3, 7, 15),
        multiscale_dropout: float = 0.1,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        self.projector = EmbeddingProjector(
            input_size=input_size,
            proj_size=proj_size,
            dropout=dropout_projector,
        )
        self.multiscale = MultiScaleConv1DBlock(
            channels=proj_size,
            kernels=multiscale_kernels,
            dropout=multiscale_dropout,
        )
        self.backbone = LSTMCNN(
            input_size=proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        x = self.multiscale(x)
        return self.backbone(x, mask)


class LSTMCNNCRFProjectorMultiScale(CRFBaseModel):
    """LSTM-CNN + CRF with trainable projector and multi-scale Conv1d front-end."""
    def __init__(
        self,
        input_size: int = 1280,
        proj_size: int = 256,
        dropout_projector: float = 0.4,
        multiscale_kernels=(3, 7, 15),
        multiscale_dropout: float = 0.1,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = MultiScaleProjectedLSTMCNN(
            input_size=input_size,
            proj_size=proj_size,
            dropout_projector=dropout_projector,
            multiscale_kernels=multiscale_kernels,
            multiscale_dropout=multiscale_dropout,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class GatedResidualConvMultiScaleProjectedLSTMCNN(nn.Module):
    """GatedResidualConvSplitProjector -> MultiScaleConv1DBlock -> existing LSTMCNN backbone."""
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.2,
        multiscale_kernels=(3, 7, 15),
        multiscale_dropout: float = 0.1,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ):
        super().__init__()
        if input_size != seq_input_size + struct_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + struct_input_size ({seq_input_size + struct_input_size})"
            )

        self.projector = GatedResidualConvSplitProjector(
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )
        self.multiscale = MultiScaleConv1DBlock(
            channels=seq_proj_size,
            kernels=multiscale_kernels,
            dropout=multiscale_dropout,
        )
        self.backbone = LSTMCNN(
            input_size=seq_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        x = self.multiscale(x)
        return self.backbone(x, mask)


class LSTMCNNCRFGated3DiResidualConvMultiScale(CRFBaseModel):
    """LSTM-CNN + CRF with ESM2 main branch, gated 3Di conv residual, and multi-scale Conv1d on the fused main branch."""
    def __init__(
        self,
        input_size: int = 1300,
        seq_input_size: int = 1280,
        struct_input_size: int = 20,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.4,
        multiscale_kernels=(3, 7, 15),
        multiscale_dropout: float = 0.1,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = GatedResidualConvMultiScaleProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            multiscale_kernels=multiscale_kernels,
            multiscale_dropout=multiscale_dropout,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class TriBranchResidualProjector(nn.Module):
    """
    ESM2 main branch + residue10 residual + optional 3Di residual with Conv1d.

    Input layout:
      [seq | residue | struct]
    where:
      seq     = ESM2
      residue = additional residue-level params
      struct  = optional 3Di channels

    Shapes:
      embeddings: [B, C_total, L]
      output:     [B, seq_proj_size, L]
    """
    def __init__(
        self,
        seq_input_size: int,
        residue_input_size: int,
        struct_input_size: int = 0,
        seq_proj_size: int = 256,
        residue_proj_size: int = 16,
        struct_proj_size: int = 16,
        dropout: float = 0.2,
        residue_residual_scale: float = 0.05,
        struct_residual_scale: float = 0.10,
        residue_branch_dropout: float = 0.2,
        struct_branch_dropout: float = 0.3,
        residue_gate_bias: float = -2.5,
        struct_gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.residue_input_size = residue_input_size
        self.struct_input_size = struct_input_size

        self.residue_residual_scale = residue_residual_scale
        self.struct_residual_scale = struct_residual_scale
        self.residue_branch_dropout = residue_branch_dropout
        self.struct_branch_dropout = struct_branch_dropout

        self.seq_projector = EmbeddingProjector(
            input_size=seq_input_size,
            proj_size=seq_proj_size,
            dropout=dropout,
        )
        self.residue_projector = EmbeddingProjector(
            input_size=residue_input_size,
            proj_size=residue_proj_size,
            dropout=dropout,
        )

        self.residue_to_seq = nn.Conv1d(residue_proj_size, seq_proj_size, kernel_size=1)
        self.residue_gate_ln = nn.LayerNorm(seq_proj_size + residue_proj_size)
        self.residue_gate_dropout = nn.Dropout(dropout)
        self.residue_gate = nn.Linear(seq_proj_size + residue_proj_size, seq_proj_size)

        self.has_struct = struct_input_size > 0
        if self.has_struct:
            pad = struct_conv_kernel // 2
            self.struct_projector = EmbeddingProjector(
                input_size=struct_input_size,
                proj_size=struct_proj_size,
                dropout=dropout,
            )
            self.struct_conv = nn.Conv1d(
                struct_proj_size,
                struct_proj_size,
                kernel_size=struct_conv_kernel,
                padding=pad,
            )
            self.struct_conv_act = nn.GELU()
            self.struct_conv_dropout = nn.Dropout(dropout)

            self.struct_to_seq = nn.Conv1d(struct_proj_size, seq_proj_size, kernel_size=1)
            self.struct_gate_ln = nn.LayerNorm(seq_proj_size + struct_proj_size)
            self.struct_gate_dropout = nn.Dropout(dropout)
            self.struct_gate = nn.Linear(seq_proj_size + struct_proj_size, seq_proj_size)

            nn.init.zeros_(self.struct_gate.weight)
            nn.init.constant_(self.struct_gate.bias, struct_gate_bias)

        self.out_ln = nn.LayerNorm(seq_proj_size)

        nn.init.zeros_(self.residue_gate.weight)
        nn.init.constant_(self.residue_gate.bias, residue_gate_bias)

    def _drop_branch(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if self.training and p > 0:
            keep = (torch.rand(x.size(0), 1, 1, device=x.device) >= p).to(x.dtype)
            x = x * keep
        return x

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected = self.seq_input_size + self.residue_input_size + self.struct_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected concatenated embeddings with C={expected} "
                f"(= {self.seq_input_size}+{self.residue_input_size}+{self.struct_input_size}), "
                f"got {embeddings.size(1)}"
            )

        s0 = 0
        s1 = self.seq_input_size
        s2 = s1 + self.residue_input_size
        s3 = s2 + self.struct_input_size

        seq = embeddings[:, s0:s1, :]
        residue = embeddings[:, s1:s2, :]
        struct = embeddings[:, s2:s3, :] if self.has_struct else None

        seq = self.seq_projector(seq)              # [B, D_seq, L]
        residue = self.residue_projector(residue)  # [B, D_res, L]
        residue = self._drop_branch(residue, self.residue_branch_dropout)

        seq_t = seq.transpose(1, 2)                                # [B, L, D_seq]
        residue_t = residue.transpose(1, 2)                        # [B, L, D_res]
        residue_up_t = self.residue_to_seq(residue).transpose(1, 2)  # [B, L, D_seq]

        residue_gate_in = torch.cat([seq_t, residue_t], dim=-1)
        residue_gate = torch.sigmoid(
            self.residue_gate(
                self.residue_gate_dropout(
                    self.residue_gate_ln(residue_gate_in)
                )
            )
        )

        fused_t = seq_t + self.residue_residual_scale * residue_gate * residue_up_t

        if self.has_struct:
            struct = self.struct_projector(struct)     # [B, D_str, L]
            struct = self.struct_conv(struct)
            struct = self.struct_conv_act(struct)
            struct = self.struct_conv_dropout(struct)
            struct = self._drop_branch(struct, self.struct_branch_dropout)

            struct_t = struct.transpose(1, 2)                          # [B, L, D_str]
            struct_up_t = self.struct_to_seq(struct).transpose(1, 2)  # [B, L, D_seq]

            struct_gate_in = torch.cat([fused_t, struct_t], dim=-1)
            struct_gate = torch.sigmoid(
                self.struct_gate(
                    self.struct_gate_dropout(
                        self.struct_gate_ln(struct_gate_in)
                    )
                )
            )

            fused_t = fused_t + self.struct_residual_scale * struct_gate * struct_up_t

        fused_t = self.out_ln(fused_t)
        return fused_t.transpose(1, 2)  # [B, D_seq, L]


class TriBranchProjectedLSTMCNN(nn.Module):
    """TriBranchResidualProjector + existing LSTMCNN backbone."""
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        residue_input_size: int,
        struct_input_size: int = 0,
        seq_proj_size: int = 256,
        residue_proj_size: int = 16,
        struct_proj_size: int = 16,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        residue_residual_scale: float = 0.05,
        struct_residual_scale: float = 0.10,
        residue_branch_dropout: float = 0.2,
        struct_branch_dropout: float = 0.3,
        residue_gate_bias: float = -2.5,
        struct_gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ):
        super().__init__()
        expected = seq_input_size + residue_input_size + struct_input_size
        if input_size != expected:
            raise ValueError(
                f"input_size ({input_size}) must equal "
                f"seq_input_size + residue_input_size + struct_input_size ({expected})"
            )

        self.projector = TriBranchResidualProjector(
            seq_input_size=seq_input_size,
            residue_input_size=residue_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            residue_proj_size=residue_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
            residue_residual_scale=residue_residual_scale,
            struct_residual_scale=struct_residual_scale,
            residue_branch_dropout=residue_branch_dropout,
            struct_branch_dropout=struct_branch_dropout,
            residue_gate_bias=residue_gate_bias,
            struct_gate_bias=struct_gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )

        self.backbone = LSTMCNN(
            input_size=seq_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)


class LSTMCNNCRFTriBranchResidual(CRFBaseModel):
    """
    LSTM-CNN + CRF with:
      - ESM2 main branch
      - residue10 gated residual branch
      - optional 3Di gated residual branch with Conv1d
    """
    def __init__(
        self,
        input_size: int = 1290,
        seq_input_size: int = 1280,
        residue_input_size: int = 10,
        struct_input_size: int = 0,
        seq_proj_size: int = 256,
        residue_proj_size: int = 16,
        struct_proj_size: int = 16,
        dropout_projector: float = 0.4,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        residue_residual_scale: float = 0.05,
        struct_residual_scale: float = 0.10,
        residue_branch_dropout: float = 0.2,
        struct_branch_dropout: float = 0.3,
        residue_gate_bias: float = -2.5,
        struct_gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = TriBranchProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            residue_input_size=residue_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            residue_proj_size=residue_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            residue_residual_scale=residue_residual_scale,
            struct_residual_scale=struct_residual_scale,
            residue_branch_dropout=residue_branch_dropout,
            struct_branch_dropout=struct_branch_dropout,
            residue_gate_bias=residue_gate_bias,
            struct_gate_bias=struct_gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class ProjectedLSTMCNN(nn.Module):
    """EmbeddingProjector + existing LSTMCNN feature extractor.

    Keeps the same forward signature as LSTMCNN: (embeddings [B,C,L], mask [B,L]) -> features [B,L,F].
    """
    def __init__(
        self,
        input_size: int,
        proj_size: int = 512,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        self.projector = EmbeddingProjector(input_size=input_size, proj_size=proj_size, dropout=dropout_projector)
        self.backbone = LSTMCNN(
            input_size=proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        # Expose biLSTM so existing training code can still call flatten_parameters()
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)



class SplitEmbeddingProjector(nn.Module):
    """Project sequence and structural channels separately, then concatenate them back.

    Expects concatenated embeddings shaped [B, C_total, L] where
    C_total = seq_input_size + struct_input_size and the layout is [seq | struct].
    """
    def __init__(
        self,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.struct_input_size = struct_input_size
        self.seq_projector = EmbeddingProjector(
            input_size=seq_input_size,
            proj_size=seq_proj_size,
            dropout=dropout,
        )
        self.struct_projector = EmbeddingProjector(
            input_size=struct_input_size,
            proj_size=struct_proj_size,
            dropout=dropout,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected = self.seq_input_size + self.struct_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected concatenated embeddings with C={expected} (= {self.seq_input_size}+{self.struct_input_size}), got {embeddings.size(1)}"
            )

        seq = embeddings[:, : self.seq_input_size, :]
        struct = embeddings[:, self.seq_input_size : expected, :]

        seq = self.seq_projector(seq)
        struct = self.struct_projector(struct)
        return torch.cat([seq, struct], dim=1)


class SplitProjectedLSTMCNN(nn.Module):
    """SplitEmbeddingProjector + existing LSTMCNN feature extractor.

    Use this when you store one concatenated per-residue tensor [ESM | 3Di-onehot]
    and want to keep the dataset unchanged.
    """
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 64,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        if input_size != seq_input_size + struct_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + struct_input_size ({seq_input_size + struct_input_size})"
            )

        self.projector = SplitEmbeddingProjector(
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
        )
        self.backbone = LSTMCNN(
            input_size=seq_proj_size + struct_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)


class LSTMCNNCRFSplitProjector(CRFBaseModel):
    """LSTM-CNN + CRF with separate trainable projectors for sequence and structural channels.

    Input layout must be a single concatenated tensor [seq | struct] with shape [B, C, L].
    This keeps the dataset unchanged while allowing different compression for each branch.
    """
    def __init__(
        self,
        input_size: int = 1300,
        seq_input_size: int = 1280,
        struct_input_size: int = 20,
        seq_proj_size: int = 256,
        struct_proj_size: int = 64,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = SplitProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class GatedResidualSplitProjector(nn.Module):
    """Project sequence and structural channels separately, then add a gated structural residual to the sequence branch.

    Input layout is one concatenated tensor [seq | struct] with shape [B, C_total, L].
    Output shape is [B, seq_proj_size, L], so ESM2 remains the main branch and
    3Di only injects a controlled residual signal.
    """
    def __init__(
        self,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout: float = 0.2,
        residual_scale: float = 0.2,
        struct_branch_dropout: float = 0.0,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.struct_input_size = struct_input_size
        self.residual_scale = residual_scale
        self.struct_branch_dropout = struct_branch_dropout

        self.seq_projector = EmbeddingProjector(
            input_size=seq_input_size,
            proj_size=seq_proj_size,
            dropout=dropout,
        )
        self.struct_projector = EmbeddingProjector(
            input_size=struct_input_size,
            proj_size=struct_proj_size,
            dropout=dropout,
        )

        self.struct_to_seq = nn.Conv1d(struct_proj_size, seq_proj_size, kernel_size=1)
        self.gate_ln = nn.LayerNorm(seq_proj_size + struct_proj_size)
        self.gate_dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(seq_proj_size + struct_proj_size, seq_proj_size)
        self.out_ln = nn.LayerNorm(seq_proj_size)

        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected = self.seq_input_size + self.struct_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected concatenated embeddings with C={expected} (= {self.seq_input_size}+{self.struct_input_size}), got {embeddings.size(1)}"
            )

        seq = embeddings[:, : self.seq_input_size, :]
        struct = embeddings[:, self.seq_input_size : expected, :]

        seq = self.seq_projector(seq)
        struct = self.struct_projector(struct)

        if self.training and self.struct_branch_dropout > 0:
            keep = (torch.rand(struct.size(0), 1, 1, device=struct.device) >= self.struct_branch_dropout).to(struct.dtype)
            struct = struct * keep

        seq_t = seq.transpose(1, 2)
        struct_t = struct.transpose(1, 2)
        struct_up_t = self.struct_to_seq(struct).transpose(1, 2)

        gate_in = torch.cat([seq_t, struct_t], dim=-1)
        gate = torch.sigmoid(self.gate(self.gate_dropout(self.gate_ln(gate_in))))

        fused = self.out_ln(seq_t + self.residual_scale * gate * struct_up_t)
        return fused.transpose(1, 2)


class GatedResidualConvSplitProjector(nn.Module):
    """Project ESM2 and 3Di separately, run a small Conv1d over the structural branch,
    then add a gated structural residual to the ESM2 branch.

    Input layout: [B, C_total, L] where C_total = seq_input_size + struct_input_size
    and channels are concatenated as [seq | struct].
    """
    def __init__(
        self,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout: float = 0.2,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.struct_input_size = struct_input_size
        self.residual_scale = residual_scale
        self.struct_branch_dropout = struct_branch_dropout

        self.seq_projector = EmbeddingProjector(
            input_size=seq_input_size,
            proj_size=seq_proj_size,
            dropout=dropout,
        )
        self.struct_projector = EmbeddingProjector(
            input_size=struct_input_size,
            proj_size=struct_proj_size,
            dropout=dropout,
        )

        pad = struct_conv_kernel // 2
        self.struct_conv = nn.Conv1d(
            struct_proj_size,
            struct_proj_size,
            kernel_size=struct_conv_kernel,
            padding=pad,
        )
        self.struct_conv_act = nn.GELU()
        self.struct_conv_dropout = nn.Dropout(dropout)

        self.struct_to_seq = nn.Conv1d(struct_proj_size, seq_proj_size, kernel_size=1)

        self.gate_ln = nn.LayerNorm(seq_proj_size + struct_proj_size)
        self.gate_dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(seq_proj_size + struct_proj_size, seq_proj_size)
        self.out_ln = nn.LayerNorm(seq_proj_size)

        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected = self.seq_input_size + self.struct_input_size
        if embeddings.size(1) != expected:
            raise ValueError(
                f"Expected concatenated embeddings with C={expected} (= {self.seq_input_size}+{self.struct_input_size}), got {embeddings.size(1)}"
            )

        seq = embeddings[:, : self.seq_input_size, :]
        struct = embeddings[:, self.seq_input_size : expected, :]

        seq = self.seq_projector(seq)
        struct = self.struct_projector(struct)

        # local structural patterns before gating
        struct = self.struct_conv(struct)
        struct = self.struct_conv_act(struct)
        struct = self.struct_conv_dropout(struct)

        if self.training and self.struct_branch_dropout > 0:
            keep = (
                torch.rand(struct.size(0), 1, 1, device=struct.device) >= self.struct_branch_dropout
            ).to(struct.dtype)
            struct = struct * keep

        seq_t = seq.transpose(1, 2)                       # [B, L, D_seq]
        struct_t = struct.transpose(1, 2)                # [B, L, D_str]
        struct_up_t = self.struct_to_seq(struct).transpose(1, 2)  # [B, L, D_seq]

        gate_in = torch.cat([seq_t, struct_t], dim=-1)
        gate = torch.sigmoid(self.gate(self.gate_dropout(self.gate_ln(gate_in))))

        fused = self.out_ln(seq_t + self.residual_scale * gate * struct_up_t)
        return fused.transpose(1, 2)                     # [B, D_seq, L]


class GatedResidualConvProjectedLSTMCNN(nn.Module):
    """Conv-enhanced gated residual projector + existing LSTMCNN backbone."""
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ):
        super().__init__()
        if input_size != seq_input_size + struct_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + struct_input_size ({seq_input_size + struct_input_size})"
            )

        self.projector = GatedResidualConvSplitProjector(
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )

        self.backbone = LSTMCNN(
            input_size=seq_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)


class LSTMCNNCRFGated3DiResidualConv(CRFBaseModel):
    """LSTM-CNN + CRF with ESM2 backbone and gated 3Di residual fusion,
    plus a small Conv1d on the structural branch before gating.
    """
    def __init__(
        self,
        input_size: int = 1300,
        seq_input_size: int = 1280,
        struct_input_size: int = 20,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.4,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        residual_scale: float = 0.1,
        struct_branch_dropout: float = 0.3,
        gate_bias: float = -2.5,
        struct_conv_kernel: int = 5,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = GatedResidualConvProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
            struct_conv_kernel=struct_conv_kernel,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )
        

class GatedResidualProjectedLSTMCNN(nn.Module):
    """GatedResidualSplitProjector + existing LSTMCNN feature extractor.

    Keeps ESM2 as the main branch and injects 3Di as a small gated residual before the backbone.
    """
    def __init__(
        self,
        input_size: int,
        seq_input_size: int,
        struct_input_size: int,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        residual_scale: float = 0.2,
        struct_branch_dropout: float = 0.0,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        if input_size != seq_input_size + struct_input_size:
            raise ValueError(
                f"input_size ({input_size}) must equal seq_input_size + struct_input_size ({seq_input_size + struct_input_size})"
            )

        self.projector = GatedResidualSplitProjector(
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout=dropout_projector,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
        )
        self.backbone = LSTMCNN(
            input_size=seq_proj_size,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_conv1=dropout_conv1,
            n_tissues=0,
        )
        self.biLSTM = self.backbone.biLSTM

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projector(embeddings)
        return self.backbone(x, mask)


class LSTMCNNCRFGated3DiResidual(CRFBaseModel):
    """LSTM-CNN + CRF with ESM2 backbone and gated 3Di residual fusion.

    Input layout must be a single concatenated tensor [seq | struct] with shape [B, C, L].
    The model projects ESM2 and 3Di separately, then adds a learned gated structural residual
    to the projected ESM2 branch before the existing LSTMCNN feature extractor.
    """
    def __init__(
        self,
        input_size: int = 1300,
        seq_input_size: int = 1280,
        struct_input_size: int = 20,
        seq_proj_size: int = 256,
        struct_proj_size: int = 32,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
        residual_scale: float = 0.2,
        struct_branch_dropout: float = 0.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = GatedResidualProjectedLSTMCNN(
            input_size=input_size,
            seq_input_size=seq_input_size,
            struct_input_size=struct_input_size,
            seq_proj_size=seq_proj_size,
            struct_proj_size=struct_proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            residual_scale=residual_scale,
            struct_branch_dropout=struct_branch_dropout,
            gate_bias=gate_bias,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )

class LSTMCNNCRF(CRFBaseModel):
    '''LSTM-CNN feature extractor + multistate CRF.'''
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int =3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers : int = 1,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61, # total number of states in the state space model
        feature_extractor = 'LSTMCNN'
        ) -> None:


        super().__init__(num_labels, num_states)

        if feature_extractor == 'LSTMCNN':
            self.feature_extractor = LSTMCNN(input_size=input_size, dropout_input=dropout_input, n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_lstm_layers=1, dropout_conv1=dropout_conv1, n_tissues=0)
        else:
            self.feature_extractor = CustomTransformerWrapper(input_dim=input_size)
        self.features_to_emissions = nn.Linear(n_filters*2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(self.max_len, self.min_len, n_branches=2 if num_labels==3 else 1)
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)


class LSTMCNNCRFProjector(CRFBaseModel):
    """LSTM-CNN feature extractor + multistate CRF, with a small trainable projector on the input embeddings.

    Use this when you feed precomputed embeddings [B, C_in, L] (e.g., ESM2/ESMC/ProstT5),
    but want a lightweight trainable adaptation before the LSTMCNN.
    """
    def __init__(
        self,
        input_size: int = 1280,
        proj_size: int = 512,
        dropout_projector: float = 0.2,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int = 3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_labels: int = 2,
        num_states: int = 61,
    ) -> None:
        super().__init__(num_labels, num_states)

        self.feature_extractor = ProjectedLSTMCNN(
            input_size=input_size,
            proj_size=proj_size,
            dropout_projector=dropout_projector,
            dropout_input=dropout_input,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout_conv1=dropout_conv1,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
        )
        self.features_to_emissions = nn.Linear(n_filters * 2, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(
            self.max_len, self.min_len, n_branches=2 if num_labels == 3 else 1
        )
        self.crf = CRF(
            num_states,
            batch_first=True,
            allowed_transitions=allowed_transitions,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
        )


class SimpleLSTMCNNCRF(CRFBaseModel):
    '''LSTM-CNN feature extractor with simple 2-state CRF model.'''
    def __init__(
        self,
        input_size: int = 1280,
        dropout_input: float = 0.25,
        n_filters: int = 64,
        filter_size: int =3,
        dropout_conv1: float = 0.15,
        hidden_size: int = 128,
        num_lstm_layers : int = 1,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 2 # total number of states in the state space model
        ) -> None:


        super().__init__()

        self.feature_extractor = LSTMCNN(input_size=input_size, dropout_input=dropout_input, n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_lstm_layers=1, dropout_conv1=dropout_conv1, n_tissues=0)
        self.features_to_emissions = nn.Linear(n_filters*2, num_labels)
        self.num_states = num_states
        self.crf = CRF(num_states, batch_first=True) # no constraints on CRF.


    # redefine forward because no emission repeating.
    def forward(self, embeddings, mask, targets = None, skip_marginals: bool = False):
        features = self.feature_extractor(embeddings, mask) # (batch_size, seq_len, feature_dim)
        emissions = self.features_to_emissions(features) # (batch_size, seq_len, num_labels)
        
        viterbi_paths, probs = self.crf.decode(emissions=emissions, mask = mask.bool())

        #pad the viterbi paths
        # max_pad_len = max([len(x) for x in viterbi_paths])
        # pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 
        # pos_preds = torch.tensor(pos_preds, device = emissions.device) #Tensor conversion is just for compatibility with downstream metric functions

        probs = self.crf.compute_marginal_probabilities(emissions, mask.bool()) if not skip_marginals else torch.softmax(emissions, dim=-1)

        if targets is not None:
            loss = self.crf(emissions = emissions, tags=targets.long(), mask = mask.bool(), reduction='mean') *-1
            return (probs, viterbi_paths, loss)
        else:
            return probs, viterbi_paths


class SelfAttentionFeatureNet(nn.Module):

    def __init__(self,
        input_size: float = 1280,
        hidden_size: float = 640,
        dropout_input: float = 0.25,
        n_heads: int = 4,
        attn_dropout: float = 0.1, 
        ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout_input)

        if hidden_size != input_size:
            self.projection = nn.Linear(input_size, hidden_size)
        self.mha = nn.MultiheadAttention(hidden_size,n_heads, attn_dropout, batch_first=True)


    def forward(self, inputs, mask):

        inputs = inputs.transpose(2,1)
        inputs = self.dropout(inputs)
        attn_mask = 1 -mask
        # key_padding_mask – If specified, a mask of shape (N, S)
        # indicating which elements within key to ignore for the purpose of attention (i.e. treat as “padding”). 
        # For unbatched query, shape should be (S)(S). Binary and byte masks are supported. 
        # For a binary mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention. 
        # For a byte mask, a non-zero value indicates that the corresponding key value will be ignored
        #attn_mask = torch.
        if self.hidden_size != self.input_size:
            inputs = self.projection(inputs)

        out, attn = self.mha(inputs, inputs, inputs, key_padding_mask = attn_mask)

        return out


class SelfAttentionCRF(CRFBaseModel):
    '''Attention feature extractor + multistate CRF.'''
    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 128,
        dropout_input: float = 0.25,
        n_heads: int = 4,
        attn_dropout: float = 0.15,
        num_labels: int = 2, #logits (=emissions) to produce by the NN
        num_states = 61 # total number of states in the state space model
        ) -> None:


        super().__init__(num_labels, num_states)

        self.feature_extractor = SelfAttentionFeatureNet(input_size, hidden_size, dropout_input, n_heads, attn_dropout)

        self.features_to_emissions = nn.Linear(hidden_size, num_labels)
        self.num_states = num_states

        allowed_transitions, allowed_start, allowed_end = self.get_crf_constraints(n_branches=2 if num_labels==3 else 1)
        self.crf = CRF(num_states, batch_first=True, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)