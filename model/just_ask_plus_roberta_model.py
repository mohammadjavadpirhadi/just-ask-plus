import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from model.multimodal_transformer import Transformer

class JustAskPlusRoBERTaConfig(PretrainedConfig):

    def __init__(
        self,
        n_layers=3,
        n_heads=12,
        dim=768,
        attended_question_dim=512,
        hidden_dim=4 * 768,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        **kwargs
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.attended_question_dim = attended_question_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation

        super().__init__(**kwargs)


@dataclass
class JustAskPlusRoBERTaOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class JustAskPlusRoBERTaModel(PreTrainedModel):

    config_class = JustAskPlusRoBERTaConfig

    def __init__(self, config: JustAskPlusRoBERTaConfig):
        super().__init__(config)
        
        self.q_t_encoder = Transformer(config=config)
        q_q_a_config = JustAskPlusRoBERTaConfig(
            n_layers=config.n_layers,
            n_heads=8,
            dim=512,
            attended_question_dim=config.attended_question_dim,
            hidden_dim=4 * 512,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
        )
        self.attended_question_transcript_proj = nn.Linear(in_features=config.dim, out_features=config.attended_question_dim)
        self.q_q_a_endcoder = Transformer(config=q_q_a_config)

    def forward(
        self,
        question_features: torch.FloatTensor, 
        attended_question_features: torch.FloatTensor,
        answer_features: torch.FloatTensor,
        transcript_features: torch.FloatTensor,
        transcript_attention_mask: torch.FloatTensor,
        q_q_a_encoder_attention_mask: torch.FloatTensor,
        answer_id: Optional[torch.LongTensor] = None,
    ):
        batch_size, num_sentences, feature_dim = transcript_features.shape
        question_features = question_features.unsqueeze(dim=1)
        q_t_encoder_input = torch.cat((question_features, transcript_features), dim=1)
        q_t_attention_mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    1 # non-zero to avoid being masked
                ).cuda(),
                transcript_attention_mask,
            ],
            1,
        ) # mask sentence length
        attended_question_transcript = self.q_t_encoder(q_t_encoder_input, attn_mask=q_t_attention_mask, return_dict=True).last_hidden_state[:, :1, :]
        attended_question_transcript_projected = self.attended_question_transcript_proj(attended_question_transcript)

        attended_question_features = attended_question_features.unsqueeze(dim=1)
        q_q_a_encoder_input = torch.cat((attended_question_features, answer_features, attended_question_transcript_projected), dim=1)
        final_layer_output = self.q_q_a_endcoder(q_q_a_encoder_input, attn_mask=q_q_a_encoder_attention_mask, return_dict=True).last_hidden_state

        question_video_output = final_layer_output[:, :1, :]
        answer_output = final_layer_output[:, 1:-1, :]
        question_transcript_output = final_layer_output[:, -1:, :]
        
        logits = torch.bmm(question_video_output, torch.transpose(answer_output, -1, -2)).squeeze(1)

        loss = None
        if answer_id is not None:
            ce_loss_fct = nn.CrossEntropyLoss()
            loss = ce_loss_fct(logits, answer_id)

        return JustAskPlusRoBERTaOutput(
            loss=loss,
            logits=logits,
        )
