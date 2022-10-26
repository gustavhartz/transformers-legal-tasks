import transformers as tfs
from torch import nn
import torch
# Transformer model for question answering using a custom q&a head for predicting the answer span


class BaseModel(nn.Module):
    """The baseline model for reuse of code across models 

    """

    def __init__(self,
                 hparams,
                 transformerQA):

        super(BaseModel, self).__init__()
        self.hparams = hparams
        self.model = getattr(transformerQA, hparams.get('model_type'))

        # 768 and 2 from the normal bert config
        self.linearOut = transformerQA.qa_outputs

    def calculate_base_loss(self, start_logits, end_logits, start_positions, end_positions):
        """Take the output start- and end-logits obtained from the linear out layer and calculates the loss from it as done in thesis and CUAD paper

        Args:
            start_logits (_type_): _description_
            end_logits (_type_): _description_
            start_positions (_type_): _description_
            end_positions (_type_): _description_

        Returns:
            float: Loss
        """
        # Calculate the loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # If BCE loss
            if self.hparams.get('loss_type') == 'bce':
                loss_fct = nn.BCEWithLogitsLoss()
                start_positions_bce = torch.zeros(
                    (start_logits.shape[0], start_logits.shape[1])).to(start_logits.device)
                for btc, start_position in enumerate(start_positions):
                    start_positions_bce[btc, start_position] = 1
                end_positions_bce = torch.zeros(
                    (end_logits.shape[0], end_logits.shape[1])).to(end_logits.device)
                for btc, end_position in enumerate(end_positions):
                    end_positions_bce[btc, end_position] = 1

                # Duplicate code to allow for easier experiments
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
            else:
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=ignored_index, label_smoothing=self.hparams.get('label_smoothing', 0.0))
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
        return total_loss

    def get_transformer_outputs(self,
                                input_ids=None,
                                attention_mask=None,
                                token_type_ids=None,
                                position_ids=None,
                                head_mask=None,
                                inputs_embeds=None,
                                output_attentions=None,
                                output_hidden_states=None,
                                return_dict=None):

        if (self.hparams.get('model_type') == 'deberta'):
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return outputs

    def forward(self):
        raise NotImplementedError


class QAModel(BaseModel):
    def __init__(self,
                 hparams,
                 transformerQA):

        super(QAModel, self).__init__(hparams, transformerQA)
        # TODO: How to init this one

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.get_transformer_outputs(input_ids, attention_mask, token_type_ids, position_ids,
                                               head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)

        # From bert outputs to our predictions
        sequence_output = outputs[0]
        logits = self.linearOut(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        base_loss = self.calculate_base_loss(
            start_logits, end_logits, start_positions, end_positions)

        # Additions to base loss
        total_loss = base_loss

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return tfs.modeling_outputs.QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QAModelPOA1(BaseModel):
    def __init__(self,
                 hparams,
                 transformerQA):

        super(QAModelPOA1, self).__init__(hparams, transformerQA)
        self.hparams = hparams
        self.model = getattr(transformerQA, hparams.get('model_type'))

        # 768 and 2 from the normal bert config
        self.linearOut = transformerQA.qa_outputs

        # Part of answer Architecture

        # hiddensize to 10 because why not
        self.seq_reduction = nn.Linear(hparams.get('hidden_size'), 8)

        # Reduce a bit more
        self.seq_reduction2 = nn.Linear(
            8*hparams.get('max_seq_length'), hparams.get('max_seq_length'))

        # Final size
        self.seq_reduction_linearOut = nn.Linear(hparams.get(
            'max_seq_length'), hparams.get('max_seq_length'))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.get_transformer_outputs(input_ids, attention_mask, token_type_ids, position_ids,
                                               head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)

        # From bert outputs to our predictions
        sequence_output = outputs[0]
        logits = self.linearOut(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        base_loss = self.calculate_base_loss(
            start_logits, end_logits, start_positions, end_positions)

        # Add the other loss

        total_loss = base_loss  # PLUS other loss

        # TODO: Ensure both parts of the loss gets returned for logging purposes

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return tfs.modeling_outputs.QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
