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

                # Reshape target logits
                loss_fct = nn.BCEWithLogitsLoss()
                # TODO: Potential issue this might alter the start_positions outside the loop
                start_positions = self._target_logits_bce(
                    start_positions, (start_logits.shape[0], start_logits.shape[1])).to(start_logits.device)

                end_positions = self._target_logits_bce(
                    end_positions, (end_logits.shape[0], end_logits.shape[1])).to(end_logits.device)

            else:
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=ignored_index, label_smoothing=self.hparams.get('label_smoothing', 0.0))

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss

    def calculate_poa_loss(self, logits, start_positions, end_positions):
        batch_size = logits.shape[0]
        poa_loss = None
        # Calculate loss if part of answer like we usually do
        if start_positions is not None and end_positions is not None:
            target = torch.zeros(
                size=(batch_size, self.hparams.get('max_seq_length')))
            for btc in range(batch_size):
                fencepost = 1
                if end_positions.squeeze()[btc] == self.hparams.get('max_seq_length'):
                    fencepost = 0

                target[btc, start_positions.squeeze()[btc]:end_positions.squeeze()[
                    btc]+fencepost] = 1
            loss_fct = nn.BCEWithLogitsLoss()
            # Work on distributed and single
            if len(logits.shape) > len(target.shape):
                target = target.unsqueeze(-1)
            poa_loss = loss_fct(logits, target)
        return poa_loss

    def _target_logits_bce(self, logits, shape):
        positions_bce = torch.zeros(shape)
        for btc, position in enumerate(logits):
            positions_bce[btc, position] = 1
        return positions_bce


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

        loss = {"total_loss": total_loss}
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if total_loss is not None else output

        return tfs.modeling_outputs.QuestionAnsweringModelOutput(
            loss=loss,
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

        # ______________Part of answer Architecture______________
        # hiddensize to out_par because why not

        # some value 2**2
        self.out_par = 4
        self.seq_reduction = nn.Linear(hparams.get(
            'hidden_size'), self.out_par)

        # Reduce single value pr. token
        self.seq_reduction_final = nn.Linear(
            self.out_par*hparams.get('max_seq_length'), hparams.get('max_seq_length'))

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

        # _________________Part of Answer Loss_______________

        # Get batch_size from data as we don't drop the last batch
        batch_size = sequence_output.shape[0]

        # B, Max_seq, out_par
        poa_logits = self.seq_reduction(sequence_output)
        poa_logits = self.seq_reduction_final(
            poa_logits.reshape(batch_size, -1))
        # 1 value pr. token
        poa_logits = poa_logits.reshape(
            batch_size, self.hparams.get('max_seq_length'), 1)

        # POA loss
        part_of_answer_loss = self.calculate_poa_loss(
            poa_logits, start_positions, end_positions)

        total_loss = base_loss + 0.1*part_of_answer_loss  # PLUS other loss

        loss = {"total_loss": total_loss,
                "qa_base_loss": base_loss, "poa_loss": 0.1*part_of_answer_loss}

        # TODO: Ensure both parts of the loss gets returned for logging purposes

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if total_loss is not None else output

        return tfs.modeling_outputs.QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
