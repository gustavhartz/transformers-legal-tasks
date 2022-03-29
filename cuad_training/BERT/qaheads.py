from torch import nn

# Linear Q&A head for predicting the answer span of the output of a transformer model
# using linear layers


# Q&A head
class linearQuestionAnsweringHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.linear.weight)
        self.num_labels = num_labels
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    # Start loss based on positions is not calculated here
    def forward(self, hidden_states):
        # Copied from transformers directyly
        outputs = hidden_states
        sequence_output = outputs[0]

        logits = self.linear(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits
