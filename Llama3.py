
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def answer_question(question, context):
    # Tokenize the input
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')

    # Get the model's predictions
    outputs = model(**inputs)

    # Get the start and end scores
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the start and end positions
    start_position = torch.argmax(start_scores)
    end_position = torch.argmax(end_scores)

    # Get the answer
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_position:end_position+1]))

    return answer