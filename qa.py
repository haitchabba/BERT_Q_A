from flask import Flask, render_template, request
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
bert_model = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
bert_tokenizer = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')


def answer_question(question, context):

    input_ids = torch.tensor(bert_tokenizer.encode(question, context))

    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    sep_index = input_ids.index(bert_tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    assert len(segment_ids) == len(input_ids)

    start_scores, end_scores = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    ans_tokens = input_ids[torch.argmax(start_scores): torch.argmax(end_scores) + 1]

    answer_tokens = bert_tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)

    answer_tokens_to_string = bert_tokenizer.convert_tokens_to_string(answer_tokens)

    return answer_tokens_to_string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        ques = request.form['question']
        cont = request.form['context']
        return render_template('index.html', print=answer_question(ques, cont))


