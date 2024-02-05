import gradio as gr
import fasttext

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import numpy as np
import pandas as pd
import torch

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

class LanguageIdentification:
    def __init__(self):
        pretrained_lang_model = "./lid.176.ftz"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=200) # returns top 200 matching languages
        return predictions

LANGUAGE = LanguageIdentification()

def tokenized_data(tokenizer, inputs):
    return tokenizer.batch_encode_plus(
        [inputs],
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True)

eng_model_name = "roberta-base"
eng_step = 1900
eng_tokenizer = AutoTokenizer.from_pretrained(eng_model_name)
eng_file_name = "{}-{}.pt".format(eng_model_name, eng_step)
eng_state_dict = torch.load(eng_file_name)
eng_model = AutoModelForSequenceClassification.from_pretrained(
    eng_model_name, num_labels=2, id2label=id2label, label2id=label2id,
    state_dict=eng_state_dict
)

kor_model_name = "klue/roberta-small"
kor_step = 2400
kor_tokenizer = AutoTokenizer.from_pretrained(kor_model_name)
kor_file_name = "{}-{}.pt".format(kor_model_name.replace('/', '_'), kor_step)
kor_state_dict = torch.load(kor_file_name)
kor_model = AutoModelForSequenceClassification.from_pretrained(
    kor_model_name, num_labels=2, id2label=id2label, label2id=label2id,
    state_dict=kor_state_dict
)

def builder(Lang, Text):
    percent_kor, percent_eng = 0, 0
    text_list = Text.split(' ')

    # [ output_1 ]
    if Lang == '언어감지 기능 사용':
        pred = LANGUAGE.predict_lang(Text)
        if '__label__en' in pred[0]:
            Lang = 'Eng'
            idx = pred[0].index('__label__en')
            p_eng = pred[1][idx]
        if '__label__ko' in pred[0]:
            Lang = 'Kor'
            idx = pred[0].index('__label__ko')
            p_kor = pred[1][idx]
        percent_kor = p_kor / (p_kor+p_eng)
        percent_eng = p_eng / (p_kor+p_eng)

    if Lang == 'Eng':
        model = eng_model
        tokenizer = eng_tokenizer
        if percent_eng==0: percent_eng=1

    if Lang == 'Kor':
        model = kor_model
        tokenizer = kor_tokenizer
        if percent_kor==0: percent_kor=1
        

    # [ output_2 ]
    inputs = tokenized_data(tokenizer, Text)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']).logits
    
    m = torch.nn.Softmax(dim=1)
    output = m(logits)

    # [ output_3 ]
    output_analysis = []
    for word in text_list:
        tokenized_word = tokenized_data(tokenizer, word)
        with torch.no_grad():
            logit = model(input_ids=tokenized_word['input_ids'], 
                attention_mask=tokenized_word['attention_mask']).logits
        word_output = m(logit)
        if word_output[0][1] > 0.99:
            output_analysis.append( (word, '+++') )
        elif word_output[0][1] > 0.9:
            output_analysis.append( (word, '++') )
        elif word_output[0][1] > 0.8:
            output_analysis.append( (word, '+') )
        elif word_output[0][1] < 0.01:
            output_analysis.append( (word, '---') )
        elif word_output[0][1] < 0.1:
            output_analysis.append( (word, '--') )
        elif word_output[0][1] < 0.2:
            output_analysis.append( (word, '-') )
        else:
            output_analysis.append( (word, None) )

    return [ {'Kor': percent_kor, 'Eng': percent_eng}, 
            {id2label[1]: output[0][1].item(), id2label[0]: output[0][0].item()}, 
            output_analysis ]

with gr.Blocks() as app:
    gr.Markdown(
    """
    <h1 align="center">
    영화 리뷰 점수 판별기
    </h1>
    """)

    gr.Markdown(
    """
    - 영화 리뷰를 입력하면, 리뷰가 긍정인지 부정인지 판별해주는 모델
        - 영어와 한글을 지원하며, 언어를 직접 선택할수도, 혹은 모델이 언어감지를 직접 하도록 할 수 있음
    - 사용자가 리뷰를 입력하면, (1) 감지된 언어, (2) 긍정일 확률과 부정일 확률, (3) 입력된 리뷰의 어떤 단어가 긍정/부정 결정에 영향을 주었는지 확인
        - 긍정일 경우 빨강색, 부정일 경우 파란색으로 표시
    """)

    with gr.Accordion(label="모델에 대한 설명 ( 여기를 클릭 하시오. )", open=False):
        gr.Markdown(
        """
        - 언어감지는 `fasttext`의 `language detector`을 사용
        - 영어는 `bert-base-uncased` 기반으로, 영어 영화 리뷰 분석 데이터셋인 `SST-2`로 학습 및 평가(92.8%의 정확도)
        - 한국어는 `klue/roberta-base` 기반이으로, 네이버 영화의 리뷰를 크롤링해서 영화 리뷰 분석 데이터셋을 제작하고, 이를 이용하여 모델을 학습 및 평가(94%의 정확도)
        - 단어별 영향력은, 단어 각각을 모델에 넣었을 때 결과가 긍정으로 나오는지 부정으로 나오는지를 바탕으로 측정하였다.
        """)

    with gr.Row():
        with gr.Column():
            inputs_1 = gr.Dropdown(choices=['언어감지 기능 사용', 'Eng', 'Kor'], value='언어감지 기능 사용', label='Lang')
            inputs_2 = gr.Textbox(placeholder="리뷰를 입력하시오.", label='Text')
            with gr.Row():
                btn = gr.Button("제출하기")
        with gr.Column():
            output_1 = gr.Label(num_top_classes=3, label='Lang')
            output_2 = gr.Label(num_top_classes=2, label='Result')
            output_3 = gr.HighlightedText(label="Analysis", combine_adjacent=False).style(color_map={"+++": "#CF0000", "++": "#FF3232", "+": "#FFD4D4", "---": "#0004FE", "--": "#4C47FF", "-": "#BEBDFF"})
    
    btn.click(fn=builder, inputs=[inputs_1, inputs_2], outputs=[output_1, output_2, output_3])

if __name__ == "__main__":
    app.launch()