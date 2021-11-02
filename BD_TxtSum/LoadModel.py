# -*- coding: utf-8 -*-
"""
Load Model
## Read Me
- CPU 환경에서도 사용 가능합니다.
- set_model에서 파일 경로를 확인해주세요.
- 모델 파일은 총 한 개 입니다. (model_v1.pt)
- 이 파일은 pytorch, pytorch_lightning, ke-t5를 기반으로 만들어졌습니다.

- 이 파일은 한 개의 class와 두 개의 함수로 구성되어 있습니다.
  1) MinutesSummaryModel(pl.LightningModule)
    - 모델 아키텍처 구성을 위한 class입니다.
    - set_model에서 사전학습된 모델을 불러오기 위한 class입니다.
  2) set_model(model_name, path)
    - 전이학습을 위한 모델과 토크나이저를 불러오는 함수입니다.
    - 이 함수에서 만들어지는 두 개의 변수(model, tokenizer)를 광역변수로 선언합니다.
  3) summarize(text)
    - text에 대한 요약을 수행하기 위한 함수입니다.
    - set_model을 실행하여 model과 tokenizer을 선언한 이후에 이 함수를 사용할 수 있습니다.

- 다음과 같은 패키지가 필요합니다.
    - transformers==4.5.0
    - pytorch-lightning==1.2.7
    - sentencepiece

@misc{ke_t5,
    author       = {KETI AIRC},
    title        = {KE-T5: Korean English T5},
    month        = mar,
    year         = 2021,
    url          = {https://github.com/AIRC-KETI/ke-t5}
}

"""

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pytorch_lightning as pl
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer

pl.seed_everything(42)

# 모델 클래스 선언
class MinutesSummaryModel(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(('KETI-AIR/ke-t5-small-ko'), return_dict=True)

  def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):

    output = self.model(
        input_ids,
        attention_mask = attention_mask,
        labels=labels,
        decoder_attention_mask = decoder_attention_mask
    )

    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
    input_ids = batch['text_input_ids']
    attention_mask =  batch['text_attention_mask']
    labels = batch['labels']
    labels_attention_mask = batch['labels_attention_mask']

    loss, outputs = self(input_ids = input_ids, 
                         attention_mask = attention_mask, 
                         decoder_attention_mask = labels_attention_mask,
                         labels=labels)
    
    self.log('train_loss', loss, prog_bar = True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids = batch['text_input_ids']
    attention_mask =  batch['text_attention_mask']
    labels = batch['labels']
    labels_attention_mask = batch['labels_attention_mask']

    loss, outputs = self(input_ids = input_ids, 
                         attention_mask = attention_mask, 
                         decoder_attention_mask = labels_attention_mask,
                         labels=labels)
    
    self.log('val_loss', loss, prog_bar = True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch['text_input_ids']
    attention_mask =  batch['text_attention_mask']
    labels = batch['labels']
    labels_attention_mask = batch['labels_attention_mask']

    loss, outputs = self(input_ids = input_ids, 
                         attention_mask = attention_mask, 
                         decoder_attention_mask = labels_attention_mask,
                         labels=labels)
    
    self.log('test_loss', loss, prog_bar = True, logger=True)
    return loss

  def configure_optimizers(self):
    return AdamW(self.parameters(), lr=0.0001)


# 모델 및 토크나이저 선언
def set_model(model_name, path):
  global tokenizer, model
  PATH = path
  tokenizer = T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-small-ko')
  model = MinutesSummaryModel()
  model.load_state_dict((torch.load(str(PATH+model_name))))
  model.eval()


# 요약 함수 선언
def summarize(text):
  tokenizer = T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-small-ko')

  text_encoding = tokenizer(
    text,
    max_length = 512,
    padding = "max_length",
    truncation = True,
    return_attention_mask = True,
    return_tensors='pt'
    )
  
  generated_ids = model.model.generate(
    input_ids = text_encoding['input_ids'],
    attention_mask = text_encoding['attention_mask'],
    max_length=150,
    num_beams=2,
    repetition_penalty = 2.5,
    length_penalty = 1.0,
    early_stopping=True
    )

  preds = [
          tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
          for gen_id in generated_ids
          ]

  return "".join(preds)