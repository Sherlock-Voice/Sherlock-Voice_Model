#참고 
#!pip install mxnet-mkl==1.6.0 numpy==1.23.1
#!pip install gluonnlp==0.8.0
#!pip install tqdm pandas
#!pip install sentencepiece
#!pip install transformers
#!pip install torch>=1.8.1

#추천
#pip3 install mxnet 
#pip3 install gluonnlp
#pip3 install tqdm pandas
#pip3 install sentencepiece
#pip3 install transformers
#pip3 install pytorch



import torch
from KoBERT_model import run



if __name__=="__main__":
    run('이따가 뚝섬 가서 맛있는거 먹자. 오늘 비온다고 했으니까 혹시 모르니 우산도 챙기고.')