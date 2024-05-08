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
    run("""안녕하세요, 중앙지검검찰청에서 전화드렸습니다. 000씨 맞으실까요? 혹시 주변에 000씨라고 지인 되실까요?
다름이 아니라 이 분이 지금 부정부패 혐의로 조사를 받고 있는데 000씨에게 입금이 된 내역이 조사중 발견 되어 연락드렸습니다.
    """)