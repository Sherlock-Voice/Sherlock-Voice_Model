#참고 
#!pip install mxnet-mkl==1.6.0 numpy==1.23.1
#!pip install gluonnlp==0.8.0
#!pip install tqdm pandas
#!pip install sentencepiece
#!pip install transformers
#!pip install torch>=1.8.1

#참고
#pip3 install numpy==1.16.5
#pip3 install tqdm pandas
#pip3 install sentencepiece
#pip3 install transformers==4.27.4
#pip3 install pytorch
#pip3 install gluonnlp==0.8.0

#확인된 방법
#python==3.8.19
#tqdm 
#sentencepiece==0.1.95
#transformers==4.24.0
#pytorch==2.3.0
#   conda install pytorch=2.3.0 torchvision torchaudio cpuonly -c pytorch-test
#gluonnlp (pip)(==0.10.0)
#mxnet (pip)
#numpy==1.17.4 (pip)

from KoBERT_model import run, inf

if __name__=="__main__":
    inf("""검찰청 근처 은행 갔다가 가서 조금 늦을것 같다. 
        지하철 타고 가면 금방 갈 것 같아. 금융 감독원 맞은편에 있는 고깃집 맞지?""")
    
#    run("안녕하세요, 중앙지검검찰청에서 전화드렸습니다. 000씨 맞으실까요? 혹시 주변에 000씨라고 지인 되실까요? 다름이 아니라 이 분이 지금 부정부패 혐의로 조사를 받고 있는데 000씨에게 입금이 된 내역이 조사중 발견 되어 연락드렸습니다.")

