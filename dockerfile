FROM python:3.10.12

WORKDIR / C:\Users\er112\VSCode\HUFS\capstone\KoBERT_model

COPY requirements.txt .

RUN pip install --no-deps --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio

COPY . . 

CMD [ "python", "./run.py"]