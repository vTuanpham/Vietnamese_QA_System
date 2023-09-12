# This directory include parser for parsing data from different source to AdvanceQA format


## Context augmentation
* Contain context injectors for random context augmentation.
* This is done to make the model more robust and pay attention to random position in the docs.
* Pre-trained model often train on news, media where important information
  lie at the start or the end of the text, this will introduce bias which is not what we want.
* This problem was researched in the paper [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/pdf/2307.03172.pdf)
* Context augmentation is fetch from random page of vietnamese wikipedia

## Data translation 
* Contain translator from vinai or google api
* Support large data translation via multithreading and chunk data split 
* Support restarting fail thread with its specific chunk
* Translation that can take up to 1000 hours with 200k examples in ELI5 can be reduce down to a couple hours.
  * Note: Be sure to have good internet connection
### Spliting data into chunk
![image](https://github.com/vTuanpham/Vietnamese_QA_System/assets/82665400/dab63062-9a76-4d88-aa3c-8c2b58a5a1ee)

### Restarting fail thread
![image](https://github.com/vTuanpham/Vietnamese_QA_System/assets/82665400/e9da4e69-c7f7-4cdc-9ae4-22025e2a88f9)
