import os
import torch
from torch.utils.data import Dataset 
import cv2 
import json 
from transformers import AutoTokenizer 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
root = '/DATATWO/users/mincut/Object-Centric-VideoAnswering/data'
##use Ber

class VidQA(Dataset):
    
    def __init__(self,mode='train'):
        
        self.dtst_fldr = os.path.join(root,mode)
        self.mode = mode
        self.ques_file = os.path.join(self.dtst_fldr,f"{mode}.json")
        self.ques,self.ques_types,self.ques_vid,self.responses = self.read_questions()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.vid_fldr = os.listdir(os.path.join(root,'extracted_frames',mode))
        if mode == 'train':
            self.gen_vocab()
    
    def read_frames(self,idx):
        frame_fldr = os.path.join(root,'extracted_frames',self.mode,self.vid_fldr[idx])
        frm_seq = []
        for fl in os.scandir(frame_fldr):
            img = cv2.imread(fl.path)
            # Convert the image from BGR to RGB format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert image to torch tensor
            img = torch.from_numpy(img.transpose((2, 0, 1)))
            frm_seq.append(img)

        return torch.stack(frm_seq)
        
    def read_questions(self):
        mode = self.mode 
        q_path = os.path.join(root,mode,f"{mode}.json")
        cat = dict()
        q_list = []
        qt_list = []
        qt_vid = []
        qt_resp =[]
        dtst = json.load(open(q_path))
        for dt_pt in dtst:
            vid_idx = dt_pt['video_filename'].split('.')[0].split('_')[-1]
            vid_idx = int(vid_idx)
            for q in dt_pt['questions']:
                q_list.append(q['question'])
                qt_list.append(q['question_id'])
                qt_vid.append(vid_idx)
                if 'answer' in q.keys():
                    qt_resp.append(q['answer'])
                else:
                    for choice in q['choices']:
                        if choice['answer'] == "correct":
                            qt_resp.append(choice['choice'])
                
        return q_list,qt_list,qt_vid,qt_resp
    
    def gen_vocab(self):
        ##not required for bert based tokenisation
        pass 
        
    def tokenize_sentences(self,sentence):
        #use bert tokenizer 
        return self.tokenizer.encode(sentence)

    def __len__(self):
        return len(self.ques)
    
    def __getitem__(self,idx):
        
        frm = self.read_frames(self.ques_vid[idx])
        sentence = self.tokenize_sentences(self.ques[idx])
        response = self.tokenize_sentences(self.responses[idx])
        return frm,sentence,response
    

#dataloader boilerplate
def get_dataloader(mode='train',batch_size=32,shuffle=True):
        
        dataset = VidQA(mode)
        print(f"DATASET of len: {len(dataset)}")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader

def collate_fn(batch):
    
    frms = [x[0] for x in batch]
    frm_mask = [torch.ones(x.size(0))*i for i,x in enumerate(frms)]
    frms = torch.cat(frms,0)
    frm_mask = torch.cat(frm_mask,0)
    sentences = [x[1] for x in batch]
    q_lengths = [len(cap) for cap in sentences]
    queries = torch.zeros(len(sentences), max(q_lengths)).long()
    for i, cap in enumerate(sentences):
        end = q_lengths[i]
        queries[i, :end] = torch.tensor(cap[:end])
    
    sentences = [x[2] for x in batch]
    r_lengths = [len(cap) for cap in sentences]
    responses = torch.zeros(len(sentences), max(r_lengths)).long()
    for i, cap in enumerate(responses):
        end = r_lengths[i]
        responses[i, :end] = torch.tensor(cap[:end])
    
    batch = {
        "frames" : frms,
        "frame_mask" : frm_mask,
        "queries" : queries,
        "q_lengths" : q_lengths,
        "responses" : responses,
        "r_lengths" : r_lengths
    }
    return batch
    
for batch in get_dataloader():
    breakpoint()