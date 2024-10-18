
import json

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []
        self.docids = []
        self.sentids = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            instruction = '''
            Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            입력에 대해서 적합한 응답을 생성하시오. 생성 기준을 꼼꼼히 읽고 이해하는 것이 중요합니다.
            생성 기준:
            1 - 당신은 문장에 대한 도메인을 분류하는 챗봇입니다.
            2 - 도메인은 appropriate, inappropriate 중에서 생성하시오.
            3 - inappropriate은 부적절성 개념을 포함하는 문장입니다. 부적절성은 공격성, 편향성, 비하성 등의 부정적 특성을 포함합니다.
            4 - appropriate은 부적절성 내용이 없는 문장입니다.
            5 - 도메인이 appropriate이면 '### Response:appropriate'을 생성하고, inappropriate이면 '### Response:inappropriate'을 생성하시오.
            6 - 출력은 '### Response:appropriate'과 '### Response:inappropriate' 중에서 1개만 생성하시오.
            '''
            sentence = f"### Input: {inp['sentence']}\n"

            chat = instruction + "\n\n" + sentence + '\n\n### Response:\n'

            return chat
        
        for example in data:
            docid = example["id"]
            if len(example["output"]) == 0:
                for sent in example["input"]["sentences"]:
                    example["output"].append({"id": sent["id"], "lable": ""})
            for curr_sentence, curr_output in zip(example["input"]["sentences"], example["output"]):
                sent_id = curr_sentence["id"]
                output_id = curr_output["id"]
                if sent_id != output_id:
                    print(f'[ERR] id mismatch!! -> sent_id:{sent_id}, output_id:{output_id}')
                    continue

                chat = make_chat(curr_sentence)
                message = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": chat},
                ]
        
                source = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                if "label" in curr_output:
                    target = curr_output["label"]
                else:
                    target = ""
                target = f'### Response:{target}'
                if target != "":
                    target += tokenizer.eos_token
                
                target = tokenizer(target,
                        return_attention_mask=False,
                        add_special_tokens=False,
                        return_tensors="pt")
                target["input_ids"] = target["input_ids"].type(torch.int64)

                input_ids = torch.concat((source[0], target["input_ids"][0]))
                labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
                self.inp.append(input_ids)
                self.label.append(labels)
                self.docids.append(docid)
                self.sentids.append(sent_id)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.docids[idx], self.sentids[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
