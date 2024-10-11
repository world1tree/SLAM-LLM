import os.path as osp
import random
import json, yaml
import copy

import numpy as np
from scipy import signal
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset
import whisper
from slam_llm.utils.compute_utils import calculate_output_length_1d


class STTRDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.ds_rate = self.dataset_config.ds_rate
        self.audio_root = self.dataset_config.audio_root
        self.split = split
        if self.split == "train":
            print(f"ds_rate: {self.ds_rate}")
            print(f"audio_root: {self.audio_root}")
        self.tokenizer = tokenizer

        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 

        self.data_list = []
        if self.split == "train":
            data_path = dataset_config.train_data_path
        else:
            data_path = dataset_config.val_data_path

        with open(data_path, encoding='utf-8') as fin:
            for line in fin:
                data_dict = json.loads(line.strip())
                self.data_list.append(data_dict)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio = data_dict.get("audio")
        key = data_dict.get("key")

        audio = osp.join(self.audio_root, audio)
        audio_path, frame_start, frame_offset = audio.split(':')
        frame_start = int(frame_start)
        frame_offset = int(frame_offset)

        # 采样率必须是16000
        audio_raw = whisper.load_audio(audio_path)
        audio_raw = audio_raw[frame_start:frame_start+frame_offset]
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320 # ad-hoc for fairseq 320x downsample
            audio_length = audio_length // self.ds_rate # ad-hoc for 5x fc downsample
        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            # audio_raw = np.concatenate((np.zeros(random.randint(0, 16000)), audio_raw, np.zeros(random.randint(0, 16000)))).astype(audio_raw.dtype)[:16000*30]
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // self.ds_rate # ad-hoc for 5x fc downsample
            # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
        else:
            raise ValueError(f"dataset_config.input_type: {self.input_type} not found.")
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        # -1在后面经过embedding前会被替换
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        prompt = data_dict.get("instruction")
        prompt = self.prompt_template.format(prompt)
        # for debug
        if self.split == "train" and index == 0 or index == 1:
            print(f"prompt-{index}: {prompt}")
            print('-'*10)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)

        answer = data_dict.get("answer", None)
        if self.inference_mode:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "target": answer,
                "prompt_length": prompt_length,
                "key": key,
            }

        answer = self.answer_template.format(answer)
        example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.
        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(
            example_ids, dtype=torch.int64
        )
        example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[:audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        # 和下面padding的区别在于此处的max_length是最终长度
        if isinstance(sequence, list):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, list):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    # 最后一个维度，左边不填充，右边填充padding_length
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        # torch.cat不指定维度是在第0维进行拼接
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 
        input_prompt_lengths = [s["audio_length"] + s['prompt_length'] for s in samples] #[120, 48, 82, 42]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s['prompt_length'] for s in samples]  #[0, 0, 0, 0]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)

        # audio + prompt共同组成了prompt首先需要padding到同一个长度，然后answer再padding在后面，为什么要这样设置?
        # 相比于直接对audio+prompt+answerpading有什么区别？是为了方便批量推理
        # 注意：左侧需要left_padding, 而右侧需要right_padding
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        # True表示当前位置有值
        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])


        if self.input_type == "raw":
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel":
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            # audio_mel_post_mask标记了经过whisper之后的语音长度
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1

        # modality_mask标记了语音的范围
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index]
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "modality_mask": modality_mask,
                "targets": targets,
                "keys": keys,
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask
        }



def get_sttr_dataset(dataset_config, tokenizer, split):
    dataset = STTRDatasetJsonl(dataset_config, tokenizer, split)

    return dataset
