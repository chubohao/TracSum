import unsloth, torch, re
from pathlib import Path
from transformers import (
    EvalPrediction,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
import pandas as pd
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, f1_score
    
class Track:
    def __init__(self):

        self.current_path = Path(__file__).parent 
        # load model and tokenizer
        self.model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=128,
            load_in_4bit=True,
            dtype=None
        )

        # apply chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        )

    def __mapping(self, aspect):
        map = {
            "a": "Aims or Objective",
            "i": "Intervention or Methods",
            "o": "Outcomes or Results",
            "p": "Patients Information",
            "m": "Medicines Information",
            "d": "Treatment Duration",
            "s": "Side Effects"
        }

        return map[aspect]
    
    # need your action
    def __formatting_prompts_func(self, batch):
        sentences = batch['sentence']
        aspects = batch['aspect']
        labels = batch['label']

        texts = []
        for sentence, aspect, label in zip(sentences, aspects, labels):
            message = [
                {"role": "user", "content": (
                    "Please determine whether the provided sentence contains information relevant to the given aspect.\n"
                    f"Sentence: {sentence}\n"
                    f"Aspect: {self.__mapping(aspect)}.\n"
                )},
                {"role": "assistant", "content": f"Relevance:{'Yes' if label == 1 else 'No'}."}
            ]
            texts.append(self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False))

        return {"text": texts,}

    # load dataset
    def __load_data(self):
        dataset = load_dataset("anonymous-data-review/sentence-aspect-label")
        train_dataset = dataset["train"]

        train_positive = train_dataset[train_dataset["label"] == 1]          # 取出 label == 1 的行
        train_positive_augmented = pd.concat([train_positive] * 6, ignore_index=True)  # 复制 10 次

        train_dataset = pd.concat([train_dataset, train_positive_augmented], ignore_index=True)
        train_dataset = Dataset.from_pandas(train_dataset).shuffle(seed=42)

        test_dataset = dataset['test']
        train_dataset = train_dataset.map(
            self.__formatting_prompts_func,
            batched = True,
            batch_size=32,
            remove_columns=train_dataset.column_names
        )
        test_dataset = test_dataset.map(
            self.__formatting_prompts_func,
            batched = True,
            batch_size=32,
            remove_columns=test_dataset.column_names
        )
        return train_dataset, test_dataset

    def __load_model(self):
        model = unsloth.FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state = 32,
            use_rslora=True,
            loftq_config = None
        )
        print(model.print_trainable_parameters())

        return model, self.tokenizer
    
    def __compute_metrics(self, eval_pred: EvalPrediction):
        predictions = eval_pred.predictions[0]
        predictions[predictions == -100] = self.tokenizer.pad_token_id

        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)


        # 正则提取 Label:0. 或 Label:1.
        extract_label = lambda text: 1 if re.search(r"Relevance\s*:\s*Yes", text, re.IGNORECASE) else \
                             0 if re.search(r"Relevance\s*:\s*No", text, re.IGNORECASE) else -1
        pred_labels = [extract_label(pred) for pred in decoded_preds]
        true_labels = [extract_label(label) for label in decoded_labels]

        # 打印前 5 条样本对
        for i in range(min(3, len(decoded_preds))):
            print("+++++ SAMPLE", i, "+++++")
            print("PRED TEXT:", decoded_preds[i])
            print("TRUE TEXT:", decoded_labels[i])
            print("PRED LABEL:", pred_labels[i])
            print("TRUE LABEL:", true_labels[i])
            print("-----------------------------")

        total = len(pred_labels)
        failed = sum(p not in [0, 1] for p in pred_labels)

        # 保留能成功提取的对
        filtered = [(p, t) for p, t in zip(pred_labels, true_labels) if p in [0, 1] and t in [0, 1]]
        if not filtered:
            return {"accuracy": 0.0, "failure_rate": 1.0, "label_0_acc": 0.0, "label_1_acc": 0.0}

        pred_labels, true_labels = zip(*filtered)
        acc = accuracy_score(true_labels, pred_labels)

        # 分别计算 Label 0 和 Label 1 的准确率
        label_0_indices = [i for i, t in enumerate(true_labels) if t == 0]
        label_1_indices = [i for i, t in enumerate(true_labels) if t == 1]

        label_0_acc = accuracy_score([true_labels[i] for i in label_0_indices],
                                    [pred_labels[i] for i in label_0_indices]) if label_0_indices else 0.0
        label_1_acc = accuracy_score([true_labels[i] for i in label_1_indices],
                                    [pred_labels[i] for i in label_1_indices]) if label_1_indices else 0.0

        return {
            "accuracy": acc,
            "failure_rate": failed / total,
            "label_0_acc": label_0_acc,
            "label_1_acc": label_1_acc
        }

    def __preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels
    
    def train(self):
        train_dataset, test_dataset = self.__load_data()
        model, tokenizer = self.__load_model()
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            dataset_text_field="text",
            max_seq_length=128,
            dataset_num_proc=2,
            packing=False,
            compute_metrics=self.__compute_metrics,
            preprocess_logits_for_metrics = self.__preprocess_logits_for_metrics,
            args=TrainingArguments(
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                eval_accumulation_steps=2,
                gradient_accumulation_steps=2,
                eval_strategy="steps",
                eval_steps=32,
                warmup_steps=200,
                num_train_epochs=3,
                learning_rate=1e-5,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=4,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                save_strategy="steps",
                save_steps=32,
                save_total_limit=5,
                load_best_model_at_end=True,
                output_dir=f"{self.current_path}/output"
            ),
        )

        
        
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>",
            response_part = "<|start_header_id|>assistant<|end_header_id|>",
        )
        trainer.train()
        trainer.save_model(f"{self.current_path}/best_models")

    def merge(self):
        # 路径指向你保存的 LoRA adapter 目录
        adapter_path = f"{self.current_path}/best_models"

        # 加载 adapter 配置，获取 base model 名称
        peft_config = PeftConfig.from_pretrained(adapter_path)

        # 加载 base model
        base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)

        # 加载 adapter 权重
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # 融合 adapter 到 base model 中
        merged_model = model.merge_and_unload()

        # 保存完整模型（不再需要 PEFT 加载方式）
        merged_model.save_pretrained(f"{self.current_path}/best_models_merged")
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        tokenizer.save_pretrained(f"{self.current_path}/best_models_merged")

if __name__ == "__main__":
    track = Track()
    track.train()

    