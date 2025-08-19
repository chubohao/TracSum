import unsloth
import torch,os,re
from transformers import (
    EvalPrediction,
    TrainingArguments
)
from unsloth.chat_templates import train_on_responses_only
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from evaluate import load
from prompts import prompts
from datasets import load_dataset,concatenate_datasets
from pathlib import Path

class SUM:
    def __init__(self):
        self.current_path = Path(__file__).parent 

        # load model and tokenizer
        self.model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None
        )

        # apply chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template = "llama-3",
        )


    # need your action
    def __formatting_prompts_func(self, batch):
        sentences = batch['sentences']
        aspcets = batch['aspect']
        summaries = batch['summary']

        texts = []
        for sentences, aspect, summary in zip(sentences, aspcets, summaries):
            message = [
                {"role": "system", "content": "You are a great summarization assistant. Please follow the instructions strictly."},
                {"role": "user", "content": prompts(sentences, aspect=aspect)},
                {"role": "assistant", "content": f"Summary: {summary}"}
            ]
            texts.append(self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False))

        return {"text": texts,}
    
    def __load_data(self):
        # need your action
        dataset = load_dataset("anonymous-data-review/tracsum")

        # split dataset to train and test
        train_dataset = dataset["train"]

        test_dataset = dataset["test"]
        
        train_dataset = train_dataset.map(
            self.__formatting_prompts_func,
            batched = True,
            remove_columns=train_dataset.column_names
        )

        test_dataset = test_dataset.map(
            self.__formatting_prompts_func,
            batched = True,
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

        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        for i in range(min(3, len(decoded_predictions))):
            print("+" * 100)
            print(f"[{i}] RAW PRED:\n{decoded_predictions[i]}")
            print("=" * 100)
            print(f"[{i}] RAW REF:\n{decoded_references[i]}")

        def extract_summary(text):
            match = re.search(r"Summary\s*:\s*(.*)", text, re.DOTALL)
            return match.group(1).strip() if match else None

        extracted_predictions = []
        extracted_references = []
        for pred, ref in zip(decoded_predictions, decoded_references):
            pred_summary = extract_summary(pred)
            ref_summary = extract_summary(ref)
            if pred_summary is not None and ref_summary is not None:
                extracted_predictions.append(pred_summary)
                extracted_references.append(ref_summary)
        if not extracted_predictions or not extracted_references:
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeLsum": 0.0,
                "bertscore": 0.0,
                "empty_warning": True
            }
        # load evaluation metric
        rouge = load("rouge")
        bert_score = load("bertscore", model_type="distilbert-base-uncased")

        # Calculate metrics
        rouge_result = rouge.compute(predictions=extracted_predictions, references=extracted_references)
        bertscore_result = bert_score.compute(predictions=extracted_predictions, references=extracted_references, lang="en")

        # Return calculated metrics
        return {
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeLsum": rouge_result["rougeLsum"],
            "bertscore": bertscore_result["f1"][0]
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
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,
            compute_metrics=self.__compute_metrics,
            preprocess_logits_for_metrics = self.__preprocess_logits_for_metrics,
            args=TrainingArguments(
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                eval_accumulation_steps=2,
                gradient_accumulation_steps=2,
                eval_strategy="steps",
                eval_steps=32,
                warmup_steps=200,
                num_train_epochs=10,
                learning_rate=1e-5,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=8,
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

if __name__ == "__main__":
    sum = SUM()
    sum.train()

    