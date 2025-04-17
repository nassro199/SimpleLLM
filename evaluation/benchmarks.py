"""
Benchmark evaluation pipeline for MoE LLM.
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from tqdm.auto import tqdm

from ..model.model import MoELLM
from .metrics import calculate_metrics

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Evaluator for running benchmarks on MoE LLM.
    """
    def __init__(
        self,
        model: MoELLM,
        tokenizer,
        device: Optional[torch.device] = None,
        output_dir: str = "benchmark_results"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_mmlu(
        self,
        data_path: str = "cais/mmlu",
        num_few_shot: int = 5,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate the model on the MMLU benchmark.
        
        Args:
            data_path: Path to the MMLU dataset
            num_few_shot: Number of few-shot examples to use
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on MMLU with {num_few_shot}-shot examples...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets not installed. Run `pip install datasets`.")
        
        # Load MMLU dataset
        try:
            dataset = load_dataset(data_path)
        except Exception as e:
            logger.error(f"Failed to load MMLU dataset: {e}")
            return {"mmlu_accuracy": 0.0}
        
        # Get validation and test sets
        validation_set = dataset["validation"]
        test_set = dataset["test"]
        
        # Get subjects
        subjects = list(set(test_set["subject"]))
        
        # Initialize metrics
        all_metrics = {}
        
        # Evaluate on each subject
        for subject in tqdm(subjects, desc="Evaluating MMLU subjects"):
            # Filter validation and test sets for this subject
            subject_validation = validation_set.filter(lambda x: x["subject"] == subject)
            subject_test = test_set.filter(lambda x: x["subject"] == subject)
            
            # Get few-shot examples
            few_shot_examples = []
            for i in range(min(num_few_shot, len(subject_validation))):
                example = subject_validation[i]
                question = example["question"]
                choices = [example[f"choice{j}"] for j in range(4)]
                answer = choices[example["answer"]]
                
                few_shot_examples.append({
                    "question": question,
                    "choices": choices,
                    "answer": answer
                })
            
            # Evaluate on test set
            correct = 0
            total = 0
            
            # Process in batches
            for i in range(0, len(subject_test), batch_size):
                batch = subject_test[i:i + batch_size]
                
                batch_questions = []
                batch_choices = []
                batch_answers = []
                
                for example in batch:
                    question = example["question"]
                    choices = [example[f"choice{j}"] for j in range(4)]
                    answer_idx = example["answer"]
                    
                    batch_questions.append(question)
                    batch_choices.append(choices)
                    batch_answers.append(answer_idx)
                
                # Create prompts with few-shot examples
                prompts = []
                for question, choices in zip(batch_questions, batch_choices):
                    prompt = f"The following are multiple-choice questions about {subject}.\n\n"
                    
                    # Add few-shot examples
                    for ex in few_shot_examples:
                        prompt += f"Question: {ex['question']}\n"
                        prompt += "Choices:\n"
                        for j, choice in enumerate(ex["choices"]):
                            prompt += f"{chr(65 + j)}. {choice}\n"
                        prompt += f"Answer: {ex['answer']}\n\n"
                    
                    # Add current question
                    prompt += f"Question: {question}\n"
                    prompt += "Choices:\n"
                    for j, choice in enumerate(choices):
                        prompt += f"{chr(65 + j)}. {choice}\n"
                    prompt += "Answer:"
                    
                    prompts.append(prompt)
                
                # Tokenize prompts
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate answers
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs["input_ids"].shape[1] + 5,
                        do_sample=False
                    )
                
                # Decode outputs
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Extract answers
                for gen_text, answer_idx in zip(generated_texts, batch_answers):
                    # Extract the generated answer (A, B, C, or D)
                    answer_text = gen_text.split("Answer:")[-1].strip()
                    
                    # Check if the answer is correct
                    if answer_text and answer_text[0] in "ABCD" and ord(answer_text[0]) - ord("A") == answer_idx:
                        correct += 1
                    
                    total += 1
            
            # Calculate accuracy for this subject
            subject_accuracy = correct / total if total > 0 else 0.0
            all_metrics[f"mmlu_{subject}_accuracy"] = subject_accuracy
            
            logger.info(f"MMLU {subject} accuracy: {subject_accuracy:.4f}")
        
        # Calculate overall accuracy
        overall_accuracy = sum(all_metrics.values()) / len(all_metrics)
        all_metrics["mmlu_accuracy"] = overall_accuracy
        
        logger.info(f"MMLU overall accuracy: {overall_accuracy:.4f}")
        
        # Save results
        with open(os.path.join(self.output_dir, "mmlu_results.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        return all_metrics
    
    def evaluate_gsm8k(
        self,
        data_path: str = "gsm8k",
        split: str = "test",
        num_few_shot: int = 8,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Evaluate the model on the GSM8K benchmark.
        
        Args:
            data_path: Path to the GSM8K dataset
            split: Split to evaluate on
            num_few_shot: Number of few-shot examples to use
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on GSM8K with {num_few_shot}-shot examples...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets not installed. Run `pip install datasets`.")
        
        # Load GSM8K dataset
        try:
            dataset = load_dataset(data_path)
        except Exception as e:
            logger.error(f"Failed to load GSM8K dataset: {e}")
            return {"gsm8k_accuracy": 0.0}
        
        # Get train and test sets
        train_set = dataset["train"]
        test_set = dataset[split]
        
        # Get few-shot examples
        few_shot_examples = []
        for i in range(min(num_few_shot, len(train_set))):
            example = train_set[i]
            question = example["question"]
            answer = example["answer"]
            
            few_shot_examples.append({
                "question": question,
                "answer": answer
            })
        
        # Evaluate on test set
        correct = 0
        total = 0
        
        # Process in batches
        for i in tqdm(range(0, len(test_set), batch_size), desc="Evaluating GSM8K"):
            batch = test_set[i:i + batch_size]
            
            batch_questions = []
            batch_answers = []
            
            for example in batch:
                question = example["question"]
                answer = example["answer"]
                
                batch_questions.append(question)
                batch_answers.append(answer)
            
            # Create prompts with few-shot examples
            prompts = []
            for question in batch_questions:
                prompt = "Solve the following math problems step by step:\n\n"
                
                # Add few-shot examples
                for ex in few_shot_examples:
                    prompt += f"Question: {ex['question']}\n"
                    prompt += f"Answer: {ex['answer']}\n\n"
                
                # Add current question
                prompt += f"Question: {question}\n"
                prompt += "Answer:"
                
                prompts.append(prompt)
            
            # Tokenize prompts
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate answers
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + 200,
                    do_sample=False
                )
            
            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract answers
            for gen_text, answer in zip(generated_texts, batch_answers):
                # Extract the generated answer
                gen_answer = gen_text.split("Answer:")[-1].strip()
                
                # Extract the final answer (usually the last number)
                import re
                gen_numbers = re.findall(r"(\d+)", gen_answer)
                ref_numbers = re.findall(r"(\d+)", answer)
                
                if gen_numbers and ref_numbers and gen_numbers[-1] == ref_numbers[-1]:
                    correct += 1
                
                total += 1
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0.0
        metrics = {"gsm8k_accuracy": accuracy}
        
        logger.info(f"GSM8K accuracy: {accuracy:.4f}")
        
        # Save results
        with open(os.path.join(self.output_dir, "gsm8k_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def evaluate_math(
        self,
        data_path: str = "hendrycks_math",
        split: str = "test",
        num_few_shot: int = 4,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Evaluate the model on the MATH benchmark.
        
        Args:
            data_path: Path to the MATH dataset
            split: Split to evaluate on
            num_few_shot: Number of few-shot examples to use
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on MATH with {num_few_shot}-shot examples...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets not installed. Run `pip install datasets`.")
        
        # Load MATH dataset
        try:
            dataset = load_dataset(data_path)
        except Exception as e:
            logger.error(f"Failed to load MATH dataset: {e}")
            return {"math_accuracy": 0.0}
        
        # Get train and test sets
        train_set = dataset["train"]
        test_set = dataset[split]
        
        # Get subjects
        subjects = list(set(test_set["type"]))
        
        # Initialize metrics
        all_metrics = {}
        
        # Evaluate on each subject
        for subject in tqdm(subjects, desc="Evaluating MATH subjects"):
            # Filter train and test sets for this subject
            subject_train = train_set.filter(lambda x: x["type"] == subject)
            subject_test = test_set.filter(lambda x: x["type"] == subject)
            
            # Get few-shot examples
            few_shot_examples = []
            for i in range(min(num_few_shot, len(subject_train))):
                example = subject_train[i]
                problem = example["problem"]
                solution = example["solution"]
                
                few_shot_examples.append({
                    "problem": problem,
                    "solution": solution
                })
            
            # Evaluate on test set
            correct = 0
            total = 0
            
            # Process in batches
            for i in range(0, len(subject_test), batch_size):
                batch = subject_test[i:i + batch_size]
                
                batch_problems = []
                batch_solutions = []
                batch_answers = []
                
                for example in batch:
                    problem = example["problem"]
                    solution = example["solution"]
                    answer = example["answer"]
                    
                    batch_problems.append(problem)
                    batch_solutions.append(solution)
                    batch_answers.append(answer)
                
                # Create prompts with few-shot examples
                prompts = []
                for problem in batch_problems:
                    prompt = f"Solve the following {subject} problems step by step:\n\n"
                    
                    # Add few-shot examples
                    for ex in few_shot_examples:
                        prompt += f"Problem: {ex['problem']}\n"
                        prompt += f"Solution: {ex['solution']}\n\n"
                    
                    # Add current problem
                    prompt += f"Problem: {problem}\n"
                    prompt += "Solution:"
                    
                    prompts.append(prompt)
                
                # Tokenize prompts
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate solutions
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs["input_ids"].shape[1] + 300,
                        do_sample=False
                    )
                
                # Decode outputs
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Extract solutions and check answers
                for gen_text, answer in zip(generated_texts, batch_answers):
                    # Extract the generated solution
                    gen_solution = gen_text.split("Solution:")[-1].strip()
                    
                    # Check if the answer is correct
                    if answer in gen_solution:
                        correct += 1
                    
                    total += 1
            
            # Calculate accuracy for this subject
            subject_accuracy = correct / total if total > 0 else 0.0
            all_metrics[f"math_{subject}_accuracy"] = subject_accuracy
            
            logger.info(f"MATH {subject} accuracy: {subject_accuracy:.4f}")
        
        # Calculate overall accuracy
        overall_accuracy = sum(all_metrics.values()) / len(all_metrics)
        all_metrics["math_accuracy"] = overall_accuracy
        
        logger.info(f"MATH overall accuracy: {overall_accuracy:.4f}")
        
        # Save results
        with open(os.path.join(self.output_dir, "math_results.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        return all_metrics
    
    def evaluate_bbh(
        self,
        data_path: str = "lukaemon/bbh",
        num_few_shot: int = 3,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate the model on the Big Bench Hard (BBH) benchmark.
        
        Args:
            data_path: Path to the BBH dataset
            num_few_shot: Number of few-shot examples to use
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on BBH with {num_few_shot}-shot examples...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets not installed. Run `pip install datasets`.")
        
        # Load BBH dataset
        try:
            dataset = load_dataset(data_path)
        except Exception as e:
            logger.error(f"Failed to load BBH dataset: {e}")
            return {"bbh_accuracy": 0.0}
        
        # Get tasks
        tasks = list(dataset.keys())
        
        # Initialize metrics
        all_metrics = {}
        
        # Evaluate on each task
        for task in tqdm(tasks, desc="Evaluating BBH tasks"):
            # Get task dataset
            task_dataset = dataset[task]
            
            # Get few-shot examples
            few_shot_examples = []
            for i in range(min(num_few_shot, len(task_dataset["train"]))):
                example = task_dataset["train"][i]
                input_text = example["input"]
                target = example["target"]
                
                few_shot_examples.append({
                    "input": input_text,
                    "target": target
                })
            
            # Evaluate on test set
            correct = 0
            total = 0
            
            # Process in batches
            for i in range(0, len(task_dataset["test"]), batch_size):
                batch = task_dataset["test"][i:i + batch_size]
                
                batch_inputs = []
                batch_targets = []
                
                for example in batch:
                    input_text = example["input"]
                    target = example["target"]
                    
                    batch_inputs.append(input_text)
                    batch_targets.append(target)
                
                # Create prompts with few-shot examples
                prompts = []
                for input_text in batch_inputs:
                    prompt = f"Task: {task}\n\n"
                    
                    # Add few-shot examples
                    for ex in few_shot_examples:
                        prompt += f"Input: {ex['input']}\n"
                        prompt += f"Output: {ex['target']}\n\n"
                    
                    # Add current input
                    prompt += f"Input: {input_text}\n"
                    prompt += "Output:"
                    
                    prompts.append(prompt)
                
                # Tokenize prompts
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate outputs
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs["input_ids"].shape[1] + 50,
                        do_sample=False
                    )
                
                # Decode outputs
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Extract outputs and check targets
                for gen_text, target in zip(generated_texts, batch_targets):
                    # Extract the generated output
                    gen_output = gen_text.split("Output:")[-1].strip()
                    
                    # Check if the output is correct
                    if gen_output == target:
                        correct += 1
                    
                    total += 1
            
            # Calculate accuracy for this task
            task_accuracy = correct / total if total > 0 else 0.0
            all_metrics[f"bbh_{task}_accuracy"] = task_accuracy
            
            logger.info(f"BBH {task} accuracy: {task_accuracy:.4f}")
        
        # Calculate overall accuracy
        overall_accuracy = sum(all_metrics.values()) / len(all_metrics)
        all_metrics["bbh_accuracy"] = overall_accuracy
        
        logger.info(f"BBH overall accuracy: {overall_accuracy:.4f}")
        
        # Save results
        with open(os.path.join(self.output_dir, "bbh_results.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        return all_metrics
    
    def evaluate_all_benchmarks(self) -> Dict[str, float]:
        """
        Evaluate the model on all benchmarks.
        
        Returns:
            Dictionary of metrics from all benchmarks
        """
        logger.info("Evaluating on all benchmarks...")
        
        # Initialize metrics
        all_metrics = {}
        
        # Evaluate on MMLU
        mmlu_metrics = self.evaluate_mmlu()
        all_metrics.update(mmlu_metrics)
        
        # Evaluate on GSM8K
        gsm8k_metrics = self.evaluate_gsm8k()
        all_metrics.update(gsm8k_metrics)
        
        # Evaluate on MATH
        math_metrics = self.evaluate_math()
        all_metrics.update(math_metrics)
        
        # Evaluate on BBH
        bbh_metrics = self.evaluate_bbh()
        all_metrics.update(bbh_metrics)
        
        # Save all results
        with open(os.path.join(self.output_dir, "all_benchmark_results.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        return all_metrics
