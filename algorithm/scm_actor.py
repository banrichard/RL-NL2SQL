import ray
import torch
import re # Needed for parsing
from typing import List, Dict,Tuple, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
from .scm_processor import SCM_Config, SCM_Unified_Model
from openrlhf.trainer.ray.launcher import BaseModelActor
import torch.nn.functional as F
from openrlhf.utils.deepspeed import DeepspeedStrategy

def parse_schema(schema_string: str) -> dict:
    """Parses CREATE TABLE string into tables and columns."""
    if not schema_string: return {"tables": [], "columns": []}
    tables, columns = [], []
    table_defs = re.findall(r"CREATE TABLE\s+[`\"']?(\w+)[`\"']?\s*\((.*?)\)\s*;?", schema_string, re.DOTALL | re.IGNORECASE)
    for name, content in table_defs:
        tables.append(name)
        for line in content.strip().split('\n'):
            col_match = re.match(r"[`\"'](\w+)[`\"']", line.strip())
            if col_match and col_match.group(1).upper() not in ["PRIMARY", "FOREIGN", "CONSTRAINT"]:
                columns.append(f"{name}.{col_match.group(1)}")
    return {"tables": tables, "columns": columns}

def find_subsequence_indices(main_list: list, sub_list: list) -> int:
    """Finds start index of sublist in main list."""
    len_sub = len(sub_list);
    if len_sub == 0: return -1
    for i in range(len(main_list) - len_sub + 1):
        if main_list[i:i+len_sub] == sub_list: return i
    return -1

@ray.remote(num_gpus=1) # Decorator to make it a Ray Actor with 1 GPU
class SCM_ModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain: str):
        """
        Initializes the SCM model and tokenizer within the Ray Actor environment.
        Follows the pattern of RewardModelActor.
        """
        self._setup_distributed(strategy)  # Setup distributed environment using strategy
        # print(f"[{self.__class__.__name__}] Initializing SCM model from: {pretrain} on rank {strategy.args.local_rank}")

        try:
            # 1. Load SCM Config and Model
            self.config = SCM_Config.from_pretrained(pretrain)
            self.model = SCM_Unified_Model.from_pretrained(
                pretrain,
                config=self.config,
                # Add any necessary kwargs based on strategy (e.g., dtype)
                # torch_dtype=strategy.model_dtype
            )

            # 2. Load Tokenizer (Force Fast)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name,
                # trust_remote_code=strategy.args.trust_remote_code # If needed
            )

            # Set pad token if missing (important for padding)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"[{self.__class__.__name__}] Set pad_token to eos_token: {self.tokenizer.eos_token}")

            # 3. Prepare model with DeepSpeed strategy
            # Use is_rlhf=False or check if SCM needs specific wrapping
            self.model = self.strategy.prepare(self.model, is_rlhf=False)
            self.model.eval()  # Set to evaluation mode

        except Exception as e:
            print(f"[{self.__class__.__name__}] ERROR during model initialization: {e}")
            raise e

    @torch.no_grad()
    def forward(
            self,
            sequences_data: List[Dict],  # Input format defined in SCM_Integration
            max_length: int = 4096  # Use a reasonable default or get from config
    ) -> List[Dict]:
        """
        Performs SCM analysis on a batch of sequences generated during rollout.
        This replaces the previous 'analyze_sequences' method.
        """
        results = []
        # Process data one by one (batching could be added for efficiency if needed)
        for data in sequences_data:
            question = data.get("question", "")
            schema_str = data.get("schema", "")
            full_text = data.get("full_text", "")  # Decoded text from actor

            if not question or not schema_str or not full_text:
                print(f"[{self.__class__.__name__}] Warning: Missing data in input dict, skipping.")
                results.append({"entropies": [0.0], "quality_scores": [0.0]})  # Return default
                continue

            # 1. Parse Schema and Steps
            parsed_schema = parse_schema(schema_str)
            schema_columns = parsed_schema["columns"]
            steps = self._parse_steps(full_text)

            entropies = []
            quality_scores = []
            cot_history = ""

            # 2. Analyze Initial State (t=0)
            try:
                h0, q0 = self._get_analysis_for_step(question, schema_str, "", schema_columns, max_length)
                entropies.append(h0)
                quality_scores.append(q0)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error analyzing initial state: {e}")
                entropies.append(0.0);
                quality_scores.append(0.0)  # Default on error

            # 3. Step-wise Analysis
            for step_text in steps:
                cot_history += step_text + " "  # Append current step text
                try:
                    ht, qt = self._get_analysis_for_step(question, schema_str, cot_history, schema_columns, max_length)
                    entropies.append(ht)
                    quality_scores.append(qt)
                except Exception as e:
                    print(f"[{self.__class__.__name__}] Error analyzing step: {e}")
                    # Use previous values or default on error
                    entropies.append(entropies[-1] if entropies else 0.0)
                    quality_scores.append(quality_scores[-1] if quality_scores else 0.0)

            results.append({
                "entropies": entropies,
                "quality_scores": quality_scores
            })

        return results  # Return results as a list of dicts

    def _parse_steps(self, full_text: str) -> List[str]:
        """Parses the CoT text into logical steps."""
        # Simple split based on "### Step" - adjust as needed for your CoT format
        steps = []
        # Split keeping the delimiters
        parts = re.split(r'(### Step \d+:)', full_text)
        current_step = ""
        if parts[0] and not parts[0].startswith("### Step"):  # Handle text before the first step
            current_step = parts[0]

        for i in range(1, len(parts), 2):  # Iterate over delimiter and content pairs
            delimiter = parts[i]
            content = parts[i + 1] if (i + 1) < len(parts) else ""
            if current_step:  # Add the previous step
                steps.append(current_step.strip())
            current_step = delimiter + content  # Start the new step

        if current_step:  # Add the last step
            steps.append(current_step.strip())

        # If no steps found, maybe return the whole text as one step?
        return steps if steps else [full_text.strip()]

    def _get_analysis_for_step(self, question: str, schema_str: str, cot_history: str, schema_columns: List[str],
                               max_length: int) -> Tuple[float, float]:
        """
        Analyzes a single state (Q, S, CoT_history) using the SCM model.
        """
        # 1. Prepare Input Text
        schema_columns_str_for_input = " [SEP] ".join(schema_columns)
        input_text = f"[CLS] {question} [SEP] {schema_columns_str_for_input} [SEP] {cot_history.strip()} [SEP]"

        # 2. Tokenize and Calculate Indices
        encoding = self.tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",  # Padding is needed if batching inputs
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        ).to(self.model.device)  # Move inputs to model's device

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()

        column_char_indices = [input_text.find(col) for col in schema_columns]
        column_token_indices = []
        for char_start_index in column_char_indices:
            token_index = 0
            found = False
            if char_start_index != -1:
                for i, offset in enumerate(offset_mapping):
                    if isinstance(offset, (list, tuple)) and len(offset) == 2:
                        start, end = offset
                        if start <= char_start_index < end:
                            token_index = i;
                            found = True;
                            break
            column_token_indices.append(token_index if found else 0)

        # Convert indices to tensor for the model
        column_token_indices_tensor = torch.tensor([column_token_indices], dtype=torch.long).to(self.model.device)

        # 3. Model Forward Pass
        outputs = self.model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            column_token_indices=column_token_indices_tensor
        )

        # 4. Calculate Entropy and Quality Score
        total_entropy = self._calculate_total_entropy(outputs, len(schema_columns))
        quality_score = self._calculate_quality_score(outputs)

        return total_entropy, quality_score

    def _calculate_total_entropy(self, outputs: Dict, num_cols: int) -> float:
        """Calculates total entropy from model logits."""
        total_entropy = 0.0

        if "aggregate_logits" in outputs:
            probs = F.softmax(outputs["aggregate_logits"], dim=-1)
            total_entropy += torch.sum(-probs * torch.log(probs + 1e-9), dim=-1).item()

        if "column_logits" in outputs:
            valid_logits = outputs["column_logits"][:, :num_cols]
            probs = torch.sigmoid(valid_logits)
            col_entropy = -(probs * torch.log(probs + 1e-9) + (1 - probs) * torch.log(1 - probs + 1e-9))
            total_entropy += torch.sum(col_entropy, dim=-1).item()

        if "orderby_direction_logits" in outputs:
            probs = F.softmax(outputs["orderby_direction_logits"], dim=-1)
            total_entropy += torch.sum(-probs * torch.log(probs + 1e-9), dim=-1).item()

        return total_entropy if total_entropy > 0 else 0.0  # Ensure non-negative

    def _calculate_quality_score(self, outputs: Dict) -> float:
        """Calculates average quality score from model predictions."""
        if "predicted_quality_scores" in outputs:
            scores = outputs["predicted_quality_scores"][0].clamp(0, 10).tolist()  # Clamp scores to [0, 10] range
            return sum(scores) / len(scores) if scores else 0.0
        return 0.0