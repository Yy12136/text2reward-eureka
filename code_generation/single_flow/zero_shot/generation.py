import re
import time
import os
from typing import Any, List, Mapping, Optional, Tuple

import torch
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from openai import OpenAI

from code_generation.post_process.post_process import RewardFunctionConverter


class HuggingFaceLLM(LLM):
    def __init__(self, name: str, temperature: float = 0, **kwargs):
        super().__init__()
        self.name = name
        self.temperature = temperature
        
    @property
    def _llm_type(self) -> str:
        return self.name

    def __call__(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:
        return self._call(prompt, stop, **kwargs)

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:
        name_map = {
            "codellama_34b": "codellama/CodeLlama-34b-Instruct-hf",
            "llama_2_70b": "meta-llama/Llama-2-70b-chat-hf",
            "deepseek-v3-241226": "deepseek-ai/deepseek-v3-241226"
        }
        assert self.name in name_map, f"Model name {self.name} not supported!"
        model = name_map[self.name]
        
        try:
            # 尝试以离线模式加载
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                local_files_only=True,
                trust_remote_code=True
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading model in offline mode: {e}")
            print("Please make sure you have downloaded the model files locally.")
            raise

        pipe.tokenizer.pad_token_id = tokenizer.eos_token_id

        chat = [
            {"role": "user", "content": prompt},
        ]

        prompt = tokenizer.apply_chat_template(chat, tokenize=False)

        raw_results = pipe(
            [prompt],
            do_sample=False,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4096,
            batch_size=1
        )
        return raw_results[0][0]["generated_text"][len(prompt):]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"name": self.name, "temperature": self.temperature}


class ZeroShotGenerator:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="f03c5260-8425-465c-b6c8-c929568a7e60"
        )

    def generate_code(self, instruction: str, map_dict: dict) -> Tuple[str, str]:
        try:
            # 根据映射字典判断当前环境
            if "self.goal_height" in str(map_dict):
                # LiftCube-v0 环境
                available_vars = """
                - self.tcp.pose.p: end-effector position
                - self.obj.pose.p: object position
                - self.agent.check_grasp(self.obj): check if object is grasped
                - self.goal_height: target lifting height
                - self.agent.robot.get_qpos()[:-2]: robot joint positions
                - self.agent.robot.get_qvel()[:-2]: robot joint velocities
                """
            elif "self.goal_pos" in str(map_dict):
                # PickCube-v0 环境
                available_vars = """
                - self.tcp.pose.p: end-effector position
                - self.obj.pose.p: object position
                - self.agent.check_grasp(self.obj): check if object is grasped
                - self.goal_pos: target position
                - self.agent.robot.get_qpos()[:-2]: robot joint positions
                - self.agent.robot.get_qvel()[:-2]: robot joint velocities
                """
            else:
                # 其他环境
                available_vars = "\n".join([f"- {k}: {v}" for k, v in map_dict.items()])

            specific_prompt = f"""
            Generate a specific reward function for the following task:
            {instruction}
            
            The reward function should:
            1. Be dense and informative
            2. Guide the robot through each step
            3. Include specific milestones for the task
            
            Only use these available variables and methods:
            {available_vars}
            
            Do NOT use any other variables not listed above.
            
            Example format:
            ```python
            def compute_dense_reward(self, action) -> float:
                # Initialize reward
                reward = 0.0
                
                # Get positions
                ee_pos = self.tcp.pose.p
                obj_pos = self.obj.pose.p
                
                # Check grasp
                grasp_success = self.agent.check_grasp(self.obj)
                
                # Your reward computation here
                
                return reward
            ```
            """
            
            specific_response = self.client.chat.completions.create(
                model="deepseek-v3-241226",
                messages=[{"role": "user", "content": specific_prompt}]
            )
            specific_code = specific_response.choices[0].message.content
            
            # 生成通用的奖励函数
            general_prompt = f"""
            Generate a general reward function for robotic manipulation:
            {instruction}
            
            The reward function should:
            1. Be sparse but meaningful
            2. Focus on task completion
            3. Be applicable to similar tasks
            """
            
            general_response = self.client.chat.completions.create(
                model="deepseek-v3-241226",
                messages=[{"role": "user", "content": general_prompt}]
            )
            general_code = general_response.choices[0].message.content
            
            # 提取实际的函数代码
            def extract_function(code: str) -> str:
                """从生成的文本中提取实际的函数代码"""
                # 只保留实际的 Python 代码
                code_template = """import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Your reward computation code here
    
    return reward"""
                
                # 查找 Python 代码块
                import re
                pattern = r"```python\n(.*?)```"
                match = re.search(pattern, code, re.DOTALL)
                if match:
                    function_code = match.group(1).strip()
                    # 如果代码中没有 import，添加必要的导入
                    if "import" not in function_code:
                        function_code = "import numpy as np\n\n" + function_code
                    return function_code
                return code_template  # 如果没找到代码块，返回模板
                
            # 处理生成的代码
            if specific_code:
                specific_code = extract_function(specific_code)
                # 添加代码转换
                converter = RewardFunctionConverter(map_dict)
                specific_code = converter.general_to_specific(specific_code)
            
            if general_code:
                general_code = extract_function(general_code)
            
            return specific_code, general_code
            
        except Exception as e:
            print(f"生成代码出错: {e}")
            return "", ""