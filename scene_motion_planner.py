
"""
Full pipeline: Scene understanding to motion generation
1. Scene analysis with QwenVL
2. Route and action planning based on scene analysis
3. Generate MotionLLM motion prompts
4. Call MotionLLM to generate the motion sequence
"""

import os
import sys
import requests
import json
import subprocess
import tempfile

def convert_github_url_to_raw(github_url):
    """Convert GitHub blob link to raw link"""
    if 'github.com' in github_url and '/blob/' in github_url:
        raw_url = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        return raw_url
    return github_url

class SceneMotionPlanner:
    def __init__(self, api_key=None):
        """
        Initialize Scene-to-Motion Planner
        
        Args:
            api_key: QwenVL API Key
        """
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or "sk-4046430e513f44c68beec5635a02d97f"
        self.qwen_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        
    def analyze_scene(self, image_url):
        """
        Analyze a scene image using QwenVL
        
        Args:
            image_url: Image URL
            
        Returns:
            Scene analysis text
        """
        # Convert GitHub link if necessary
        processed_url = convert_github_url_to_raw(image_url)
        print(f"🖼️ Analyzing scene image: {processed_url}")
        
        # prompt
        scene_analysis_prompt = """
Please analyze this scene image in detail, focusing on:
1. Scene type (indoor/outdoor, room type, etc.)
2. Spatial layout and locations of major objects
3. Walkable paths and areas
4. Obstacles and areas to avoid
5. Key landmarks and reference points
6. Suitable movement modes (walking, running, crawling, etc.)

Provide a structured description to support subsequent route planning.
"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen-vl-max",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": processed_url
                            },
                            {
                                "text": scene_analysis_prompt
                            }
                        ]
                    }
                ]
            },
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.8,
                "max_tokens": 1500
            }
        }
        
        try:
            response = requests.post(self.qwen_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                return f"❌ Scene analysis failed: HTTP {response.status_code} - {response.text}"
            
            result = response.json()
            
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    try:
                        parsed_content = json.loads(content)
                        if isinstance(parsed_content, list) and len(parsed_content) > 0:
                            if isinstance(parsed_content[0], dict) and 'text' in parsed_content[0]:
                                return parsed_content[0]['text']
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
                return content
            else:
                return f"❌ Scene analysis response format error: {result}"
                
        except Exception as e:
            return f"❌ Scene analysis failed: {e}"
    
    def plan_route_and_actions(self, scene_analysis, start_point=None, end_point=None, task_description=None):
        """
        Plan route and actions based on scene analysis
        
        Args:
            scene_analysis: Scene analysis result
            start_point: Start point description (optional)
            end_point: End point description (optional)
            task_description: Task description (optional)
            
        Returns:
            Route plan and action sequence
        """
        print("🗺️ Planning route and actions based on scene analysis...")
        
        # prompt
        route_planning_prompt = f"""
Based on the following scene analysis, plan a reasonable movement route and action sequence:

Scene analysis:
{scene_analysis}

Planning requirements:
- Start point: {start_point or 'an appropriate starting position in the scene'}
- End point: {end_point or 'an appropriate target position in the scene'}
- Task: {task_description or 'explore and move within the scene'}

Please provide:
1. A detailed route plan (including key waypoints)
2. Three consecutive action stages, each described in English and suitable for MotionLLM generation:
   - Stage 1: initial action (e.g., "a person starts walking forward slowly")
   - Stage 2: main movement (e.g., "a person walks around obstacles and turns left")
   - Stage 3: ending action (e.g., "a person approaches the target and stops")

Action description requirements:
- Use English
- Describe specific human motions
- Consider obstacles and spatial constraints in the scene
- Ensure actions are coherent and physically plausible
"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen-vl-max",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": route_planning_prompt
                            }
                        ]
                    }
                ]
            },
            "parameters": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        try:
            response = requests.post(self.qwen_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                return f"❌ Route planning failed: HTTP {response.status_code} - {response.text}"
            
            result = response.json()
            
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    try:
                        parsed_content = json.loads(content)
                        if isinstance(parsed_content, list) and len(parsed_content) > 0:
                            if isinstance(parsed_content[0], dict) and 'text' in parsed_content[0]:
                                return parsed_content[0]['text']
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
                return content
            else:
                return f"❌ Route planning response format error: {result}"
                
        except Exception as e:
            return f"❌ Route planning failed: {e}"
    
    def extract_motion_prompts(self, route_plan):
        """
        Extract motion descriptions from the route plan
        
        Args:
            route_plan: Route plan result
            
        Returns:
            List of three motion descriptions
        """
        print("📝 Extracting motion descriptions...")
        
        
        extract_prompt = f"""
From the following route plan text, extract three consecutive English motion descriptions. Each description should:
1. Be in English
2. Describe concrete human actions
3. Be suitable for motion generation models
4. Be in chronological order

Route plan text:
{route_plan}

Output strictly in the following format with motions only, nothing else:
Stage 1: [English motion]
Stage 2: [English motion]
Stage 3: [English motion]

Example:
Stage 1: a person starts walking forward slowly
Stage 2: a person walks around obstacles and turns left
Stage 3: a person approaches the target and stops
"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen-vl-max",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": extract_prompt
                            }
                        ]
                    }
                ]
            },
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.8,
                "max_tokens": 500
            }
        }
        
        try:
            response = requests.post(self.qwen_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                print(f"❌ Motion extraction failed, using default motions: HTTP {response.status_code}")
                return self._get_default_motions()
            
            result = response.json()
            
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    try:
                        parsed_content = json.loads(content)
                        if isinstance(parsed_content, list) and len(parsed_content) > 0:
                            if isinstance(parsed_content[0], dict) and 'text' in parsed_content[0]:
                                content = parsed_content[0]['text']
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
                
                # Parse motion descriptions
                motions = []
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if (line.lower().startswith('阶段') or line.lower().startswith('stage')) and ':' in line:
                        motion = line.split(':', 1)[1].strip()
                        if motion:
                            motions.append(motion)
                
                if len(motions) >= 3:
                    return motions[:3]
                else:
                    print("❌ Extracted motions fewer than 3, using default motions")
                    return self._get_default_motions()
            else:
                print("❌ Motion extraction response format error, using default motions")
                return self._get_default_motions()
                
        except Exception as e:
            print(f"❌ Motion extraction failed, using default motions: {e}")
            return self._get_default_motions()
    
    def _get_default_motions(self):
        """Get default motion descriptions"""
        return [
            "a person starts walking forward slowly",
            "a person walks straight and looks around",
            "a person stops and stands still"
        ]
    
    def generate_motions_with_motionllm(self, motion_prompts):
        """
        Generate motion sequence with MotionLLM
        
        Args:
            motion_prompts: List of motion descriptions
            
        Returns:
            Generation result
        """
        print("🎬 Generating motion sequence with MotionLLM...")
        
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for prompt in motion_prompts:
                f.write(prompt + '\n')
            temp_input_file = f.name
        
        try:
            # Modify auto_generate_and_concat.py to support file input
            modified_script = self._create_modified_motionllm_script(motion_prompts)
            
            # Run the modified script
            result = subprocess.run([
                'python3', modified_script
            ], capture_output=True, text=True, cwd='motionllm/Motion-Agent')
            
            if result.returncode == 0:
                print("✅ MotionLLM motion generation completed!")
                return "Motion generation succeeded. Please check motionllm/Motion-Agent/demo directory"
            else:
                print(f"❌ MotionLLM execution failed: {result.stderr}")
                return f"Motion generation failed: {result.stderr}"
                
        except Exception as e:
            print(f"❌ Calling MotionLLM failed: {e}")
            return f"Calling MotionLLM failed: {e}"
        finally:
            # Cleanup temp file
            if os.path.exists(temp_input_file):
                os.unlink(temp_input_file)
    
    def _create_modified_motionllm_script(self, motion_prompts):
        """Create modified MotionLLM script"""
        script_content = f'''import os
import torch
import numpy as np
from models.mllm import MotionLLM
from options.option_llm import get_args_parser
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
from concat_motions import concat_motions

    def process_scene_task(self, scene_image, task):
        """
        处理场景任务 - 兼容main.py的接口
        
        Args:
            scene_image: 场景图像路径或URL
            task: 任务描述
            
        Returns:
            处理结果字典
        """
        try:
            # 步骤1: 分析场景
            scene_analysis = self.analyze_scene(scene_image)
            
            # 步骤2: 规划路线和动作
            route_plan = self.plan_route_and_actions(
                scene_analysis, 
                task_description=task
            )
            
            # 步骤3: 提取动作描述
            motion_prompts = self.extract_motion_prompts(route_plan)
            
            # 步骤4: 生成动作
            result = self.generate_motions_with_motionllm(motion_prompts)
            
            return {
                "status": "success",
                "task": task,
                "scene_image": scene_image,
                "output_dir": "./output",
                "total_steps": len(motion_prompts),
                "motion_prompts": motion_prompts,
                "result": result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "task": task,
                "scene_image": scene_image
            }

def main():
 
    prompts = {motion_prompts}
    
    save_dir = './demo'
    os.makedirs(save_dir, exist_ok=True)

    # 初始化模型
    args = get_args_parser()
    args.llm_backbone = "/home/g2563110552/gemma2b"
    args.ckpt = "ckpt/motionllm.pth"
    args.save_dir = save_dir
    args.device = 'cuda:0'
    args.dataname = 't2m'
    args.window_size = 196
    model = MotionLLM(args).to(args.device)
    model.load_model(args.ckpt)
    model.llm.eval()

    npy_list = []
    for idx, caption in enumerate(prompts, 1):
        print(f"\\n[阶段{{idx}}] 生成动作: {{caption}}")
        with torch.no_grad():
            motion_tokens = model.generate(caption)
            motion = model.net.forward_decoder(motion_tokens.unsqueeze(0))
        motion_np = model.denormalize(motion.detach().cpu().numpy())
        if motion_np.ndim == 3:
            motion_np = motion_np[0]
        motion_tensor = torch.from_numpy(motion_np).float().to(args.device)
        motion_recovered = recover_from_ric(motion_tensor, joints_num=22)
        save_mp4_path = os.path.join(save_dir, f"scene_motion_{{idx}}.mp4")
        save_npy_path = os.path.join(save_dir, f"scene_motion_{{idx}}.npy")
        plot_3d_motion(save_mp4_path, t2m_kinematic_chain, motion_recovered.squeeze().cpu().numpy(), title=caption, fps=20, radius=4)
        np.save(save_npy_path, motion_np)
        print(f"[✅] 阶段{{idx}}已保存: {{save_mp4_path}}  {{save_npy_path}}")
        npy_list.append(save_npy_path)

    # 拼接所有动作
    print("\\n[INFO] 开始拼接所有阶段动作...")
    concat_motions(npy_list, save_dir, out_name="scene_based_motion")
    print("[✅] 场景基础动作生成完成！最终拼接动作和视频已生成。\\n")

if __name__ == "__main__":
    main()
'''
        
        # Save modified script
        script_path = 'motionllm/Motion-Agent/scene_motion_generator.py'
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return script_path

def main():
    """Main function"""
    print("🎯 Scene-to-Motion Generation System")
    print("=" * 50)
    
    # Initialize planner
    planner = SceneToMotionPlanner()
    
    # Get user input
    image_url = input("Enter scene image URL: ").strip()
    if not image_url:
        print("❌ Image URL cannot be empty!")
        return
    
    start_point = input("Enter start point (optional, press Enter to skip): ").strip()
    end_point = input("Enter end point (optional, press Enter to skip): ").strip()
    task_description = input("Enter task description (optional, press Enter to skip): ").strip()
    
    print("\n" + "=" * 50)
    
    # Step 1: Scene analysis
    print("📊 Step 1: Scene analysis")
    scene_analysis = planner.analyze_scene(image_url)
    print(f"Scene analysis result:\n{scene_analysis}\n")
    
    # Step 2: Route planning
    print("🗺️ Step 2: Route planning")
    route_plan = planner.plan_route_and_actions(
        scene_analysis, start_point, end_point, task_description
    )
    print(f"Route plan:\n{route_plan}\n")
    
    # Step 3: Extract motion descriptions
    print("📝 Step 3: Extract motion descriptions")
    motion_prompts = planner.extract_motion_prompts(route_plan)
    print("Extracted motion descriptions:")
    for i, prompt in enumerate(motion_prompts, 1):
        print(f"  Stage {i}: {prompt}")
    print()
    
    # Step 4: Generate motion
    print("🎬 Step 4: Generate motion sequence")
    result = planner.generate_motions_with_motionllm(motion_prompts)
    print(f"Generation result: {result}")
    
    print("\n🎉 Full pipeline completed!")

if __name__ == "__main__":
    main()