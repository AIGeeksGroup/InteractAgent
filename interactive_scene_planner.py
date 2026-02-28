import os
import sys
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
from interactive_qwenvl import QwenVLChat
from prompt_templates import PromptTemplates
from enhanced_motion_generator import EnhancedMotionGenerator


class InteractiveScenePlanner:
    """Interactive Scene Motion Planner"""
    
    def __init__(self):
        """Initialize interactive planner"""
        self.qwen_chat = None
        self.motion_generator = None
        self.prompt_templates = PromptTemplates()
        self.current_session = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "scene_image": None,
            "scene_description": None,
            "tasks": [],
            "generated_motions": []
        }
        
        print("🎉 Welcome to Interactive Scene Motion Planner!")
        print("=" * 60)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize various components"""
        print("🔧 Initializing system components...")
        
        # Initialize QwenVL
        try:
            self.qwen_chat = QwenVLChat()
            print("✅ QwenVL scene recognition module loaded")
        except Exception as e:
            print(f"❌ QwenVL initialization failed: {e}")
            return False
        
        # Initialize Enhanced Motion Generator
        try:
            from enhanced_motion_generator import EnhancedMotionGenerator
            self.motion_generator = EnhancedMotionGenerator(
                output_dir=f"./session_output/{self.current_session['session_id']}"
            )
            print("✅ Enhanced Motion Generator module loaded")
            print("🎯 Using Motion-Agent architecture with token concatenation")
        except Exception as e:
            print(f"❌ Enhanced Motion Generator initialization failed: {e}")
            print("💡 Please check if enhanced_motion_generator.py exists")
            return False
        
        print("🎯 System initialization completed!")
        return True
    
    def show_main_menu(self):
        """Show main menu"""
        print(f"\n" + "="*60)
        print(f"📋 Main Menu - Session ID: {self.current_session['session_id']}")
        print(f"="*60)
        print("1. 🖼️  Set Scene Image")
        print("2. 🔍 Analyze Current Scene")
        print("3. 📝 Create Motion Task")
        print("4. 🎬 Generate Motion Sequence")
        print("5. 📊 View Session Status")
        print("6. 💾 Save Session")
        print("7. 📁 Load Session")
        print("8. 🧹 Clean Old Files")
        print("9. ❓ Help Information")
        print("10. 🔁 Improve Previous Prompt (Qwen Reflection)")
        print("11. 🧪 Add Custom Prompt to Task List")
        print("0. 👋 Exit Program")
        print("-" * 60)
    
    def set_scene_image(self):
        """Set scene image"""
        print(f"\n🖼️  Set Scene Image")
        print("-" * 30)
        
        # Provide image setting options
        print("💡 Image Setting Options:")
        print("1. 🖼️  Single Image")
        print("2. 🎯 Dual-view Images (Bird's-eye view + First-person view)")
        print("3. 📷 View Current Images")
        print("4. 🗑️  Clear All Images")
        
        choice = input("Select operation (1-4): ").strip()
        
        if choice == "1":
            self._set_single_image()
        elif choice == "2":
            self._set_dual_view_images()
        elif choice == "3":
            self._show_current_images()
        elif choice == "4":
            self._clear_images()
        else:
            print("❌ Invalid selection")
    
    def _set_single_image(self):
        """Set single image"""
        print("\n📷 Set Single Image")
        print("-" * 20)
        
        # Provide sample image options
        print("💡 Sample Images:")
        print("1. VSCode icon (for testing)")
        print("2. Indoor scene image")
        print("3. Custom URL")
        
        choice = input("Select image source (1-3): ").strip()
        
        if choice == "1":
            image_url = "https://raw.githubusercontent.com/microsoft/vscode/main/resources/linux/code.png"
            image_type = "main"
        elif choice == "2":
            # image_url = "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800"
            image_url = "https://i.postimg.cc/MZL3ZLSL/image.png"
            image_type = "main"
        elif choice == "3":
            image_url = input("Enter image URL: ").strip()
            image_type = input("Enter image type identifier (e.g., main, scene): ").strip() or "main"
        else:
            print("❌ Invalid selection")
            return
        
        if image_url:
            try:
                self.qwen_chat.set_image(image_url, image_type)
                self.current_session["scene_image"] = image_url
                self.current_session["image_type"] = image_type
                print(f"✅ {image_type} image set: {image_url}")
            except Exception as e:
                print(f"❌ Failed to set image: {e}")
    
    def _set_dual_view_images(self):
        """Set dual-view images"""
        print("\n🎯 Set Dual-view Images")
        print("-" * 20)
        print("💡 Dual-view Image Description:")
        print("• Bird's-eye view: Top-down angle, shows overall room layout")
        print("• First-person view: Shot from eye level, shows current field of vision")
        print("• Combining both images provides more accurate spatial information")
        
        # Provide sample dual-view images
        print("\n💡 Sample Dual-view Images:")
        print("1. Use sample images (for testing)")
        print("2. Custom dual-view images")
        
        choice = input("Select image source (1-2): ").strip()
        
        if choice == "1":
            # Sample dual-view images (placeholders, need real dual-view images for actual use)
            birdview_url = "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800&h=600&fit=crop"
            egoview_url = "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800&h=600&fit=crop"
            print("⚠️  Note: These are sample images, please provide real bird's-eye and first-person view images for actual use")
        elif choice == "2":
            birdview_url = input("Enter bird's-eye view image URL: ").strip()
            egoview_url = input("Enter first-person view image URL: ").strip()
        else:
            print("❌ Invalid selection")
            return
        
        if birdview_url and egoview_url:
            try:
                self.qwen_chat.set_dual_view_images(birdview_url, egoview_url)
                self.current_session["birdview_image"] = birdview_url
                self.current_session["egoview_image"] = birdview_url
                self.current_session["image_type"] = "dual_view"
                print("✅ Dual-view images set successfully")
            except Exception as e:
                print(f"❌ Failed to set dual-view images: {e}")
        else:
            print("❌ Please provide both image URLs")
    
    def _show_current_images(self):
        """Show currently set images"""
        print("\n📷 Current Image Status")
        print("-" * 20)
        
        if self.qwen_chat:
            image_info = self.qwen_chat.list_images()
            print(image_info)
            
            if self.current_session.get("image_type") == "dual_view":
                print(f"🎯 Image Type: Dual-view Images")
                print(f"  • Bird's-eye view: {self.current_session.get('birdview_image', 'Not set')}")
                print(f"  • First-person view: {self.current_session.get('egoview_image', 'Not set')}")
            else:
                print(f"🖼️  Image Type: Single Image")
                print(f"  • Image: {self.current_session.get('scene_image', 'Not set')}")
        else:
            print("❌ QwenVL not initialized")
    
    def _clear_images(self):
        """Clear all images"""
        if self.qwen_chat:
            self.qwen_chat.clear_images()
            self.current_session["scene_image"] = None
            self.current_session["birdview_image"] = None
            self.current_session["egoview_image"] = None
            self.current_session["image_type"] = None
            print("✅ All images cleared")
        else:
            print("❌ QwenVL not initialized")
    
    def analyze_scene(self):
        """Analyze current scene"""
        print(f"\n🔍 Analyze Current Scene")
        print("-" * 30)
        
        # Check if images are set
        if not self.qwen_chat or self.qwen_chat.get_image_count() == 0:
            print("❌ Please set scene images first!")
            return
        
        print("🤔 Analyzing scene, please wait...")
        
        try:
            # Select appropriate analysis prompt based on image type
            image_type = self.current_session.get("image_type")
            
            if image_type == "dual_view":
                print("🎯 Using dual-view scene analysis...")
                analysis_prompt = self.prompt_templates.get_dual_view_scene_analysis_prompt()
            else:
                print("🖼️  Using single-view scene analysis...")
                analysis_prompt = self.prompt_templates.get_scene_analysis_prompt()
            
            # Call QwenVL to analyze scene
            scene_description = self.qwen_chat.ask(analysis_prompt)
            
            if "❌" in scene_description:
                print(f"❌ Scene analysis failed: {scene_description}")
                return
            
            self.current_session["scene_description"] = scene_description
            
            print(f"✅ Scene analysis completed!")
            print(f"\n📄 Scene Description:")
            print("-" * 40)
            print(scene_description)
            print("-" * 40)
            
            # Ask if analysis results should be saved
            save = input("\nSave scene analysis results? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                self._save_scene_analysis(scene_description)
            
        except Exception as e:
            print(f"❌ Scene analysis error: {e}")
    
    def create_motion_task(self):
        """Create motion task"""
        print(f"\n📝 Create Motion Task")
        print("-" * 30)
        
        # Get task description
        task_description = input("\nPlease describe the task you want to execute (e.g., walk from bed to sofa): ").strip()
        
        if not task_description:
            print("❌ Task description cannot be empty")
            return
        
        # Select complexity level
        print(f"\nSelect motion complexity level:")
        print("1. Simple (3-5 steps)")
        print("2. Medium (5-8 steps)")  
        print("3. Complex (8-12 steps)")
        
        complexity_choice = input("Please select (1-3): ").strip()
        complexity_map = {"1": "simple", "2": "medium", "3": "complex"}
        complexity = complexity_map.get(complexity_choice, "medium")
        
        print(f"\n🤔 Generating motion planning, please wait...")
        
        try:
            # Check if detailed scene description exists
            if self.current_session["scene_description"]:
                # Select appropriate motion planning prompt based on image type
                image_type = self.current_session.get("image_type")
                
                if image_type == "dual_view":
                    # Use dual-view motion planning
                    planning_prompt = self.prompt_templates.get_dual_view_motion_planning_prompt(
                        self.current_session["scene_description"],
                        task_description
                    )
                    print("🎯 Using dual-view motion planning...")
                else:
                    # Use dual-view motion planning
                    planning_prompt = self.prompt_templates.get_dual_view_motion_planning_prompt(
                        self.current_session["scene_description"],
                        task_description
                    )
                    print("🎯 Using dual-view motion planning...")
            else:
                # Require scene analysis first
                print("❌ Cannot create motion task: missing scene description")
                print("💡 Please select option 2 for scene analysis first, then create motion task after getting detailed scene information")
                return
            
            # Call QwenVL to generate motion planning
            planning_response = self.qwen_chat.ask(planning_prompt)
            
            if "❌" in planning_response:
                print(f"❌ Motion planning generation failed: {planning_response}")
                return
            
            # Parse motion steps
            motion_steps = self.prompt_templates.parse_motion_response(planning_response)
            
            if not motion_steps:
                print("❌ Failed to parse valid motion steps")
                print(f"Original response: {planning_response}")
                return
            
            print(f"\n✅ Motion planning completed!")
            print(f"📊 Step count: {len(motion_steps)}")
            
            if len(motion_steps) < 2:
                print(f"⚠️  Warning: Too few motion steps, may not be detailed enough")
            elif len(motion_steps) > 15:
                print(f"⚠️  Warning: Too many motion steps, may be overly complex")
            
            print(f"\n📋 Motion Steps:")
            print("-" * 40)
            for i, step in enumerate(motion_steps, 1):
                print(f"{i:2d}. {step}")
            print("-" * 40)
            
            # Show motion sequence summary
            print(f"\n🎯 Motion Sequence Summary:")
            print(f"   • Total steps: {len(motion_steps)}")
            print(f"   • Estimated duration: {len(motion_steps) * 3:.1f} seconds")
            print(f"   • Complexity: {'Simple' if len(motion_steps) <= 5 else 'Medium' if len(motion_steps) <= 10 else 'Complex'}")
            
            # Ask for task confirmation
            confirm = input("\nConfirm this motion planning? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                task_info = {
                    "task_id": len(self.current_session["tasks"]) + 1,
                    "description": task_description,
                    "complexity": complexity,
                    "motion_steps": motion_steps,
                    "raw_response": planning_response,
                    "step_count": len(motion_steps),
                    "timestamp": datetime.now().isoformat(),
                    "status": "planned"
                }
                
                self.current_session["tasks"].append(task_info)
                print(f"✅ Task added to session (Task ID: {task_info['task_id']})")
                
                # Provide next steps guidance
                print(f"\n💡 Next Steps:")
                print(f"   • Use option 4 to generate the actual motion sequence")
                print(f"   • Each step will be processed by MotionLLM")
                print(f"   • The system will combine all steps into a complete motion")
            else:
                print("❌ Task cancelled")
                
        except Exception as e:
            print(f"❌ Motion task creation failed: {e}")
    
    def generate_motion_sequence(self):
        """Generate motion sequence"""
        print(f"\n🎬 Generate Motion Sequence")
        print("-" * 30)
        
        if not self.current_session["tasks"]:
            print("❌ No available motion tasks, please create tasks first")
            return
        
        # Show available tasks
        print("📋 Available Tasks:")
        for task in self.current_session["tasks"]:
            status_icon = "✅" if task["status"] == "completed" else "⏳" if task["status"] == "generating" else "📝"
            print(f"  {status_icon} Task {task['task_id']}: {task['description']} ({len(task['motion_steps'])} steps)")
        
        # Select task
        try:
            task_id = int(input("\nSelect task ID to generate: ").strip())
            selected_task = None
            
            for task in self.current_session["tasks"]:
                if task["task_id"] == task_id:
                    selected_task = task
                    break
            
            if not selected_task:
                print("❌ Invalid task ID")
                return
            
            if selected_task["status"] == "completed":
                regenerate = input("This task is completed, regenerate? (y/n): ").strip().lower()
                if regenerate not in ['y', 'yes']:
                    return
            
            print(f"\n🎭 Starting task generation: {selected_task['description']}")
            print(f"📊 Contains {len(selected_task['motion_steps'])} motion steps")
            
            # Update task status
            selected_task["status"] = "generating"
            
            # Generate motion sequence
            sequence_name = f"task_{task_id}_{datetime.now().strftime('%H%M%S')}"
            
            print(f"\n⏳ Generating motion sequence, this may take several minutes...")
            start_time = time.time()
            
            # import pdb; pdb.set_trace()  # Yujia Debug
            result = self.motion_generator.generate_motion_sequence(
                selected_task["motion_steps"],
                sequence_name
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            if "error" in result:
                print(f"❌ Motion generation failed: {result['error']}")
                selected_task["status"] = "failed"
                return
            
            # Update task status
            selected_task["status"] = "completed"
            selected_task["generation_result"] = result
            selected_task["generation_time"] = generation_time
            
            # Record to session
            self.current_session["generated_motions"].append({
                "task_id": task_id,
                "sequence_name": sequence_name,
                "result": result,
                "generation_time": generation_time
            })
            
            print(f"\n🎉 Motion sequence generation completed!")
            print(f"⏱️  Generation time: {generation_time:.1f} seconds")
            print(f"📁 Output directory: {result['concat_result']['output_dir']}")
            print(f"🎬 Total frames: {result['total_frames']}")
            print(f"⭐ Average quality: {result['average_quality']:.3f}")
            
            # Show generated files
            print(f"\n📄 Generated files:")
            for i, motion in enumerate(result['individual_motions'], 1):
                print(f"  Step {i}: {motion['files']['video']}")
            
            if 'concat_result' in result:
                print(f"  Concatenated video: {result['concat_result'].get('final_video', 'N/A')}")
            
            # Step 2: Convert to SMPL-X format for Blender visualization
            print(f"\n🔄 Converting to SMPL-X format for Blender visualization...")
            try:
                # Get the NPY file path from the result
                npy_file_path = result['concat_result'].get('motion_data', None)
                if npy_file_path and os.path.exists(npy_file_path):
                    print(f"📊 Using joint data file: {npy_file_path}")
                    success = self._convert_to_smplx_format(npy_file_path, result['concat_result']['output_dir'])
                    if success:
                        print("✅ SMPL-X conversion completed successfully!")
                        print(f"📁 Check output directory: {result['concat_result']['output_dir']}")
                    else:
                        print("❌ SMPL-X conversion failed")
                else:
                    print("⚠️  Joint data file not found, skipping SMPL-X conversion")
                    print(f"💡 Expected file: {npy_file_path}")
            except Exception as e:
                print(f"❌ SMPL-X conversion error: {e}")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Motion generation error: {e}")
    
    def _convert_to_smplx_format(self, npy_path, output_dir):
        """Convert .npy motion file to SMPL-X format (from convert_motionllm_fresh.py)"""
        try:
            # Add trumans_utils to path # TODO: Yujia Debug
            # Use absolute path for trumans_utils
            current_dir = os.path.dirname(os.path.abspath(__file__))
            trumans_utils_path = os.path.join(current_dir, "trumans_utils")
            if trumans_utils_path not in sys.path:
                sys.path.insert(0, trumans_utils_path)
            print(f"📁 Added trumans_utils path: {trumans_utils_path}")

            
            # Import required modules
            import pickle as pkl
            import torch
            import numpy as np
            import yaml
            
            # Change to trumans_utils directory for imports and execution
            original_cwd = os.getcwd()
            os.chdir(trumans_utils_path)
            
            try:
                # Force reload modules
                if 'models.joints_to_smplx' in sys.modules:
                    del sys.modules['models.joints_to_smplx']
                if 'utils' in sys.modules:
                    del sys.modules['utils']
                
                # Try absolute import first
                # import pdb; pdb.set_trace() # Yujia Debug
                try:
                    from models.joints_to_smplx import joints_to_smpl, JointsToSMPLX
                    from utils import dotDict
                except ImportError:
                    # Fallback to direct file import
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("joints_to_smplx", os.path.join(trumans_utils_path, "models/joints_to_smplx.py"))
                    joints_to_smplx_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(joints_to_smplx_module)
                    
                    spec = importlib.util.spec_from_file_location("utils", os.path.join(trumans_utils_path, "utils.py"))
                    utils_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(utils_module)
                    
                    joints_to_smpl = joints_to_smplx_module.joints_to_smpl
                    JointsToSMPLX = joints_to_smplx_module.JointsToSMPLX
                    dotDict = utils_module.dotDict
                
                # Fix the hardcoded path in joints_to_smplx.py
                import smplx
                original_create = smplx.create
                
                def fixed_smplx_create(path, *args, **kwargs):
                    if path == './smpl_models':
                        # Use the absolute path from config
                        fixed_path = os.path.join(trumans_utils_path, "smpl_models")
                        print(f"🔧 Fixed hardcoded path: ./smpl_models -> {fixed_path}")
                        return original_create(fixed_path, *args, **kwargs)
                    else:
                        return original_create(path, *args, **kwargs)
                
                # Apply the monkey patch
                smplx.create = fixed_smplx_create
                
                print("✅ Successfully imported SMPL-X conversion modules")
                
                # Load configuration
                config_path = os.path.join(trumans_utils_path, "config/config_sample_synhsi.yaml")
                print(f"📋 Loading configuration: {config_path}")
                
                if not os.path.exists(config_path):
                    print(f"❌ Configuration file not found: {config_path}")
                    return False
                    
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                
                # Fix relative paths to absolute paths
                if 'ckpt_dir' in cfg:
                    cfg['ckpt_dir'] = os.path.join(trumans_utils_path, cfg['ckpt_dir'].lstrip('./'))
                if 'smpl_dir' in cfg:
                    cfg['smpl_dir'] = os.path.join(trumans_utils_path, cfg['smpl_dir'].lstrip('./'))
                if 'model' in cfg and 'model_smplx' in cfg['model'] and 'ckpt' in cfg['model']['model_smplx']:
                    cfg['model']['model_smplx']['ckpt'] = os.path.join(trumans_utils_path, cfg['model']['model_smplx']['ckpt'].lstrip('./'))
                if 'dataset' in cfg and 'folder' in cfg['dataset']:
                    cfg['dataset']['folder'] = os.path.join(trumans_utils_path, cfg['dataset']['folder'].lstrip('./'))
                
                cfg = dotDict(cfg)
                print(f"🔧 Fixed configuration paths to absolute paths")
                
                # Setup device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"🚀 Using device: {device}")
                
                # Load JointsToSMPLX model
                print("📦 Initializing JointsToSMPLX model...")
                model_path = cfg.model.model_smplx.ckpt
                if not os.path.exists(model_path):
                    print(f"❌ Model file not found: {model_path}")
                    return False
                    
                model_joints_to_smplx = JointsToSMPLX(**cfg.model.model_smplx)
                model_joints_to_smplx.load_state_dict(torch.load(model_path, map_location=device))
                model_joints_to_smplx.to(device)
                model_joints_to_smplx.eval()
                print("✅ Model loaded successfully.")
                
                # Load motion data - convert relative path to absolute path
                if not os.path.isabs(npy_path):
                    # Convert relative path to absolute path based on original working directory
                    abs_npy_path = os.path.join(original_cwd, npy_path)
                else:
                    abs_npy_path = npy_path
                
                print(f"📊 Loading motion data: {abs_npy_path}")
                points_all = np.load(abs_npy_path)
                print(f"📊 Loaded motion data shape: {points_all.shape}")
                
                # Extend 22 joints to 24 (as required by the model)
                if points_all.shape[1] == 22:
                    print("🔧 Extending 22 joints to 24 joints...")
                    left_wrist = points_all[:, 20, :]  # left_wrist
                    right_wrist = points_all[:, 21, :]  # right_wrist
                    
                    extended_joints = np.zeros((points_all.shape[0], 24, 3))
                    extended_joints[:, :22, :] = points_all
                    extended_joints[:, 22, :] = left_wrist
                    extended_joints[:, 23, :] = right_wrist
                    points_all = extended_joints
                    print(f"🔧 Extended data shape: {points_all.shape}")
                
                # Convert to SMPL-X parameters
                print("🔄 Converting joint positions to SMPL-X parameters...")
                keypoint_gene_torch = torch.from_numpy(points_all).float()
                keypoint_gene_torch = keypoint_gene_torch.reshape(-1, cfg.dataset.nb_joints * 3).to(device)
                
                pose, transl, left_hand, right_hand, vertices = joints_to_smpl(
                    model_joints_to_smplx,
                    keypoint_gene_torch,
                    cfg.dataset.joints_ind,
                    cfg.interp_s
                )
                print("✅ SMPL-X conversion completed!")
                
                # Prepare output data
                output_data = {
                    'transl': transl,
                    'body_pose': pose[:, 3:],
                    'global_orient': pose[:, :3],
                    'left_hand_pose': left_hand if left_hand is not None else np.zeros((transl.shape[0], 45)),
                    'right_hand_pose': right_hand if right_hand is not None else np.zeros((transl.shape[0], 45)),
                    'betas': np.zeros(10),
                    'gender': 'male',
                    'jaw_pose': np.zeros((transl.shape[0], 3)),
                    'expression': np.zeros((transl.shape[0], 10)),
                    'conversion_info': {
                        'method': 'interactive_motionllm_to_smplx',
                        'source_file': npy_path,
                        'num_frames': transl.shape[0],
                        'model_used': 'JointsToSMPLX',
                        'note': 'Integrated conversion from interactive MotionLLM to SMPL-X'
                    }
                }
                
                # Save .pkl file - convert relative path to absolute path
                if not os.path.isabs(output_dir):
                    abs_output_dir = os.path.join(original_cwd, output_dir)
                else:
                    abs_output_dir = output_dir
                
                pkl_output_path = os.path.join(abs_output_dir, "motionllm_interactive_smplx.pkl")
                print(f"💾 Saving SMPL-X data to: {pkl_output_path}")
                
                with open(pkl_output_path, 'wb') as f:
                    pkl.dump(output_data, f)
                
                # Create Blender test script
                blender_script_path = os.path.join(abs_output_dir, "test_motionllm_interactive.py")
                self._create_blender_test_script(blender_script_path, "motionllm_interactive_smplx.pkl")
                
                print(f"✅ SMPL-X conversion successful!")
                print(f"📁 SMPL-X file: {pkl_output_path}")
                print(f"📁 Blender test script: {blender_script_path}")
                
                return True
                
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
            

            
        except Exception as e:
            print(f"❌ SMPL-X conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_blender_test_script(self, script_path, pkl_filename):
        """Create Blender test script for the converted SMPL-X data"""
        script_content = f'''
import bpy
import sys
import os

# Add trumans_utils path
sys.path.append('../trumans_utils/visualize_smplx_motion')

# Import visualization function
from load_smplx_animatioin_clear import load_smplx_animation_new

def test_interactive_motionllm_data():
    """Test interactive MotionLLM to SMPL-X conversion"""
    
    pkl_file_path = "{pkl_filename}"
    
    print("🔍 Testing interactive MotionLLM to SMPL-X conversion...")
    
    # Find SMPL-X object
    smplx_mesh = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and ("SMPLX" in obj.name.upper() or "SMPL" in obj.name.upper()):
            smplx_mesh = obj
            break
    
    if smplx_mesh is None:
        print("❌ SMPL-X mesh object not found")
        return
    
    print(f"✅ Found SMPL-X mesh: {{smplx_mesh.name}}")
    
    # Setup scene
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 80
    
    # Load animation
    print("📥 Loading interactive MotionLLM animation data...")
    result = load_smplx_animation_new(pkl_file_path, smplx_mesh, load_hand=False)
    
    if result == {{'FINISHED'}}:
        print("🎉 Interactive MotionLLM animation loaded successfully!")
        print("\\nCheck points:")
        print("1. Legs should be pointing downward (not upward)")
        print("2. Character should be standing normally")
        print("3. Legs should have movement")
        print("4. Overall pose should be natural")
        print("5. Skeleton should align with model")
        print("6. This should solve all previous issues!")
    else:
        print("❌ Animation loading failed")

# Execute test
if __name__ == "__main__":
    test_interactive_motionllm_data()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Blender test script created: {script_path}")
    
    def show_session_status(self):
        """Show session status"""
        print(f"\n📊 Session Status")
        print("-" * 30)
        
        session = self.current_session
        
        print(f"🆔 Session ID: {session['session_id']}")
        print(f"🖼️  Scene Image: {'Set' if session['scene_image'] else 'Not set'}")
        print(f"🔍 Scene Analysis: {'Completed' if session['scene_description'] else 'Not completed'}")
        print(f"📝 Task Count: {len(session['tasks'])}")
        print(f"🎬 Generated Motions: {len(session['generated_motions'])}")
        
        if session['tasks']:
            print(f"\n📋 Task List:")
            for task in session['tasks']:
                status_map = {
                    "planned": "📝 Planned",
                    "generating": "⏳ Generating", 
                    "completed": "✅ Completed",
                    "failed": "❌ Failed"
                }
                status = status_map.get(task['status'], task['status'])
                print(f"  Task {task['task_id']}: {task['description']} - {status}")
    
    def _save_scene_analysis(self, scene_description):
        """Save scene analysis results"""
        try:
            analysis_dir = f"./session_output/{self.current_session['session_id']}/scene_analysis"
            os.makedirs(analysis_dir, exist_ok=True)
            
            analysis_file = os.path.join(analysis_dir, "scene_analysis.txt")
            
            # Ensure scene_description is a string
            if isinstance(scene_description, list):
                scene_description = "\n".join(str(item) for item in scene_description)
            else:
                scene_description = str(scene_description)
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(f"Scene Analysis Results\n")
                f.write(f"=" * 40 + "\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Image: {self.current_session.get('scene_image', 'Not set')}\n")
                f.write(f"\nAnalysis Content:\n")
                f.write(scene_description)
            
            print(f"✅ Scene analysis saved: {analysis_file}")
            
        except Exception as e:
            print(f"❌ Failed to save scene analysis: {e}")

    def improve_previous_prompt(self):
        """Input previous prompt + video URL, let Qwen reflect and produce an improved prompt"""
        print("\n🔁 Improve Previous Prompt (Qwen Reflection)")
        print("-" * 40)
        if not self.qwen_chat:
            print("❌ QwenVL not initialized")
            return

        try:
            # Ask for original task intent
            original_task = input("Please enter your original task goal (e.g., walk to the yellow chair): ").strip()
            if not original_task:
                print("❌ Original task cannot be empty")
                return

            # Step 1: Collect previous attempts information
            print("\n📚 Step 1: Collecting previous attempts information...")
            print("Please provide information about your previous attempts (you can leave empty if not available):")
            print("💡 Qwen will automatically analyze the videos to identify problems and deviations!")
            
            # First attempt
            print("\n--- First Attempt ---")
            first_prompt = input("First MotionLLM instruction (A person...; A person...): ").strip()
            first_video_url = input("First video URL (leave empty if none): ").strip()
            
            # Second attempt  
            print("\n--- Second Attempt ---")
            second_prompt = input("Second MotionLLM instruction (A person...; A person...): ").strip()
            second_video_url = input("Second video URL (leave empty if none): ").strip()
            
            # Current/latest attempt
            print("\n--- Current/Latest Attempt ---")
            current_prompt = input("Current MotionLLM instruction (A person...; A person...): ").strip()
            current_video_url = input("Current video URL (leave empty if none): ").strip()

            # Success experience learning
            print("\n🎯 Success Experience Learning (Optional)")
            print("Please provide successful examples that achieved the goal well:")
            print("💡 These will help Qwen learn what works best!")
            
            success_examples = []
            success_count = 0
            while True:
                success_count += 1
                print(f"\n--- Success Example {success_count} ---")
                success_prompt = input(f"Successful MotionLLM instruction {success_count} (A person...; A person...): ").strip()
                if not success_prompt:
                    break
                success_video_url = input(f"Success video URL {success_count} (leave empty if none): ").strip()
                success_examples.append({
                    'prompt': success_prompt,
                    'video_url': success_video_url
                })
                print(f"✅ Added success example {success_count}")
                
                # Ask if user wants to add more
                more_success = input("Add another success example? (y/n): ").strip().lower()
                if more_success not in ['y', 'yes']:
                    break
            
            print(f"📚 Total success examples collected: {len(success_examples)}")

            # Temporarily set a default image to avoid Qwen requiring images
            if self.qwen_chat.get_image_count() == 0:
                print("📷 Temporarily setting default image for text conversation...")
                self.qwen_chat.set_image("https://raw.githubusercontent.com/microsoft/vscode/main/resources/linux/code.png", "temp")

            # Step 2: Multi-step analysis and reflection for EACH attempt
            print("\n🔍 Step 2: Multi-step analysis and reflection for EACH attempt...")
            
            # Build comprehensive context for Qwen
            attempts_info = []
            if first_prompt:
                attempts_info.append(f"First attempt: {first_prompt}")
                if first_video_url:
                    attempts_info.append(f"First video: {first_video_url}")
            
            if second_prompt:
                attempts_info.append(f"Second attempt: {second_prompt}")
                if second_video_url:
                    attempts_info.append(f"Second video: {second_video_url}")
            
            if current_prompt:
                attempts_info.append(f"Current attempt: {current_prompt}")
                if current_video_url:
                    attempts_info.append(f"Current video: {current_video_url}")
            
            attempts_summary = "\n".join(attempts_info) if attempts_info else "No previous attempts available"
            
            # Add success examples to summary
            if success_examples:
                success_info = ["\n🎯 SUCCESS EXAMPLES:"]
                for i, example in enumerate(success_examples, 1):
                    success_info.append(f"Success {i}: {example['prompt']}")
                    if example['video_url']:
                        success_info.append(f"Success {i} video: {example['video_url']}")
                attempts_summary += "\n" + "\n".join(success_info)
            

            
            # Analyze each attempt individually with three reflections
            all_reflections = {}
            
            # Analyze First Attempt (if exists)
            if first_prompt:
                print(f"\n🔍 Analyzing FIRST ATTEMPT: {first_prompt[:50]}...")
                print("-" * 60)
                
                # First reflection for first attempt
                print("\n🤔 First reflection: Analyzing first attempt prompt-video relationship...")
                first_attempt_context = (
                    "You are an expert motion analysis specialist. Analyze this specific attempt in detail."
                )
                first_attempt_question = (
                    f"Task: {original_task}\n\n"
                    f"FAILED ATTEMPT: {first_prompt}\n"
                    f"Video: {first_video_url if first_video_url else 'No video'}\n\n"
                    "🎯 BRIEF FAILURE ANALYSIS:\n\n"
                    "In 2-3 sentences, identify:\n"
                    "1. What went wrong with each action?\n"
                    "2. What language confused MotionLLM?\n"
                    "3. How should the prompt be fixed?\n\n"
                    "CRITICAL: MotionLLM cannot understand angles, scene objects, or abstract actions.\n"
                    "Keep your response concise and focused."
                )
                
                first_attempt_first_reflection = self.qwen_chat.ask(f"Context: {first_attempt_context}\n\nQuestion: {first_attempt_question}")
                print("\n📊 First attempt - First reflection results:")
                print("-" * 40)
                print(first_attempt_first_reflection)
                print("-" * 40)
                
                # Second reflection for first attempt
                print("\n🔍 Second reflection: Learning from first attempt patterns...")
                first_attempt_second_question = (
                    f"Task: {original_task}\n\n"
                    f"Failed prompt: {first_prompt}\n"
                    f"Analysis: {first_attempt_first_reflection}\n\n"
                    "🎯 STRATEGIC IMPROVEMENT:\n\n"
                    "In 2-3 sentences, identify:\n"
                    "1. What language patterns should be avoided?\n"
                    "2. What would make the actions more goal-oriented?\n"
                    "3. How should the sequence be restructured?\n\n"
                    "Keep your response concise and focused."
                )
                
                first_attempt_second_reflection = self.qwen_chat.ask(f"Context: {first_attempt_context}\n\nQuestion: {first_attempt_second_question}")
                print("\n📊 First attempt - Second reflection results:")
                print("-" * 40)
                print(first_attempt_second_reflection)
                print("-" * 40)
                
                # Third reflection for first attempt
                print("\n🎯 Third reflection: Generating improved prompt for first attempt...")
                first_attempt_third_question = (
                    f"Task: {original_task}\n\n"
                    f"Failed prompt: {first_prompt}\n"
                    f"Problems: {first_attempt_first_reflection}\n"
                    f"Strategy: {first_attempt_second_reflection}\n\n"
                    "🎯 GENERATE IMPROVED PROMPT:\n\n"
                    "Create a better prompt that:\n"
                    "1. Fixes the identified problems\n"
                    "2. Uses concrete, simple language\n"
                    "3. Makes actions goal-oriented\n\n"
                    "CRITICAL: No angles, no scene objects, no abstract actions.\n"
                    "Use 'straightly' for walk forward actions.\n\n"
                    "Output: Only 'A person [action]' semicolon-separated strings."
                )
                
                first_attempt_third_reflection = self.qwen_chat.ask(f"Context: {first_attempt_context}\n\nQuestion: {first_attempt_third_question}")
                print("\n📊 First attempt - Third reflection results:")
                print("-" * 40)
                print(first_attempt_third_reflection)
                print("-" * 40)
                
                # Store first attempt reflections
                all_reflections['first_attempt'] = {
                    'first_reflection': first_attempt_first_reflection,
                    'second_reflection': first_attempt_second_reflection,
                    'third_reflection': first_attempt_third_reflection
                }
            
            # Analyze Second Attempt (if exists)
            if second_prompt:
                print(f"\n🔍 Analyzing SECOND ATTEMPT: {second_prompt[:50]}...")
                print("-" * 60)
                
                # First reflection for second attempt
                print("\n🤔 First reflection: Analyzing second attempt prompt-video relationship...")
                second_attempt_context = (
                    "You are an expert motion analysis specialist. Analyze this specific attempt in detail."
                )
                second_attempt_question = (
                    f"User's task goal: {original_task}\n\n"
                    f"Second attempt prompt: {second_prompt}\n"
                    f"Second attempt video: {second_video_url if second_video_url else 'No video provided'}\n\n"
                    "🎯 ATOMIC ACTION ANALYSIS REQUIRED:\n\n"
                    "You have access to the video. Please analyze each atomic action step by step:\n\n"
                    "**STEP 1: ATOMIC ACTION MAPPING**\n"
                    "For each action in the prompt, analyze what actually happened in the video:\n"
                    "1. **Action 1**: [prompt action] → What actually happened in video?\n"
                    "2. **Action 2**: [prompt action] → What actually happened in video?\n"
                    "3. **Action 3**: [prompt action] → What actually happened in video?\n"
                    "4. Continue for each action...\n\n"
                    "**STEP 2: POSITION TRACKING**\n"
                    "For each action, track the person's position:\n"
                    "- Starting position before action\n"
                    "- Final position after action\n"
                    "- How much did the person actually move?\n"
                    "- Did they move in the intended direction?\n\n"
                    "**STEP 3: GOAL ALIGNMENT ANALYSIS**\n"
                    "For each action, analyze:\n"
                    "- Did this action bring the person closer to the target goal?\n"
                    "- Did this action move them away from the target goal?\n"
                    "- How effective was this action in achieving the overall goal?\n\n"
                    "**STEP 4: PROMPT LANGUAGE PROBLEMS**\n"
                    "Identify what in the prompt language caused the mismatch:\n"
                    "- Which words confused MotionLLM?\n"
                    "- What should be changed to make the action more precise?\n"
                    "- What concrete, simple language would work better?\n\n"
                    "CRITICAL MOTIONLLM LIMITATIONS:\n"
                    "1. MotionLLM CANNOT understand specific angles (15 degrees, 90 degrees, etc.)\n"
                    "2. MotionLLM CAN understand distances (meters, steps) - this is OK\n"
                    "3. MotionLLM CANNOT understand abstract actions (check position, adjust subtly)\n"
                    "4. MotionLLM CANNOT see or understand scene objects (yellow chair, table, wall) - it has NO scene awareness\n"
                    "5. MotionLLM CANNOT understand adverbs (slightly, smoothly, steadily)\n"
                    "6. MotionLLM CAN ONLY understand simple, concrete physical actions\n"
                    "7. MotionLLM CANNOT understand directional references to objects (towards the chair, facing the wall)\n"
                    "8. MotionLLM REQUIRES 'straightly' for walk forward actions (e.g., 'walk forward 3 meters straightly')\n\n"
                    "**STEP 5: IMPROVEMENT SUGGESTIONS**\n"
                    "Based on your analysis, suggest specific improvements:\n"
                    "- What should each action instruction be changed to?\n"
                    "- How can we make each action more likely to achieve the goal?\n"
                    "- What simple, concrete language will work better?\n\n"
                    "Focus on making each atomic action more precise and goal-oriented."
                )
                
                second_attempt_first_reflection = self.qwen_chat.ask(f"Context: {second_attempt_context}\n\nQuestion: {second_attempt_question}")
                print("\n📊 Second attempt - First reflection results:")
                print("-" * 40)
                print(second_attempt_first_reflection)
                print("-" * 40)
                
                # Second reflection for second attempt
                print("\n🔍 Second reflection: Learning from second attempt patterns...")
                second_attempt_second_question = (
                    f"User's task goal: {original_task}\n\n"
                    f"Second attempt prompt: {second_prompt}\n"
                    f"Second attempt atomic analysis: {second_attempt_first_reflection}\n\n"
                    "Based on your atomic action analysis above, now think strategically:\n\n"
                    "**STRATEGIC PATTERN ANALYSIS:**\n\n"
                    "1. **What worked well in the prompt language?**\n"
                    "   - Which action instructions were clear and effective?\n"
                    "   - What language patterns helped MotionLLM understand the intent?\n\n"
                    "2. **What caused problems in the prompt language?**\n"
                    "   - Which words or phrases confused MotionLLM?\n"
                    "   - What made actions imprecise or ineffective?\n\n"
                    "3. **Goal achievement analysis:**\n"
                    "   - Which actions brought the person closer to the target?\n"
                    "   - Which actions moved them away from the target?\n"
                    "   - What was the overall effectiveness of the sequence?\n\n"
                    "4. **Action sequence optimization:**\n"
                    "   - What would be a better order of actions?\n"
                    "   - How can we make each action more goal-oriented?\n"
                    "   - What simple, concrete language patterns work best?\n\n"
                    "5. **Distance and direction optimization:**\n"
                    "   - How can we make distance specifications more effective?\n"
                    "   - How can we make direction instructions clearer?\n"
                    "   - What avoids the problems we identified?\n\n"
                    "Focus on creating a more effective action sequence that will actually reach the target goal."
                )
                
                second_attempt_second_reflection = self.qwen_chat.ask(f"Context: {second_attempt_context}\n\nQuestion: {second_attempt_second_question}")
                print("\n📊 Second attempt - Second reflection results:")
                print("-" * 40)
                print(second_attempt_second_reflection)
                print("-" * 40)
                
                # Third reflection for second attempt
                print("\n🎯 Third reflection: Generating improved prompt for second attempt...")
                second_attempt_third_question = (
                    f"User's task goal: {original_task}\n\n"
                    f"Second attempt prompt: {second_prompt}\n"
                    f"Second attempt atomic analysis: {second_attempt_first_reflection}\n"
                    f"Second attempt strategic analysis: {second_attempt_second_reflection}\n\n"
                    "🎯 IMPROVED PROMPT GENERATION:\n\n"
                    "Based on your atomic action analysis and strategic insights, generate an improved prompt that:\n\n"
                    "1. **Fixes the specific problems you identified** in each atomic action\n"
                    "2. **Uses the language patterns that worked well**\n"
                    "3. **Makes each action more goal-oriented** to reach the target\n"
                    "4. **Uses simple, concrete language** that MotionLLM can understand\n"
                    "5. **Avoids the problematic language patterns** you identified\n\n"
                    "CRITICAL MOTIONLLM LIMITATIONS:\n"
                    "1. MotionLLM CANNOT understand specific angles (15 degrees, 90 degrees, etc.)\n"
                    "2. MotionLLM CAN understand distances (meters, steps) - this is OK\n"
                    "3. MotionLLM CANNOT understand abstract actions (check position, adjust subtly)\n"
                    "4. MotionLLM CANNOT see or understand scene objects (yellow chair, table, wall) - it has NO scene awareness\n"
                    "5. MotionLLM CANNOT understand adverbs (slightly, smoothly, steadily)\n"
                    "6. MotionLLM CAN ONLY understand simple, concrete physical actions\n"
                    "7. MotionLLM CANNOT understand directional references to objects (towards the chair, facing the wall)\n"
                    "8. MotionLLM REQUIRES 'straightly' for walk forward actions (e.g., 'walk forward 3 meters straightly')\n\n"
                    "**Your task:** Generate an improved prompt that will actually achieve the goal.\n"
                    "**Output format:** Only 'A person [action]' semicolon-separated strings.\n"
                    "**Focus:** Make each action more likely to bring the person closer to the target goal."
                )
                
                second_attempt_third_reflection = self.qwen_chat.ask(f"Context: {second_attempt_context}\n\nQuestion: {second_attempt_third_question}")
                print("\n📊 Second attempt - Third reflection results:")
                print("-" * 40)
                print(second_attempt_third_reflection)
                print("-" * 40)
                
                # Store second attempt reflections
                all_reflections['second_attempt'] = {
                    'first_reflection': second_attempt_first_reflection,
                    'second_reflection': second_attempt_second_reflection,
                    'third_reflection': second_attempt_third_reflection
                }
            
            # Analyze Current Attempt (if exists)
            if current_prompt:
                print(f"\n🔍 Analyzing CURRENT ATTEMPT: {current_prompt[:50]}...")
                print("-" * 60)
                
                # First reflection for current attempt
                print("\n🤔 First reflection: Analyzing current attempt prompt-video relationship...")
                current_attempt_context = (
                    "You are an expert motion analysis specialist. Analyze this specific attempt in detail."
                )
                current_attempt_question = (
                    f"User's task goal: {original_task}\n\n"
                    f"Current attempt prompt: {current_prompt}\n"
                    f"Current attempt video: {current_video_url if current_video_url else 'No video provided'}\n\n"
                    "🎯 ATOMIC ACTION ANALYSIS REQUIRED:\n\n"
                    "You have access to the video. Please analyze each atomic action step by step:\n\n"
                    "**STEP 1: ATOMIC ACTION MAPPING**\n"
                    "For each action in the prompt, analyze what actually happened in the video:\n"
                    "1. **Action 1**: [prompt action] → What actually happened in video?\n"
                    "2. **Action 2**: [prompt action] → What actually happened in video?\n"
                    "3. **Action 3**: [prompt action] → What actually happened in video?\n"
                    "4. Continue for each action...\n\n"
                    "**STEP 2: POSITION TRACKING**\n"
                    "For each action, track the person's position:\n"
                    "- Starting position before action\n"
                    "- Final position after action\n"
                    "- How much did the person actually move?\n"
                    "- Did they move in the intended direction?\n\n"
                    "**STEP 3: GOAL ALIGNMENT ANALYSIS**\n"
                    "For each action, analyze:\n"
                    "- Did this action bring the person closer to the target goal?\n"
                    "- Did this action move them away from the target goal?\n"
                    "- How effective was this action in achieving the overall goal?\n\n"
                    "**STEP 4: PROMPT LANGUAGE PROBLEMS**\n"
                    "Identify what in the prompt language caused the mismatch:\n"
                    "- Which words confused MotionLLM?\n"
                    "- What should be changed to make the action more precise?\n"
                    "- What concrete, simple language would work better?\n\n"
                    "CRITICAL MOTIONLLM LIMITATIONS:\n"
                    "1. MotionLLM CANNOT understand specific angles (15 degrees, 90 degrees, etc.)\n"
                    "2. MotionLLM CAN understand distances (meters, steps) - this is OK\n"
                    "3. MotionLLM CANNOT understand abstract actions (check position, adjust subtly)\n"
                    "4. MotionLLM CANNOT see or understand scene objects (yellow chair, table, wall) - it has NO scene awareness\n"
                    "5. MotionLLM CANNOT understand adverbs (slightly, smoothly, steadily)\n"
                    "6. MotionLLM CAN ONLY understand simple, concrete physical actions\n"
                    "7. MotionLLM CANNOT understand directional references to objects (towards the chair, facing the wall)\n"
                    "8. MotionLLM REQUIRES 'straightly' for walk forward actions (e.g., 'walk forward 3 meters straightly')\n\n"
                    "**STEP 5: IMPROVEMENT SUGGESTIONS**\n"
                    "Based on your analysis, suggest specific improvements:\n"
                    "- What should each action instruction be changed to?\n"
                    "- How can we make each action more likely to achieve the goal?\n"
                    "- What simple, concrete language will work better?\n\n"
                    "Focus on making each atomic action more precise and goal-oriented."
                )
                
                current_attempt_first_reflection = self.qwen_chat.ask(f"Context: {current_attempt_context}\n\nQuestion: {current_attempt_question}")
                print("\n📊 Current attempt - First reflection results:")
                print("-" * 40)
                print(current_attempt_first_reflection)
                print("-" * 40)
                
                # Second reflection for current attempt
                print("\n🔍 Second reflection: Learning from current attempt patterns...")
                current_attempt_second_question = (
                    f"User's task goal: {original_task}\n\n"
                    f"Current attempt prompt: {current_prompt}\n"
                    f"Current attempt atomic analysis: {current_attempt_first_reflection}\n\n"
                    "Based on your atomic action analysis above, now think strategically:\n\n"
                    "**STRATEGIC PATTERN ANALYSIS:**\n\n"
                    "1. **What worked well in the prompt language?**\n"
                    "   - Which action instructions were clear and effective?\n"
                    "   - What language patterns helped MotionLLM understand the intent?\n\n"
                    "2. **What caused problems in the prompt language?**\n"
                    "   - Which words or phrases confused MotionLLM?\n"
                    "   - What made actions imprecise or ineffective?\n\n"
                    "3. **Goal achievement analysis:**\n"
                    "   - Which actions brought the person closer to the target?\n"
                    "   - Which actions moved them away from the target?\n"
                    "   - What was the overall effectiveness of the sequence?\n\n"
                    "4. **Action sequence optimization:**\n"
                    "   - What would be a better order of actions?\n"
                    "   - How can we make each action more goal-oriented?\n"
                    "   - What simple, concrete language patterns work best?\n\n"
                    "5. **Distance and direction optimization:**\n"
                    "   - How can we make distance specifications more effective?\n"
                    "   - How can we make direction instructions clearer?\n"
                    "   - What avoids the problems we identified?\n\n"
                    "Focus on creating a more effective action sequence that will actually reach the target goal."
                )
                
                current_attempt_second_reflection = self.qwen_chat.ask(f"Context: {current_attempt_context}\n\nQuestion: {current_attempt_second_question}")
                print("\n📊 Current attempt - Second reflection results:")
                print("-" * 40)
                print(current_attempt_second_reflection)
                print("-" * 40)
                
                # Third reflection for current attempt
                print("\n🎯 Third reflection: Generating improved prompt for current attempt...")
                current_attempt_third_question = (
                    f"User's task goal: {original_task}\n\n"
                    f"Current attempt prompt: {current_prompt}\n"
                    f"Current attempt atomic analysis: {current_attempt_first_reflection}\n"
                    f"Current attempt strategic analysis: {current_attempt_second_reflection}\n\n"
                    "🎯 IMPROVED PROMPT GENERATION:\n\n"
                    "Based on your atomic action analysis and strategic insights, generate an improved prompt that:\n\n"
                    "1. **Fixes the specific problems you identified** in each atomic action\n"
                    "2. **Uses the language patterns that worked well**\n"
                    "3. **Makes each action more goal-oriented** to reach the target\n"
                    "4. **Uses simple, concrete language** that MotionLLM can understand\n"
                    "5. **Avoids the problematic language patterns** you identified\n\n"
                    "CRITICAL MOTIONLLM LIMITATIONS:\n"
                    "1. MotionLLM CANNOT understand specific angles (15 degrees, 90 degrees, etc.)\n"
                    "2. MotionLLM CAN understand distances (meters, steps) - this is OK\n"
                    "3. MotionLLM CANNOT understand abstract actions (check position, adjust subtly)\n"
                    "4. MotionLLM CANNOT see or understand scene objects (yellow chair, table, wall) - it has NO scene awareness\n"
                    "5. MotionLLM CANNOT understand adverbs (slightly, smoothly, steadily)\n"
                    "6. MotionLLM CAN ONLY understand simple, concrete physical actions\n"
                    "7. MotionLLM CANNOT understand directional references to objects (towards the chair, facing the wall)\n"
                    "8. MotionLLM REQUIRES 'straightly' for walk forward actions (e.g., 'walk forward 3 meters straightly')\n\n"
                    "**Your task:** Generate an improved prompt that will actually achieve the goal.\n"
                    "**Output format:** Only 'A person [action]' semicolon-separated strings.\n"
                    "**Focus:** Make each action more likely to bring the person closer to the target goal."
                )
                
                current_attempt_third_reflection = self.qwen_chat.ask(f"Context: {current_attempt_context}\n\nQuestion: {current_attempt_third_question}")
                print("\n📊 Current attempt - Third reflection results:")
                print("-" * 40)
                print(current_attempt_third_reflection)
                print("-" * 40)
                
                # Store current attempt reflections
                all_reflections['current_attempt'] = {
                    'first_reflection': current_attempt_first_reflection,
                    'second_reflection': current_attempt_second_reflection,
                    'third_reflection': current_attempt_third_reflection
                }
            
            # Analyze success examples (if provided)
            if success_examples:
                print(f"\n🎯 Analyzing SUCCESS EXAMPLES to learn what works...")
                print("-" * 80)
                
                success_analysis = []
                for i, example in enumerate(success_examples, 1):
                    print(f"\n🔍 Analyzing Success Example {i}: {example['prompt'][:50]}...")
                    
                    success_context = (
                        "You are an expert motion analysis specialist. Analyze this SUCCESSFUL example to understand what made it work."
                    )
                    
                    success_question = (
                        f"Task: {original_task}\n\n"
                        f"SUCCESS EXAMPLE {i}: {example['prompt']}\n"
                        f"Video: {example['video_url'] if example['video_url'] else 'No video'}\n\n"
                        "🎯 BRIEF SUCCESS ANALYSIS:\n\n"
                        "In 2-3 sentences, identify:\n"
                        "1. What language patterns made this prompt effective?\n"
                        "2. What action structure worked well?\n"
                        "3. What should be replicated in future prompts?\n\n"
                        "Keep your response concise and focused."
                    )
                    
                    success_reflection = self.qwen_chat.ask(f"Context: {success_context}\n\nQuestion: {success_question}")
                    print(f"\n📊 Success Example {i} - Analysis results:")
                    print("-" * 40)
                    print(success_reflection)
                    print("-" * 40)
                    
                    success_analysis.append({
                        'example_number': i,
                        'prompt': example['prompt'],
                        'video_url': example['video_url'],
                        'analysis': success_reflection
                    })
                
                # Store success analysis
                all_reflections['success_examples'] = success_analysis
                print(f"\n✅ Success examples analysis completed!")
            
            # Final comprehensive reflection combining all attempts
            print(f"\n🎯 FINAL COMPREHENSIVE REFLECTION: Combining insights from all attempts...")
            print("-" * 80)
            
            final_reflection_context = (
                "You are an expert motion prompt engineer. You have analyzed multiple attempts individually "
                "and success examples to understand what works. Now create the ultimate optimized prompt "
                "that incorporates ALL learned insights, prioritizing successful patterns."
            )
            
            # Build success examples analysis text
            success_analysis_text = ""
            if 'success_examples' in all_reflections:
                success_lines = []
                for analysis in all_reflections['success_examples']:
                    success_lines.append(f"Success {analysis.get('example_number')}: {analysis.get('analysis')}")
                success_analysis_text = "\n".join(success_lines)
            
            # Prepare problems text outside of f-string to avoid syntax issues
            problems_text = chr(10).join([f'{attempt_name}: {reflections.get("first_reflection", "")[:100]}...' for attempt_name, reflections in all_reflections.items() if attempt_name != 'success_examples'])
            
            final_reflection_question = (
                f"Task: {original_task}\n\n"
                f"Key insights from analysis:\n"
                f"- Problems: {problems_text}\n"
                f"- Success patterns: {success_analysis_text[:200]}...\n\n"
                "🎯 GENERATE ULTIMATE PROMPT:\n\n"
                "Create the best prompt that:\n"
                "1. Avoids all identified problems\n"
                "2. Uses successful language patterns\n"
                "3. Makes actions goal-oriented\n\n"
                "CRITICAL: No angles, no scene objects, no abstract actions.\n"
                "Use 'straightly' for walk forward actions.\n\n"
                "Output: Only 'A person [action]' semicolon-separated strings."
            )
            
            final_reflection = self.qwen_chat.ask(f"Context: {final_reflection_context}\n\nQuestion: {final_reflection_question}")
            print("\n🎉 FINAL COMPREHENSIVE REFLECTION RESULTS:")
            print("-" * 80)
            print(final_reflection)
            print("-" * 80)
            
            # Final step: Parse and validate the improved instructions
            print("\n✅ Final step: Parsing and validating improved instructions...")
            steps = self.prompt_templates.parse_motion_response(final_reflection)
            if not steps:
                print("❌ Failed to parse valid action strings. Original output:\n")
                print(final_reflection)
                return

            improved_prompt = "; ".join(steps)
            print("\n🎉 Final improved action instructions:")
            print("-" * 40)
            print(improved_prompt)
            print("-" * 40)
            print(f"📊 Total action steps: {len(steps)}")

            # Optionally save to current session for later generation
            save_choice = input("\nSave this instruction as a new task for generation? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                task_info = {
                    "task_id": len(self.current_session["tasks"]) + 1,
                    "description": f"Multi-reflection refined: {original_task}",
                    "complexity": "medium",
                    "motion_steps": steps,
                    "raw_response": final_reflection,
                    "step_count": len(steps),
                    "timestamp": datetime.now().isoformat(),
                    "status": "planned",
                    "reflection_history": all_reflections
                }
                self.current_session["tasks"].append(task_info)
                print(f"✅ Saved as task (Task ID: {task_info['task_id']}), can be generated directly in menu 4")
                print(f"📝 Task description: {original_task}")
                print(f"📊 Action step count: {len(steps)}")
                print(f"💡 This task includes comprehensive reflection analysis for better results")
            
            # Clean up temporary images AFTER user input
            if self.qwen_chat.get_image_count() > 0:
                self.qwen_chat.clear_images()
                
        except Exception as e:
            print(f"❌ Improvement process error: {e}")
            import traceback
            traceback.print_exc()

    def test_custom_prompt(self):
        """Add custom prompt to task list for later generation"""
        print("\n🧪 Add Custom Prompt to Task List")
        print("-" * 40)

        try:
            # Get custom prompt from user
            print("Please enter your custom MotionLLM prompt:")
            print("Format: 'A person [action]; A person [action]; ...'")
            print("Example: 'A person turns to the right; A person walks forward 3 meters; A person sits down'")
            print("-" * 40)
            
            custom_prompt = input("Your custom prompt: ").strip()
            if not custom_prompt:
                print("❌ Prompt cannot be empty")
                return

            # Parse the prompt to get motion steps
            print("\n🔍 Parsing your custom prompt...")
            steps = self.prompt_templates.parse_motion_response(custom_prompt)
            if not steps:
                print("❌ Failed to parse valid motion steps")
                print(f"Original prompt: {custom_prompt}")
                return

            print(f"✅ Successfully parsed {len(steps)} motion steps:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")

            # Add to task list
            task_info = {
                "task_id": len(self.current_session["tasks"]) + 1,
                "description": f"Custom prompt: {custom_prompt[:50]}...",
                "complexity": "custom",
                "motion_steps": steps,
                "raw_response": custom_prompt,
                "step_count": len(steps),
                "timestamp": datetime.now().isoformat(),
                "status": "planned"
            }
            
            self.current_session["tasks"].append(task_info)
            print(f"\n✅ Custom prompt added as Task ID: {task_info['task_id']}")
            print(f"📝 Description: {custom_prompt[:50]}...")
            print(f"📊 Action steps: {len(steps)}")
            print(f"💡 Now you can use option 4 to generate motion for this task!")
                
        except Exception as e:
            print(f"❌ Failed to add custom prompt: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Run main program"""
        if not self.qwen_chat or not self.motion_generator:
            print("❌ System initialization failed, program exiting")
            return
        
        print(f"\n💡 Tip: It's recommended to set scene images first, then analyze the scene, then create motion tasks")
        
        while True:
            try:
                self.show_main_menu()
                choice = input("\nSelect operation (0-11): ").strip()
                
                if choice == '1':
                    self.set_scene_image()
                elif choice == '2':
                    self.analyze_scene()
                elif choice == '3':
                    self.create_motion_task()
                elif choice == '4':
                    self.generate_motion_sequence()
                elif choice == '5':
                    self.show_session_status()
                elif choice == '10':
                    self.improve_previous_prompt()
                elif choice == '11':
                    self.test_custom_prompt()
                elif choice == '0':
                    print(f"\n👋 Thank you for using Interactive Scene Motion Planner!")
                    break
                else:
                    print("❌ Invalid selection, please enter 0-11")
                
                # Pause to let user see results
                if choice != '0':
                    input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Ctrl+C detected, program exiting")
                break
            except Exception as e:
                print(f"\n❌ Program error: {e}")
                print("💡 Please retry or contact technical support")


def main():
    """Main function"""
    try:
        planner = InteractiveScenePlanner()
        planner.run()
    except Exception as e:
        print(f"❌ Program startup failed: {e}")
        print("💡 Please check if dependencies are correctly installed")


if __name__ == "__main__":
    main()
