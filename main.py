import os
import sys
import argparse
import json
from datetime import datetime
from typing import Optional

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import config, validate_environment, check_dependencies, show_system_info
    from logger import motion_logger, LogContext
    from interactive_scene_planner import InteractiveScenePlanner
except ImportError as e:
    print(f"❌ Module import failed: {e}")
    print("Please ensure all necessary files are in the correct location")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Scene Motion Planner - Generate human motion sequences based on scene images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python main.py                          # Start interactive interface
  python main.py --check-env              # Check runtime environment and exit
  python main.py --config config.json    # Use custom configuration file
  python main.py --batch scene.jpg task.txt  # Batch processing mode
  python main.py --motion-gen "walk forward"  # Generate motion from text
  python main.py --motion-gen "walk forward" --convert-smplx  # Generate and convert to SMPL-X
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["interactive", "batch", "api", "motion-generation", "closed-loop"],
        default="interactive",
        help="Run mode: interactive(interactive), batch(batch), api(API service), motion-generation(MotionLLM generation), closed-loop(Qwen→MotionLLM→Blender→Qwen)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Custom configuration file path"
    )
    
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check runtime environment and exit"
    )
    
    parser.add_argument(
        "--scene-image",
        type=str,
        help="Scene image URL or path (batch mode)"
    )
    
    parser.add_argument(
        "--task-file",
        type=str,
        help="Task description file path (batch mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory path"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU mode"
    )
    
    # Motion generation specific arguments
    parser.add_argument(
        "--motion-gen",
        type=str,
        help="Generate motion from text description"
    )
    
    parser.add_argument(
        "--convert-smplx",
        action="store_true",
        help="Convert generated motion to SMPL-X format for Blender visualization"
    )
    
    parser.add_argument(
        "--motion-output-dir",
        type=str,
        default="./demo",
        help="Output directory for motion files (default: ./demo)"
    )
    
    parser.add_argument(
        "--trumans-utils-path",
        type=str,
        default="./trumans_utils",
        help="Path to trumans_utils directory for SMPL-X conversion"
    )

    # Closed-loop specific arguments
    parser.add_argument(
        "--task",
        type=str,
        help="High-level navigation/motion task description (closed-loop mode)"
    )
    parser.add_argument(
        "--scene-image-url",
        type=str,
        help="Publicly accessible scene image URL for Qwen (closed-loop mode)"
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default="qwen-vl-max",
        help="Qwen model name for scene understanding and reflection (closed-loop mode)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Closed-loop iterations: Qwen→MotionLLM→Qwen reflection cycles"
    )
    parser.add_argument(
        "--rendered-video-url",
        type=str,
        help="Manually provided public URL of the rendered video for Qwen reflection"
    )
    parser.add_argument(
        "--ask-video-url",
        action="store_true",
        help="Interactively prompt to input rendered video URL for reflection"
    )
    parser.add_argument(
        "--prev-prompt",
        type=str,
        help="Previous MotionLLM prompt to critique and improve before new generation"
    )
    parser.add_argument(
        "--ask-prev-prompt",
        action="store_true",
        help="Interactively prompt to input previous MotionLLM prompt"
    )
    parser.add_argument(
        "--history-file",
        type=str,
        default=os.path.join("./logs", "closed_loop_history.jsonl"),
        help="Path to history jsonl file for reflection context"
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=5,
        help="Number of recent history items to include into reflection context"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"Scene Motion Planner {config.VERSION}"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup runtime environment"""
    motion_logger.info("Setting up runtime environment...")
    
    # Load custom configuration
    if args.config:
        if os.path.exists(args.config):
            config.load_config_from_file(args.config)
            motion_logger.info(f"Configuration file loaded: {args.config}")
        else:
            motion_logger.error(f"Configuration file not found: {args.config}")
            return False
    
    # Update configuration
    if args.output_dir:
        config.OUTPUT_BASE_DIR = args.output_dir
    
    if args.log_level:
        config.LOG_LEVEL = args.log_level
    
    if args.no_gpu:
        config.MOTION_DEVICE = "cpu"
        motion_logger.info("Forcing CPU mode")
    
    # Recreate directories
    config._create_directories()
    
    return True


def run_environment_check():
    """Run environment check"""
    print("🔍 Runtime Environment Check")
    print("=" * 50)
    
    # Show system information
    show_system_info()
    
    print("\n" + "=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Validate environment
    if not validate_environment():
        return False
    
    print("\n🎉 Environment check completed, system can run normally!")
    return True


def run_motion_generation_mode(args):
    """Run motion generation mode - integrated from run_local.py and convert_motionllm_fresh.py"""
    motion_logger.info("Starting motion generation mode")
    
    if not args.motion_gen:
        print("❌ Motion generation mode requires --motion-gen parameter")
        return False
    
    try:
        # Step 1: Generate motion using MotionLLM (from run_local.py)
        print("🎯 MotionLLM Motion Generation")
        print("=" * 50)
        
        # Import MotionLLM components
        sys.path.append('Motion-Agent')
        # Change working directory to Motion-Agent for relative paths
        original_cwd = os.getcwd()
        os.chdir('Motion-Agent')
        from models.mllm import MotionLLM
        from options.option_llm import get_args_parser
        from utils.motion_utils import recover_from_ric, plot_3d_motion
        from utils.paramUtil import t2m_kinematic_chain
        import torch
        import numpy as np
        
        # Setup MotionLLM arguments
        old_argv = sys.argv
        sys.argv = ['main.py']  # clear argv for sub-parser
        motion_args = get_args_parser()
        sys.argv = old_argv  # restore original argv
        motion_args.llm_backbone = "gemma2b" # Yujia Debug
        motion_args.ckpt = "ckpt/motionllm.pth"
        motion_args.save_dir = args.motion_output_dir
        motion_args.device = 'cuda:0' if not args.no_gpu else 'cpu'
        motion_args.dataname = 't2m'
        motion_args.window_size = 196
        
        os.makedirs(motion_args.save_dir, exist_ok=True)
        
        # Load MotionLLM model
        print("📦 Loading MotionLLM model...")
        model = MotionLLM(motion_args).to(motion_args.device)
        model.load_model(motion_args.ckpt)
        model.llm.eval()
        
        # Generate motion
        caption = args.motion_gen
        print(f"🎬 Generating motion for: '{caption}'")
        
        with torch.no_grad():
            motion_tokens = model.generate(caption)
            motion = model.net.forward_decoder(motion_tokens.unsqueeze(0))  # [1, seq_len, 263]
        
        print(f"📊 Raw generated motion shape: {motion.shape}")
        motion_np = model.denormalize(motion.detach().cpu().numpy())
        
        # Check and adjust motion shape
        if motion_np.ndim != 3:
            raise ValueError(f"Expected 3D array but got shape {motion_np.shape}")
        
        batch_size, seq_len, feat_dim = motion_np.shape
        motion_np = motion_np[0]  # Take first batch
        print(f"📊 Taking first batch, new shape: {motion_np.shape}")
        
        # Convert to joint positions
        motion_tensor = torch.from_numpy(motion_np).float().to(motion_args.device)
        motion_recovered = recover_from_ric(motion_tensor, joints_num=22)
        print(f"📊 After recover_from_ric shape: {motion_recovered.shape}")
        
        # Save motion files
        motion_data = motion_recovered.squeeze().cpu().numpy()
        save_mp4_path = os.path.join(motion_args.save_dir, "motionllm_generated.mp4")
        save_npy_path = os.path.join(motion_args.save_dir, "motionllm_generated.npy")
        
        plot_3d_motion(save_mp4_path, t2m_kinematic_chain, motion_data, title=caption, fps=20, radius=4)
        np.save(save_npy_path, motion_data)
        
        print(f"✅ Motion generated and saved:")
        print(f"   📹 Video: {save_mp4_path}")
        print(f"   📊 Data: {save_npy_path}")
        
        # Step 2: Convert to SMPL-X format if requested
        if args.convert_smplx:
            print("\n🔄 Converting to SMPL-X format for Blender visualization")
            print("=" * 50)
            
            success = convert_to_smplx_format(save_npy_path, args.trumans_utils_path, motion_args.save_dir)
            if success:
                print("✅ SMPL-X conversion completed successfully!")
                print(f"📁 Check output directory: {motion_args.save_dir}")
            else:
                print("❌ SMPL-X conversion failed")
                return False
        else:
            print("\n💡 Tip: Use --convert-smplx to convert generated NPY to SMPL-X for Blender visualization")
        
        # Restore original working directory
        os.chdir(original_cwd)
        return True
        
    except Exception as e:
        motion_logger.error("Motion generation failed", exception=e)
        print(f"❌ Motion generation failed: {e}")
        import traceback
        traceback.print_exc()
        # Restore original working directory in case of error
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        return False


def convert_to_smplx_format(npy_path, trumans_utils_path, output_dir):
    """Convert .npy motion file to SMPL-X format (from convert_motionllm_fresh.py)"""
    try:
        # Add trumans_utils to path
        sys.path.append(trumans_utils_path)
        
        # Import required modules
        import pickle as pkl
        import torch
        import numpy as np
        import yaml
        
        from models.joints_to_smplx import joints_to_smpl, JointsToSMPLX
        from utils import dotDict
        
        print("✅ Successfully imported SMPL-X conversion modules")
        
        # Load configuration
        config_path = os.path.join(trumans_utils_path, "config/config_sample_synhsi.yaml")
        print(f"📋 Loading configuration: {config_path}")
        
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg = dotDict(cfg)
        
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
        
        # Load motion data
        print(f"📊 Loading motion data: {npy_path}")
        points_all = np.load(npy_path)
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
                'method': 'integrated_motionllm_to_smplx',
                'source_file': npy_path,
                'num_frames': transl.shape[0],
                'model_used': 'JointsToSMPLX',
                'note': 'Integrated conversion from MotionLLM to SMPL-X'
            }
        }
        
        # Save .pkl file
        pkl_output_path = os.path.join(output_dir, "motionllm_generated_smplx.pkl")
        print(f"💾 Saving SMPL-X data to: {pkl_output_path}")
        
        with open(pkl_output_path, 'wb') as f:
            pkl.dump(output_data, f)
        
        # Create Blender test script
        blender_script_path = os.path.join(output_dir, "test_motionllm_integrated.py")
        create_blender_test_script(blender_script_path, "motionllm_generated_smplx.pkl")
        
        print(f"✅ SMPL-X conversion successful!")
        print(f"📁 SMPL-X file: {pkl_output_path}")
        print(f"📁 Blender test script: {blender_script_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ SMPL-X conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_blender_test_script(script_path, pkl_filename):
    """Create Blender test script for the converted SMPL-X data"""
    script_content = f'''
import bpy
import sys
import os

# Add trumans_utils path
sys.path.append('./trumans_utils/visualize_smplx_motion')

# Import visualization function
from load_smplx_animatioin_clear import load_smplx_animation_new

def test_integrated_motionllm_data():
    """Test integrated MotionLLM to SMPL-X conversion"""
    
    pkl_file_path = "{pkl_filename}"
    
    print("🔍 Testing integrated MotionLLM to SMPL-X conversion...")
    
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
    print("📥 Loading integrated MotionLLM animation data...")
    result = load_smplx_animation_new(pkl_file_path, smplx_mesh, load_hand=False)
    
    if result == {{'FINISHED'}}:
        print("🎉 Integrated MotionLLM animation loaded successfully!")
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
    test_integrated_motionllm_data()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Blender test script created: {script_path}")


def run_interactive_mode():
    """Run interactive mode"""
    motion_logger.info("Starting interactive mode")
    
    try:
        with LogContext("interactive_session", motion_logger):
            planner = InteractiveScenePlanner()
            planner.run()
        
        motion_logger.info("Interactive session ended")
        return True
        
    except KeyboardInterrupt:
        motion_logger.info("User interrupted program")
        return True
    except Exception as e:
        motion_logger.error("Interactive mode run failed", exception=e)
        return False


def run_batch_mode(scene_image: str, task_file: str):
    """Run batch mode"""
    motion_logger.info("Starting batch mode", scene_image=scene_image, task_file=task_file)
    
    if not os.path.exists(task_file):
        motion_logger.error(f"Task file not found: {task_file}")
        return False
    
    try:
        # Read task file
        with open(task_file, 'r', encoding='utf-8') as f:
            tasks = [line.strip() for line in f if line.strip()]
        
        if not tasks:
            motion_logger.error("Task file is empty")
            return False
        
        motion_logger.info(f"Read {len(tasks)} tasks")
        
        # Create planner
        from scene_motion_planner import SceneMotionPlanner
        planner = SceneMotionPlanner()
        
        # Process each task
        for i, task in enumerate(tasks, 1):
            motion_logger.info(f"Processing task {i}/{len(tasks)}: {task}")
            
            with LogContext(f"batch_task_{i}", motion_logger, task=task):
                result = planner.process_scene_task(scene_image, task)
                
                if "error" in result:
                    motion_logger.error(f"任务 {i} 失败: {result['error']}")
                else:
                    motion_logger.info(f"任务 {i} 完成", 
                                     output_dir=result['output_dir'],
                                     steps=result['total_steps'])
        
        motion_logger.info("批处理完成")
        return True
        
    except Exception as e:
        motion_logger.error("批处理模式运行失败", exception=e)
        return False


def run_api_mode():
    """运行API服务模式"""
    motion_logger.info("Starting API service mode")
    
    try:
        # Flask/FastAPI service can be implemented here
        motion_logger.warning("API mode not yet implemented")
        return False
        
    except Exception as e:
        motion_logger.error("API mode run failed", exception=e)
        return False


def run_closed_loop_mode(args):
    """Closed-loop pipeline: Qwen scene → prompt → MotionLLM → render → Qwen reflection"""
    motion_logger.info("Starting closed-loop mode")
    try:
        # Preconditions
        if not args.scene_image_url:
            print("❌ Closed-loop mode requires --scene-image-url (public URL for Qwen)")
            return False
        if not args.task:
            print("❌ Closed-loop mode requires --task (high-level motion instruction)")
            return False

        # Import Qwen chat and prompt templates
        from interactive_qwenvl import QwenVLChat
        from prompt_templates import PromptTemplates

        templates = PromptTemplates()
        chat = QwenVLChat(model=args.qwen_model)
        chat.set_image(args.scene_image_url, "main")

        best_prompt = None
        best_artifacts = None

        num_iterations = max(1, int(args.iterations))
        for iter_idx in range(1, num_iterations + 1):
            with LogContext(f"closed_loop_iter_{iter_idx}", motion_logger):
                print(f"\n🔁 Closed-loop iteration {iter_idx}/{num_iterations}")

                # Step 0 (optional): critique previous prompt + video to produce improved prompt
                prev_prompt = getattr(args, "prev_prompt", None)
                if getattr(args, "ask_prev_prompt", False) and not prev_prompt:
                    try:
                        user_prev = input("Enter previous MotionLLM prompt (leave empty to skip): ").strip()
                        if user_prev:
                            args.prev_prompt = user_prev
                            prev_prompt = user_prev
                    except Exception:
                        pass

                steps = []
                motion_text = None
                if prev_prompt:
                    video_hint_prev = getattr(args, "rendered_video_url", None)
                    critique_ctx = (
                        "You must critique the previous motion instructions and corresponding video, "
                        "and produce improved motion instructions. Output must ONLY be a semicolon-separated string of 'A person [action]'."
                    )
                    critique_q = (
                        f"Previous prompt: {prev_prompt}\n" +
                        (f"Rendered video (public URL): {video_hint_prev}\n" if video_hint_prev else "") +
                        "Please provide a better motion sequence based on issues observed in the video."
                    )
                    improved = chat.ask_with_context(critique_q, critique_ctx)
                    steps = templates.parse_motion_response(improved)

                # Step 1: If no improved steps from previous prompt, do scene analysis path
                if not steps:
                    scene_prompt = templates.get_dual_view_scene_analysis_prompt()
                    scene_desc = chat.ask_with_context(
                        question="Please output concise scene analysis key points according to the above template.",
                        context=scene_prompt
                    )
                    motion_planning_prompt = templates.get_dual_view_motion_planning_prompt(
                        scene_description=scene_desc,
                        task_description=args.task
                    )
                    motion_plan_text = chat.ask(motion_planning_prompt)
                    steps = templates.parse_motion_response(motion_plan_text)
                if not steps:
                    print("❌ Failed to parse valid motion steps from Qwen, skipping this iteration")
                    continue

                motion_text = "; ".join(steps)
                print(f"📝 Motion prompt: {motion_text}")
                best_prompt = motion_text

                # Step 2: MotionLLM generation (reuse motion-generation helper)
                class SimpleArgs:
                    pass
                mg_args = SimpleArgs()
                mg_args.motion_gen = motion_text
                mg_args.motion_output_dir = args.motion_output_dir or "./demo"
                mg_args.no_gpu = args.no_gpu
                mg_args.convert_smplx = args.convert_smplx
                mg_args.trumans_utils_path = args.trumans_utils_path
                ok = run_motion_generation_mode(mg_args)
                if not ok:
                    print("❌ MotionLLM generation failed, skip reflection")
                    continue

                # Collect artifacts
                video_path = os.path.join(mg_args.motion_output_dir, "motionllm_generated.mp4")
                npy_path = os.path.join(mg_args.motion_output_dir, "motionllm_generated.npy")
                best_artifacts = {"video": video_path, "npy": npy_path}

                # Optional: prompt for manual video URL if requested and not provided yet
                if getattr(args, "ask_video_url", False) and not getattr(args, "rendered_video_url", None):
                    try:
                        user_input_url = input("Enter rendered video public URL (leave empty to skip): ").strip()
                        if user_input_url:
                            args.rendered_video_url = user_input_url
                            print(f"✅ Rendered video URL set: {args.rendered_video_url}")
                    except Exception:
                        pass

                # Step 3: Qwen reflection (support manual uploaded video URL)
                if args.scene_image_url and args.qwen_model:
                    reflection_context = (
                        "You are a motion planning reviewer. Given the scene analysis, task, and rendered result (if accessible), "
                        "assess whether the motion meets the task, provide improvement suggestions, and output an improved concise MotionLLM prompt, "
                        "which ONLY contains a semicolon-separated string of 'A person [action]'."
                    )
                    # Use user-provided public video URL if present
                    video_hint = ""
                    if getattr(args, "rendered_video_url", None):
                        video_hint = f"渲染视频(公网URL): {args.rendered_video_url}\n"

                    reflection_question = (
                        f"场景要点: {scene_desc}\n任务: {args.task}\n"
                        f"当前指令: {motion_text}\n{video_hint}"
                        "若能访问渲染视频，请基于其进行判断。"
                    )
                    # Load recent history and enrich context
                    try:
                        history_snippets = []
                        if args.history_file and os.path.exists(args.history_file):
                            with open(args.history_file, 'r', encoding='utf-8') as hf:
                                lines = hf.readlines()
                            include = lines[-max(0, int(args.history_limit)):] if args.history_limit else lines
                            for ln in include:
                                try:
                                    item = json.loads(ln)
                                    history_snippets.append(
                                        f"[time]={item.get('time')} [task]={item.get('task')}\n"
                                        f"[prompt]={item.get('prompt')}\n[video]={item.get('video_url') or item.get('video_path')}"
                                    )
                                except Exception:
                                    continue
                        if history_snippets:
                            reflection_context = (
                                reflection_context + "\n\nRecent history (new → old):\n" + "\n\n".join(reversed(history_snippets))
                            )
                    except Exception:
                        pass
                    reflection = chat.ask_with_context(reflection_question, reflection_context)
                    improved_steps = templates.parse_motion_response(reflection)
                    if improved_steps:
                        improved_prompt = "; ".join(improved_steps)
                        print(f"✨ Improved motion prompt: {improved_prompt}")
                        best_prompt = improved_prompt
                    else:
                        print("ℹ️ Reflection did not produce structured improvements, keeping current prompt")

        # Summary
        print("\n✅ Closed-loop completed")
        if best_prompt:
            print(f"🧠 Best motion prompt: {best_prompt}")
        if best_artifacts:
            print(f"📹 Video: {best_artifacts.get('video')}")
            print(f"📊 Data: {best_artifacts.get('npy')}")
        # Append one history record
        try:
            if args.history_file:
                os.makedirs(os.path.dirname(args.history_file), exist_ok=True)
                record = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "task": args.task,
                    "prompt": best_prompt,
                    "video_path": best_artifacts.get('video') if best_artifacts else None,
                    "npy_path": best_artifacts.get('npy') if best_artifacts else None,
                    "video_url": getattr(args, 'rendered_video_url', None),
                    "scene_image_url": args.scene_image_url,
                    "qwen_model": args.qwen_model,
                    "iterations": args.iterations
                }
                with open(args.history_file, 'a', encoding='utf-8') as hf:
                    hf.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"📝 History saved: {args.history_file}")
        except Exception as e:
            print(f"⚠️ 历史记录写入失败: {e}")
        return True

    except Exception as e:
        motion_logger.error("Closed-loop mode failed", exception=e)
        print(f"❌ Closed-loop mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_on_exit():
    """Cleanup on exit"""
    try:
        # Save log statistics
        motion_logger.save_stats()
        
        # Show session summary
        motion_logger.print_summary()
        
        # Clean temporary files
        temp_dir = config.TEMP_DIR
        if os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
                motion_logger.info("Temporary files cleaned")
            except Exception:
                pass
        
    except Exception as e:
        print(f"Error during cleanup: {e}")


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Show startup information
    print("🎉 Scene Motion Planner")
    print("=" * 50)
    print("Integrating QwenVL scene recognition and MotionLLM motion generation")
    print(f"Version: {config.VERSION}")
    print("=" * 50)
    
    try:
        # Setup environment
        if not setup_environment(args):
            print("❌ Environment setup failed")
            return 1
        
        # Environment check mode
        if args.check_env:
            success = run_environment_check()
            return 0 if success else 1
        
        # Run environment validation
        if not validate_environment():
            print("❌ Environment validation failed, please use --check-env for detailed information")
            return 1
        
        motion_logger.info("System startup", mode=args.mode)
        
        # Run based on mode
        success = False
        
        if args.mode == "interactive":
            success = run_interactive_mode()
            
        elif args.mode == "batch":
            if not args.scene_image or not args.task_file:
                print("❌ Batch mode requires --scene-image and --task-file parameters")
                return 1
            success = run_batch_mode(args.scene_image, args.task_file)
            
        elif args.mode == "api":
            success = run_api_mode()
            
        elif args.mode == "motion-generation":
            success = run_motion_generation_mode(args)
        
        elif args.mode == "closed-loop":
            success = run_closed_loop_mode(args)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user")
        return 0
    except Exception as e:
        motion_logger.error("Program run failed", exception=e)
        print(f"❌ Program run failed: {e}")
        return 1
    finally:
        cleanup_on_exit()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)



