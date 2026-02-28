import os
import sys
import torch
import numpy as np
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

# Add Motion-Agent path to system path
motion_agent_path = os.path.join(os.path.dirname(__file__), 'Motion-Agent')
if motion_agent_path not in sys.path:
    sys.path.insert(0, motion_agent_path)

class EnhancedMotionGenerator:
    """
    Enhanced motion generator that implements Motion-Agent's architecture:
    1. Generate motion tokens for each action
    2. Concatenate tokens to form a sequence
    3. Use VQVAE decoder to generate complete motion
    4. Render the final motion sequence
    """
    
    def __init__(self, output_dir: str = "./enhanced_output"):
        """
        Initialize the enhanced motion generator
        
        Args:
            output_dir: Directory to save generated motions
        """
        self.output_dir = output_dir
        self.motion_history = {}  # Cache for motion tokens
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MotionLLM and VQVAE models
        self.motionllm_model = None
        self.vqvae_model = None
        self._initialize_models()
        
        print(f"🎯 Enhanced Motion Generator initialized on {self.device}")
        print(f"📁 Output directory: {output_dir}")
    
    def _initialize_models(self):
        """Initialize MotionLLM and VQVAE models"""
        try:
            print("🔧 Initializing MotionLLM model...")
            
            # Import Motion-Agent modules
            try:
                from models.mllm import MotionLLM
                from options.option_llm import get_args_parser
                print("✅ Modules imported successfully")
            except Exception as e:
                print(f"❌ Module import failed: {e}")
                return False
            
            # Get arguments for MotionLLM
            try:
                import sys
                old_argv = sys.argv
                sys.argv = ['enhanced_motion_generator.py']  # clear argv for sub-parser
                args = get_args_parser()
                sys.argv = old_argv  # restore original argv
                args.llm_backbone = os.path.abspath("gemma2b")  # Update with your model path
                args.device = str(self.device)
                args.dataname = 't2m'
                args.window_size = 196
                print("✅ Arguments configured")
            except Exception as e:
                print(f"❌ Argument configuration failed: {e}")
                return False
            
            # Fix the working directory for MotionLLM initialization
            original_cwd = os.getcwd()
            os.chdir(motion_agent_path)
            
            try:
                # Initialize MotionLLM
                self.motionllm_model = MotionLLM(args).to(self.device)
                
                # Load pre-trained model
                ckpt_path = os.path.join(motion_agent_path, "ckpt/motionllm.pth")
                if os.path.exists(ckpt_path):
                    self.motionllm_model.load_model(ckpt_path)
                    self.motionllm_model.llm.eval()
                    print("✅ MotionLLM model loaded successfully")
                else:
                    print(f"⚠️  MotionLLM checkpoint not found at {ckpt_path}")
                    print("💡 Please ensure the checkpoint file exists")
                    return False
                
                # VQVAE model is already integrated in MotionLLM
                self.vqvae_model = self.motionllm_model.net
                print("✅ VQVAE model initialized (integrated in MotionLLM)")
                
                return True
                
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            print("💡 Please check Motion-Agent installation and model paths")
            return False
    
    def generate_motion_sequence(self, motion_steps: List[str], sequence_name: str) -> Dict[str, Any]:
        """
        Generate a complete motion sequence using Motion-Agent's approach
        
        Args:
            motion_steps: List of motion step descriptions
            sequence_name: Name for the sequence
            
        Returns:
            Dictionary containing generation results
        """
        print(f"🚀 Starting enhanced motion sequence generation: {sequence_name}")
        print(f"📊 Total steps: {len(motion_steps)}")
        
        if self.motionllm_model is None:
            return {"error": "MotionLLM model not initialized"}
        
        start_time = time.time()
        
        try:
            # Step 1: Generate motion tokens for each step
            print("🔄 Step 1: Generating motion tokens for each action...")
            motion_tokens_list = self._generate_motion_tokens(motion_steps)
            
            if not motion_tokens_list:
                return {"error": "Failed to generate motion tokens"}
            
            # Step 2: Concatenate all motion tokens
            print("🔗 Step 2: Concatenating motion tokens...")
            concatenated_tokens = self._concatenate_motion_tokens(motion_tokens_list)
            
            # Step 3: Generate complete motion through VQVAE decoder
            print("🎭 Step 3: Generating complete motion sequence...")
            complete_motion = self._generate_complete_motion(concatenated_tokens)
            
            if complete_motion is None:
                return {"error": "Failed to generate complete motion"}
            
            # Step 4: Process and save the motion
            print("💾 Step 4: Processing and saving motion...")
            result = self._process_and_save_motion(
                complete_motion, 
                motion_steps, 
                sequence_name
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Add metadata
            result.update({
                "sequence_name": sequence_name,
                "step_count": len(motion_steps),
                "generation_time": generation_time,
                "motion_steps": motion_steps,
                "total_frames": result.get("total_frames", 0),
                "average_quality": result.get("average_quality", 0.0)
            })
            
            print(f"✅ Enhanced motion sequence generation completed in {generation_time:.1f} seconds")
            return result
            
        except Exception as e:
            print(f"❌ Enhanced motion generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_motion_tokens(self, motion_steps: List[str]) -> List[torch.Tensor]:
        """
        Generate motion tokens for each action step
        
        Args:
            motion_steps: List of motion descriptions
            
        Returns:
            List of motion tokens for each step
        """
        motion_tokens_list = []
        
        for i, step_description in enumerate(motion_steps, 1):
            print(f"  📝 Step {i}/{len(motion_steps)}: {step_description}")
            
            # Check cache first
            if step_description in self.motion_history:
                print(f"    ✅ Using cached tokens for: {step_description}")
                motion_tokens_list.append(self.motion_history[step_description])
                continue
            
            # Generate new tokens using real MotionLLM
            motion_tokens = self._generate_single_motion_tokens(step_description)
            
            if motion_tokens is not None:
                # Cache the tokens
                self.motion_history[step_description] = motion_tokens
                motion_tokens_list.append(motion_tokens)
                print(f"    ✅ Generated tokens for: {step_description}")
            else:
                print(f"    ❌ Failed to generate tokens for: {step_description}")
                return []
        
        return motion_tokens_list
    
    def _generate_single_motion_tokens(self, description: str) -> Optional[torch.Tensor]:
        """
        Generate motion tokens for a single action description using real MotionLLM
        
        Args:
            description: Action description
            
        Returns:
            Motion tokens tensor or None if failed
        """
        try:
            # Use real MotionLLM to generate tokens
            with torch.no_grad():
                motion_tokens = self.motionllm_model.generate(description)
            
            print(f"    📊 Generated {len(motion_tokens)} motion tokens")
            return motion_tokens
            
        except Exception as e:
            print(f"    ❌ Token generation error: {e}")
            return None
    
    def _concatenate_motion_tokens(self, motion_tokens_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate all motion tokens to form a sequence
        
        Args:
            motion_tokens_list: List of motion tokens for each step
            
        Returns:
            Concatenated motion tokens
        """
        if not motion_tokens_list:
            raise ValueError("No motion tokens to concatenate")
        
        print(f"    🔗 Concatenating {len(motion_tokens_list)} token sequences...")
        
        # Concatenate along the time dimension
        concatenated = torch.cat(motion_tokens_list, dim=0)
        
        print(f"    ✅ Concatenated tokens shape: {concatenated.shape}")
        return concatenated
    
    def _generate_complete_motion(self, concatenated_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Generate complete motion through VQVAE decoder
        
        Args:
            concatenated_tokens: Concatenated motion tokens
            
        Returns:
            Complete motion tensor or None if failed
        """
        try:
            # Use real VQVAE decoder to generate motion
            with torch.no_grad():
                # Add batch dimension and move to device
                tokens_batch = concatenated_tokens.unsqueeze(0).to(self.device)
                motion = self.vqvae_model.forward_decoder(tokens_batch)
            
            print(f"    ✅ Generated motion shape: {motion.shape}")
            return motion
            
        except Exception as e:
            print(f"    ❌ Motion generation error: {e}")
            return None
    
    def _process_and_save_motion(self, motion: torch.Tensor, motion_steps: List[str], sequence_name: str) -> Dict[str, Any]:
        """
        Process and save the generated motion
        
        Args:
            motion: Generated motion tensor
            motion_steps: List of motion steps
            sequence_name: Name for the sequence
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Create sequence directory
            sequence_dir = os.path.join(self.output_dir, sequence_name)
            os.makedirs(sequence_dir, exist_ok=True)
            
            # Denormalize motion data
            motion_np = self.motionllm_model.denormalize(motion.detach().cpu().numpy())
            
            # Save motion data (both 263-dim features and 22-joint positions)
            motion_npy_path = os.path.join(sequence_dir, f"{sequence_name}.npy")
            np.save(motion_npy_path, motion_np)  # Save 263-dim features
            
            # Also save 22-joint positions for SMPL-X conversion
            try:
                from utils.motion_utils import recover_from_ric
                motion_tensor = torch.from_numpy(motion_np).float().to(self.device)
                motion_recovered = recover_from_ric(motion_tensor, joints_num=22)
                joints_npy_path = os.path.join(sequence_dir, f"{sequence_name}_joints.npy")
                np.save(joints_npy_path, motion_recovered.squeeze().cpu().numpy())
                print(f"    📊 Joint positions saved to: {joints_npy_path}")
            except Exception as e:
                print(f"    ⚠️  Joint conversion failed: {e}")
                joints_npy_path = None
            
            # Auto-save as BVH file
            try:
                from simple_bvh_writer import convert_motionllm_output_to_bvh
                bvh_file = convert_motionllm_output_to_bvh(motion_np, sequence_dir, sequence_name)
                print(f"    🎬 BVH file generated: {bvh_file}")
            except Exception as e:
                print(f"    ⚠️  BVH conversion failed: {e}")
            
            # Auto-generate GIF animation
            try:
                from simple_gif_visualizer import create_motion_gif
                gif_file = os.path.join(sequence_dir, f"{sequence_name}.gif")
                create_motion_gif(motion_np, gif_file)
                print(f"    🎬 GIF generated: {gif_file}")
            except Exception as e:
                print(f"    ⚠️  GIF generation failed: {e}")
            
            # Auto-generate HTML visualization
            try:
                from html_visualizer import create_html_visualization
                html_file = os.path.join(sequence_dir, f"{sequence_name}.html")
                create_html_visualization(motion_np, html_file)
                print(f"    🌐 HTML visualization generated: {html_file}")
            except Exception as e:
                print(f"    ⚠️  HTML generation failed: {e}")
            
            # Use smart visualizer (recommended)
            try:
                from smart_visualizer import create_smart_visualization
                smart_result = create_smart_visualization(motion_np, sequence_dir, sequence_name)
                print(f"    🧠 Smart visualization completed!")
                print(f"       📊 Report: {smart_result['files']['report']}")
                print(f"       🎬 GIF: {smart_result['files']['gif']}")
                print(f"       🌐 HTML: {smart_result['files']['html']}")
                print(f"       📈 Trajectory plot: {smart_result['files']['plot']}")
            except Exception as e:
                print(f"    ⚠️  Smart visualization failed: {e}")
            
            # Render video using Motion-Agent's visualization
            video_path = os.path.join(sequence_dir, f"{sequence_name}.mp4")
            try:
                # Import Motion-Agent visualization modules
                from utils.motion_utils import recover_from_ric, plot_3d_motion
                from utils.paramUtil import t2m_kinematic_chain
                
                # Process motion for visualization
                motion_np_processed = self._prepare_motion_for_visualization(motion_np)
                motion_tensor = torch.from_numpy(motion_np_processed).float().to(self.device)
                motion_recovered = recover_from_ric(motion_tensor, joints_num=22)
                
                # Generate video title
                video_title = f"Motion Sequence: {' -> '.join(motion_steps)}"
                
                # Render 3D motion video
                plot_3d_motion(
                    video_path, 
                    t2m_kinematic_chain, 
                    motion_recovered.squeeze().cpu().numpy(), 
                    title=video_title, 
                    fps=20, 
                    radius=4
                )
                
                print(f"    🎬 Video rendered: {video_path}")
                
            except Exception as e:
                print(f"    ⚠️  Video rendering failed: {e}")
                video_path = "N/A"
            
            # Generate metadata
            metadata = {
                "sequence_name": sequence_name,
                "generation_time": datetime.now().isoformat(),
                "motion_steps": motion_steps,
                "motion_shape": motion_np.shape,
                "total_frames": motion_np.shape[1] if len(motion_np.shape) > 1 else 1,
                "feature_dim": motion_np.shape[-1] if len(motion_np.shape) > 1 else motion_np.shape[0],
                "device": str(self.device),
                "model_info": {
                    "motionllm_loaded": self.motionllm_model is not None,
                    "vqvae_loaded": self.vqvae_model is not None
                }
            }
            
            # Save metadata
            metadata_path = os.path.join(sequence_dir, f"{sequence_name}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"    💾 Motion data saved to: {motion_npy_path}")
            print(f"    📄 Metadata saved to: {metadata_path}")
            
            # Return result in the format expected by interactive_scene_planner.py
            return {
                "success": True,
                "motion_path": motion_npy_path,
                "metadata_path": metadata_path,
                "total_frames": metadata["total_frames"],
                "average_quality": 0.8,  # Placeholder quality score
                "motion_shape": motion_np.shape,
                # Add the expected concat_result field
                "concat_result": {
                    "output_dir": sequence_dir,
                    "final_motion": motion_npy_path,
                    "motion_data": joints_npy_path if joints_npy_path else motion_npy_path,  # Use joints data for SMPL-X conversion
                    "final_video": video_path,
                    "success": True
                },
                # Add individual_motions for compatibility
                "individual_motions": [
                    {
                        "step": i + 1,
                        "description": step,
                        "files": {
                            "motion": motion_npy_path,
                            "video": video_path
                        }
                    }
                    for i, step in enumerate(motion_steps)
                ]
            }
            
        except Exception as e:
            print(f"    ❌ Motion processing error: {e}")
            return {"error": str(e)}
    
    def _prepare_motion_for_visualization(self, motion_np: np.ndarray) -> np.ndarray:
        """
        Prepare motion data for visualization
        
        Args:
            motion_np: Motion numpy array
            
        Returns:
            Processed motion array
        """
        # Ensure motion has correct shape for visualization
        if motion_np.ndim == 3:
            # Take first batch if multiple batches
            motion_np = motion_np[0]  # (seq_len, feat_dim)
        
        # Ensure feature dimension is sufficient for 22 joints * 3 coordinates
        if motion_np.shape[-1] < 22 * 3:
            print(f"⚠️  Feature dimension {motion_np.shape[-1]} is less than required {22*3}")
            # Pad or truncate as needed
            if motion_np.shape[-1] > 22 * 3:
                motion_np = motion_np[:, :22*3]
            else:
                # Pad with zeros
                padding = np.zeros((motion_np.shape[0], 22*3 - motion_np.shape[1]))
                motion_np = np.concatenate([motion_np, padding], axis=1)
        
        return motion_np
    
    def clear_cache(self):
        """Clear the motion token cache"""
        self.motion_history.clear()
        print("🗑️  Motion token cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the motion token cache"""
        return {
            "cache_size": len(self.motion_history),
            "cached_descriptions": list(self.motion_history.keys()),
            "total_tokens": sum(len(tokens) for tokens in self.motion_history.values())
        }
    
    def test_motionllm_connection(self, test_description: str = "A person walks forward") -> Dict[str, Any]:
        """
        Test MotionLLM connection and token generation
        
        Args:
            test_description: Test motion description
            
        Returns:
            Test results
        """
        print(f"🧪 Testing MotionLLM with: {test_description}")
        
        try:
            # Test token generation
            tokens = self._generate_single_motion_tokens(test_description)
            
            if tokens is not None:
                # Test motion generation
                motion = self._generate_complete_motion(tokens)
                
                if motion is not None:
                    return {
                        "success": True,
                        "tokens_shape": tokens.shape,
                        "motion_shape": motion.shape,
                        "message": "MotionLLM integration working correctly"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Motion generation failed",
                        "tokens_shape": tokens.shape
                    }
            else:
                return {
                    "success": False,
                    "error": "Token generation failed"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Test failed: {str(e)}"
            }


def main():
    """Test the enhanced motion generator"""
    generator = EnhancedMotionGenerator()
    
    # Test motion steps
    test_steps = [
        "A person turns to the right",
        "A person walks forward 3 meters",
        "A person turns to the left",
        "A person walks forward 2 meters"
    ]
    
    print("🧪 Testing Enhanced Motion Generator...")
    result = generator.generate_motion_sequence(test_steps, "test_sequence")
    
    if "error" not in result:
        print("✅ Test completed successfully!")
        print(f"📊 Result: {result}")
    else:
        print(f"❌ Test failed: {result['error']}")


if __name__ == "__main__":
    main()
