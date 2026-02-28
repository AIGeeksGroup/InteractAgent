
"""
Enhanced Prompt Template System - Learning from Motion-Agent Project
Focused on accurate motion planning with structured output
"""

import re
from typing import List


class PromptTemplates:
    """Enhanced Prompt Template Management Class - Learning from Motion-Agent"""
    
    def __init__(self):
        """Initialize template system"""
        pass
    
    def get_dual_view_scene_analysis_prompt(self) -> str:
        """
        Generate dual-view scene analysis prompt (bird's-eye view + first-person view)
        
        Returns:
            Dual-view scene analysis prompt
        """
        return """
Please analyze these two images: the first is a bird's-eye view, the second is a first-person view.

Please provide the following information:

1. Room type and layout
2. Room dimensions (length x width, in meters)
3. Current viewing direction (facing where)
4. Distance from current position to various furniture (in meters)
5. Required turning angle to reach target position

Important: The first-person view shows your current facing direction, which is the starting point for motion planning. Please accurately determine the current orientation and calculate the required turning angle to reach the target position.

Note: In first-person view, what you see ahead is your forward direction (0 degrees), left is -90 degrees, right is +90 degrees. Please calculate turning angles based on this coordinate system.
"""

    def get_scene_analysis_prompt(self) -> str:
        """
        Generate scene analysis prompt for single image
        
        Returns:
            Scene analysis prompt
        """
        return """
Please analyze this scene image and provide a detailed description including:

1. Spatial layout: Room type and overall layout
2. Key objects: Key objects and furniture in the scene
3. Dimensions: Room dimensions and object sizes (in meters)
4. Walkable areas: Walkable areas and paths
5. Obstacles: Obstacles and their positions
6. Height info: Height information for objects and surfaces
7. Starting position: Suggest a reasonable starting position for the person (e.g., "entrance", "center of room", "near the door")

Please provide a comprehensive analysis that can be used for motion planning and trajectory generation.
"""

    def get_trajectory_planning_prompt(self, scene_description: str, task_description: str) -> str:
        """
        Generate trajectory planning prompt for Trumans motion generation
        
        Args:
            scene_description: Scene description
            task_description: Task description
            
        Returns:
            Trajectory planning prompt
        """
        return f"""You are a trajectory planning expert that helps users generate precise 3D trajectory coordinates for human motion tasks. You will generate a sequence of 3D coordinate points that represent the person's path.

Task: {task_description}

Scene Analysis: {scene_description}

Instructions:
1. Generate a sequence of 3D coordinate points representing the person's trajectory
2. Each point should have x, y, z coordinates in meters
3. Start from a reasonable position (e.g., entrance at x=0, y=0, z=0)
4. Consider the spatial relationships from the scene analysis
5. Use realistic distances and smooth path transitions
6. Include enough points to create a smooth motion (typically 8-20 points)
7. **IMPORTANT**: Make the trajectory natural and realistic, not just a straight line!

TRAJECTORY DESIGN PRINCIPLES:
- Add natural curves and turns (not just straight lines)
- Include brief pauses or direction changes at key points
- Vary walking speed by adjusting point density (closer points = slower movement)
- Consider obstacles and create realistic navigation paths
- Add small lateral movements (x-axis variations) to simulate natural walking

COORDINATE SYSTEM:
- x: left/right (positive = right)
- y: forward/backward (positive = forward) 
- z: up/down (keep at 0 for ground level)

Response Format:
You should respond with a JSON array of coordinate points. Make sure the trajectory is NOT a straight line:

[
  {{"x": 0.0, "y": 0.0, "z": 0.0}},
  {{"x": 0.2, "y": 0.5, "z": 0.0}},
  {{"x": 0.8, "y": 1.0, "z": 0.0}},
  {{"x": 1.5, "y": 1.2, "z": 0.0}},
  {{"x": 2.2, "y": 1.8, "z": 0.0}},
  {{"x": 2.8, "y": 2.5, "z": 0.0}},
  {{"x": 3.0, "y": 3.0, "z": 0.0}}
]

Example for "walk to the yellow chair" (assuming chair is at x=3, y=3):
Notice how the path curves naturally and includes lateral movement, not just forward movement.

Generate your trajectory coordinates now. Remember: make it natural and curved, not a straight line!"""

    def get_dual_view_motion_planning_prompt(self, scene_description: str, task_description: str) -> str:
        """
        Generate motion planning prompt based on dual-view scene description
        Learning from Motion-Agent's structured approach
        
        Args:
            scene_description: Dual-view scene description
            task_description: Task description
            
        Returns:
            Motion planning prompt
        """
        return f"""You are a motion planning expert that helps users generate precise human motion sequences for navigation tasks. You have access to MotionLLM, which can generate simple, atomic 3D human motions based on textual descriptions.

Task: {task_description}

Scene Analysis: {scene_description}

Instructions:
1. Break down the navigation task into simple, atomic motion steps
2. Each motion step should be described in the format: "A person [action]"
3. Focus on navigation-related actions: walking, turning, reaching, etc.
4. Ensure each step is independent and can be executed by MotionLLM
5. Consider the spatial relationships from the scene analysis

Response Format:
You should ONLY respond in JSON format, following this template:
{{
  "plan": "A numbered list of motion steps, each in 'A person [action]' format",
  "reasoning": "Brief explanation of the motion sequence logic"
}}

Example Response:
{{
  "plan": "1. A person turns to the right; 2. A person walks forward 3 meters; 3. A person turns to the left; 4. A person walks forward 2 meters",
  "reasoning": "The person needs to navigate around obstacles to reach the target location, using precise turning and walking motions."
}}

Generate your motion plan now:"""
    
    def parse_motion_response(self, response: str) -> List[str]:
        """
        Parse MotionLLM response to extract action steps
        Enhanced to handle structured JSON responses
        
        Args:
            response: MotionLLM response text
            
        Returns:
            List of action steps
        """
        if not response:
            return []
        
        # Try to parse as JSON first (following Motion-Agent approach)
        try:
            import json
            parsed = json.loads(response)
            if "plan" in parsed:
                plan = parsed["plan"]
                # Extract motion steps from the plan
                steps = []
                # Split by semicolon and clean up
                for step in plan.split(";"):
                    step = step.strip()
                    if step and "A person" in step:
                        # Remove numbering (e.g., "1. ", "2. ")
                        step = re.sub(r'^\d+\.\s*', '', step)
                        # Extract the action part
                        if "A person" in step:
                            action = step.split("A person")[1].strip()
                            if action:
                                steps.append(f"A person {action}")
                if steps:
                    print(f"✅ Parsed {len(steps)} motion steps from JSON plan")
                    return steps
        except Exception as e:
            print(f"⚠️  JSON parsing failed: {e}")
            pass
        
        # Try to parse as quasi-JSON format (JSON fields without braces)
        try:
            # Look for "plan": "..." pattern
            import re
            plan_match = re.search(r'"plan":\s*"([^"]+)"', response)
            if plan_match:
                plan = plan_match.group(1)
                # Extract motion steps from the plan
                steps = []
                # Split by semicolon and clean up
                for step in plan.split(";"):
                    step = step.strip()
                    if step and "A person" in step:
                        # Remove numbering (e.g., "1. ", "2. ")
                        step = re.sub(r'^\d+\.\s*', '', step)
                        # Extract the action part
                        if "A person" in step:
                            action = step.split("A person")[1].strip()
                            if action:
                                steps.append(f"A person {action}")
                if steps:
                    print(f"✅ Parsed {len(steps)} motion steps from quasi-JSON plan")
                    return steps
        except Exception as e:
            print(f"⚠️  Quasi-JSON parsing failed: {e}")
            pass
        
        # Fallback to original parsing logic
        response = response.strip()
        lines = response.split('\n')
        action_steps = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match standard action format: action type + distance/angle + direction
            # Example: walk forward 2.5 meters, turn left 90 degrees, turn to the right
            if (any(word in line.lower() for word in ['walk', 'turn', 'step', 'move']) and
                any(word in line.lower() for word in ['forward', 'backward', 'left', 'right', 'meters', 'degrees']) and
                not any(word in line.lower() for word in ['action type', 'turning angle', 'movement distance', 'explanation', 'direction'])):
                
                clean_line = re.sub(r'^\*\*|\*\*$', '', line)
                clean_line = re.sub(r'\[.*?\]', '', clean_line)
                clean_line = clean_line.strip()
                
                if clean_line and len(clean_line) > 5 and not clean_line.startswith('**'):
                    action_steps.append(clean_line)
                    print(f"✅ Parsed action: {clean_line}")
        
        print(f"📊 Total parsed actions: {len(action_steps)}")
        return action_steps
