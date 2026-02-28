# InteractAgent: Agentic Human Motion Interaction with Memory-Augmented LLMs

This is the official repository for the paper:

> **InteractAgent: Agentic Human Motion Interaction with Memory-Augmented LLMs**
>
> Tao Zhang\*, Senhe Zhang\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*†, Yujia Zhang, Dong Gong‡
>
> \*Equal contribution. †Project lead. ‡Corresponding author.
>
> ### [Paper](paper.pdf)




https://github.com/user-attachments/assets/843d9b59-f4c6-4f49-86c0-83970161b779





## Visualization


<table>
  <tr>
    <td align="center" valign="top">
      <video src="https://github.com/user-attachments/assets/e3e97c9e-de03-41f5-8efc-bcc1eafb20e2"
             width="500" height="500" controls></video><br>
      <b>
        A 3D-rendered figure walks through a modern, open-plan bedroom and living area with terrazzo flooring, contemporary furniture, and soft natural lighting. The character pauses to sit in a mustard-yellow armchair, then stands and stretches before walking toward the sofa, giving a sense of scale and daily life within the stylish interior space.
      </b>
    </td>
    <td align="center" valign="top">
      <video src="https://github.com/user-attachments/assets/d478b3e5-beb5-4114-8a0d-56247ed2a73f"
             width="500" height="500" controls></video><br>
      <b>
        A 3D-rendered figure walks from the bedroom area through a stylish open-plan space with terrazzo flooring, past a mustard armchair and modern sofa, heading toward a raised marble-tiled zone near a decorative umbrella and potted plant — showcasing spatial flow and interior design in a minimalist, contemporary home.
      </b>
    </td>
  </tr>
</table>




## Overview

InteractAgent is a training-free framework for generating long-horizon, scene-aware human motion from images and natural-language goals. The paper formulates the system around three core stages:

1. **Scene understanding** with a multimodal LLM.
2. **Atomic action planning** in natural language.
3. **Motion execution** with a frozen MotionLLM executor.

The paper introduces two reflection mechanisms that make the system robust over long horizons:

- **Online Observation**: step-wise evaluation of each executed action to decide whether to continue, redo, or replan.
- **Memory-Augmented Reflection**: trajectory-level refinement across multiple attempts, with adaptive stopping when the task is already solved.

In this repository, the practical implementation centers on Qwen-VL based scene reasoning and Motion-Agent based motion generation. The code also includes an interactive planner, a closed-loop prompt refinement mode, and utilities for converting generated motion into SMPL-X parameters for downstream visualization.

## What is in this repository

- `main.py`: main entry point with `interactive`, `motion-generation`, and `closed-loop` modes.
- `interactive_scene_planner.py`: menu-driven workflow for scene analysis, task creation, motion generation, and multi-round prompt reflection.
- `interactive_qwenvl.py`: Qwen-VL API wrapper for image-conditioned reasoning.
- `prompt_templates.py`: scene-analysis, planning, trajectory, and reflection prompt templates plus response parsing.
- `enhanced_motion_generator.py`: Motion-Agent based motion generation with token concatenation and sequence decoding.
- `scene_motion_planner.py`: earlier end-to-end scene-to-motion pipeline for batch-style processing.
- `trumans_utils/`: utilities and configs for joint-to-SMPL-X conversion and TRUMANS-style assets.
- `Motion-Agent/`: bundled Motion-Agent code used as the low-level motion executor.

## Method Summary

The paper describes InteractAgent as a dual-loop system:

- The **forward pipeline** takes scene images and a task, extracts a scene description, decomposes the goal into short executable actions, and sends those actions to a frozen MotionLLM executor.
- **Online Observation** evaluates each rendered atomic action and can either accept it, redo it, or replan the remaining steps to prevent early mistakes from cascading.
- **Memory-Augmented Reflection** stores planning-execution history across attempts and lets the multimodal model improve the next plan using previous failures and successes.

This repository implements the same overall idea as a research codebase, but with a practical emphasis on Qwen-driven planning and Motion-Agent execution. The paper's main experiments use GPT-4o as the multimodal planner/evaluator, while the current scripts default to `qwen-vl-max` for scene analysis and reflection.

## Setup

### 1. Create the environment

```bash
conda create -n interactagent python=3.10
conda activate interactagent
pip install -r requirements.txt
```

### 2. Configure the multimodal API

Set a valid Qwen API key before running the planner:

```bash
export QWEN_API_KEY=your_api_key_here
```

The current code validates this key in `config.py`, and most planner paths rely on it.

### 3. Prepare the MotionLLM backbone and checkpoints

This repository includes the Motion-Agent source code, but the required model weights are not bundled.

- The motion generation path in `main.py` expects a MotionLLM checkpoint at `Motion-Agent/ckpt/motionllm.pth`.
- The enhanced generator also expects access to a Gemma backbone directory named `gemma2b`.
- The SMPL-X conversion path expects TRUMANS-related assets under `trumans_utils/`, including checkpoint files referenced by `trumans_utils/config/config_sample_synhsi.yaml`.

In other words, the source code is present, but you still need to place the pretrained weights in the expected locations before motion generation or SMPL-X conversion can run successfully.

### 4. Optional: verify the runtime environment

```bash
python main.py --check-env
```

This checks core Python dependencies and validates that the Qwen key and motion checkpoint path are available.

## Usage

### Interactive planning

Launch the interactive menu:

```bash
python main.py --mode interactive
```

This mode is the most complete end-to-end demo in the repository. It lets you:

1. Provide a single image or dual-view images.
2. Ask Qwen-VL to analyze the scene.
3. Generate an atomic motion plan in `"A person [action]"` format.
4. Execute the plan with MotionLLM.
5. Optionally convert the resulting motion to SMPL-X.

### Direct motion generation

If you already have a motion prompt and want to run MotionLLM directly:

```bash
python main.py \
  --mode motion-generation \
  --motion-gen "A person turns to the right; A person walks forward 3 meters" \
  --motion-output-dir ./demo
```

To also convert the generated motion to SMPL-X:

```bash
python main.py \
  --mode motion-generation \
  --motion-gen "A person walks forward 2 meters" \
  --motion-output-dir ./demo \
  --convert-smplx
```

### Closed-loop refinement

The repository also exposes a lightweight closed-loop mode that alternates between scene reasoning, MotionLLM generation, and prompt refinement:

```bash
python main.py \
  --mode closed-loop \
  --scene-image-url "https://your-public-image-url" \
  --task "Walk from the bed to the sofa and sit down" \
  --iterations 3 \
  --motion-output-dir ./demo
```

Useful options:

- `--qwen-model`: choose the Qwen model used for planning and reflection.
- `--rendered-video-url`: provide a public video URL so Qwen can critique an already rendered result.
- `--ask-video-url`: enter the render URL interactively after generation.
- `--prev-prompt`: seed the loop with a previous MotionLLM prompt that should be improved.
- `--history-file`: store reflection history as JSONL for later reuse.

## Inputs and Outputs

### Inputs

- Scene images are generally expected to be public URLs so the multimodal API can access them.
- Tasks are natural-language instructions such as navigation, reaching, sitting, or multi-step indoor interaction goals.
- Motion prompts are decomposed into short atomic steps, usually in the form `"A person ..."` for compatibility with the parser and MotionLLM workflow.

### Outputs

Depending on the mode and available assets, the pipeline can generate:

- `motionllm_generated.mp4`: a rendered motion video.
- `motionllm_generated.npy`: generated joint motion data.
- `motionllm_generated_smplx.pkl`: SMPL-X parameters for downstream visualization.
- `test_motionllm_integrated.py`: a helper Blender script generated during SMPL-X export.
- `logs/closed_loop_history.jsonl`: prompt/reflection history from closed-loop runs.

## Notes on the Current Implementation

- The repository is a research codebase, not a polished package. Some paths are hard-coded and assume local checkpoints are placed exactly where the scripts expect them.
- `api` mode is declared in `main.py` but is not implemented.
- The closed-loop mode in this repo performs prompt-level reflection and history-aware replanning, but it is still a simplified version of the full paper formulation with rendered trajectory feedback and adaptive stopping.
- The paper evaluates the framework on long-horizon human-scene interaction benchmarks including TRUMANS and UniHSI. This repository mainly exposes the generation and reflection pipeline used to build that system.

## Acknowledgements

This project builds on:

- [Motion-Agent](Motion-Agent/README.md) for the MotionLLM executor and token-based motion generation backbone.
- TRUMANS-related assets and utilities in `trumans_utils/` for joint-to-SMPL-X conversion.


