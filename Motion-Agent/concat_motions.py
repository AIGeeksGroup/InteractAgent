#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作拼接模块 - 将多个动作序列拼接成一个完整的动作
"""

import os
import numpy as np
import torch
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain


def concat_motions(motion_files, output_dir, out_name="concat_motion"):
    """
    拼接多个动作文件
    
    Args:
        motion_files: 动作文件路径列表
        output_dir: 输出目录
        out_name: 输出文件名前缀
        
    Returns:
        拼接结果字典
    """
    try:
        print(f"🔗 开始拼接 {len(motion_files)} 个动作文件...")
        
        # 加载所有动作数据
        motions = []
        for i, file_path in enumerate(motion_files):
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                continue
                
            motion_data = np.load(file_path)
            print(f"📁 加载动作 {i+1}: {motion_data.shape}")
            motions.append(motion_data)
        
        if not motions:
            return {"error": "没有有效的动作文件"}
        
        # 拼接动作数据
        concatenated_motion = np.concatenate(motions, axis=0)
        print(f"✅ 拼接完成，总帧数: {concatenated_motion.shape[0]}")
        
        # 保存拼接后的数据
        concat_npy_path = os.path.join(output_dir, f"{out_name}.npy")
        np.save(concat_npy_path, concatenated_motion)
        
        # 生成拼接后的视频
        concat_mp4_path = os.path.join(output_dir, f"{out_name}.mp4")
        
        # 恢复3D关节数据
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        motion_tensor = torch.from_numpy(concatenated_motion).float().to(device)
        motion_recovered = recover_from_ric(motion_tensor, joints_num=22)
        
        # 生成视频
        plot_3d_motion(
            concat_mp4_path,
            t2m_kinematic_chain,
            motion_recovered.squeeze().cpu().numpy(),
            title=f"Concatenated Motion: {out_name}",
            fps=20,
            radius=4
        )
        
        result = {
            "output_dir": output_dir,
            "concat_npy": concat_npy_path,
            "final_video": concat_mp4_path,
            "total_frames": concatenated_motion.shape[0],
            "input_files": motion_files,
            "success": True
        }
        
        print(f"🎉 动作拼接完成！")
        print(f"📁 数据文件: {concat_npy_path}")
        print(f"🎬 视频文件: {concat_mp4_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ 动作拼接失败: {e}")
        return {"error": f"动作拼接失败: {e}"}


def smooth_transition(motion1, motion2, transition_frames=5):
    """
    在两个动作之间创建平滑过渡
    
    Args:
        motion1: 第一个动作数据
        motion2: 第二个动作数据
        transition_frames: 过渡帧数
        
    Returns:
        平滑过渡后的动作数据
    """
    if transition_frames <= 0:
        return np.concatenate([motion1, motion2], axis=0)
    
    # 获取过渡区域
    end_frames = motion1[-transition_frames:]
    start_frames = motion2[:transition_frames]
    
    # 创建平滑过渡
    transition = np.zeros((transition_frames, motion1.shape[1]))
    
    for i in range(transition_frames):
        alpha = i / (transition_frames - 1)
        transition[i] = (1 - alpha) * end_frames[i] + alpha * start_frames[i]
    
    # 拼接：motion1[:-transition_frames] + transition + motion2[transition_frames:]
    result = np.concatenate([
        motion1[:-transition_frames],
        transition,
        motion2[transition_frames:]
    ], axis=0)
    
    return result


def concat_motions_with_smooth_transition(motion_files, output_dir, out_name="smooth_concat_motion", transition_frames=5):
    """
    带平滑过渡的动作拼接
    
    Args:
        motion_files: 动作文件路径列表
        output_dir: 输出目录
        out_name: 输出文件名前缀
        transition_frames: 过渡帧数
        
    Returns:
        拼接结果字典
    """
    try:
        print(f"🔗 开始平滑拼接 {len(motion_files)} 个动作文件...")
        
        # 加载所有动作数据
        motions = []
        for i, file_path in enumerate(motion_files):
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                continue
                
            motion_data = np.load(file_path)
            print(f"📁 加载动作 {i+1}: {motion_data.shape}")
            motions.append(motion_data)
        
        if not motions:
            return {"error": "没有有效的动作文件"}
        
        if len(motions) == 1:
            concatenated_motion = motions[0]
        else:
            # 逐步平滑拼接
            concatenated_motion = motions[0]
            for i in range(1, len(motions)):
                concatenated_motion = smooth_transition(
                    concatenated_motion, 
                    motions[i], 
                    transition_frames
                )
                print(f"✅ 已拼接动作 {i+1}, 当前总帧数: {concatenated_motion.shape[0]}")
        
        print(f"🎉 平滑拼接完成，总帧数: {concatenated_motion.shape[0]}")
        
        # 保存拼接后的数据
        concat_npy_path = os.path.join(output_dir, f"{out_name}.npy")
        np.save(concat_npy_path, concatenated_motion)
        
        # 生成拼接后的视频
        concat_mp4_path = os.path.join(output_dir, f"{out_name}.mp4")
        
        # 恢复3D关节数据
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        motion_tensor = torch.from_numpy(concatenated_motion).float().to(device)
        motion_recovered = recover_from_ric(motion_tensor, joints_num=22)
        
        # 生成视频
        plot_3d_motion(
            motion_recovered.squeeze().cpu().numpy(),
            concat_mp4_path,
            title=f"Smooth Concatenated Motion: {out_name}",
            fps=20,
            radius=4
        )
        
        result = {
            "output_dir": output_dir,
            "concat_npy": concat_npy_path,
            "final_video": concat_mp4_path,
            "total_frames": concatenated_motion.shape[0],
            "input_files": motion_files,
            "transition_frames": transition_frames,
            "success": True
        }
        
        print(f"🎬 平滑拼接视频已生成: {concat_mp4_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ 平滑拼接失败: {e}")
        return {"error": f"平滑拼接失败: {e}"}


if __name__ == "__main__":
    # 测试函数
    print("🧪 测试动作拼接功能...")
    
    # 这里可以添加测试代码
    test_files = [
        "motion_step_1.npy",
        "motion_step_2.npy", 
        "motion_step_3.npy"
    ]
    
    # 检查文件是否存在
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        print(f"找到 {len(existing_files)} 个测试文件")
        result = concat_motions(existing_files, "./", "test_concat")
        print(f"测试结果: {result}")
    else:
        print("没有找到测试文件，跳过测试")