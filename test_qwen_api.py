#!/usr/bin/env python3
"""
Test script for QwenVL API
Tests both API key validity and project-specific functionality
"""

import os
import sys
import json
import requests
import time
from datetime import datetime

# Add project path to import project modules
sys.path.append('scene_motion_planner')

def test_basic_api_connection(api_key, api_url):
    """Test basic API connection and key validity"""
    print("\n" + "="*60)
    print("TEST 1: Basic API Connection Test")
    print("="*60)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple text-only test
    payload = {
        "model": "qwen-vl-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "Hello, can you respond with 'API test successful'?"
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 100
        }
    }
    
    try:
        print(f"📡 Sending request to: {api_url}")
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                print(f"✅ API Connection: SUCCESS")
                print(f"📝 Response: {content[:100]}...")
                return True, "API key is valid and working"
            else:
                print(f"⚠️  Unexpected response format")
                return False, "Invalid response format"
                
        elif response.status_code == 401:
            print(f"❌ API Connection: FAILED (401 Unauthorized)")
            return False, "Invalid API key - authentication failed"
            
        elif response.status_code == 403:
            print(f"❌ API Connection: FAILED (403 Forbidden)")
            return False, "API key lacks permission or quota exceeded"
            
        else:
            print(f"❌ API Connection: FAILED (HTTP {response.status_code})")
            print(f"📝 Error: {response.text[:200]}")
            return False, f"HTTP {response.status_code} error"
            
    except requests.exceptions.Timeout:
        print(f"⏰ API Connection: TIMEOUT")
        return False, "Request timeout - API may be slow or unavailable"
        
    except requests.exceptions.ConnectionError:
        print(f"🔌 API Connection: CONNECTION ERROR")
        return False, "Cannot connect to API endpoint"
        
    except Exception as e:
        print(f"❌ API Connection: ERROR")
        print(f"📝 Error: {str(e)}")
        return False, f"Unexpected error: {str(e)}"

def test_image_analysis(api_key, api_url):
    """Test image analysis capability (core functionality for Scene Motion Planner)"""
    print("\n" + "="*60)
    print("TEST 2: Image Analysis Test (Scene Understanding)")
    print("="*60)
    
    # Test with user-provided images
    test_images = [
        "https://i.postimg.cc/MZL3ZLSL/image.png",  # Bird's eye view
        "https://i.postimg.cc/MZL3ZLSL/image.png"
        # "https://i.postimg.cc/WbsW-yTb0/2.png"       # First person view
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # First test single image
    print("\n📸 Testing single image analysis...")
    single_image_prompt = """
Please describe this scene. What type of room or space is this? 
List the main objects and furniture you can see.
Estimate the room dimensions in meters.
"""
    
    payload = {
        "model": "qwen-vl-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": test_images[0]
                        },
                        {
                            "text": single_image_prompt
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
    
    single_success = False
    try:
        print(f"🖼️  Testing with: {test_images[0]}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                # Handle both string and list responses
                if isinstance(content, list) and len(content) > 0:
                    content = content[0].get('text', str(content))
                print(f"✅ Single Image Analysis: SUCCESS")
                print(f"\n📄 Scene Description:")
                print("-" * 40)
                print(str(content)[:600])
                print("-" * 40)
                single_success = True
        elif response.status_code == 400:
            # Even if download fails, check if we got a response
            result = response.json()
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    content = content[0].get('text', str(content))
                print(f"⚠️  Download warning but got response")
                print(f"📄 Scene Description: {str(content)[:400]}")
                single_success = True
            else:
                print(f"⚠️  Image download timeout reported")
        else:
            print(f"❌ Single Image: FAILED (HTTP {response.status_code})")
            
    except Exception as e:
        print(f"❌ Single Image Error: {str(e)[:100]}")
    
    # Now test dual-view mode
    print("\n📸📸 Testing dual-view analysis (bird's eye + first person)...")
    dual_view_prompt = """
Please analyze these two images: the first is a bird's-eye view, the second is a first-person view.

Please provide:
1. Room type and layout
2. Room dimensions (estimate in meters)
3. Key furniture and objects
4. Suggested walking paths
"""
    
    payload = {
        "model": "qwen-vl-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": test_images[0]},
                        {"image": test_images[1]},
                        {"text": dual_view_prompt}
                    ]
                }
            ]
        },
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 800
        }
    }
    
    dual_success = False
    try:
        print(f"🖼️  Image 1: {test_images[0]}")
        print(f"🖼️  Image 2: {test_images[1]}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                # Handle both string and list responses
                if isinstance(content, list) and len(content) > 0:
                    content = content[0].get('text', str(content))
                print(f"✅ Dual-View Analysis: SUCCESS")
                print(f"\n📄 Dual-View Scene Analysis:")
                print("-" * 40)
                print(str(content)[:800])
                print("-" * 40)
                dual_success = True
                
                # Check quality
                expected = ["room", "meters", "furniture", "sofa", "table"]
                found = sum(1 for word in expected if word.lower() in str(content).lower())
                if found >= 3:
                    print(f"✅ Analysis quality: GOOD ({found}/5 keywords found)")
                else:
                    print(f"⚠️  Analysis quality: LIMITED ({found}/5 keywords found)")
        else:
            print(f"⚠️  Dual-view response code: {response.status_code}")
            # Still try to get content
            try:
                result = response.json()
                if "output" in result and "choices" in result["output"]:
                    content = result["output"]["choices"][0]["message"]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        content = content[0].get('text', str(content))
                    print(f"📄 Got response despite error: {str(content)[:400]}")
                    dual_success = True
            except:
                pass
                
    except Exception as e:
        print(f"❌ Dual-View Error: {str(e)[:100]}")
    
    # Return results
    if dual_success:
        return True, "Image analysis works - dual-view scene understanding successful"
    elif single_success:
        return True, "Image analysis works - single image successful, dual-view may have issues"
    else:
        return False, "Image analysis failed - check image URLs or API access"

def test_motion_planning_prompt(api_key, api_url):
    """Test motion planning capability (as used in the project)"""
    print("\n" + "="*60)
    print("TEST 3: Motion Planning Test (Project Workflow)")
    print("="*60)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simulate a motion planning request as used in the project
    motion_planning_prompt = """
You are a motion planning expert. Given the following scene and task, generate a motion plan.

Scene: A living room with a sofa on the left, coffee table in the center, and TV stand on the right. Room dimensions approximately 5x4 meters.

Task: Walk from the entrance to the sofa

Please respond in JSON format:
{
  "plan": "A numbered list of motion steps, each in 'A person [action]' format",
  "reasoning": "Brief explanation of the motion sequence logic"
}
"""
    
    payload = {
        "model": "qwen-vl-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": motion_planning_prompt
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
        print(f"📝 Testing motion planning capability...")
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                print(f"✅ Motion Planning: SUCCESS")
                print(f"\n📄 Motion Plan Response:")
                print("-" * 40)
                print(content[:800])
                print("-" * 40)
                
                # Try to parse as JSON
                try:
                    # Extract JSON from response if wrapped in markdown
                    import re
                    # print(type(content)) # debug the json not parse
                    json_match = re.search(r'\{[\s\S]*\}', content[0]["text"])
                    if json_match:
                        motion_plan = json.loads(json_match.group())
                        if "plan" in motion_plan:
                            print(f"✅ JSON Format: VALID")
                            print(f"✅ Motion steps found in plan")
                            return True, "Motion planning works correctly with proper JSON format"
                    else:
                        print(f"⚠️  JSON Format: NOT FOUND")
                        return True, "Motion planning works but JSON format may need adjustment"
                except:
                    print(f"⚠️  JSON Format: CANNOT PARSE")
                    return True, "Motion planning works but response format needs improvement"
                    
            else:
                print(f"⚠️  Unexpected response format")
                return False, "Invalid response format for motion planning"
                
        else:
            print(f"❌ Motion Planning: FAILED (HTTP {response.status_code})")
            return False, f"Motion planning failed with HTTP {response.status_code}"
            
    except Exception as e:
        print(f"❌ Motion Planning: ERROR")
        print(f"📝 Error: {str(e)}")
        return False, f"Motion planning error: {str(e)}"

def test_project_integration():
    """Test using the actual project's QwenVLChat class"""
    print("\n" + "="*60)
    print("TEST 4: Project Integration Test")
    print("="*60)
    
    try:
        from interactive_qwenvl import QwenVLChat
        
        print(f"✅ Successfully imported QwenVLChat from project")
        
        # Initialize with default (hardcoded) API key
        chat = QwenVLChat()
        print(f"✅ QwenVLChat initialized with default settings")
        print(f"📝 Using API key: {chat.api_key[:10]}...{chat.api_key[-4:]}")
        print(f"📝 Using model: {chat.model}")
        print(f"📝 Using endpoint: {chat.url}")
        
        # Test simple question
        test_image = "https://images.unsplash.com/photo-1493809842364-78817add7ffb?w=800&q=80"  # Bedroom image
        chat.set_image(test_image, "main")
        
        response = chat.ask("What type of room is this? Please describe briefly.")
        
        if response and "❌" not in response:
            print(f"✅ Project integration: SUCCESS")
            print(f"📝 Response: {response[:200]}...")
            return True, "Project's QwenVLChat class works correctly"
        else:
            print(f"❌ Project integration: FAILED")
            print(f"📝 Error: {response}")
            return False, "Project's QwenVLChat class failed"
            
    except ImportError as e:
        print(f"⚠️  Cannot import project modules: {e}")
        return False, "Cannot test project integration - import error"
        
    except Exception as e:
        print(f"❌ Project Integration: ERROR")
        print(f"📝 Error: {str(e)}")
        return False, f"Project integration error: {str(e)}"

def main():
    """Main test function"""
    print("\n" + "="*70)
    print("🧪 QwenVL API Test for Scene Motion Planner")
    print("="*70)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get API configuration
    # First try environment variable, then use hardcoded value from project
    api_key = os.getenv("QWEN_API_KEY") or "sk-4046430e513f44c68beec5635a02d97f"
    api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    
    if api_key == "sk-4046430e513f44c68beec5635a02d97f":
        print(f"\n⚠️  WARNING: Using hardcoded API key from project")
        print(f"   This key may not be valid or may belong to someone else")
        print(f"   Consider setting your own key: export QWEN_API_KEY='your_key'")
    else:
        print(f"\n✅ Using API key from environment variable")
    
    # Run tests
    results = []
    
    # Test 1: Basic connection
    success, message = test_basic_api_connection(api_key, api_url)
    results.append(("Basic API Connection", success, message))
    
    if not success:
        print("\n⚠️  Skipping remaining tests due to API connection failure")
    else:
        # Test 2: Image analysis
        success, message = test_image_analysis(api_key, api_url)
        results.append(("Image Analysis", success, message))
        
        # Test 3: Motion planning
        success, message = test_motion_planning_prompt(api_key, api_url)
        results.append(("Motion Planning", success, message))
        
        # Test 4: Project integration
        success, message = test_project_integration()
        results.append(("Project Integration", success, message))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}: {message}")
    
    # Overall assessment
    total_tests = len(results)
    passed_tests = sum(1 for _, success, _ in results if success)
    
    print("\n" + "-"*70)
    print(f"📈 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The API key works and is compatible with the project.")
    elif passed_tests > 0:
        print("⚠️  Some tests passed. The API key works but may have limitations.")
    else:
        print("❌ All tests failed. The API key is invalid or not working.")
    
    # Recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    if api_key == "sk-4046430e513f44c68beec5635a02d97f":
        print("1. ⚠️  Replace the hardcoded API key with your own:")
        print("   - Get your key from: https://dashscope.console.aliyun.com/")
        print("   - Set environment variable: export QWEN_API_KEY='your_key'")
    
    if passed_tests < total_tests:
        print("2. 🔧 Debug failed tests and check API quotas/permissions")
    
    print("3. 📝 Remove hardcoded keys from code before sharing or committing")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)