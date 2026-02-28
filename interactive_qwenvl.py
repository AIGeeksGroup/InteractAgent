import os
import requests
import json

def convert_github_url_to_raw(github_url):
    """
    Convert GitHub blob link to raw link
    
    Args:
        github_url: GitHub blob link
    
    Returns:
        raw link
    """
    if 'github.com' in github_url and '/blob/' in github_url:
        raw_url = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        return raw_url
    return github_url

class QwenVLChat:
    def __init__(self, api_key=None, model="qwen-vl-max"):
        """
        Initialize QwenVL chat client
        
        Args:
            api_key: API Key, if empty will get from environment variable
            model: model name
        """
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or "sk-4046430e513f44c68beec5635a02d97f"
        self.model = model
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.current_images = {}  # Store multiple images
        self.image_order = []     # Image order
        
    def set_image(self, image_url, image_type="main"):
        """
        Set single image
        
        Args:
            image_url: Image URL
            image_type: Image type identifier (e.g., "main", "birdview", "egoview")
        """
        # If it's a GitHub link, convert to raw link
        processed_url = convert_github_url_to_raw(image_url)
        self.current_images[image_type] = processed_url
        
        # If it's a new type, add to order list
        if image_type not in self.image_order:
            self.image_order.append(image_type)
            
        if processed_url != image_url:
            print(f"✅ {image_type} image set: {image_url}")
            print(f"🔄 Converted to Raw link: {processed_url}")
        else:
            print(f"✅ {image_type} image set: {image_url}")
    
    def set_dual_view_images(self, birdview_url, egoview_url):
        """
        Set dual-view images (bird's-eye view + first-person view)
        
        Args:
            birdview_url: Bird's-eye view image URL
            egoview_url: First-person view image URL
        """
        self.set_image(birdview_url, "birdview")
        self.set_image(egoview_url, "egoview")
        print("🎯 Dual-view images set: Bird's-eye view + First-person view")
    
    def clear_images(self):
        """Clear all images"""
        self.current_images.clear()
        self.image_order.clear()
        print("🗑️ All images cleared")
    
    def get_image_count(self):
        """Get current image count"""
        return len(self.current_images)
    
    def list_images(self):
        """List all current images"""
        if not self.current_images:
            return "📷 No images are currently set"
        
        result = "📷 Currently set images:\n"
        for img_type in self.image_order:
            if img_type in self.current_images:
                result += f"  • {img_type}: {self.current_images[img_type]}\n"
        return result
        
    def check_network_connection(self):
        """检查网络连接状态"""
        try:
            import requests
            # Test basic network connection
            test_response = requests.get("https://www.baidu.com", timeout=10)
            if test_response.status_code == 200:
                print("✅ Network connection normal")
                return True
            else:
                print("⚠️ Network connection abnormal")
                return False
        except Exception as e:
            print(f"❌ Network connection failed: {e}")
            return False
    
    def check_api_endpoint(self):
        """Check API endpoint accessibility"""
        try:
            import requests
            # Test API endpoint (using HEAD request, more lightweight)
            test_response = requests.head("https://dashscope.aliyuncs.com", timeout=10)
            # Any response indicates endpoint is accessible (including 403, 401, etc.)
            print(f"✅ API endpoint accessible - Status code: {test_response.status_code}")
            return True
        except requests.exceptions.Timeout:
            print("⏰ API endpoint access timeout")
            return False
        except requests.exceptions.ConnectionError:
            print("❌ API endpoint connection error")
            return False
        except Exception as e:
            print(f"❌ API endpoint access failed: {e}")
            return False

    def ask(self, question):
        """
        Ask the model questions about current images
        
        Args:
            question: Question content
            
        Returns:
            Model's answer
        """
        if not self.current_images:
            return "❌ Please set images first! Use set_image() or set_dual_view_images() methods to set images."
            
        if not question or not question.strip():
            return "❌ Please enter a valid question!"
        
        # Check network connection
        print("🔍 Checking network connection...")
        if not self.check_network_connection():
            print("⚠️ Network connection check failed, but continuing with API call...")
        
        # Simplify API endpoint check, directly try API call
        print("🔄 Preparing to send API request...")
        
        # Build multimodal input
        content = []
        
        # Add images in order
        for img_type in self.image_order:
            if img_type in self.current_images:
                content.append({
                    "image": self.current_images[img_type]
                })
        
        # Add text question
        content.append({
            "text": question.strip()
        })
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            },
            "parameters": {
                "temperature": 0.1,  # Reduce randomness
                "top_p": 0.8,
                "max_tokens": 1500  # Increase token count to support multi-image analysis
            }
        }
        
        try:
            print(f"🔄 Sending API request to: {self.url}")
            print(f"📷 Image count: {len(self.current_images)}")
            for img_type in self.image_order:
                if img_type in self.current_images:
                    print(f"  • {img_type}: {self.current_images[img_type]}")
            print(f"❓ Question: {question.strip()}")
            
            # Increase timeout to 120 seconds and add retry mechanism
            max_retries = 3
            timeout = 120  # Increase to 120 seconds
            
            for attempt in range(max_retries):
                try:
                    print(f"🔄 Attempt {attempt + 1}/{max_retries}...")
                    response = requests.post(self.url, headers=headers, json=payload, timeout=timeout)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if "output" in result and "choices" in result["output"]:
                            answer = result["output"]["choices"][0]["message"]["content"]
                            print(f"✅ Received answer, length: {len(answer)} characters")
                            
                            # Handle different response formats
                            if isinstance(answer, list):
                                # If it's a list format, extract text content
                                text_parts = []
                                for item in answer:
                                    if isinstance(item, dict) and 'text' in item:
                                        text_parts.append(item['text'])
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                answer = '\n'.join(text_parts)
                                print(f"🔄 Processing list format response, extracted text length: {len(answer)} characters")
                            elif isinstance(answer, str):
                                # If already a string, use directly
                                pass
                            else:
                                # Other formats, convert to string
                                answer = str(answer)
                                print(f"🔄 Converting response format, length: {len(answer)} characters")
                            
                            # Ensure answer is string type
                            if not isinstance(answer, str):
                                answer = str(answer)
                            
                            return str(answer)  # Ensure return string
                        else:
                            print(f"⚠️ API response format abnormal: {result}")
                            return f"❌ API response format abnormal: {str(result)}"  # Ensure convert to string
                    else:
                        print(f"❌ API request failed, status code: {response.status_code}")
                        print(f"Error message: {response.text}")
                        if attempt < max_retries - 1:
                            print(f"⏳ Waiting 5 seconds before retry...")
                            import time
                            time.sleep(5)
                            continue
                        else:
                            return f"❌ API request failed, status code: {response.status_code}, error: {response.text}"
                            
                except requests.exceptions.Timeout:
                    print(f"⏰ Request timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        print(f"⏳ Waiting 10 seconds before retry...")
                        import time
                        time.sleep(10)
                        continue
                    else:
                        return "❌ Request timeout, please try again later"
                        
        except requests.exceptions.RequestException as e:
            print(f"❌ Network request exception: {e}")
            return f"❌ Network request exception: {e}"
        except Exception as e:
            print(f"❌ Unknown error: {e}")
            return f"❌ Unknown error: {e}"
    
    def ask_with_context(self, question, context=""):
        """
        Ask questions with context
        
        Args:
            question: Question content
            context: Context information
            
        Returns:
            Model's answer
        """
        if context:
            full_question = f"Context: {context}\n\nQuestion: {question}"
        else:
            full_question = question
            
        return self.ask(full_question)

def main():
    """Main interaction loop"""
    print("🎉 Welcome to QwenVL Real-time Image Recognition Chat Program!")
    print("=" * 50)
    
    # Initialize chat client
    chat = QwenVLChat()
    
    # Set default image (using GitHub image address)
    default_image = "https://github.com/ghppcx/Object-Detection/blob/main/%E6%97%A0%E6%A0%87%E9%A2%98.png"
    chat.set_image(default_image)
    
    print("\n📋 Available commands:")
    print("  - Direct question: Ask questions about current images")
    print("  - set_image <url>: Set new image link")
    print("  - help: Show help information")
    print("  - quit/exit: Exit program")
    print("  - clear: Clear screen")
    
    print(f"\n🖼️  Current image: {chat.current_image}")
    print("\n💬 Start chatting! (Enter 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤔 You: ").strip()
            
            # Handle exit command
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thank you for using QwenVL Image Recognition Program!")
                break
                
            # Handle empty input
            if not user_input:
                print("💡 Please enter a question or command...")
                continue
                
            # Handle help command
            if user_input.lower() in ['help', 'h']:
                print("\n📋 Available commands:")
                print("  - Direct question: Ask questions about current images")
                print("  - set_image <url>: Set new image link")
                print("  - help: Show help information")
                print("  - quit/exit: Exit program")
                print("  - clear: Clear screen")
                print(f"\n🖼️  Current image: {chat.current_image}")
                continue
                
            # Handle clear screen command
            if user_input.lower() in ['clear', 'cls']:
                os.system('clear' if os.name == 'posix' else 'cls')
                print("🎉 QwenVL Real-time Image Recognition Chat Program")
                print(f"🖼️  Current image: {chat.current_image}")
                continue
                
            # Handle set image command
            if user_input.lower().startswith('set_image '):
                new_image = user_input[10:].strip()
                if new_image:
                    chat.set_image(new_image)
                else:
                    print("❌ Please provide image link! Example: set_image https://example.com/image.jpg")
                continue
                
            # Handle regular questions
            print("\n🤖 QwenVL: Analyzing image, please wait...")
            
            # Call model
            response = chat.ask(user_input)
            
            # Show answer
            print(f"\n🤖 QwenVL: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Ctrl+C detected, program exiting. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Program error: {e}")
            print("💡 Please retry or enter 'quit' to exit program")

if __name__ == "__main__":
    main()