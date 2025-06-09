import os
import replicate
from dotenv import load_dotenv

load_dotenv()

def test_latentsync():
    """Test if LatentSync model is working"""
    
    print("Testing LatentSync model...")
    print("="*50)
    
    # Test 1: Try with demo URLs from Replicate docs
    print("\n1. Testing with Replicate's demo files...")
    try:
        output = replicate.run(
            "bytedance/latentsync:9d95ee5d66c993bbd3e0779dacd2dd6af6f542de93403aae36c6343455e0ca04",
            input={
                "video": "https://replicate.delivery/pbxt/MGZuEgzJZh6avv1LDEMppJZXLP9avGXqRuH7iAb7MBAz0Wu4/demo2_video.mp4",
                "audio": "https://replicate.delivery/pbxt/MGZuENopzAwWcpFsZ7SwoZ7itP4gvqasswPeEJwbRHTxtkwF/demo2_audio.wav"
            }
        )
        print(f"✓ Success! Output type: {type(output)}")
        print(f"Output: {str(output)[:100]}...")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        print(f"Error type: {type(e)}")
        
        # Check if it's an HTTP error
        if hasattr(e, '__cause__'):
            cause = e.__cause__
            if hasattr(cause, 'response'):
                print(f"HTTP Status: {cause.response.status_code if hasattr(cause.response, 'status_code') else 'Unknown'}")
                print(f"Response: {cause.response.text if hasattr(cause.response, 'text') else 'No text'}")
    
    # Test 2: Check model info
    print("\n2. Checking model information...")
    try:
        client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
        model = client.models.get("bytedance/latentsync")
        print(f"✓ Model found: {model.name}")
        print(f"Description: {model.description[:100]}...")
        print(f"Latest version: {model.latest_version.id if model.latest_version else 'Unknown'}")
        
    except Exception as e:
        print(f"✗ Model info failed: {e}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_latentsync()
