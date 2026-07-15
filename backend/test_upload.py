import requests
import time
import os

def run_upload_test():
    # 1. Download a tiny sample video to test with
    video_filename = "test_bunny.mp4"
    # Using a reliable, public domain sample video link
    video_url = "https://www.w3schools.com/html/mov_bbb.mp4"

    if not os.path.exists(video_filename):
        print(f"📥 Downloading sample video to current folder...")
        # Use requests with stream=True for robust downloading
        dl_response = requests.get(video_url, stream=True)
        dl_response.raise_for_status()  # Check if the download URL is valid
        
        with open(video_filename, 'wb') as f:
            for chunk in dl_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Download complete.")

    # 2. Send the video to our FastAPI endpoint
    print("🚀 Uploading video to FastAPI server (http://localhost:8080/api/v1/video/upload)...")
    
    with open(video_filename, "rb") as f:
        # We mimic a multipart/form-data browser upload
        files = {"file": (video_filename, f, "video/mp4")}
        response = requests.post("http://localhost:8080/api/v1/video/upload", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed with status {response.status_code}: {response.text}")
        return

    data = response.json()
    task_id = data["task_id"]
    print(f"✅ Upload accepted! Task ID: {task_id}")
    print(f"📂 Server reports file saved at: {data['file_path']}")

    # 3. Poll the background task status
    print("\n⏳ Polling Pixeltable background processing status...")
    while True:
        status_res = requests.get(f"http://localhost:8080/api/v1/video/task-status/{task_id}")
        status_data = status_res.json()
        status = status_data["status"]
        
        print(f"   Current Status: [{status.upper()}]")
        
        if status in ["completed", "failed"]:
            break
            
        time.sleep(3) # Wait 3 seconds before checking again

    if status == "completed":
        print("\n🎉 SUCCESS! The video was ingested, saved, and processed.")
        print("👉 Go check your project's root folder. You should see `storage/videos/test_bunny.mp4`!")
    else:
        print("\n⚠️ Processing failed. Check your Uvicorn terminal logs for the crash details.")

if __name__ == "__main__":
    run_upload_test()