#!/usr/bin/env python3
"""
Optimized Video Caption Generator
================================

This is an optimized version that focuses on memory efficiency and speed.
Features:
- Smaller batch sizes to avoid CUDA OOM
- Aggressive memory cleanup
- Progress tracking with ETA
- Resume capability
"""

import os
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.cuda
from transformers import AutoProcessor, AutoModelForVision2Seq
from decord import VideoReader, cpu
import time
import traceback
from datetime import datetime, timedelta
import gc
from tqdm import tqdm


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def load_video_frames(video_path: str, max_frames_num: int = 256) -> Tuple[np.ndarray, List[float], float]:
    """Load uniform sampled frames from video"""
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()
        video_duration = total_frame_num / avg_fps
        
        # Uniform sampling
        if total_frame_num <= max_frames_num:
            frame_indices = list(range(total_frame_num))
        else:
            frame_indices = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
        
        # Get frame timestamps
        frame_times = [idx / avg_fps for idx in frame_indices]
        
        # Extract frames
        frames = vr.get_batch(frame_indices).asnumpy()
        
        return frames, frame_times, video_duration
        
    except Exception as e:
        raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")


class OptimizedCaptionGenerator:
    """Memory-optimized caption generator"""
    
    def __init__(self, device_id: int, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        self.model_name = model_name
        
        # Set environment variables for memory optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Set CUDA device
        torch.cuda.set_device(device_id)
        
        # Load model
        self._load_model()
        
        # Caption prompt for single image
        self.caption_prompt_single = (
            "You are an image captioning assistant for a Video Question Answering system. "
            "Your task is to produce a detailed descriptive caption for the image between 40 and 100 tokens. "
            "Your description cannot exceed 100 tokens. "
            "You must capture all important visual information without omission. "
            "Describe concrete elements such as objects, people, actions, background, colors, symbols, logos, or visible text with their positions. "
            "If some categories are absent (e.g., no text, no people), omit them without inventing details. "
            "Do not mention layout, alignment, design style, or atmosphere. "
            "The description must be factual, comprehensive, and cover every essential element that is visibly present. "
            "Output only the caption text."
        )
        
        # Caption prompt for batch processing
        self.caption_prompt_batch = (
            "You are an image captioning assistant for a Video Question Answering system. "
            "You will be given multiple images from a video sequence. "
            "For EACH image, produce a detailed descriptive caption between 40 and 100 tokens. "
            "Each description cannot exceed 100 tokens. "
            "You must capture all important visual information without omission for each image. "
            "Describe concrete elements such as objects, people, actions, background, colors, symbols, logos, or visible text with their positions. "
            "If some categories are absent (e.g., no text, no people), omit them without inventing details. "
            "Do not mention layout, alignment, design style, or atmosphere. "
            "Each description must be factual, comprehensive, and cover every essential element that is visibly present in that specific image. "
            "Provide one caption per image in the order they are presented. Output only the caption text for each image."
        )
    
    def _load_model(self):
        """Load model with memory optimizations"""
        print(f"[GPU {self.device_id}] Loading optimized model...")
        
        # Clear cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Load model with optimizations
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use fp16 instead of bf16 for better memory
                device_map={"": self.device},
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            
            self.model.eval()
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"[GPU {self.device_id}] Model loaded successfully")
            
        except Exception as e:
            print(f"[GPU {self.device_id}] Model loading failed: {e}")
            raise
    
    def generate_caption_single(self, image: Image.Image) -> str:
        """Generate caption for single image with aggressive memory management"""
        try:
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.caption_prompt_single}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                images=[image], 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate with memory optimization
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    temperature=None,  # Remove temperature to avoid warning
                )
            
            # Decode response
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            caption = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Aggressive cleanup
            del inputs, outputs, generated_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return caption if caption else "No caption generated"
            
        except torch.cuda.OutOfMemoryError as e:
            # Handle OOM gracefully
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return f"CUDA OOM Error: {str(e)}"
        except Exception as e:
            return f"Error generating caption: {str(e)}"
    
    def generate_captions_batch(self, images: List[Image.Image], batch_size: int = 4) -> List[str]:
        """Generate captions for multiple images with true batching and aggressive memory management"""
        captions = []
        
        print(f"[GPU {self.device_id}] Processing {len(images)} images in batches of {batch_size}")
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            try:
                # For batch processing, use a single conversation with multiple images
                if len(batch_images) == 1:
                    # Single image case
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": batch_images[0]},
                                {"type": "text", "text": self.caption_prompt_single}
                            ]
                        }
                    ]
                    text = self.processor.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    inputs = self.processor(
                        text=[text],
                        images=batch_images,
                        return_tensors="pt",
                        padding=True
                    )
                else:
                    # Multiple images - create one conversation with all images
                    content = []
                    for j, image in enumerate(batch_images):
                        content.append({"type": "image", "image": image})
                    
                    batch_instruction = (
                        "You are an image captioning assistant for a Video Question Answering system.\n"
                        f"Generate a detailed caption for each of the {len(batch_images)} images (40-100 tokens each). "
                        "You must capture all important visual information without omission.\n"
                        "Describe concrete elements such as objects, people, actions, background, colors, symbols, logos, "
                        "or visible text with their positions.\n"
                        f"Describe objects, people, actions, background, colors, symbols, logos, text with positions. "
                        "Do not mention layout, alignment, design style, or atmosphere.\n"
                        "The description must be factual, comprehensive, and cover every essential element that is visibly present.\n"
                        f"Provide {len(batch_images)} separate captions, one for each image in order."
                    )
                    content.append({"type": "text", "text": batch_instruction})
                    
                    messages = [{"role": "user", "content": content}]
                    text = self.processor.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    inputs = self.processor(
                        text=[text],
                        images=batch_images,
                        return_tensors="pt",
                        padding=True
                    )
                
                # Move to device
                inputs = {k: v.to(self.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Generate batch
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100 * len(batch_images),  # More tokens for multiple captions
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        temperature=None,  # Remove temperature to avoid warning
                    )
                
                # Decode the response
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                full_response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                # Parse multiple captions from the response
                if len(batch_images) == 1:
                    batch_captions = [full_response if full_response else "No caption generated"]
                else:
                    # Split the response into multiple captions
                    batch_captions = self._parse_batch_response(full_response, len(batch_images))
                
                captions.extend(batch_captions)
                
                # Aggressive cleanup
                del inputs, outputs, generated_tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                print(f"[GPU {self.device_id}] Batch {i//batch_size + 1} completed")
                
            except torch.cuda.OutOfMemoryError:
                print(f"[GPU {self.device_id}] Batch OOM, falling back to individual processing")
                
                # Cleanup first
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Process individually
                for image in batch_images:
                    caption = self.generate_caption_single(image)
                    captions.append(caption)
            
            except Exception as e:
                print(f"[GPU {self.device_id}] Batch processing error: {e}, falling back to individual")
                
                for image in batch_images:
                    caption = self.generate_caption_single(image)
                    captions.append(caption)
        
        return captions

    def _parse_batch_response(self, response: str, expected_count: int) -> List[str]:
        """Parse batch response into individual captions"""
        captions = []
        
        # Try splitting by common patterns
        patterns = [
            '\n\n',  # Double newline
            '\n',    # Single newline
            '. ',    # Period space (less reliable)
        ]
        
        best_split = None
        for pattern in patterns:
            parts = [part.strip() for part in response.split(pattern) if part.strip()]
            if len(parts) == expected_count:
                best_split = parts
                break
            elif len(parts) > expected_count:
                # Take first expected_count parts
                best_split = parts[:expected_count]
                break
        
        if best_split:
            captions = best_split
        else:
            # Fallback: split roughly by length
            avg_length = len(response) // expected_count
            captions = []
            for i in range(expected_count):
                start = i * avg_length
                end = (i + 1) * avg_length if i < expected_count - 1 else len(response)
                caption = response[start:end].strip()
                captions.append(caption if caption else f"Caption {i+1} parsing failed")
        
        # Ensure we have exactly the expected number of captions
        while len(captions) < expected_count:
            captions.append("Caption generation failed")
        
        return captions[:expected_count]

    def process_frames(self, frames: np.ndarray, batch_size: int = 1) -> List[str]:
        """Process all frames with batching and memory management"""
        # Convert all frames to PIL images
        images = [Image.fromarray(frame).convert('RGB') for frame in frames]
        
        # Use batch processing (no verbose output here)
        captions = self.generate_captions_batch(images, batch_size)
        
        return captions


def process_videos_worker(device_id: int, video_files: List[str], input_dir: str, output_dir: str, 
                         progress_queue: mp.Queue = None):
    """Worker function for processing videos with progress tracking"""
    
    try:
        # Initialize generator
        generator = OptimizedCaptionGenerator(device_id)
        
        # Process each video
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(input_dir, video_file)
            output_file = os.path.join(output_dir, f"{video_file}.json")
            
            # Skip if already exists
            if os.path.exists(output_file):
                if progress_queue:
                    progress_queue.put({
                        'device_id': device_id,
                        'video_file': video_file,
                        'status': 'skipped',
                        'progress': f"{i+1}/{len(video_files)}",
                        'time': 0
                    })
                continue
            
            start_time = time.time()
            
            try:
                # Load video with fewer frames to reduce memory usage
                frames, frame_times, video_duration = load_video_frames(video_path, 128)
                
                # Generate captions with conservative batch size for 7B model
                batch_size = 1  # Conservative batch size to avoid OOM on 7B model
                captions = generator.process_frames(frames, batch_size)
                
                # Prepare output
                output_data = {}
                for j, (frame_time, caption) in enumerate(zip(frame_times, captions)):
                    output_data[str(j)] = {
                        "ts": round(frame_time, 2),
                        "ts_formatted": format_timestamp(frame_time),
                        "caption": caption,
                        "frame_index": j  # Add explicit frame index for clarity
                    }
                
                # Add metadata
                metadata = {
                    "video_path": video_path,
                    "video_name": video_file,
                    "total_frames": len(frames),
                    "video_duration": round(video_duration, 2),
                    "video_duration_formatted": format_timestamp(video_duration),
                    "processed_at": datetime.now().isoformat(),
                    "model_name": generator.model_name,
                    "device_id": device_id
                }
                
                final_output = {
                    "metadata": metadata,
                    "frames": output_data
                }
                
                # Save
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, ensure_ascii=False, indent=2)
                
                processing_time = time.time() - start_time
                
                # Send progress update
                if progress_queue:
                    progress_queue.put({
                        'device_id': device_id,
                        'video_file': video_file,
                        'status': 'completed',
                        'progress': f"{i+1}/{len(video_files)}",
                        'time': processing_time,
                        'frames': len(frames),
                        'duration': video_duration
                    })
                
                # Cleanup after each video
                del frames, captions, output_data, final_output
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                processing_time = time.time() - start_time
                if progress_queue:
                    progress_queue.put({
                        'device_id': device_id,
                        'video_file': video_file,
                        'status': 'error',
                        'progress': f"{i+1}/{len(video_files)}",
                        'time': processing_time,
                        'error': str(e)
                    })
                continue
        
        # Send completion signal
        if progress_queue:
            progress_queue.put({
                'device_id': device_id,
                'status': 'worker_completed'
            })
        
    except Exception as e:
        if progress_queue:
            progress_queue.put({
                'device_id': device_id,
                'status': 'worker_error',
                'error': str(e)
            })


def progress_monitor(total_videos: int, num_gpus: int, progress_queue: mp.Queue):
    """Monitor and display progress from all workers"""
    completed_videos = 0
    skipped_videos = 0
    error_videos = 0
    workers_completed = 0
    
    start_time = time.time()
    video_times = []
    
    # Create progress bar
    pbar = tqdm(total=total_videos, desc="Processing videos", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    while workers_completed < num_gpus:
        try:
            # Get progress update with timeout
            update = progress_queue.get(timeout=1)
            
            if update['status'] == 'completed':
                completed_videos += 1
                video_times.append(update['time'])
                
                # Update progress bar with detailed info
                avg_time = sum(video_times) / len(video_times)
                remaining_videos = total_videos - completed_videos - skipped_videos - error_videos
                eta_seconds = remaining_videos * avg_time / num_gpus if video_times else 0
                
                pbar.set_postfix({
                    'GPU': update['device_id'],
                    'Time': f"{update['time']:.1f}s",
                    'Avg': f"{avg_time:.1f}s",
                    'ETA': f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                })
                pbar.update(1)
                
                # Detailed log (less frequent)
                if completed_videos % 10 == 0 or completed_videos == 1:
                    elapsed = time.time() - start_time
                    tqdm.write(f"\n✅ Progress Update:")
                    tqdm.write(f"   📹 Videos: {completed_videos} completed, {skipped_videos} skipped, {error_videos} errors")
                    tqdm.write(f"   ⏱️  Time: {elapsed/60:.1f}m elapsed, {avg_time:.1f}s/video average")
                    tqdm.write(f"   🎯 ETA: {eta_seconds/60:.1f}m remaining")
            
            elif update['status'] == 'skipped':
                skipped_videos += 1
                pbar.update(1)
                
            elif update['status'] == 'error':
                error_videos += 1
                tqdm.write(f"\n❌ GPU {update['device_id']}: Error processing {update['video_file']}")
                tqdm.write(f"   Error: {update.get('error', 'Unknown error')}")
                pbar.update(1)
                
            elif update['status'] == 'worker_completed':
                workers_completed += 1
                tqdm.write(f"\n🎉 GPU {update['device_id']} completed all assigned videos")
                
            elif update['status'] == 'worker_error':
                workers_completed += 1
                tqdm.write(f"\n💥 GPU {update['device_id']} worker failed: {update.get('error', 'Unknown error')}")
                
        except:
            # Timeout - continue monitoring
            continue
    
    pbar.close()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 Processing Complete!")
    print(f"=" * 50)
    print(f"📹 Total videos: {total_videos}")
    print(f"✅ Completed: {completed_videos}")
    print(f"⏭️  Skipped: {skipped_videos}")
    print(f"❌ Errors: {error_videos}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    if video_times:
        print(f"📊 Average per video: {sum(video_times)/len(video_times):.1f} seconds")
        print(f"🚀 Throughput: {len(video_times)/(total_time/60):.1f} videos/minute")
    print(f"=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Optimized video caption generator")
    parser.add_argument("--input_dir", type=str, default="/home/syh/.cache/huggingface/videomme/data")
    parser.add_argument("--output_dir", type=str, default="/home/syh/work-d/test/video/lmms-eval/qwen_videomme_captions")
    parser.add_argument("--num_gpus", type=int, default=8)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    video_files = [f for f in os.listdir(args.input_dir) 
                   if Path(f).suffix.lower() in video_extensions]
    
    print(f"🎬 Found {len(video_files)} video files")
    print(f"🖥️  Using {args.num_gpus} GPUs")
    print(f"📁 Input: {args.input_dir}")
    print(f"💾 Output: {args.output_dir}")
    print(f"=" * 50)
    
    # Split files among GPUs
    chunk_size = len(video_files) // args.num_gpus
    remainder = len(video_files) % args.num_gpus
    
    file_chunks = []
    start = 0
    for i in range(args.num_gpus):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_chunk_size
        file_chunks.append(video_files[start:end])
        print(f"GPU {i}: {len(file_chunks[i])} videos assigned")
        start = end
    
    # Create progress queue
    progress_queue = mp.Queue()
    
    # Start workers
    processes = []
    for i in range(args.num_gpus):
        if file_chunks[i]:
            p = mp.Process(
                target=process_videos_worker,
                args=(i, file_chunks[i], args.input_dir, args.output_dir, progress_queue)
            )
            p.start()
            processes.append(p)
    
    # Start progress monitor
    monitor_process = mp.Process(
        target=progress_monitor,
        args=(len(video_files), args.num_gpus, progress_queue)
    )
    monitor_process.start()
    
    # Wait for all workers to complete
    for p in processes:
        p.join()
    
    # Stop progress monitor
    monitor_process.join()
    
    print("🏁 All workers completed!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
