import argparse
import torch
from PIL import Image
import time

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an image with nanoVLM")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-230M-8k",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--image", type=str, default="assets/image.png",
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of tokens per output")
    parser.add_argument("--measure_vram", action="store_true",
                        help="Measure and display VRAM usage during model loading and generation")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for faster inference (first run will be slow)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for sampling (None for greedy decoding)")
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    # Measure model loading time
    load_start = time.time()
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()
    
    # Optional: compile model for faster inference
    if args.compile:
        print("Compiling model... (first run will be slow)")
        model = torch.compile(model)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_time = time.time() - load_start
    print(f"Model loading time: {load_time:.2f}s")
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.synchronize()
        model_vram_bytes = torch.cuda.memory_allocated(device)
        model_vram_mb = model_vram_bytes / (1024 ** 2)
        print(f"VRAM used after loading model: {model_vram_mb:.2f} MB")

    # Get tokenizer and image processor from model config if not provided
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    resize_to_max_side_len = False
    if hasattr(model.cfg, "resize_to_max_side_len"):
        resize_to_max_side_len = model.cfg.resize_to_max_side_len
    image_processor = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size, resize_to_max_side_len)

    # Measure image processing time
    preprocess_start = time.time()
    img = Image.open(args.image).convert("RGB")
    processed_image, splitted_image_ratio = image_processor(img)
    if not hasattr(tokenizer, "global_image_token") and splitted_image_ratio[0]*splitted_image_ratio[1] == len(processed_image) - 1:
        # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
        processed_image = processed_image[1:]

    image_string = get_image_string(tokenizer, [splitted_image_ratio], model.cfg.mp_image_token_length)

    messages = [{"role": "user", "content": image_string + args.prompt}]
    encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
    tokens = torch.tensor(encoded_prompt).to(device)
    img_t = processed_image.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    preprocess_time = time.time() - preprocess_start
    print(f"Image preprocessing time: {preprocess_time:.3f}s")

    print("\nInput:\n ", args.prompt, "\n\nOutput:")
    
    generation_times = []
    total_start = time.time()
    
    for i in range(args.generations):
        gen_start = time.time()
        
        # Generate with optional temperature control
        if args.temperature is not None:
            # Sampling mode
            gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens, 
                               temperature=args.temperature, greedy=False)
        else:
            # Greedy decoding (faster, deterministic)
            gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens, greedy=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.time() - gen_start
        generation_times.append(gen_time)
        
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        
        if args.measure_vram and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_bytes = torch.cuda.max_memory_allocated(device)
            peak_vram_mb = peak_vram_bytes / (1024 ** 2)
            current_vram_bytes = torch.cuda.memory_allocated(device)
            current_vram_mb = current_vram_bytes / (1024 ** 2)
            print(f"  >> Generation {i+1} ({gen_time:.2f}s): {out}")
            print(f"     VRAM - Peak: {peak_vram_mb:.2f} MB, Current: {current_vram_mb:.2f} MB")
        else:
            print(f"  >> Generation {i+1} ({gen_time:.2f}s): {out}")
    
    total_time = time.time() - total_start
    avg_time = sum(generation_times) / len(generation_times)
    
    print(f"\n{'='*60}")
    print(f"Timing Summary:")
    print(f"  Model loading: {load_time:.2f}s")
    print(f"  Image preprocessing: {preprocess_time:.3f}s")
    print(f"  Average generation time: {avg_time:.2f}s")
    print(f"  Total generation time: {total_time:.2f}s")
    print(f"  Tokens/second (approx): {args.max_new_tokens / avg_time:.1f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
