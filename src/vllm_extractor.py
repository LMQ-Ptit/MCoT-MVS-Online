import os
import re
import json
import base64
from io import BytesIO
from typing import Union, Optional, Dict, Any
from PIL import Image
import torch

COT_SYSTEM_PROMPT = (
    "You are an expert multimodal reasoning assistant for Composed Image Retrieval (CIR).\n"
    "Your task is to analyze a Reference Image and a user's Modification Instruction, "
    "and perform multi-step Chain-of-Thought reasoning to decompose the user's intent."
)

COT_USER_PROMPT_TEMPLATE = (
    'You are given a Reference Image and a user\'s Modification Instruction: "{modification_text}".\n'
    "Carefully reason step-by-step:\n"
    "1. Identify the key attributes of the clothing in the Reference Image (color, sleeves, style).\n"
    "2. Identify what specific changes are requested in the Modification Instruction.\n"
    "3. Imagine the Target Image after applying the changes.\n"
    "4. Distinguish which attributes are RETAINED from the reference, and which are REMOVED/CHANGED.\n"
    "Finally, summarize your conclusion in this JSON block:\n"
    "```json\n"
    "{\n"
    '  "target": "<detailed caption of the modified target clothing>",\n'
    '  "retained": "<attributes kept from the original image>",\n'
    '  "deleted": "<attributes from the original image that were replaced or removed>"\n'
    "}\n"
    "```"
)


class MCoTExtractor:
    """
    Multimodal Chain-of-Thought (MCoT) Extractor for MCoT-MVS (WWW 2026).
    Given:
        - Reference Image (PIL.Image or file path)
        - Modification Text (str)
    Returns:
        dict with:
            - 'target': Inferred description of the target image (TT)
            - 'retained': Visual elements kept from reference image (RT)
            - 'deleted': Visual elements removed / altered (DT)
    """

    def __init__(
        self,
        mode: str = "local",
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda:1",
        torch_dtype: Optional[torch.dtype] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        mode:
            - 'local': Uses HuggingFace Transformers (Qwen2.5-VL-3B-Instruct)
            - 'api': Uses OpenAI-compatible API (vLLM server, Ollama, OpenRouter, etc.)
            - 'mock': For pipeline debugging / testing without loading weights
        device:
            - Defaults to 'cuda:1' to keep 'cuda:0' free for the Segment pipeline.
        """
        self.mode = mode.lower()
        self.model_name = model_name

        # Validate device availability
        if "cuda" in device and not torch.cuda.is_available():
            print("⚠️ CUDA not available. Falling back to CPU for MCoTExtractor.")
            self.device = "cpu"
        elif "cuda:1" in device and torch.cuda.device_count() < 2:
            print("⚠️ GPU 1 (cuda:1) requested but only 1 GPU detected. Using cuda:0.")
            self.device = "cuda:0"
        else:
            self.device = device

        self.torch_dtype = torch_dtype or (
            torch.float16 if "cuda" in self.device else torch.float32
        )

        if self.mode == "local":
            self._init_local_model()
        elif self.mode == "api":
            self.api_base = api_base or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
            self._init_api_client()
        elif self.mode == "mock":
            print("ℹ️ MCoTExtractor initialized in [mock] mode.")
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'local', 'api', or 'mock'.")

    def _init_local_model(self):
        print(f"⏳ Loading local VLLM model: [{self.model_name}] on [{self.device}]...")
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Load Qwen2.5-VL model with conditional generation
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText
                model_cls = AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForCausalLM
                model_cls = AutoModelForCausalLM

        self.model = model_cls.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map={"": self.device},
            trust_remote_code=True,
        ).eval()

        if "cuda" in self.device:
            gpu_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            vram_mb = torch.cuda.memory_allocated(gpu_idx) / (1024**2)
            print(f"✅ VLLM Model loaded successfully on GPU [{self.device}]! VRAM used: {vram_mb:.1f} MB")
        else:
            print("✅ VLLM Model loaded successfully on CPU!")

    def _init_api_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
            print(f"✅ API Client initialized: {self.api_base} (Model: {self.model_name})")
        except ImportError:
            self.client = None
            print("⚠️ 'openai' package not installed. Will use direct 'requests' fallback for API calls.")

    def warmup(self):
        """Pre-warm GPU kernels with a dummy forward pass to eliminate first-query latency."""
        if self.mode != "local":
            return
        print("⏳ Pre-warming VLLM GPU kernels...")
        try:
            dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            self.extract(dummy_img, "is red and has short sleeves")
            if "cuda" in self.device:
                torch.cuda.synchronize()
            print("✅ Warmup completed! Ready for real-time inference.")
        except Exception as e:
            print(f"⚠️ Warmup notice: {e}")

    @staticmethod
    def _encode_image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def _parse_json_response(raw_text: str, default_caption: str = "") -> Dict[str, str]:
        """Robust parser to extract target, retained, and deleted fields from VLLM output."""
        # 1. Try markdown JSON block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if all(k in data for k in ["target", "retained", "deleted"]):
                    return {
                        "target": str(data["target"]).strip(),
                        "retained": str(data["retained"]).strip(),
                        "deleted": str(data["deleted"]).strip(),
                    }
            except Exception:
                pass

        # 2. Try bare JSON object
        raw_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if raw_match:
            try:
                data = json.loads(raw_match.group(1))
                if all(k in data for k in ["target", "retained", "deleted"]):
                    return {
                        "target": str(data["target"]).strip(),
                        "retained": str(data["retained"]).strip(),
                        "deleted": str(data["deleted"]).strip(),
                    }
            except Exception:
                pass

        # 3. Fallback regex search for individual keys
        target_m = re.search(r'"target"\s*:\s*"(.*?)"', raw_text)
        retained_m = re.search(r'"retained"\s*:\s*"(.*?)"', raw_text)
        deleted_m = re.search(r'"deleted"\s*:\s*"(.*?)"', raw_text)

        target = target_m.group(1).strip() if target_m else (default_caption or "modified clothing")
        retained = retained_m.group(1).strip() if retained_m else "original clothing shape and base style"
        deleted = deleted_m.group(1).strip() if deleted_m else "original elements replaced by modification"

        return {"target": target, "retained": retained, "deleted": deleted}

    def extract(self, image: Union[str, Image.Image], modification_text: str) -> Dict[str, str]:
        """
        Main extraction function.
        Returns:
            dict: { 'target': str, 'retained': str, 'deleted': str }
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError("Input 'image' must be a PIL Image or file path.")

        if self.mode == "mock":
            return {
                "target": f"a photo with {modification_text}",
                "retained": "clothing style, base shape",
                "deleted": "original color and sleeve details",
            }
        elif self.mode == "api":
            return self._extract_via_api(image, modification_text)
        elif self.mode == "local":
            return self._extract_via_local(image, modification_text)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _extract_via_local(self, image: Image.Image, modification_text: str) -> Dict[str, str]:
        user_prompt = COT_USER_PROMPT_TEMPLATE.replace("{modification_text}", modification_text)

        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        try:
            from qwen_vl_utils import process_vision_info
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        except ImportError:
            # Fallback if qwen_vl_utils is not installed
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
            )

        # Slice off input tokens
        generated_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return self._parse_json_response(response, default_caption=modification_text)

    def _extract_via_api(self, image: Image.Image, modification_text: str) -> Dict[str, str]:
        user_prompt = COT_USER_PROMPT_TEMPLATE.replace("{modification_text}", modification_text)
        b64_img = self._encode_image_to_base64(image)

        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        if self.client:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=256,
                temperature=0.2,
            )
            raw_text = response.choices[0].message.content
        else:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0.2,
            }
            resp = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"]

        return self._parse_json_response(raw_text, default_caption=modification_text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MCoT VLLM Extractor")
    parser.add_argument("--mode", type=str, default="mock", choices=["local", "api", "mock"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--image_path",
        type=str,
        default="../data/fiq/fashionIQ_dataset/images/B005X4PL1G.png",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="is shiny and silver with shorter sleeves and fit and flare",
    )
    args = parser.parse_args()

    print(f"--- Testing MCoTExtractor in [{args.mode}] mode on [{args.device}] ---")
    extractor = MCoTExtractor(
        mode=args.mode,
        model_name=args.model_name,
        device=args.device,
    )

    if os.path.exists(args.image_path):
        img_input = args.image_path
    else:
        print(f"⚠️ Image not found at {args.image_path}, testing with synthetic image:")
        img_input = Image.new("RGB", (224, 224), color=(200, 100, 50))

    result = extractor.extract(img_input, args.text)
    print("\n🎉 Extracted MCoT Output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
