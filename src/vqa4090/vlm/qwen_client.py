from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vqa4090.utils.model_resolver import resolve_model_source


class QwenVLClient:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 128,
        load_in_4bit: bool = False,
        temperature: float = 0.0,
        max_evidence: int = 5,
        evidence_chars: int = 320,
    ) -> None:
        kwargs: dict = {"trust_remote_code": True}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        resolved_model, provider = resolve_model_source(model_name)
        print(f"[QwenVLClient] model source: {provider} -> {resolved_model}")
        self.processor = AutoProcessor.from_pretrained(resolved_model, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(resolved_model, **kwargs)
        if not load_in_4bit:
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_evidence = max_evidence
        self.evidence_chars = evidence_chars

    def answer(self, question: str, evidence_texts: list[str], image_paths: list[str] | None = None) -> str:
        evidence = "\n".join(
            f"[{i+1}] {t[: self.evidence_chars]}" for i, t in enumerate(evidence_texts[: self.max_evidence])
        )
        text_prompt = (
            "You are a precise VQA assistant. "
            "Use only the image and the numbered evidence snippets. "
            "Return the shortest exact phrase, number, date, or name that answers the question. "
            "Do not explain. If the evidence is insufficient, answer exactly 'I cannot answer from evidence.'\n\n"
            f"Question: {question}\nEvidence:\n{evidence}\nAnswer:"
        )

        messages = [{"role": "user", "content": []}]
        valid_images: list[Image.Image] = []
        if image_paths:
            for p in image_paths[:1]:
                if p and Path(p).exists():
                    try:
                        img = Image.open(p).convert("RGB")
                        w, h = img.size
                        max_side = max(w, h)
                        if max_side > 896:
                            scale = 896.0 / max_side
                            img = img.resize((int(w * scale), int(h * scale)))
                        valid_images.append(img)
                        messages[0]["content"].append({"type": "image"})
                    except Exception:
                        continue

        messages[0]["content"].append({"type": "text", "text": text_prompt})
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[rendered],
            images=valid_images if valid_images else None,
            return_tensors="pt",
            padding=True,
        )
        device = self.model.device if hasattr(self.model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}

        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                gen_kwargs["temperature"] = max(self.temperature, 1e-5)
            out = self.model.generate(**inputs, **gen_kwargs)

        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        txt = self.processor.decode(gen_ids, skip_special_tokens=True)
        return txt.strip()
