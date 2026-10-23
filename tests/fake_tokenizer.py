class FakeTokenizer:
    """Tiny deterministic ChatML tokenizer used only by synthetic unit tests."""
    im_start_token = "<|im_start|>"
    im_end_token = "<|im_end|>"
    pad_token_id = 0
    eos_token_id = 0
    is_fast = True
    chat_template = "synthetic-chatml-v1"
    _special_to_id = {im_start_token: 1, im_end_token: 2}
    _id_to_special = {1: im_start_token, 2: im_end_token}

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, return_tensors=None, **kwargs):
        rendered = "".join(f"{self.im_start_token}{m['role']}\n{m['content']}{self.im_end_token}\n" for m in messages)
        if add_generation_prompt:
            rendered += f"{self.im_start_token}assistant\n"
        if tokenize:
            ids = self.encode(rendered, add_special_tokens=False)
            if return_tensors == "pt":
                import torch
                return torch.tensor([ids], dtype=torch.long)
            return ids
        return rendered

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            matched = next((token for token in (self.im_start_token, self.im_end_token) if text.startswith(token, i)), None)
            if matched:
                ids.append(self._special_to_id[matched])
                i += len(matched)
            else:
                ids.append(ord(text[i]) + 1000)
                i += 1
        return ids

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        ids = []
        offsets = []
        i = 0
        while i < len(text):
            matched = next((token for token in (self.im_start_token, self.im_end_token) if text.startswith(token, i)), None)
            if matched:
                ids.append(self._special_to_id[matched])
                offsets.append((i, i + len(matched)))
                i += len(matched)
            else:
                ids.append(ord(text[i]) + 1000)
                offsets.append((i, i + 1))
                i += 1
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        output = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self._id_to_special:
                if not skip_special_tokens:
                    output.append(self._id_to_special[token_id])
            else:
                output.append(chr(token_id - 1000))
        return "".join(output)
