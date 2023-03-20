from __future__ import annotations

import torch


def _align_tokens(
    text: str,
    tokens: list[str],
    special_tokens: list[str],
) -> list[tuple[int, int] | None]:
    special_tokens = set(special_tokens)
    alignments = [None] * len(tokens)

    text_idx = 0
    token_idx = 0
    while text_idx < len(text) and token_idx < len(tokens):
        token = tokens[token_idx]

        if token in special_tokens:
            token_idx += 1
            continue

        if token.startswith('##'):
            prev_token_start_idx = alignments[token_idx - 1][0]
            token_end_idx = alignments[token_idx - 1][1] + len(token) - 2
            alignments[token_idx] = (prev_token_start_idx, token_end_idx)

            token_idx += 1
            text_idx = token_end_idx + 1
            continue

        # token is general word
        token_s_idx = 0
        start_idx = -1
        end_idx = -1
        while text_idx < len(text) and token_s_idx < len(token):
            if text[text_idx] != token[token_s_idx]:
                start_idx = -1
                end_idx = -1
                text_idx += 1
                token_s_idx = 0
                continue

            # is match
            if start_idx == -1:
                start_idx = text_idx
            end_idx = text_idx

            text_idx += 1
            token_s_idx += 1

        if token_s_idx == len(token):
            alignments[token_idx] = (start_idx, end_idx)
        token_idx += 1

    return alignments


def get_contacts_sequencies(
    text: str,
    attention_threshold: float,
    model,
    tokenizer,
) -> None | list[tuple[int, int]]:
    tokens_dict = tokenizer(text, truncation=True, return_tensors='pt')
    attention_map = model(**tokens_dict)[0, :, 0, :].mean(0)
    contacts_mask = torch.where(attention_map > attention_threshold, 1, 0).squeeze().tolist()

    # align tokens with raw text
    tokens = tokenizer.convert_ids_to_tokens(tokens_dict['input_ids'].squeeze().tolist())
    tokens_alignments = _align_tokens(text, tokens, tokenizer.all_special_tokens)

    # mine intervals with contact info
    contacts_seqs_raw = []
    for idx, is_contact in enumerate(contacts_mask):
        if is_contact == 1 and tokens_alignments[idx]:
            contacts_seqs_raw.append(tokens_alignments[idx])

    if not contacts_seqs_raw:
        return None

    contacts_seqs_agg = []
    prev_start_pos, prev_end_pos = contacts_seqs_raw[0]
    for start_pos, end_pos in contacts_seqs_raw[1:]:
        if prev_end_pos - start_pos >= -1:
            prev_end_pos = end_pos
        else:
            contacts_seqs_agg.append((prev_start_pos, prev_end_pos))
            prev_start_pos = start_pos
            prev_end_pos = end_pos
    contacts_seqs_agg.append((prev_start_pos, prev_end_pos))

    return contacts_seqs_agg
