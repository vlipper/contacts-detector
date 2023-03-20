import hydra
import streamlit as st
from omegaconf import OmegaConf

from model import load_model, load_tokenizer
from inference import get_contacts_sequencies


def inference(
    model_path: str,
    tokenizer_path: str,
    attention_threshold: float,
) -> None:
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    text = st.text_input(
        label='Write something with out without contact info:',
        value='Продаю машину. Звоните: 89087071820',
    )

    contacts_seqs = get_contacts_sequencies(text, attention_threshold, model, tokenizer)
    if not contacts_seqs:
        st.text('There are no contacts')
    else:
        text_mod = ''
        prev_end_idx = -1
        for start_idx, end_idx in contacts_seqs:
            text_mod += f'{text[prev_end_idx + 1 : start_idx]}'
            text_mod += f':blue[{text[start_idx : end_idx + 1]}]'
            prev_end_idx = end_idx
        text_mod += f'{text[prev_end_idx + 1 :]}'

        st.text('There are contacts:')
        st.markdown(text_mod)


@hydra.main(config_path='../conf/', config_name='config', version_base='1.3')
def main(conf) -> None:
    config = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
    inference(**config)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
