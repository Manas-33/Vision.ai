from share4v.train.train import train
from share4v.train.llama_flash_attn_monkey_patch import \
    replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()


if __name__ == "__main__":
    train()
