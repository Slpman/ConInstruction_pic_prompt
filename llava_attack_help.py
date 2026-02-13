import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import cosine_embedding_loss
import torch.optim as optim
import argparse
import torch
import numpy as np
import os
import json
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from torchvision import transforms
import pickle
from utils import get_logger, prepare_prompt, get_target_data, save_json

# 设置设备与随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
empty_id = 29871
image_shape = (1, 3, 336, 336)
image_token_len = 576

torch.manual_seed(42)
np.random.seed(42)


def prompt_attack(target_prompt, index_prompt, model, tokenizer, args, logger, next_prompt=" "):
    """
    针对单个恶意指令和对应的 help_text 进行图像优化攻击
    next_prompt 现在接收的是数据集中特定的一行 help_text [cite: 121, 138]
    """
    exp_history = {}

    # 1. 准备目标指令的嵌入向量 [cite: 129, 131]
    input_ids = tokenizer_image_token(target_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
        0).to(model.device)

    input_embeds = model.get_model().embed_tokens(input_ids[:, 1:]).to(model.device)
    empty_embed = model.get_model().embed_tokens(torch.tensor([[empty_id]]).to(model.device))
    empty_embeds = empty_embed.repeat(1, image_token_len - input_ids.shape[1] - 1, 1)
    padded_input_embeds = torch.cat((empty_embeds, input_embeds), dim=1).to(model.device)

    # 2. 初始化对抗图像张量 [cite: 129]
    image_tensor = torch.randn(image_shape).to(device).requires_grad_(True)

    best_loss = 100
    optimizer = optim.Adam([image_tensor], lr=args.lr)
    cos_loss_fun = nn.CosineEmbeddingLoss()

    model.train()
    for param in model.parameters():
        param.requires_grad = False

    # 3. 迭代优化循环 [cite: 131]
    for step in range(args.num_steps):
        optimizer.zero_grad()
        image_embeds = model.encode_images(image_tensor)

        # 使用论文推荐的 "part" 模式，对齐最后几位嵌入 [cite: 135]
        if args.mode == "part":
            len_prompt_token = input_embeds.shape[1]
            target_ones = torch.ones(padded_input_embeds.shape[1])[-len_prompt_token:].to("cuda")
            part_prompt_embeds = padded_input_embeds[0][-len_prompt_token:].to("cuda")
            part_image_embeds = image_embeds[0][-len_prompt_token:].to("cuda")

            if args.loss == "l2":
                loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
            elif args.loss == "cosine":
                loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
            elif args.loss == "both":
                # 结合 L2 和余弦相似度以获得最佳效果 [cite: 136, 334]
                l2_loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
                cos_loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
                loss = l2_loss + cos_loss

            loss.backward(retain_graph=True)
            optimizer.step()

        # 4. 定期测试并保存记录 [cite: 139]
        if step % int(args.num_steps / args.num_saves) == 0:
            if loss.item() < best_loss:
                best_loss = loss.item()
                logger.info("Step {}, Loss: {}".format(step, loss.item()))

            curr_name = f"task_{index_prompt}_step_{step}_prompt.bin"
            curr_path = os.path.join(args.exp_path, curr_name)
            pickle.dump(image_tensor.detach().cpu(), open(curr_path, "wb"))

            # 使用当前问题专属的 specific help text 构建最终 Prompt [cite: 138, 302]
            second_prompt = prepare_prompt(next_prompt, model, args.model_name, image=True)
            second_ids = tokenizer_image_token(second_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                               return_tensors='pt').unsqueeze(0).to("cuda")

            with torch.inference_mode():
                output_ids = model.generate(
                    second_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            second_outputs = tokenizer.decode(output_ids[0]).strip()
            second_outputs = second_outputs.replace("<s>", "").replace("</s>", "").strip()
            logger.info(f"Step {step}: Direct Output is {second_outputs}")

            exp_rec = {
                "loss": loss.item(),
                "predict": second_outputs,
                "image_path": curr_path
            }
            exp_history[step] = exp_rec

    return exp_history


def main(args):
    disable_torch_init()
    full_exp_record = dict()

    # 修改：获取 target_prompts 和对应的 help_texts 列表
    # 确保你的 utils.py 已经修改为返回两个列表
    target_prompts, target_help_texts = get_target_data(args.task)

    args.model_name = get_model_name_from_path(args.model_path)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger = get_logger(os.path.join("logs", "_".join([args.task, args.model_name, args.suffix])))

    if not os.path.exists("exps"):
        os.makedirs("exps")

    args.exp_path = os.path.join("exps", "_".join([args.task, args.model_name, args.suffix]))
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    save_file = os.path.join(args.exp_path, "attack.json")
    if os.path.exists(save_file):
        full_exp_record = json.load(open(save_file, encoding="utf-8"))

    # 加载预训练模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, args.model_name, load_4bit=True, device=args.device)

    # 修改：同步遍历指令和特定的辅助文本
    for index_prompt, (target_prompt, specific_help_text) in enumerate(zip(target_prompts, target_help_texts)):

        if str(index_prompt) in full_exp_record:
            continue

        # 处理前后缀逻辑（如果仍然需要基于 args.suffix 的简单修饰）
        if "prefix" in args.suffix:
            num_prefix = int(args.suffix.split("_")[1])
            target_prompt = ("     " * num_prefix) + target_prompt

        if "suffix" in args.suffix:
            suffix = args.suffix.split("_")[1]
            target_prompt = target_prompt + " " + suffix

        logger.info("Prompt Index: {}, Target: {}".format(index_prompt, target_prompt))
        logger.info("Using Specific Help Text: {}".format(specific_help_text))

        exp_record = {"prompt": target_prompt}

        # 将特定的 help_text 传入 prompt_attack
        exp_record["history"] = prompt_attack(
            target_prompt,
            index_prompt,
            model=model,
            tokenizer=tokenizer,
            args=args,
            logger=logger,
            next_prompt=specific_help_text
        )

        full_exp_record[index_prompt] = exp_record

        if args.save_records:
            save_json(full_exp_record, save_file)
            logger.info(f"{save_file} saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="custom_help_text", help="用于标识实验名称的后缀")
    parser.add_argument("--optimizer_name", type=str, default="Adam")
    parser.add_argument("--mode", type=str, default="part")
    parser.add_argument("--loss", type=str, default="both")
    parser.add_argument("--task", type=str, default="safebench_tiny")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=8001)
    parser.add_argument("--num-saves", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save_records", type=str, default="True")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)