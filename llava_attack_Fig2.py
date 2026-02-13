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
import pandas as pd  # 必须引入 pandas

# from nltk.translate.bleu_score import sentence_bleu

# 尝试导入 llava，兼容不同的包名情况
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
except ImportError:
    import sys

    sys.path.append("llava_backup_v1.5")
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from torchvision import transforms
import pickle
from utils import get_logger, prepare_prompt, save_json

# 注意：这里不再使用 get_target_data，因为我们要自定义读取 instruction 列

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

empty_id = 29871
image_shape = (1, 3, 336, 336)
image_token_len = 576

torch.manual_seed(42)
np.random.seed(42)

tp = transforms.ToPILImage()


# ==============================================================================================
# prompt_attack 函数 (保持不变，逻辑通用)
# 接收 target_prompt 进行优化。在 Fig2 版本中，传入的将是 instruction。
# ==============================================================================================
def prompt_attack(target_prompt, index_prompt, model, tokenizer, args, logger, next_prompt=" ", figstep_image_path=None,
                  image_processor=None):
    exp_history = {}

    # 1. 准备攻击目标 (Target): 将 instruction 转换为 embedding
    input_ids = tokenizer_image_token(target_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
        0).to(model.device)

    input_embeds = model.get_model().embed_tokens(input_ids[:, 1:]).to(model.device)
    empty_embed = model.get_model().embed_tokens(torch.tensor([[empty_id]]).to(model.device))
    empty_embeds = empty_embed.repeat(1, image_token_len - input_ids.shape[1] - 1, 1)
    padded_input_embeds = torch.cat((empty_embeds, input_embeds), dim=1).to(model.device)

    # 2. 初始化: 加载 FigStep 图片 (Warm Start)
    image_tensor = None
    if figstep_image_path is not None and os.path.exists(figstep_image_path):
        try:
            logger.info(f"正在加载 FigStep 图片: {figstep_image_path}")
            image_pil = Image.open(figstep_image_path).convert('RGB')

            if image_processor is not None:
                image_tensor_init = image_processor.preprocess(image_pil, return_tensors='pt')['pixel_values'][0]
                image_tensor = image_tensor_init.unsqueeze(0).to(device).clone().detach().requires_grad_(True)
            else:
                logger.warning("未提供 image_processor，回退到随机初始化！")
                image_tensor = torch.randn(image_shape).to(device).requires_grad_(True)
        except Exception as e:
            logger.error(f"加载图片失败: {e}，回退到随机初始化。")
            image_tensor = torch.randn(image_shape).to(device).requires_grad_(True)
    else:
        if figstep_image_path:
            logger.warning(f"找不到图片路径: {figstep_image_path}")
        image_tensor = torch.randn(image_shape).to(device).requires_grad_(True)

    best_loss = 100
    best_idx = 0
    best_tensor = None

    optimizer = optim.Adam([image_tensor], lr=args.lr)
    cos_loss_fun = nn.CosineEmbeddingLoss()

    model.train()
    for param in model.parameters():
        param.requires_grad = False

    # 开始优化循环
    for step in range(args.num_steps):
        optimizer.zero_grad()
        image_embeds = model.encode_images(image_tensor)
        loss = None
        if args.mode == "full":
            if args.loss == "l2":
                diff = image_embeds - padded_input_embeds
                loss = (diff ** 2).mean()

            elif args.loss == "cosine":
                target_ones = torch.ones(padded_input_embeds.shape[1]).to("cuda")
                loss = cos_loss_fun(image_embeds[0], padded_input_embeds[0], target_ones)

            elif args.loss == "both":
                l2_loss = ((image_embeds - padded_input_embeds) ** 2).mean()
                target_ones = torch.ones(padded_input_embeds.shape[1]).to("cuda")
                cos_loss = cos_loss_fun(image_embeds[0], padded_input_embeds[0], target_ones)
                loss = l2_loss + cos_loss

            loss.backward(retain_graph=True)
            optimizer.step()

        elif args.mode == "part":
            len_prompt_token = input_embeds.shape[1]
            target_ones = torch.ones(padded_input_embeds.shape[1])[-len_prompt_token:].to("cuda")
            part_prompt_embeds = padded_input_embeds[0][-len_prompt_token:].to("cuda")
            part_image_embeds = image_embeds[0][-len_prompt_token:].to("cuda")

            if args.loss == "l2":
                loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
            elif args.loss == "cosine":
                loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
            elif args.loss == "both":
                l2_loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
                cos_loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
                loss = l2_loss + cos_loss
            loss.backward(retain_graph=True)
            optimizer.step()

        if step % int(args.num_steps / args.num_saves) == 0:
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_idx = step
                best_tensor = image_tensor.detach().cpu()
                logger.info("Step {}, Loss: {}".format(step, loss.item()))

            curr_name = f"task_{index_prompt}_step_{step}_prompt.bin"
            curr_path = os.path.join(args.exp_path, curr_name)
            pickle.dump(image_tensor.detach().cpu(), open(curr_path, "wb"))

            second_prompt = prepare_prompt(next_prompt, model, args.model_name, image=True)
            second_ids = tokenizer_image_token(second_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                               return_tensors='pt').unsqueeze(
                0).to("cuda")

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

            exp_rec = {}
            exp_rec["loss"] = loss.item()
            exp_rec["predict"] = second_outputs
            exp_rec["image_path"] = curr_path
            exp_history[step] = exp_rec

    return exp_history


def main(args):
    full_exp_record = dict()

    args.model_name = get_model_name_from_path(args.model_path)

    # 1. 建立日志和保存目录
    if not os.path.exists("logs"):
        os.makedirs("logs")
    # 注意：这里 suffix 最好在运行时改成 fig2_test 之类的，以区分结果
    logger = get_logger(os.path.join("logs", "_".join([args.task, args.model_name, args.suffix])))

    if not os.path.exists("exps"):
        os.makedirs("exps")

    args.exp_path = os.path.join("exps", "_".join([args.task, args.model_name, args.suffix]))
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    save_file = os.path.join(args.exp_path, "attack.json")
    if os.path.exists(save_file):
        full_exp_record = json.load(open(save_file, encoding="utf-8"))

    # 2. 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base,
                                                                           args.model_name,
                                                                           load_4bit=True, device=args.device)

    # 3. 加载数据集 (Safebench-Tiny) 并直接使用 Pandas 遍历
    # ==============================================================================================
    dataset_df = None
    csv_path = os.path.join("dataset", "safebench", "question", f"{args.task}.csv")

    # 容错：如果找不到，尝试默认文件名
    if not os.path.exists(csv_path) and "tiny" in args.task:
        csv_path = os.path.join("dataset", "safebench", "question", "safebench_tiny.csv")

    if os.path.exists(csv_path):
        logger.info(f"Loading metadata from {csv_path}")
        dataset_df = pd.read_csv(csv_path)
    else:
        logger.error(f"Critical Error: CSV not found at {csv_path}. Cannot align to instructions.")
        return  # 强制退出，因为没有 instruction 无法运行此版本代码

    # 4. 开始遍历每一个任务
    # ==============================================================================================
    # 我们不使用 get_target_data 生成的列表，而是直接遍历 dataframe
    for index, row in dataset_df.iterrows():
        index_prompt = index  # 保持和原代码一样的索引逻辑 (0, 1, 2...)

        if str(index_prompt) in full_exp_record:
            continue

        # --- 获取关键数据 ---
        # 原始问题 (仅用于日志记录，不参与计算)
        original_question = row['question']

        # 【核心修改】Target Prompt 现在是 instruction 列 (例如 "Steps to manufacture...")
        # 确保列名 'instruction' 存在 (根据你提供的截图，列名确实是 'instruction')
        target_prompt = row['instruction']

        cat_id = int(row['category_id'])
        task_id = int(row['task_id'])

        # 处理前缀后缀 (如果 args 里有的话)
        if "prefix" in args.suffix:
            num_prefix = int(args.suffix.split("_")[1])
            prefix = "     " * num_prefix
            target_prompt = prefix + target_prompt

        if "suffix" in args.suffix:
            suffix = args.suffix.split("_")[1]
            target_prompt = target_prompt + " " + suffix

        next_prompt = " "
        if "next" in args.suffix:
            next_prompt = args.suffix.split("_")[1]

        logger.info("-" * 30)
        logger.info(f"Index: {index_prompt}")
        logger.info(f"Target (Instruction): {target_prompt}")
        logger.info(f"Original Question: {original_question}")

        # --- 构建图片路径 ---
        img_name = f"query_ForbidQI_{cat_id}_{task_id}_6.png"
        figstep_path = os.path.join("dataset", "safebench", "images", "SafeBench-Tiny", img_name)

        # --- 运行攻击 ---
        exp_record = {}
        exp_record["prompt"] = target_prompt  # 记录使用的是 Instruction
        exp_record["question"] = original_question  # 额外记录原始问题供参考

        # 调用 prompt_attack，传入 instruction 作为 target
        exp_record["history"] = prompt_attack(
            target_prompt,
            index_prompt,
            model,
            tokenizer,
            args,
            logger,
            next_prompt,
            figstep_image_path=figstep_path,
            image_processor=image_processor
        )
        full_exp_record[index_prompt] = exp_record

        if args.save_records:
            save_json(full_exp_record, save_file)
            logger.info(f"{save_file} saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径依然指向你本地模型
    parser.add_argument("--model-path", type=str, default="models/llava-v1.6-vicuna-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--optimizer_name", type=str, default="Adam")
    parser.add_argument("--mode", type=str, default="part")
    parser.add_argument("--loss", type=str, default="both")
    parser.add_argument("--task", type=str, default="safebench_tiny")
    parser.add_argument("--image-file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=8001)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--num-saves", type=int, default=10)
    parser.add_argument("--pre-set", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--save_records", type=str, default="True")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)