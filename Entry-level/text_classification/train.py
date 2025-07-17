import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def data_preprocess(df):
    from sklearn.utils import resample

    # 分离多数类和少数类
    df_good = df[df['label'] == 1]  # 假设1=好评
    df_bad = df[df['label'] == 0]  # 假设0=差评

    # 下采样多数类
    df_good_downsampled = resample(df_good,
                                   replace=False,
                                   n_samples=len(df_bad),
                                   random_state=42)

    # # 上采样少数类
    # df_bad_upsampled = resample(df_bad,
    #                             replace=True,
    #                             n_samples=len(df_good),
    #                             random_state=42)

    # 组合数据集（选择一种方法）
    # 方法1：平衡数据集（各2000条）
    balanced_df = pd.concat([df_good_downsampled, df_bad])
    return balanced_df
    #
    # # 方法2：扩大数据集（各5000条）
    # expanded_df = pd.concat([df_good, df_bad_upsampled])

def main() -> None:
    # 加载数据
    df = pd.read_csv("ChnSentiCorp_htl_all.csv")
    df = df.dropna(subset=["review"])  # 删除 review 字段为空的行

    df = data_preprocess(df)

    texts = df["review"].values
    labels = df["label"].values


    # 划分训练集/验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    # 强制转换为字符串，并过滤掉非法类型
    train_texts = [str(t) if isinstance(t, (str, int, float)) and not pd.isna(t) else "" for t in train_texts]
    val_texts = [str(t) if isinstance(t, (str, int, float)) and not pd.isna(t) else "" for t in val_texts]

    # 初始化Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 编码文本
    train_encodings = tokenizer(
        list(train_texts),
        truncation=True,
        padding=True,
        max_length=128
    )

    val_encodings = tokenizer(
        list(val_texts),
        truncation=True,
        padding=True,
        max_length=128
    )

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    # 加载预训练模型（根据类别数修改num_labels）
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(set(labels))  # 类别数量
    )

    # 训练配置
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs"
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # 开始训练
    trainer.train()

    # 保存微调后的模型
    model.save_pretrained("./fine_tuned_bert")
    tokenizer.save_pretrained("./fine_tuned_bert")




if __name__ == "__main__":
    # 在训练前添加环境变量设置（Windows CMD兼容）
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步CUDA调用，精确定位错误

    main()