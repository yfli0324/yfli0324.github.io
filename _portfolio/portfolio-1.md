---
title: "基于Transformer的自杀倾向检测"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/suicide-detection-transformer
date: 2024-01-17
excerpt: "使用DistilRoBERTa模型和SimHash近似重复检测技术，对文本数据进行自杀倾向检测，实现了高准确率和鲁棒性的模型构建。"
header:
  teaser: /images/portfolio/suicide-detection-transformer/roc_curve.png
tags:
- 自然语言处理
- 文本分类
- 自杀倾向检测
- 机器学习
- Transformer
- DistilRoBERTa
- SimHash
tech_stack:
  - name: Python
  - name: Hugging Face Transformers
  - name: Scikit-learn
  - name: Pandas
  - name: NumPy
  - name: PyTorch
---

## 项目背景

自杀倾向检测是一个重要的心理健康应用领域。本项目旨在通过自然语言处理技术，自动识别文本中的自杀倾向信号，为心理健康干预提供支持。我们使用了DistilRoBERTa模型作为基础模型，并结合SimHash算法进行近似重复数据的检测和去除，以提高模型的鲁棒性和泛化能力。

## 核心实现

### 数据预处理与近似重复检测

```python
# 文本规范化函数
_whitespace = re.compile(r"\s+")
_nonword = re.compile(r"[^\\w\\s]", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = _nonword.sub(" ", s)
    s = _whitespace.sub(" ", s).strip()
    return s

# SimHash算法实现
def simhash(tokens, nbits=64) -> int:
    if not tokens:
        return 0
    v = np.zeros(nbits, dtype=np.int32)
    counts = Counter(tokens)
    for tok, w in counts.items():
        h = _fnv1a_64(tok)
        for i in range(nbits):
            bit = (h >> i) & 1
            v[i] += w if bit else -w
    out = 0
    for i in range(nbits):
        if v[i] >= 0:
            out |= (1 << i)
    return int(out)

# 分桶和聚类
def band_buckets(hashes, band_bits=16):
    assert 64 % band_bits == 0
    bands = 64 // band_bits
    mask = (1 << band_bits) - 1
    buckets = defaultdict(list)
    for idx, h in enumerate(hashes):
        for b in range(bands):
            buckets[(b, (h >> (b * band_bits)) & mask)].append(idx)
    return buckets
# 加载预训练模型
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 计算指标函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(int)
    
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    p1 = probs[:, 1]
    
    yhat = (p1 >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(labels, p1),
        "pr_auc": average_precision_score(labels, p1),
        "accuracy": float((yhat == labels).mean()),
    }

# 训练参数
training_args = TrainingArguments(
    output_dir=os.path.join(OUT_DIR, "ckpt"),
    learning_rate=LR,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
