---
title: "基于Transformer的自杀倾向检测"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/suicide-detection-transformer
date: 2024-01-17
excerpt: "使用DistilRoBERTa模型和SimHash近似重复检测技术，对文本数据进行自杀倾向检测，实现了高准确率和鲁棒性的模型构建。"
header:
  teaser: /images/ROC.png
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

# 项目背景

自杀倾向检测是一个重要的心理健康应用领域。本项目旨在通过自然语言处理技术，自动识别文本中的自杀倾向信号，为心理健康干预提供支持。我们使用了DistilRoBERTa模型作为基础模型，并结合SimHash算法进行近似重复数据的检测和去除，以提高模型的鲁棒性和泛化能力。

# 核心实现

## 数据预处理与近似重复检测

### 文本规范化函数
```python
_whitespace = re.compile(r"\s+")
_nonword = re.compile(r"[^\\w\\s]", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = _nonword.sub(" ", s)
    s = _whitespace.sub(" ", s).strip()
    return s
```
### SimHash算法实现
```python
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
```
### 分桶和聚类
```python
def band_buckets(hashes, band_bits=16):
    assert 64 % band_bits == 0
    bands = 64 // band_bits
    mask = (1 << band_bits) - 1
    buckets = defaultdict(list)
    for idx, h in enumerate(hashes):
        for b in range(bands):
            buckets[(b, (h >> (b * band_bits)) & mask)].append(idx)
    return buckets
```
## 模型训练与评估
### 加载预训练模型
```python
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
```
### 计算指标函数
```python
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
```
### 训练参数
```python
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
```
### 训练器
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```
### 开始训练
```python
trainer.train()
```
# 分析结果
## 模型性能指标
ROC-AUC: 0.9996 (95% CI: 0.9995 - 0.9997)

PR-AUC: 0.9994 (95% CI: 0.9992 - 0.9996)

Brier分数: 0.0061

最佳F1阈值: 0.632
## 混淆矩阵
![Confusion](/images/confusion.png)
## ROC曲线
![ROC](/images/ROC.png)
## PR曲线
![PR](/images/PR.png)
## 分数分布直方图
![Dist](/images/Score_dist.png)
## 精确率 / 召回率 / F1 值与阈值关系图
![Threshold](/images/threshold.png)
## 校准曲线
![Calibration](/images/calibration.png)
# 结论
本项目成功实现了一个高精度的自杀倾向检测系统。通过使用 DistilRoBERTa 模型和 SimHash 近似重复检测技术，我们获得了优秀的模型性能。模型在测试集上的 ROC-AUC 为 0.9996，PR-AUC 为 0.9994，表明模型具有出色的分类能力。
我们还通过阈值选择优化了模型的性能，最终选择了在验证集上获得最佳 F1 值的阈值 0.712。在这个阈值下，模型在测试集上的精确率为 0.9943，召回率为 0.9950，F1 值为 0.9946，达到了非常高的水平。
这个系统可以应用于社交媒体监测、心理健康热线等场景，帮助及时识别有自杀倾向的用户，为后续的干预提供支持。
