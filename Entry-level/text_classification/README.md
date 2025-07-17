文本分类简介

一个基于 BERT 的中文文本分类实现，使用 Hugging Face 的 Transformers 库进行模型训练和推理。适用于情感分析、评论分类等自然语言处理任务。

[数据集](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb)，包含五千多条中文酒店好评信息和两千多条中文酒店差评信息，训练时需要进行类别平衡，本仓库进行了下采样/上采样，也可以通过数据增强增加差评。



📁 文件结构

train.py: 使用 bert-base-uncased 预训练模型进行微调的训练脚本。

app.py: 提供一个基于 Flask 的 Web 接口，用于加载训练好的模型并进行预测。



🧪 运行环境依赖
Python >= 3.8
PyTorch
Transformers
Flask（仅用于运行 app.py）

📝 使用方式

训练模型：

   python train.py

启动服务：

   python app.py

发送预测请求（示例）：

   curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "非常好的酒店，服务态度好，卫生干净"}'