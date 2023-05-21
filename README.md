# AIproject

#### 介绍
{这是人工智能应用开发的作业，实现finetune对联任务}

#### 软件架构
所有代码都公布在kaggle上，上传的模型有三个
gpt2model中是gpt2对联模型，实现对对子任务
mengzi中是t5-mengzi模型，同样是对对子
mengzi_genrate是对联生成任务，基于mengzi


#### 安装教程

1.  pip install transformers >= 4.23
2.  pip install pytorch >= 1.7.0

#### 使用说明

1.  请注意example.py,样例代码中显示了如何使用这三个模型
2.  原始数据为70w条对联，在data文件夹下，这些用于训练前两个模型，生成数据集的代码在finetune-gpt2couplet的preprocess函数中
3.  生成对联任务由chatglm6b生成，具体代码为chatglm6b_answer_dataset.py,然后经过generate-mengzi-dataset生成
    具体参考https://www.kaggle.com/code/coolwx/chatglm-6b
           
4.  所有的训练代码都已经在kaggle上公开，包括失败的alpaca-lora训练
    gpt2微调 https://www.kaggle.com/code/coolwx/gpt2couplet
    t5-mengzi微调：https://www.kaggle.com/code/coolwx/t5-mengzi-couplet
    t5-mengzi生成任务微调：https://www.kaggle.com/code/coolwx/mengzi-couplet-new
    alpaca-lora项目：失败，似乎对中文支持很差：https://www.kaggle.com/code/coolwx/alpaca-couplet
    alpaca-lora推理，失败，但是对英文效果很好：https://www.kaggle.com/code/coolwx/generate-llama-chinese-couplet

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


