# knowledgeable-prompt-tuning

## 使用向量数据库，存储行业知识，并在大语言模型的问答中将知识注入到提示语，通过提示语微调方式，引导大语言模型根据行业知识回答用户提问。

## 依赖安装
* 基础环境：python==3.11.3
* 如果后续想多玩些大模型，推荐使用conda，目前大模型更新较快，可能不同版本依赖的某个包会有版本冲突，可以用conda来管理

```bash
pip install -r requirements.txt
```
运行后续内容前，先启动 Milvus 数据库

## Milvus运行
示例运行或服务运行，都需要先启动Milvus数据库

官方文档：https://milvus.io/docs
```bash
cd milvus
docker-compose up -d
```
Milvus启动默认端口为19530，可以通过可视化工具 [Attu](https://github.com/zilliztech/attu) 来查看

## 示例运行
指明项目路径
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/the/repo
```
知识入库
```bash
python examples/ingest.py
```
QA
```bash
python examples/qa.py
```

## 服务运行
```bash
python main.py
```
访问： http://localhost:40015 

## 容器运行
```bash
# 构建ubuntu+python3.11.3基础镜像
cd images/base
docker build . -t python:3.11.3-ubuntu20.04
# 构建服务镜像
cd ../..
sh build_image.sh
# 等待镜像构建
sh run_docker.sh
```
访问： http://localhost:4396 

## 文件说明
```
├─ api # 服务健康检测及 embedding 与 qa 接口定义
│
│
├─ app
│    │
│    │
│    ├─ core # 主函数，定义 embedding 与 qa 的方法调用；对 langchain qa_with_sources_chain 进行继承，后续可能增加流处理
│    │
│    │
│    ├─ datastore # 对 langchain 中 vectorstore 进行继承，实现过滤条件的改写方法等，并增加upsert方法，目前的更新是按删除旧内容再增加新内容处
│    │
│    │
│    └─ server # 服务启动文件，加载中间件及路由配置等
│
│
└─ examples # 直接进行 ingest 与 qa 的示例
```

## 接口文档
运行服务后，访问 http://localhost:40015/docs 即可

## 其他
1. 部分代码例如`main.py`中的`click`库，其实并没有用到。是找的脚手架中有，感觉后面可能有用，后续可能清除或更新到使用说明
1. 项目中对 openai 的请求，是通过自配的云服务器做代理。也可以启动全局的vpn代理，然后设置`HTTP_PROXY`, `HTTPS_PROXY`环境变量，例如`os.environ['HTTP_PROXY']='http://127.0.0.1:1087'`
1. 本地开发的环境变量可以放在vscode的launch.json中，可以参考：
    ```json
    {
      "configurations": [
        {
          "env": {
            "OPENAI_API_BASE": "",
            "OPENAI_API_KEY": ""
          }
        }
      ]
    }
    ```
    也可以使用`python-dotenv` + `.env`文件
1. 有任何问题，或是优化，请在issue中提出。

## TODO
* [ ] 将模型改为可选chatgpt，或本地chatglm。后续可以将预训练后量化chatglm部署本地，不经过知识库
* [ ] 分片策略优化，例如硬分片的同时利用gpt能力截取最大完整语义内容，再将剩余内容分片。使得分片数据不影响上下文语义。可以节约overlap空间，
* [ ] chatglm的ptuning进行尝试，然后梳理经验到新仓库
* [ ] 应用层与live2d的模型结合，可对话的数字奶盖🐱
* [ ] 结合应用层的提示语定制
* [ ] live2d模型编辑研究，动作编辑
* [ ] electron桌面封装，应用层功能添加（其他仓库）