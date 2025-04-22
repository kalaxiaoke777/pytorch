from fastapi import FastAPI, Request, Body

from pydantic import BaseModel
import jieba
import fasttext
import json

# 创建 FastAPI 应用实例
app = FastAPI()


# 加载自定义的停用词字典
jieba.load_userdict("work/data/stopwords.txt")

# 提供已训练好的模型路径+名字
model_save_path = "work/fasttext/toutiao_fasttext_1745331832.bin"

# 实例化fasttext对象, 并加载模型参数用于推断, 提供服务请求
model = fasttext.load_model(model_save_path)
print("FastText模型实例化完毕...")


# 定义请求体的数据模型
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None


# 定义一个简单的 GET 请求处理函数
@app.post("/")
def read_root(item: dict = Body(...)):
    # 对请求文本进行处理, 因为前面加载的是基于词的模型, 所以这里用jieba进行分词
    input_text = " ".join(jieba.lcut(item["text"]))

    # 执行预测
    res = model.predict(input_text)
    predict_name = res[0][0]
    return predict_name


# 定义一个带有路径参数的 GET 请求处理函数
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


# 定义一个 POST 请求处理函数
@app.post("/items/")
def create_item(item: Item):
    return item


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
