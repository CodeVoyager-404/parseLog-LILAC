import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class deepseekLoader:
    """大语言模型加载器"""

    def __init__(self, model_path: str, device: str = None):
        """
        初始化模型加载器
        :param model_path: 本地模型路径
        :param device: 指定运行设备(cuda/cpu)，默认自动检测
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载本地模型和分词器"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.device,
                torch_dtype="auto"
            )
            model.eval()
            self.logger.info(f"成功加载模型到设备：{self.device}")
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise

    def get_model(self):
        """获取加载的模型"""
        return self.model

    def get_tokenizer(self):
        """获取分词器"""
        return self.tokenizer

    def get_device(self):
        """获取当前设备"""
        return self.device

model_path="D:\Documents\Scholars\codes\deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B"
modelLoader=deepseekLoader(model_path)
model=modelLoader.get_model()
tokenizer=modelLoader.get_tokenizer()