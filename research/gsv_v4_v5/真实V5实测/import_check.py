"""在官方 V5 整合包的目录里，一个一个 import 官方代码要用的模块（每个单独一个进程），
看 Linux 上装的库够不够：缺了哪个（ModuleNotFoundError）一次全列出来，不用等训练到一半才发现。

用法：python import_check.py <整合包目录>
缺模块时退出码 1；别的错误（比如单独 import 时少了某个环境变量）只打印，不算缺。
"""
import os
import subprocess
import sys

MODULES = [
    # 处理文字（1A）：中文、英文、中英分段、BERT
    "text.cleaner", "text.chinese2", "text.english", "text.LangSegmenter", "transformers",
    # 声音特征、声纹、语义（1B / 1C）
    "feature_extractor.cnhubert", "module.models", "module.data_utils",
    # 训练：音色（LoRA）、语气（GPT）
    "peft", "AR.models.t2s_lightning_module", "AR.data.data_module", "pytorch_lightning",
    # 推理服务（api_v2）
    "GPT_SoVITS.TTS_infer_pack.TTS", "tools.audio_sr", "fastapi", "uvicorn", "resampy", "opencc",
]


def main() -> int:
    gsv = os.path.abspath(sys.argv[1])
    env = dict(os.environ, version="v5dev", is_half="False",
               PYTHONPATH=os.pathsep.join([gsv, os.path.join(gsv, "GPT_SoVITS")]))
    missing = []
    for mod in MODULES:
        r = subprocess.run([sys.executable, "-c", f"import {mod}"], cwd=gsv, env=env, capture_output=True,
                           text=True, timeout=600)
        last = (r.stderr.strip().splitlines() or [""])[-1]
        if r.returncode == 0:
            print(f"OK    {mod}")
        elif "ModuleNotFoundError" in last:
            print(f"MISS  {mod}: {last}")
            missing.append(last)
        else:
            print(f"ERR   {mod}: {last[:300]}")
    if missing:
        print("缺少：" + "；".join(dict.fromkeys(missing)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
