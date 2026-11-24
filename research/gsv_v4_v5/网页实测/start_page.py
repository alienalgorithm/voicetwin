"""在 GitHub 上用和 V5 整合包一样的库版本（gradio 4.44.1 等）打开声音分身的网页。
GPT-SoVITS 用测试里的替身（tests/fake_gptsovits.py：有 V5 的底模文件名、训练和合成都很快），只为了把网页从头点到尾；
真的 V5 代码和底模的实测在 ../真实V5实测/。

用法：python start_page.py <工作目录> <端口>
在 <工作目录> 里建：素材/第1课.wav + 第1课.srt（测试用的合成讲课录音）、GPT-SoVITS 替身、workspace、config.yaml，然后打开网页（一直运行）。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ.setdefault("EXPECT_VERSION", "v5dev")   # 替身的「处理文字」检查程序传的版本和官方 webui 一样是 v5dev
# 替身里没有 BERT：VoiceTwin 自己的「中英文一起」处理文字按测试的办法跑（和 tests/conftest.py 一样）。
# 不设的话它会失败，「一模一样」档就停下（不再退回会丢掉英文的官方方法），网页点不到最后。
os.environ.setdefault("VOICETWIN_TEXT_DRYRUN", "1")

import yaml  # noqa: E402

from conftest import make_lecture  # noqa: E402
from fake_gptsovits import build_fake_root  # noqa: E402

work = Path(sys.argv[1]).resolve()
port = int(sys.argv[2])
make_lecture(work / "素材" / "第1课.wav")
root = build_fake_root(work / "GPT-SoVITS")
cfg = {
    "workspace": str(work / "workspace"),
    "backend": "gptsovits",
    "prepare": {"asr": {"engine": "none"}},
    "speaker_encoder": "mfcc",
    "similarity": {"model_dir": str(work / "sv_models")},
    "backends": {"gptsovits": {"root": str(root), "python": sys.executable, "port": port + 1000, "version": "v5dev"}},
}
(work / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
print("素材：", work / "素材", flush=True)

from voicetwin.config import load_config  # noqa: E402
from voicetwin.webui.launcher import launch  # noqa: E402

os.chdir(work)
launch(load_config(str(work / "config.yaml")), host="127.0.0.1", port=port)
