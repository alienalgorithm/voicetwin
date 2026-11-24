#!/usr/bin/env bash
# .github/workflows/train-real.yml 各个任务共用的步骤（只在 GitHub 的机器上用）。
# 用法：bash research/训练实测/ci.sh prep-env | gsv-env train|eval | light-env | commit "说明" 路径...
set -euo pipefail
cd "$(dirname "$0")/../.."

# 官方 V5 整合包（RTX 50 系列版，和 gsv-v5-real.yml 一样），里面有真的 GPT-SoVITS 代码和底模
PKG=GPT-SoVITS-v5-20261005nvidia50x0.7z
SHA=27b269fc1c53823dc9d404ada83f4444b300e455011db9861ec72e01b6fc5d5e

free_disk() {
  sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL /usr/local/share/boost || true
  sudo mkdir -p /mnt/w && sudo chown "$USER" /mnt/w
  df -h / /mnt
}

case "${1:-}" in
  prep-env)
    # 准备素材：和以前转录（transcribe-old.yml / transcribe.yml）装的一样，降噪、切片用的库一样，切出来的片段（id）才一样
    free_disk
    sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg >/dev/null
    python -m pip install -U pip wheel
    python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -e ".[asr,funasr,dev,sv,denoise]" webrtcvad-wheels
    python -m pip install --no-deps resemblyzer
    ;;
  gsv-env)
    # 训练 / 打分：官方 V5 整合包的代码和底模 + Linux 的 Python 3.12，库的版本和整合包一样（和 gsv-v5-real.yml 一样）
    what="${2:-train}"
    free_disk
    sudo apt-get update -qq && sudo apt-get install -y -qq p7zip-full espeak-ng ffmpeg >/dev/null
    (
      cd /mnt/w
      curl -fL --retry 5 -o "$PKG" "https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/$PKG?download=true"
      echo "$SHA  $PKG" | sha256sum -c
      7z x -y -o/mnt/w/g "$PKG" -xr!Lib -xr!uvr5 >/dev/null
      rm -f "$PKG"
    )
    G=$(ls -d /mnt/w/g/GPT-SoVITS-*)
    # 打分时查错字要用整合包自带的中文识别模型（Paraformer）；训练用不到
    [ "$what" = eval ] || rm -rf "$G/tools/asr/models"
    echo "GSV=$G" >> "$GITHUB_ENV"
    echo "NLTK_DATA=$G/runtime/nltk_data" >> "$GITHUB_ENV"
    du -sh "$G"; df -h /mnt
    python -m pip install -U pip wheel
    python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    grep -viE '^(--no-binary|pyopenjtalk|python_mecab_ko|g2pk2|ko_pron|ToJyutping|funasr|modelscope|onnxruntime)' \
      "$G/requirements.txt" | sed -E 's/[[:space:]]*([<>=!~;].*)?$//' | grep -v '^$' > /tmp/req.txt
    printf 'onnxruntime\nmatplotlib\nresampy\n' >> /tmp/req.txt
    python -m pip install -c tests/constraints-gsv-v5.txt -r /tmp/req.txt
    python -m pip install -c tests/constraints-gsv-v5.txt -e ".[sv]" webrtcvad-wheels zhconv
    python -m pip install --no-deps resemblyzer noisereduce
    if [ "$what" = eval ]; then
      # 查错字：和老师电脑上的整合包一样的版本（中文 Paraformer 用 funasr，有英文的句子用 faster-whisper）；
      # 装不上也接着做（错字率那一列就是「—」，结果里如实写）
      python -m pip install -c tests/constraints-gsv-v5.txt faster-whisper funasr modelscope \
        || echo "::warning::查错字用的识别库没装上，这次不量错字率"
    fi
    python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
    python research/gsv_v4_v5/真实V5实测/import_check.py "$G"
    ;;
  light-env)
    # 只合并结果（排名要用声音分身自己的规则）
    sudo mkdir -p /mnt/w && sudo chown "$USER" /mnt/w
    python -m pip install -U pip wheel
    python -m pip install -e .
    ;;
  commit)
    msg="$2"; shift 2
    git config user.name "github-actions[bot]"
    git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    git add -- "$@" || true
    git commit -m "$msg" || exit 0
    # 结果文件夹只有这个工作流在写：别的提交改了同一个文件时用这次的（-X theirs 在 rebase 里就是「这次的提交」）
    for i in 1 2 3 4 5 6; do
      git pull --rebase -X theirs origin "$GITHUB_REF_NAME" && git push origin "HEAD:$GITHUB_REF_NAME" && exit 0
      sleep $((i * 7))
    done
    echo "::warning::结果没能推送到仓库（Artifacts 里有）"
    ;;
  *)
    echo "用法：bash research/训练实测/ci.sh prep-env | gsv-env train|eval | light-env | commit \"说明\" 路径..." >&2
    exit 2
    ;;
esac
