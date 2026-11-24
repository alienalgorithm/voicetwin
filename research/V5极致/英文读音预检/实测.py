# -*- coding: utf-8 -*-
"""英文读音预检的实测：VoiceTwin 的预检（voicetwin/backends/gsv_english.py）找出来的每一处，
用 GPT-SoVITS 自己的英文读音程序（GPT_SoVITS/text/english.py + en_normalization/expend.py）真的跑一遍，
看改之前、改之后引擎实际用的音素（ARPAbet）。结果写进同一个文件夹的 实测结果.txt。

需要：GPT-SoVITS 的 GPT_SoVITS 文件夹（里面的 text/ 和整合包里的大小一样，见 调研_美化_english-pron.md 第 1 节），
g2p_en 2.1.0、inflect、wordsegment、nltk（和 averaged_perceptron_tagger、cmudict 数据）。不需要显卡、不下载模型。

用法：
  NLTK_DATA=<nltk 数据> python -I 实测.py <GPT_SoVITS 文件夹> <voicetwin 仓库>
"""

import importlib.util
import re
import sys
from pathlib import Path

gsv_dir = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(gsv_dir))
from text.english import _g2p, g2p, text_normalize  # noqa: E402

spec = importlib.util.spec_from_file_location("gsv_english", repo / "voicetwin" / "backends" / "gsv_english.py")
G = importlib.util.module_from_spec(spec)
sys.modules["gsv_english"] = G
spec.loader.exec_module(G)

out_lines = []


def say(s=""):
    print(s)
    out_lines.append(s)


def phones(text):
    return " ".join(g2p(text_normalize(text)))


EN_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 '’.,;:!?\-]*")


def runs(text):
    """一句话里连着的英文（大致就是 GPT-SoVITS 切出来的英文段）。"""
    return [m.group(0).strip() for m in EN_RUN.finditer(text) if re.search(r"[A-Za-z]", m.group(0))]


corpus = repo / "voicetwin" / "data" / "lexicon" / "core_corpus.tsv"
rows = [ln.split("\t", 1) for ln in corpus.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
rows = [r for r in rows if len(r) == 2]
texts = [t for _i, t in rows]
dic = G.load_dictionary(gsv_dir / "text")

say("# 英文读音预检实测结果（实测.py 生成，不要手改）")
say(f"GPT_SoVITS 文件夹：text/english.py {(gsv_dir / 'text' / 'english.py').stat().st_size} 字节，"
    f"en_normalization/expend.py {(gsv_dir / 'text' / 'en_normalization' / 'expend.py').stat().st_size} 字节，"
    f"cmudict.rep {(gsv_dir / 'text' / 'cmudict.rep').stat().st_size} 字节")
say(f"词典：VoiceTwin 读到 {len(dic.words)} 个词、{len(dic.names)} 个人名；english.py 自己的 _g2p.cmu 有 {len(_g2p.cmu)} 个词；"
    f"两边一样：{set(dic.words) == set(_g2p.cmu)}")
say()
en_lines = [t for t in texts if re.search(r"[A-Za-z]", t)]
words = [w for t in en_lines for w in re.findall(r"[A-Za-z][A-Za-z'’]*", t)]
say(f"## 1. 母本 core_corpus.tsv：{len(texts)} 句，{len(en_lines)} 句有英文，{len(words)} 个英文词（{len({w.lower() for w in words})} 个不同的）")
for auto_fix in (True, False):
    rep = G.precheck(texts, dic, auto_fix=auto_fix)
    say(f"### auto_fix={auto_fix}（{'生成' if auto_fix else '训练'}）")
    for title, items in (("可能读错", rep.items), ("自动改好", rep.fixes), ("全大写的缩写", rep.acronyms)):
        for it in items:
            ids = [rows[n - 1][0] for n in sorted(set(it.sentences))]
            say(f"- {title} {it.kind} {it.word} ×{it.count}：{it.detail}；句子 {', '.join(ids[:6])}{' ……' if len(ids) > 6 else ''}")
    say("给老师看的话：")
    for line in ((rep.risk_line(), rep.fix_line()) if auto_fix else (rep.training_line(),)):
        if line:
            say("  " + line)
    say()

say("## 2. 每一处在引擎里实际的音素（改之前 → 改之后）")
seen = set()
for i, t in enumerate(texts):
    fs = G.find(t, dic)
    new, fixes = G.rewrite(t)
    if not fs and not fixes:
        continue
    for run in runs(t):
        if run in seen:
            continue
        hit = [f.word for f in fs if f.word in run] + [x.before for x in fixes if x.before in run]
        if not hit:
            continue
        seen.add(run)
        say(f"- {rows[i][0]}「{run}」（{', '.join(dict.fromkeys(hit))}）")
        say(f"    引擎：{phones(run)}")
        new_run = G.engine_text(run)
        if new_run != run:
            say(f"    改后「{new_run}」：{phones(new_run)}")
say()

say("## 3. 自动改的每一种写法（改之前 → 改之后，引擎实际的音素）")
for before in ["However, the data show otherwise; we must act.", "Note: this is important.", "as follows:",
               "You can call him Dr.Banner or Hulk.", "Mr. Smith and Mrs. Lee", "He is No. 1 in the class.",
               "the U.S. government", "at 9 a.m. every day", "Tom vs. Jerry", "in the 1990s", "the 1900s", "the 2000s",
               "the 2010s", "back in the ’90s", "I’m sure it’s fine.", "don’t go"]:
    after = G.engine_text(before)
    say(f"- 「{before}」：{phones(before)}")
    say(f"  「{after}」：{phones(after)}")
say()

say("## 4. 多音词：词典的第一个读音、g2p_en 的同形词（按词性）")
alts = {}
for line in (gsv_dir / "text" / "cmudict.rep").read_text(encoding="utf-8", errors="replace").splitlines():
    m = re.match(r"^([A-Z']+)\((\d)\)  (.*)$", line.strip())
    if m:
        alts.setdefault(m.group(1).lower(), []).append(m.group(3))
H = _g2p.homograph2features
for w in list(G.FIXED_READINGS) + list(G.SWAPPED) + ["read", "lives", "does", "minute", "dove"]:
    first = " ".join(_g2p.cmu.get(w, [["-"]])[0])
    hom = f"同形词 {H[w]}" if w in H else "不是同形词（总是第一个读音）"
    say(f"- {w}：第一个读音 {first}；其它 {alts.get(w, [])}；{hom}")
for s in ["This will lead to a crisis.", "live本身属于是不及物动词", "I live in Beijing.", "The wind is strong.",
          "a tear in her eye", "I object to this plan.", "an object", "Please read it.", "I have read the book.",
          "I read books every day.", "read", "They rebel against the rules.", "He is a rebel.",
          "She lives in a small city.", "A good student works hard.", "a good student",
          "Tomorrow I'm going to McDonald's.", "That boy who is talking is JimWho is talking"]:
    say(f"- 「{s}」：{phones(''.join(runs(s)) if re.search(r'[一-鿿]', s) else s)}")

(Path(__file__).resolve().parent / "实测结果.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
