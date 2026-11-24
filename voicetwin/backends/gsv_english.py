# -*- coding: utf-8 -*-
"""英文读音预检：GPT-SoVITS（V5 整合包）的英文读音程序会把讲稿里哪些英文读错；只改发给引擎的那份文字。

为什么要有它（实测见 research/V5极致/英文读音预检.md）：V5 的英文读音全靠 GPT_SoVITS/text/english.py
和 en_normalization/expend.py（整合包里的这两个文件和 GitHub 上 main 的大小一样），英文没有 BERT 特征，
读音错了就一定读错。规则全部照这两个文件：
- expend.normalize 最后一步 re.sub(r"(?<!^)(?<![\\s])([A-Z])", r" \\1", text)：词中间的每个大写字母前面加空格，
  所以全是大写的词一个字母一个字母地读（AND → A N D），大小写混着的词拆开读（McDonald → Mc Donald）；
- en_G2p.__call__：单独一个大写的 A 读成字母 A（EY1）；
- 词典（cmudict.rep 第 57 行起 + cmudict-fast.rep + engdict-hot.rep，再删掉 AE AI AR IOS HUD OS）每个词只取
  第一个读音（lead 只有「铅」的 L EH1 D）；g2p_en 的同形词按词性猜，object 的两个读音还是反的；
- 词典里没有的（qryword）：只有首字母大写时查人名词典；不超过 3 个字母的一个字母一个字母地读；更长的拆词或者
  让神经网络猜；
- _measurement_re 把数字后面的 s 当成「秒」：1990s → nineteen ninety seconds；
- text_normalize 的 rep_map 键 "[;:：，；]" 和 '["’]' 经过 re.escape 只认字面的一整串，英文里的 ; : 没有换成逗号、
  弯引号 ’ 没有换成 '，接着 normalize 把它们删掉：英文里该停的地方不停，I’m 变成 Im；
- Dr / Dr. 在词典里是 drive（D R AY1 V）；Mr. No. U.S. 里的点变成句号（中间停一下）。

只改发给引擎的那份文字（engine_text）：字幕、查错字用的参考文字、老师看到的文字都不变。
训练用的文字不改（训练只列出来给老师看，见 precheck(..., auto_fix=False)）：改了训练标签要重新训练，
等老师词典能同时用在训练和生成上时再一起做（research/V5极致/英文读音预检.md 第 5 节）。

这个文件只用 Python 标准库（和 gsv_mixed_text.py 一样）。"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

# ============================================================================ 只改发给引擎的文字
#: 不是英文字母（判断「英文词的边界」用；不能用 \b：汉字也算 \w，「是Dr.」里 D 前面没有 \b）
_NL = r"(?<![A-Za-z])"
_NR = r"(?![A-Za-z])"
_DR_RE = re.compile(_NL + r"Dr\.?\s*(?=[A-Z])")
_MR_RE = re.compile(_NL + r"(Mr|Mrs|Ms)\.\s*(?=[A-Z])")
_NO_RE = re.compile(_NL + r"[Nn][Oo]\.\s*(?=\d)")
_VS_RE = re.compile(_NL + r"[Vv]s\.(?=\s|$)")
#: 每个字母后面都有点的缩写（U.S.、U.K.、a.m.）；e.g. 和 i.e. 引擎自己会换成 for example / that is，不动
_INITIALS_RE = re.compile(r"(?<![A-Za-z.])((?:[A-Za-z]\.){2,})" + _NR)
_DECADE_RE = re.compile(r"(?<![A-Za-z0-9.,])(1[0-9]|20)([0-9])0s" + r"(?![A-Za-z0-9])")
_SHORT_DECADE_RE = re.compile(r"[’'‘]([1-9])0s(?![A-Za-z0-9])")
_CURLY_RE = re.compile(r"(?<=[A-Za-z])[’‘](?=[A-Za-z])")
_PUNCT = ";:；："
_CLOSERS = "\"'”’)]）"

_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_DECADES = {1: "tens", 2: "twenties", 3: "thirties", 4: "forties", 5: "fifties", 6: "sixties", 7: "seventies",
            8: "eighties", 9: "nineties"}


@dataclass(frozen=True)
class Fix:
    """engine_text 改了的一处：kind = punct / abbr / decade / apostrophe。"""
    kind: str
    before: str
    after: str


def _decade_words(century: int, digit: int) -> str:
    if digit == 0:
        return "two thousands" if century == 20 else f"{_ONES[century]} hundreds"
    return f"{'twenty' if century == 20 else _ONES[century]} {_DECADES[digit]}"


def _english_before(text: str, i: int) -> bool:
    """text[i] 前面（跳过空格和右引号、右括号）是英文字母：这个 ; : 在英文里。"""
    j = i - 1
    while j >= 0 and (text[j] == " " or text[j] in _CLOSERS):
        j -= 1
    return j >= 0 and bool(re.match(r"[A-Za-z]", text[j]))


def _fix_punct(text: str, fixes: List[Fix]) -> str:
    """英文后面的 ; : → 逗号。在这句话最后的：冒号 → 句号（GPT-SoVITS 把冒号当成已经有标点、不再补句号，
    删掉以后这句话就没有结尾了）；分号不动（不算标点，引擎自己会补「。」，本来就是句末的读法）。"""
    out = list(text)
    for i, ch in enumerate(text):
        if ch not in _PUNCT or not _english_before(text, i):
            continue
        last = not text[i + 1:].strip(" " + _CLOSERS)
        new = ("." if ch in ":：" else ch) if last else ","
        if new != ch:
            out[i] = new
            fixes.append(Fix("punct", ch, new))
    return "".join(out)


def _sub(rx: "re.Pattern[str]", repl: Any, text: str, kind: str, fixes: List[Fix]) -> str:
    def one(m: "re.Match[str]") -> str:
        new = repl(m) if callable(repl) else m.expand(repl)
        if new != m.group(0):
            fixes.append(Fix(kind, m.group(0).strip(), new.strip()))
        return new

    return rx.sub(one, text)


def _initials(m: "re.Match[str]") -> str:
    letters = m.group(1).replace(".", "")
    if letters.lower() in ("eg", "ie"):
        return m.group(0)
    new = " ".join(letters.upper())
    rest = m.string[m.end():]
    return new + ("." if not rest.strip(" " + _CLOSERS) else "")  # 在句子最后：句号留着


def rewrite(text: str) -> Tuple[str, List[Fix]]:
    """engine_text 的结果和改了的每一处（同样的文字改第二遍不会再变）。"""
    fixes: List[Fix] = []
    if not text or not re.search(r"[A-Za-z]", text):
        return text, fixes
    t = _fix_punct(text, fixes)
    t = _sub(_DR_RE, "Doctor ", t, "abbr", fixes)
    t = _sub(_MR_RE, r"\1 ", t, "abbr", fixes)
    t = _sub(_NO_RE, "number ", t, "abbr", fixes)
    t = _sub(_VS_RE, "versus", t, "abbr", fixes)
    t = _sub(_INITIALS_RE, _initials, t, "abbr", fixes)
    t = _sub(_DECADE_RE, lambda m: _decade_words(int(m.group(1)), int(m.group(2))), t, "decade", fixes)
    t = _sub(_SHORT_DECADE_RE, lambda m: _DECADES[int(m.group(1))], t, "decade", fixes)
    t = _sub(_CURLY_RE, "'", t, "apostrophe", fixes)
    return t, fixes


def engine_text(text: str) -> str:
    """发给 GPT-SoVITS 的文字：英文里的 ; : → 逗号，Dr. → Doctor，Mr. → Mr，No. 1 → number 1，U.S. → U S，
    vs. → versus，1990s → nineteen nineties，词中间的弯引号 ’ → '。没有英文的句子原样返回。"""
    return rewrite(text)[0]


# ============================================================================ 引擎的英文词典（整合包里的）
@dataclass(frozen=True)
class EnglishDict:
    words: FrozenSet[str]          # 小写；cmudict.rep + cmudict-fast.rep + engdict-hot.rep，删掉 english.py 删的那几个
    names: FrozenSet[str]          # 人名词典（首字母大写的词才查）
    source: str


#: english.py 的 en_G2p.__init__ 从词典里删掉的（「读音错误的几个缩写」）
_DELETED = ("ae", "ai", "ar", "ios", "hud", "os")


class _PlainUnpickler(pickle.Unpickler):
    """人名词典是 pickle：只认 dict / list / str 这些不需要找类的东西，别的一律不读（不执行文件里的任何代码）。"""

    def find_class(self, module: str, name: str) -> Any:  # noqa: D401
        raise pickle.UnpicklingError(f"不读 {module}.{name}")


def _read_words(path: Path, sep: str, start_line: int) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            if n < start_line:
                continue
            line = line.strip()
            if line:
                out.append(line.split(sep)[0].lower())
    return out


@lru_cache(maxsize=4)
def _load(text_dir: str, stamp: Tuple[Any, ...]) -> Optional[EnglishDict]:
    d = Path(text_dir)
    cmu = d / "cmudict.rep"
    if not cmu.exists():
        return None
    words = set(_read_words(cmu, "  ", 57))                 # read_dict_new：从第 57 行起，两个空格隔开
    fast = d / "cmudict-fast.rep"
    if fast.exists():
        words.update(_read_words(fast, " ", 0))
    hot = d / "engdict-hot.rep"
    if hot.exists():
        words.update(_read_words(hot, " ", 0))
    words.difference_update(_DELETED)
    names: FrozenSet[str] = frozenset()
    nd = d / "namedict_cache.pickle"
    if nd.exists():
        try:
            with open(nd, "rb") as f:
                data = _PlainUnpickler(f).load()
            if isinstance(data, dict):
                names = frozenset(str(k).lower() for k in data)
        except Exception:  # noqa: BLE001 - 读不了人名词典：首字母大写的人名可能多报，不影响别的
            names = frozenset()
    return EnglishDict(frozenset(words), names, str(d))


def load_dictionary(text_dir: Optional[Path]) -> Optional[EnglishDict]:
    """整合包的 GPT_SoVITS/text 文件夹里的英文词典；找不到 cmudict.rep 时返回 None（这时不查「词典里没有的词」）。
    按文件的修改时间缓存：同一个整合包只读一次（约 14 万个词）。"""
    if not text_dir:
        return None
    d = Path(text_dir)
    stamp = []
    for name in ("cmudict.rep", "cmudict-fast.rep", "engdict-hot.rep", "namedict_cache.pickle"):
        try:
            st = os.stat(d / name)
            stamp.append((name, st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append((name, None, None))
    if stamp[0][1] is None:
        return None
    return _load(str(d), tuple(stamp))


# ============================================================================ 预检
#: 词典里只取第一个读音、两种读法意思不一样的词（不按词性判断，总是这个读音）：引擎的读法、另一种读法。
#: 每一个都在 V5 的 english.py 上实测过（research/V5极致/英文读音预检.md）
FIXED_READINGS: Dict[str, Tuple[str, str]] = {
    "lead": ("/led/（铅）", "/liːd/（带领）"),
    "live": ("/laɪv/（活的、现场的）", "/lɪv/（住、生活）"),
    "wind": ("/waɪnd/（缠绕）", "/wɪnd/（风）"),
    "wound": ("/waʊnd/（wind 的过去式）", "/wuːnd/（伤口）"),
    "tear": ("/teə/（撕）", "/tɪə/（眼泪）"),
    "bow": ("/baʊ/（鞠躬）", "/bəʊ/（弓）"),
    "bass": ("/bæs/（鲈鱼）", "/beɪs/（低音）"),
    "sow": ("/saʊ/（母猪）", "/səʊ/（播种）"),
}
#: g2p_en 的同形词里两个读音是反的（实测：I object → AA1 B JH EH0 K T 名词的读音，an object → 动词的读音；
#: They rebel → R EH1 B AH0 L，He is a rebel → R IH0 B EH1 L）
SWAPPED = {"object": "名词、动词的读音是反的（名词会读成 /əbˈdʒekt/，动词会读成 /ˈɒbdʒɪkt/）",
           "rebel": "名词、动词的读音是反的（动词会读成 /ˈrebl/）"}
#: english.py 自己改过的同形词：按词性猜，判成现在时（VB、VBP）才读 /riːd/（实测 I read books every day. → R IY1 D），
#: 单独一个 read、Please read it.、I have read the book. 都是 R EH1 D
READ_NOTE = "按词性猜：判成现在时才读 /riːd/，单独提到、祈使句（Please read it.）会读成过去式的 /red/"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’]*")
_NUM_S_RE = re.compile(r"(?<![\d.])\d+s(?![A-Za-z0-9])")
_ST_RE = re.compile(_NL + r"St\.")
_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])")


@dataclass(frozen=True)
class Finding:
    """可能读错的一处。kind：caps（全大写的词，也是一个单词）/ acronym（全大写的缩写）/ split（大小写混着，拆开读）/
    letter_a（大写 A 读成字母）/ fixed_reading / swapped / read（多音词）/ short（词典里没有、按字母读）/
    guess（词典里没有、引擎自己猜）/ street（St.）/ num_s（数字后面的 s 读成秒）。"""
    kind: str
    word: str
    detail: str


def _pieces(word: str) -> List[str]:
    """expend.normalize 在词中间的大写字母前面加空格以后，这个词变成的几段。"""
    return [p for p in _SPLIT_RE.split(word) if p]


def _lookup(piece: str, dic: EnglishDict) -> str:
    """qryword 会怎么处理这一段：ok / short（不超过 3 个字母、按字母读）/ guess（拆词或者神经网络猜）。"""
    w = piece.replace("’", "").lower()  # 弯引号会被 normalize 删掉（rep_map 的键没生效）
    if len(w) <= 1 or w in dic.words:
        return "ok"
    if piece.istitle() and w in dic.names:
        return "ok"
    if re.fullmatch(r"[a-z]+'s", w):
        return _lookup(w[:-2], dic)
    if len(w) <= 3:
        return "short"
    return "guess"


def find(text: str, dictionary: Optional[EnglishDict] = None) -> List[Finding]:
    """一句话（已经是要发给引擎的文字）里可能读错的英文。dictionary 是 None 时不查「词典里没有的词」，
    全大写的词也分不出是单词还是缩写（都按 caps 报）。"""
    out: List[Finding] = []
    if not text or not re.search(r"[A-Za-z]", text):
        return out
    for m in _WORD_RE.finditer(text):
        word = m.group(0).rstrip("'’")
        pieces = _pieces(word)
        letters = re.sub(r"[^A-Za-z]", "", word)
        if len(letters) >= 2 and letters.isupper():
            spelled = "-".join(letters)
            if dictionary is None:
                out.append(Finding("caps", word, f"全是大写，会一个字母一个字母地读成 {spelled}"
                                                 "（是一个单词的话，写成小写才按单词读）"))
            elif letters.lower() in dictionary.words:
                out.append(Finding("caps", word, f"全是大写，会一个字母一个字母地读成 {spelled}；"
                                                 f"想读成单词就写成小写 {letters.lower()}"))
            else:
                out.append(Finding("acronym", word, "全是大写，会按字母读"))
            continue
        if (word == "A" and re.match(r"\s+[a-z]", text[m.end():])
                and not re.search(r"(?:^|[^A-Za-z])[A-Z]\s+$", text[:m.start()])):  # U S A government 不算
            out.append(Finding("letter_a", word, "单独的大写 A 会读成字母 A /eɪ/；当冠词用请写成小写 a"))
            continue
        low = word.lower()
        if len(pieces) == 1:
            if low in FIXED_READINGS:
                said, other = FIXED_READINGS[low]
                out.append(Finding("fixed_reading", word, f"引擎总是读成 {said}，读不出 {other}"))
                continue
            if low in SWAPPED:
                out.append(Finding("swapped", word, SWAPPED[low]))
                continue
            if low == "read":
                out.append(Finding("read", word, READ_NOTE))
                continue
        if dictionary is None:
            if len(pieces) > 1:
                out.append(Finding("split", word, f"大小写混着，会拆成 {'、'.join(pieces)} 分开读"))
            continue
        how = [(p, _lookup(p, dictionary)) for p in pieces]
        bad = [(p, h) for p, h in how if h != "ok"]
        if len(pieces) > 1:
            if bad:
                tail = "；".join(f"{p} {'会按字母读' if h == 'short' else '词典里没有、引擎会自己猜'}" for p, h in bad)
                out.append(Finding("split", word, f"大小写混着，会拆成 {'、'.join(pieces)} 分开读，{tail}"))
            continue
        if bad:
            h = bad[0][1]
            out.append(Finding(h, word, "词典里没有，会一个字母一个字母地读" if h == "short"
                               else "词典里没有，引擎会自己猜读音（可能读错）"))
    for m in _ST_RE.finditer(text):
        out.append(Finding("street", "St.", "会读成 street（街）并停一下；要读 Saint 就写成 Saint"))
    for m in _NUM_S_RE.finditer(text):
        out.append(Finding("num_s", m.group(0), "数字后面的 s 会读成 seconds（秒）"))
    return out


# ============================================================================ 整篇的结果、给老师看的话
_FIX_LABEL = {"punct": "英文里的分号、冒号改成逗号（句子最后的冒号改成句号；不改的话引擎会把它们删掉，该停的地方不停）",
              "apostrophe": "英文词中间的弯引号 ’ 改成 '（不改的话会被删掉，I’m 会读成 Im）"}
#: 训练时（不自动改）带点的缩写要说的话
_ABBR_TRAIN = {"Dr": "会读成 drive", "Mr": "点会变成句号，中间停一下", "Mrs": "点会变成句号，中间停一下",
               "Ms": "点会变成句号，中间停一下", "No": "会读成 no 并停一下", "vs": "点会变成句号，中间停一下"}
#: 最多列几个词（网页上一条提醒不能太长）
MAX_ITEMS = 6
#: 列出来的先后：一定会读错的在前（全大写的单词、带点的缩写、标点……），引擎自己猜的、多音词在后；同一种按出现次数
_ORDER = ("caps", "abbr", "punct", "apostrophe", "num_s", "letter_a", "swapped", "fixed_reading", "read", "split",
          "short", "guess", "street")


@dataclass
class _Item:
    kind: str
    word: str
    detail: str
    sentences: List[int] = field(default_factory=list)
    count: int = 0


@dataclass
class Precheck:
    """precheck() 的结果。items：可能读错的（按词合并，顺序 = 第一次出现）；fixes：自动改好的（auto_fix 时）。"""
    items: List[_Item]
    fixes: List[_Item]
    acronyms: List[_Item]
    auto_fix: bool
    dictionary: Optional[str]
    n_texts: int

    @staticmethod
    def _where(it: _Item, numbered: bool) -> str:
        if not numbered:
            return f"{it.count} 处"
        nums = sorted(set(it.sentences))
        head = "、".join(str(n) for n in nums[:5]) + ("……" if len(nums) > 5 else "")
        return f"第 {head} 句" + (f"，共 {it.count} 处" if it.count > len(nums) else "")

    def _list(self, items: Sequence[_Item], numbered: bool) -> str:
        rank = {k: i for i, k in enumerate(_ORDER)}
        items = sorted(items, key=lambda it: (rank.get(it.kind, len(rank)), -it.count))  # sorted 是稳定的：同样的按先后
        shown = [f"{it.word}（{self._where(it, numbered)}：{it.detail}）" for it in items[:MAX_ITEMS]]
        more = len(items) - MAX_ITEMS
        return "；".join(shown) + (f"；还有 {more} 个（都写在结果报告里）" if more > 0 else "")

    def _acronym_clause(self) -> str:
        if not self.acronyms:
            return ""
        words = "、".join(it.word for it in self.acronyms[:MAX_ITEMS]) + ("等" if len(self.acronyms) > MAX_ITEMS else "")
        return f"全是大写的缩写 {words} 会按字母读（一般就是对的）。"

    def risk_line(self) -> str:
        """生成前给老师看的一句话（没有可能读错的词时是空字符串）。"""
        if not self.items:
            return ""
        how = ("想换读法：在讲稿里改写（例如全大写的 AND 写成小写 and），或者在声音文件夹的 lexicon.txt 里加一行"
               "「AND => and」（只改读法，字幕不变）。多音词暂时没法在这里选读法，生成以后听一下这几句。")
        dic = "" if self.dictionary else "（没找到整合包的英文词典，这次没查词典里没有的词。）"
        return f"英文读音预检（生成前查的）：这些英文可能读错：{self._list(self.items, True)}。{how}{dic}"

    def fix_line(self) -> str:
        """自动改好的（只改发给引擎的文字）；全大写的缩写也在这里说一句。没有时是空字符串。"""
        parts: List[str] = []
        for it in self.fixes:
            parts.append(f"{it.detail}（{self._where(it, True)}）")
        fixed = ("英文读音预检：已经自动改好（只改发给引擎的文字，字幕和你看到的文字都不变）："
                 + "；".join(parts) + "。") if parts else ""
        clause = self._acronym_clause()
        if clause and not fixed:
            return "英文读音预检：" + clause
        return fixed + clause

    def training_line(self) -> str:
        """训练前（「② 训练模型」页的素材检查）：训练用的文字照原样，所以自动改的那些也列出来。"""
        items = list(self.items) + list(self.fixes)
        if not items:
            return ""
        return ("英文读音预检：训练素材里这些英文会按下面的读法去学，可能和你说的不一样："
                f"{self._list(items, False)}。训练用的文字照原样不改；想让它按你说的学，可以在校对表里改成你实际读的写法"
                "（例如全大写的 AND 改成 and、Dr. 改成 Doctor）。")

    def to_json(self) -> Dict[str, Any]:
        def rows(items: Sequence[_Item]) -> List[Dict[str, Any]]:
            return [{"kind": it.kind, "word": it.word, "detail": it.detail, "count": it.count,
                     "sentences": sorted(set(it.sentences))} for it in items]

        return {"items": rows(self.items), "fixed": rows(self.fixes), "acronyms": rows(self.acronyms),
                "auto_fix": self.auto_fix, "dictionary": self.dictionary, "texts": self.n_texts}


def _add(book: Dict[Tuple[str, str], _Item], order: List[_Item], kind: str, word: str, detail: str, n: int) -> None:
    """同一种情况、同一个词合成一条，记下第几句、几处。"""
    it = book.get((kind, word))
    if it is None:
        it = book[(kind, word)] = _Item(kind, word, detail)
        order.append(it)
    it.count += 1
    it.sentences.append(n)


def _train_item(fx: Fix) -> Tuple[str, str, str]:
    """训练时（文字照原样）engine_text 本来会改的一处：(kind, 词, 说明)。"""
    if fx.kind == "punct":
        return "punct", "英文里的 ; :", "会被删掉，该停的地方不停"
    if fx.kind == "apostrophe":
        return "apostrophe", "英文词中间的弯引号 ’", "会被删掉（I’m 会读成 Im）"
    if fx.kind == "decade":
        return "num_s", fx.before, "数字后面的 s 会读成 seconds（秒）"
    head = re.match(r"[A-Za-z]+", fx.before)
    return "abbr", fx.before, _ABBR_TRAIN.get(head.group(0) if head else "", "点会变成句号，中间停一下")


def precheck(texts: Sequence[str], dictionary: Optional[EnglishDict] = None, auto_fix: bool = True) -> Precheck:
    """一篇讲稿（或者全部训练素材）的英文读音预检。texts 的顺序就是句子编号（从 1 开始）。

    auto_fix=True（生成）：先按 engine_text 改，改好的记在 fixes，剩下可能读错的记在 items；
    auto_fix=False（训练：文字照原样）：engine_text 会改的那些也算可能读错，记在 items。
    全大写的缩写（词典里没有这个词，按字母读一般就是对的）单独记在 acronyms。"""
    items: List[_Item] = []
    fixes: List[_Item] = []
    acronyms: List[_Item] = []
    book: Dict[Tuple[str, str], _Item] = {}
    fbook: Dict[Tuple[str, str], _Item] = {}
    abook: Dict[Tuple[str, str], _Item] = {}
    for n, text in enumerate(texts, 1):
        text = str(text or "")
        new, done = rewrite(text)
        for fx in done:
            if not auto_fix:
                _add(book, items, *_train_item(fx), n)
            elif fx.kind in _FIX_LABEL:
                _add(fbook, fixes, fx.kind, "", _FIX_LABEL[fx.kind], n)
            else:
                _add(fbook, fixes, fx.kind, fx.before, f"{fx.before} → {fx.after}", n)
        for f in find(new if auto_fix else text, dictionary):
            if not auto_fix and f.kind == "num_s" and _DECADE_RE.fullmatch(f.word):
                continue  # 上面已经按 engine_text 会改的记过
            if f.kind == "acronym":
                _add(abook, acronyms, f.kind, f.word, f.detail, n)
            else:
                _add(book, items, f.kind, f.word, f.detail, n)
    return Precheck(items, fixes, acronyms, auto_fix, dictionary.source if dictionary else None, len(texts))
