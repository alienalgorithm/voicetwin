"""三个识别引擎投票（查错字用）：faster-whisper（主识别，现在的文字）、FunASR paraformer、SenseVoiceSmall。

为什么：在老师新录的 26 分钟（204 句，两路独立核对的定稿）上实测，三个引擎各错各的：单个引擎的字错率
faster-whisper 3.70%、FunASR 2.82%、SenseVoice 2.56%；按字投票、两个引擎听到的一样就改，错字少很多
（实测脚本和数字：research/文字校正/三引擎投票/）。

怎么投（全部是文字处理，不用模型、不联网）：
1. 三段文字都切成可以比较的单位（proofcheck.tokenize：汉字一个字一个，英文一个词一个，数字按数值，标点不算），
   另外两个引擎的结果分别和现在的文字（主识别）逐字对齐；
2. 不一样的地方一处一处投票，少数服从多数：
   - 另外两个引擎听到的一模一样、和现在的文字不一样 → 有把握的建议（「一键全部文字校正」直接采用）。
     英文也一样：两个引擎拼出来一模一样才算（实测这样改英文，改对的远多于改错的）；
   - 只有一个引擎和现在的文字不一样：两票对一票，现在的文字赢，这里不给建议（FunASR 不一样的地方照旧由
     查错字标红，老师听录音决定）。所以英文只要另外两个引擎拼得不完全一样，就一定保留 faster-whisper 的拼法
     （FunASR、SenseVoice 常把 kills 写成 chaols、that 写成 net / dead）；
   - 三个都不一样：定不了，不给建议；
   - 例外：两票对一票（或者三个都不一样）时，少数那个引擎的写法正好凑成了标准库里的语法术语（「名词型从句」
     → 「名词性从句」），按术语改，也算有把握（实测这样改 32 个字，没有改错的）；
   - 读音一样、两种写法都常见的字（他 / 它、地 / 的、在 / 再）：听录音分不出来，不改（实测按投票改，对错各半）；
   - 一处超过 6 个字 / 词：多半是那一段整个没听清，只给建议、不自动采用。
3. 另一个引擎整句和现在的文字差得太多（按字对齐不到一半，可能那段录音有问题）：它这一句不参加投票；
   参加投票的不到两个别的引擎时，不投票（返回空的）。

对外接口：
- vote(text, others, terms=None) -> List[VoteEdit]：现在的文字上要改的地方（位置按 text 算）
- apply(text, edits, sure_only=True) -> str
"""

from __future__ import annotations

import difflib
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from voicetwin.data import proofcheck as pc

#: 另一个引擎整句和现在的文字至少这么像（按字对齐的比例）才参加投票
MIN_RATIO = 0.5
#: 一处有把握的改动最多这么多个字 / 词（再长多半是那一段整个没听清，不自动改）
MAX_SURE_TOKENS = 6
#: 读音一样、两种写法都常见的字：听录音分不出来，只能看上下文，投票也定不了，不改
#: （实测：这几组按投票改，对的 7 处、错的 6 处）
_EITHER_GROUPS = ("他她它祂牠", "的地得", "在再", "做作", "哪那")
_EITHER = {ch: g for g in _EITHER_GROUPS for ch in g}
#: 少数引擎的写法凑成了这么长（至少这么多个字）的语法术语，才按术语改（两个字的「介词」「谓语」太容易碰巧凑上）
MIN_TERM_CHARS = 3

Ops = Sequence[Tuple[str, int, int, int, int]]


class VoteEdit(NamedTuple):
    start: int  # 在现在的文字里的位置 [start, end)；start == end 是补字
    end: int
    rep: str  # 换成什么（英文两边已经补好空格）
    sure: bool  # 有把握：「一键全部文字校正」直接采用
    reason: str  # 给老师看的说明
    engines: Tuple[str, ...] = ()  # 哪几个引擎听成这样


class _Cand(NamedTuple):
    name: str
    text: str  # 清理过的识别结果
    toks: List[pc.Tok]  # 这一处的字
    keys: List[str]


def _vkey(t: pc.Tok) -> str:
    """投票用的单位：数字按数值比（2 = 二），别的和查错字一样（汉字转简体、英文小写）。"""
    if t.key == "#":
        return "#" + str(t.val if t.val is not None else "")
    return t.key


def _map_start(ops: Ops, p: int, n_other: int) -> int:
    """现在的文字里从第 p 个字开始的一处，在另一个引擎里从第几个字开始（这一处前面补的字也算进来）。"""
    for tag, i1, i2, j1, _j2 in ops:
        if tag == "equal":
            if i1 <= p < i2:
                return j1 + (p - i1)
        elif i1 == p or i1 < p < i2:
            return j1
    return n_other


def _map_end(ops: Ops, p: int) -> int:
    """现在的文字里到第 p 个字为止的一处，在另一个引擎里到第几个字为止（这一处后面补的字也算进来）。"""
    for tag, i1, i2, j1, j2 in reversed(ops):
        if tag == "equal":
            if i1 < p <= i2:
                return j1 + (p - i1)
        elif i2 == p or i1 < p < i2:
            return j2
    return 0


def _regions(all_ops: Sequence[Ops]) -> List[Tuple[int, int]]:
    """所有引擎和现在的文字不一样的地方（按现在的文字的位置），重叠或者挨着的合成一处。"""
    iv = sorted((i1, i2) for ops in all_ops for tag, i1, i2, _j1, _j2 in ops if tag != "equal")
    out: List[List[int]] = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _builtin_terms() -> Tuple[str, ...]:
    try:
        from voicetwin.data.lexicon_fix import builtin_terms

        return builtin_terms()
    except Exception:  # noqa: BLE001 - 读不了标准库：只按多数投票
        return ()


def vote(text: str, others: Dict[str, Optional[str]], terms: Optional[Iterable[str]] = None) -> List[VoteEdit]:
    """三个引擎投票：返回现在的文字（text，主识别 faster-whisper 的结果）上要改的地方。

    others：{引擎名: 它听到的文字}（None、空的不算）。至少要有两个别的引擎参加（加上现在的文字一共三个）才投票，
    否则返回 []。
    terms：语法术语（默认用程序自带的标准库 grammar_terms.txt）。"""
    text = str(text or "")
    P = pc.tokenize(text)
    if not P:
        return []
    pk = [_vkey(t) for t in P]
    engines: List[Tuple[str, str, List[pc.Tok], List[str], Ops]] = []
    for name, other in (others or {}).items():
        if other is None or not str(other).strip():
            continue
        o = pc._clean_other(other)
        X = pc.tokenize(o)
        xk = [_vkey(t) for t in X]
        sm = difflib.SequenceMatcher(None, pk, xk, autojunk=False)
        if not X or sm.ratio() < MIN_RATIO:
            continue  # 整句差得太多：这一句它不参加投票
        engines.append((str(name), o, X, xk, sm.get_opcodes()))
    if len(engines) < 2:
        return []
    out: List[VoteEdit] = []
    minority: List[VoteEdit] = []  # 投票定不了的地方，每个和现在的文字不一样的引擎各一个（下面看术语）
    for a, b in _regions([e[4] for e in engines]):
        cands = []
        for name, o, X, xk, ops in engines:
            j1 = _map_start(ops, a, len(X))
            j2 = max(j1, _map_end(ops, b))
            cands.append(_Cand(name, o, X[j1:j2], xk[j1:j2]))
        won, lost = _decide(text, P, pk, a, b, cands)
        out += won
        minority += lost
    if minority:
        out += _by_terms(text, out, minority, terms)
    return sorted(out, key=lambda x: (x.start, x.end))


def _by_terms(text: str, won: Sequence[VoteEdit], minority: Sequence[VoteEdit],
              terms: Optional[Iterable[str]]) -> List[VoteEdit]:
    """投票定不了的地方：少数那个引擎的写法正好凑成了语法术语（多数投票改好以后再看），就按术语改。"""
    tl = [t for t in (_builtin_terms() if terms is None else terms)
          if sum(1 for ch in str(t) if pc._CJK_RE.match(ch)) >= MIN_TERM_CHARS]
    if not tl:
        return []
    sure = [ed for ed in won if ed.sure]
    base = apply(text, sure)
    out: List[VoteEdit] = []
    for ed in minority:
        if any(ed.start < x.end and x.start < ed.end or (ed.start, ed.end) == (x.start, x.end) for x in out):
            continue  # 这里已经按另一个引擎的术语改了
        new = apply(text, list(sure) + [ed], sure_only=False)
        hit = next((t for t in tl if new.count(t) > base.count(t)), None)
        if hit is None:
            continue
        who = pc.ENGINE_SHORT.get(ed.engines[0], ed.engines[0]) if ed.engines else ""
        out.append(ed._replace(sure=True, reason=f"{who} 听成了语法术语「{hit}」（标准库里的写法）"))
    return out


def _decide(text: str, P: Sequence[pc.Tok], pk: Sequence[str], a: int, b: int,
            cands: Sequence[_Cand]) -> Tuple[List[VoteEdit], List[VoteEdit]]:
    """一处不一样的地方投票：返回 (投出来的改法, 定不了时各个少数引擎的写法)。"""
    mine = list(pk[a:b])
    won: List[VoteEdit] = []
    lost: List[VoteEdit] = []
    if mine and all(len(c.keys) == len(mine) for c in cands):
        # 字数一样：一个字一个字投（「名词英诵句 / 名词性从句 / 名词型从句」：只有「诵 → 从」两个引擎都这么听）
        k = 0
        while k < len(mine):
            got = _winner(mine[k:k + 1], [c.keys[k:k + 1] for c in cands])
            m = k + 1  # 挨着的几个字都是同样的结果：合成一处
            while m < len(mine) and _winner(mine[m:m + 1], [c.keys[m:m + 1] for c in cands]) == got and (
                    got is not None or any(c.keys[m] != mine[m] for c in cands)):
                m += 1
            part = [_Cand(c.name, c.text, c.toks[k:m], c.keys[k:m]) for c in cands]
            if got is not None:  # 连着改的字太多：一样只给建议
                won.append(_edit(text, P, a + k, a + m, part, (got[0], got[1] and m - k <= MAX_SURE_TOKENS)))
            elif any(c.keys != mine[k:m] for c in part):
                lost += _minority(text, P, a + k, a + m, part, mine[k:m])
            k = m
        return won, lost
    got = _winner(mine, [c.keys for c in cands])
    if got is not None:
        return [_edit(text, P, a, b, cands, got)], []
    return [], _minority(text, P, a, b, cands, mine)


def _minority(text: str, P: Sequence[pc.Tok], i1: int, i2: int, cands: Sequence[_Cand],
              mine: Sequence[str]) -> List[VoteEdit]:
    out: List[VoteEdit] = []
    seen: List[List[str]] = []
    for idx, c in enumerate(cands):
        if c.keys == list(mine) or c.keys in seen:
            continue
        seen.append(c.keys)
        out.append(_edit(text, P, i1, i2, cands, (idx, False)))
    return out


def _winner(mine: Sequence[str], keys: Sequence[Sequence[str]]) -> Optional[Tuple[int, bool]]:
    """这一处谁赢：返回 (第几个别的引擎, 有没有把握)；现在的文字赢了、或者定不了，返回 None。"""
    mine = list(mine)
    if any(list(k) == mine for k in keys):
        return None  # 至少一个别的引擎和现在的文字一样：两票对一票，不改
    for i, k in enumerate(keys):
        same = sum(1 for x in keys if list(x) == list(k))
        if same >= 2 and 2 * same > len(keys):  # 另外两个引擎听到的一模一样：两票对一票
            if _either(mine, list(k)):
                return None
            return i, max(len(mine), len(k)) <= MAX_SURE_TOKENS
    return None  # 三个都不一样：定不了


def _either(mine: List[str], new: List[str]) -> bool:
    """读音一样、两种写法都常见的一个字（他 / 它、地 / 的、在 / 再）：听录音分不出来，投票也定不了。"""
    return len(mine) == len(new) == 1 and mine[0] in _EITHER and _EITHER.get(new[0]) == _EITHER[mine[0]]


def _edit(text: str, P: Sequence[pc.Tok], i1: int, i2: int, cands: Sequence[_Cand],
          got: Tuple[int, bool]) -> VoteEdit:
    idx, sure = got
    win = cands[idx]
    agree = [c for c in cands if c.keys == win.keys]
    # 换成什么：优先用 SenseVoice 的写法（带标点、英文大小写正常）
    pick = next((c for c in agree if c.name == pc.ENGINE_SENSEVOICE), win)
    rep = pick.text[pick.toks[0].start:pick.toks[-1].end] if pick.toks else ""
    if i1 < i2:
        s, e = P[i1].start, P[i2 - 1].end
    else:  # 补字：补在前一个字后面
        s = e = P[i1 - 1].end if i1 > 0 else P[0].start
    rep = pc._pad(text, s, e, rep)
    old, new = text[s:e], rep.strip()
    if not old:
        reason = f"另外两个识别引擎都听到这里还有「{pc._short(new)}」"
    elif not new:
        reason = f"另外两个识别引擎都没听到「{pc._short(old)}」"
    else:
        reason = f"另外两个识别引擎都听成「{pc._short(new)}」"
    return VoteEdit(s, e, rep, sure, reason, tuple(c.name for c in agree))


def apply(text: str, edits: Sequence[VoteEdit], sure_only: bool = True) -> str:
    """把投票的结果改进文字里（默认只改有把握的）。"""
    out = str(text or "")
    for ed in sorted(edits, key=lambda x: (x.start, x.end), reverse=True):
        if sure_only and not ed.sure:
            continue
        out = out[:ed.start] + ed.rep + out[ed.end:]
    return out


__all__ = ["VoteEdit", "vote", "apply", "MIN_RATIO", "MAX_SURE_TOKENS", "MIN_TERM_CHARS"]
