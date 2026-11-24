"""在真的浏览器（Chromium）里把声音分身的网页从头点到尾，和老师的用法一样，设置全用默认的：
打开 → ① 准备素材 → ✅ 确认训练素材 → ② 开始训练（默认「一模一样」）→ ③ 生成（默认「一模一样」）。
每一步截图；网页脚本报错、哪一步没等到结果都算不通过。

用法：python browser_check.py <网址> <素材文件夹> <结果文件夹>
"""
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

url, lecture, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
res, errs, console = [], [], []


def ok(name, cond, info=""):
    res.append(("PASS " if cond else "FAIL ") + name + (f" — {str(info)[:400]}" if info else ""))
    print(res[-1], flush=True)
    return bool(cond)


def body(pg):
    return pg.inner_text("body")


def wait_text(pg, text, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if text in body(pg):
            return round(time.time() - t0)
        pg.wait_for_timeout(2000)
    return None


def shot(pg, name):
    pg.screenshot(path=str(out / f"{name}.png"), full_page=True)


def tab(pg, name):
    pg.get_by_role("tab", name=name).click()
    pg.wait_for_timeout(1500)


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1400, "height": 1000}, locale="zh-CN")
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
    pg.on("dialog", lambda d: d.accept())   # 「确定吗？」之类的对话框：和老师一样点确定
    try:
        # gradio 一直开着心跳连接，等不到「网络空闲」：等页面加载完、标签页出来
        pg.goto(url, wait_until="load", timeout=120000)
        pg.get_by_role("tab", name="① 准备素材").wait_for(timeout=120000)
        pg.wait_for_timeout(3000)
        ok("标签页标题是「声音分身 VoiceTwin v18」", pg.title() == "声音分身 VoiceTwin v18", pg.title())
        text = body(pg)
        ok("网页标题显示 v18", "声音分身 VoiceTwin v18" in text)
        m = re.search(r"模型：[^\n]*", text)
        # 替身的底模不是官方的文件，读不出版本是对的；真的 V5 底模读出来是什么在 ../真实V5实测/ 里查
        res.append("INFO 右上角的模型型号（替身的底模）：" + (m.group(0) if m else "没找到"))
        shot(pg, "1_打开网页")

        # ① 准备素材
        voice = pg.get_by_label("声音名称（新建请直接输入名字）")
        voice.fill("网页实测")
        voice.press("Enter")
        pg.get_by_label("或者填写电脑上的文件夹路径（推荐）").fill(lecture)
        pg.mouse.click(5, 5)
        pg.get_by_role("button", name="开始准备素材").click()
        t = wait_text(pg, "素材准备好了", 900)
        ok("① 准备素材做完", t is not None, f"{t} 秒" if t is not None else body(pg)[-800:])
        shot(pg, "2_准备素材")
        pg.get_by_role("button", name="✅ 确认训练素材").click()
        t = wait_text(pg, "训练素材已确认", 120)
        ok("✅ 确认训练素材", t is not None, "" if t is not None else body(pg)[-800:])
        shot(pg, "3_确认训练素材")

        # ② 训练模型（默认「一模一样」）
        tab(pg, "② 训练模型")
        checked = pg.locator("input[type=radio]:checked").evaluate_all("els=>els.map(e=>e.parentElement.innerText)")
        ok("训练方式默认「一模一样」", any("一模一样" in c for c in checked), checked)
        pg.get_by_role("button", name="开始训练").click()
        t = wait_text(pg, "✅ 训练完成", 3600)
        text = body(pg)
        ok("② 训练完成", t is not None, f"{t} 秒" if t is not None else text[-1500:])
        ok("训练计划写的是 V5（v5dev 的 LoRA 练法）", "v5dev" in text and "LoRA" in text,
           (re.search(r"训练计划：[^\n]*", text) or [""])[0])
        m = re.search(r"实际参加训练：音色 (\d+) 条、语气 (\d+) 条（一共 (\d+) 条）", text)
        ok("每条素材都参加了训练（音色、语气都是）", m and m.group(1) == m.group(2) == m.group(3),
           m.group(0) if m else "页面上没有「实际参加训练」那句")
        ok("训练没有报错", "❌" not in text, (re.search(r"❌[^\n]*", text) or [""])[0])
        shot(pg, "4_训练完成")

        # ③ 生成讲课音频（默认「一模一样」）
        tab(pg, "③ 生成讲课音频")
        checked = pg.locator("input[type=radio]:checked").evaluate_all("els=>els.map(e=>e.parentElement.innerText)")
        ok("质量默认「一模一样」", any(c.strip().startswith("一模一样") for c in checked), checked)
        ok("文件名的例子写的是 v5dev", "第3课_10月05日09点30分_v5dev.wav" in body(pg))
        pg.get_by_placeholder("大家好，今天我们来学习……").fill(
            "大家好，今天我们继续学习定语从句。上节课我们讲了 which 和 who 的用法，那么 that 什么时候不能用呢？")
        pg.mouse.click(5, 5)
        pg.get_by_role("button", name="生成", exact=True).click()
        t = wait_text(pg, "✅ 生成好了", 3600)
        text = body(pg)
        ok("③ 生成好了", t is not None, f"{t} 秒" if t is not None else text[-1500:])
        names = re.findall(r"[^\s/\\]+_v5dev\.wav", text)
        ok("生成的文件名最后是 _v5dev", bool(names), names[:2] or text[-600:])
        shot(pg, "5_生成好了")
    except Exception as exc:  # noqa: BLE001
        ok("网页操作没有出错", False, f"{type(exc).__name__}: {exc}")
        try:
            shot(pg, "出错时")
        except Exception:  # noqa: BLE001
            pass
    ok("网页脚本没有报错（pageerror）", not errs, errs[:5])
    if console:
        res.append("INFO 浏览器控制台的报错（只记下来）：" + " | ".join(console[:10])[:1500])
    b.close()

passed = sum(r.startswith("PASS") for r in res)
failed = sum(r.startswith("FAIL") for r in res)
(out / "结果.txt").write_text("\n".join(res) + f"\n{passed} 通过，{failed} 没通过\n", encoding="utf-8")
print(f"{passed} 通过，{failed} 没通过")
sys.exit(1 if failed else 0)
