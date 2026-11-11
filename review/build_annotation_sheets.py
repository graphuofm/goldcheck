#!/usr/bin/env python3
"""Build two identical annotation sheets (annotation_rater1.xlsx, annotation_rater2.xlsx) for two
independent annotators. Items: the 32 count fan-outs and 5 NULL-top findings of GoldCheck on the
2025-11-06 BIRD Dev release, i.e. the findings whose status as an error depends on what the question
intends. Ties and empty answers are defects of the test whatever the intent and are not included.
No AI judgment is shown. Read back with score_annotation.py."""
import json, os, re
import sqlglot
from sqlglot import exp as E
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "exp")
res = json.load(open(os.path.join(EXP, "results", "goldcheck.json")))
dev = {e["question_id"]: e for e in json.load(open(os.path.join(EXP, "data_ext", "bird_dev_20251106", "dev.json")))}
CHOICES = ["标准答案对", "标准答案错", "两种都说得通", "看不懂或都不对"]


def val(rows):
    if not rows:
        return "（空）"
    v = rows[0][0] if len(rows[0]) == 1 else ", ".join(str(x) for x in rows[0])
    if isinstance(v, float):
        v = f"{v:,.2f}"
    elif isinstance(v, int):
        v = f"{v:,}"
    return str(v)


def tables(sql):
    return sorted({t.name for t in sqlglot.parse_one(sql, read="sqlite").find_all(E.Table)}, key=str.lower)


items = []
for r in sorted((r for r in res["rows"] if r["ver"] == "dev_2025"), key=lambda r: r["qid"]):
    e = dev[r["qid"]]
    if "count_fanout" in r["flags"]:
        f = r["fanout"]; t = f["column"].split(".")[0]
        others = [x for x in tables(e["SQL"]) if x.lower() != t.lower()]
        ratio = bool(re.search(r"percent|ratio|average|rate", e["question"], re.I))
        how = (f"标准答案的数法：把 {t} 表和 {'、'.join(others)} 表连起来以后数行数。一个 {t} 如果对应好几条 "
               f"{'/'.join(others)} 记录，就会被数好几次。结果 = {val(f['gold'])}。\n"
               f"另一种数法：每个 {t} 只数一次（不重复）。结果 = {val(f['with_distinct'])}。")
        if ratio:
            how += ("\n注意：这是比例或平均值的题。另一种数法只改了算式的一部分，算出来的数可能不合理（例如超过 100%），"
                    "这个数可以不看。只判断题目说的“多少个/百分之几”指的是不重复的 " + t + "，还是连表后的记录。")
        q = f"题目要的是“连表后的记录数”（标准答案对），还是“不重复的 {t} 个数”（标准答案错）？"
        items.append(("连表后重复计数", r["qid"], e, how, q))
    if "topk_null" in r["flags"]:
        how = ("题目问的是“最高/最低/最多”的那一个。标准答案把数据排序后取第一名，"
               "但排在第一名的那条记录，用来排序的那个值是空的（数据库里没有这个数据）。")
        if "topk_tie" in r["flags"]:
            how += f"\n而且有 {r['topk']['tie_complete_rows']} 条记录的这个值都是空的，标准答案只取了其中一条。"
        q = "排在第一名的是一条“没有这个数据”的记录。按题意，这样的答案可以接受吗？（可以＝标准答案对，不可以＝标准答案错）"
        items.append(("第一名是空值", r["qid"], e, how, q))

F = "Arial"
head = Font(name=F, bold=True, color="FFFFFF", size=10); body = Font(name=F, size=10); mono = Font(name="Courier New", size=8)
hfill = PatternFill("solid", fgColor="1F4E79"); yellow = PatternFill("solid", fgColor="FFFF99")
line = Side(style="thin", color="BFBFBF"); box = Border(left=line, right=line, top=line, bottom=line)
wrap = Alignment(wrap_text=True, vertical="top")

GUIDE = [
    ("标注说明（请先读完，大约 5 分钟）", ""),
    ("这是做什么的", f"一个检查工具在 BIRD 数据集（一个用来给“自然语言转 SQL”系统打分的考试题库）里，标出了 {len(items)} 道"
                     "可能有问题的标准答案。工具能证明“这道题有两种数法，结果不一样”，但判断不了题目到底要哪一种。"
                     "这件事只能由人看题目来判断。你们两人各自独立判断，最后我们算两人的一致程度。"),
    ("你只需要回答一个问题", "每一行：读英文题目（Question）和题目附带的提示（Evidence），再看中文写的两种数法，"
                          "然后判断：按题目的意思，应该用哪一种数法？"),
    ("四个选项", "标准答案对：题目要的就是标准答案那种数法。\n"
                "标准答案错：题目要的是另一种数法，标准答案数错了。\n"
                "两种都说得通：题目的英文确实有两种合理的理解，换一个认真的人也可能选另一种。\n"
                "看不懂或都不对：看不懂题目或说明，或者觉得两种数法都不对。请在备注里写一句原因。"),
    ("例子 1（不是表里的题）", "题目：How many students took a math course?\n"
                            "标准答案把“学生”和“选课记录”连起来数行数，一个学生选了 3 门数学课就被数 3 次。\n"
                            "题目问的是“多少个学生”，所以应该每个学生只数一次 → 选“标准答案错”。"),
    ("例子 2（不是表里的题）", "题目：How many math courses did Alice take?\n"
                            "标准答案同样把“学生”和“选课记录”连起来数行数，只看 Alice 一个人。\n"
                            "题目问的是“多少门课”，连表后的每一行正好是一门课 → 选“标准答案对”。"),
    ("判断时的规则", "1. 只按题目和提示的字面意思判断，不要猜出题人心里怎么想。提示（Evidence）算题目的一部分。\n"
                    "2. 题目说 how many patients / accounts / users，一般指“不重复的个数”；说 how many records / tests / times，"
                    "一般指记录条数。但要以整句话的意思为准。\n"
                    "3. 真的两种都合理才选“两种都说得通”。只是自己拿不准、没看懂，请选“看不懂或都不对”，并写原因。\n"
                    "4. 灰色的 SQL 那一列只是备查，看不懂可以完全不看。"),
    ("必须独立完成", "两人各做各的：做完之前不要讨论、不要看对方的表，也不要问 AI 或上网查答案。"
                   "做完以后把表交回，我先算一致程度，再把两人不一致的题列出来，你们再一起讨论定一个最终答案。"),
    ("怎么填", "只填“标注”页里的黄色格子：\n“你的判断”从下拉菜单里选；“备注”可以不填，选“两种都说得通”或“看不懂或都不对”时请写一句。\n"
              "最上面填你的名字。大约 37 道题，每题 1 分钟左右。"),
]


def build(path, rater):
    wb = Workbook()
    a = wb.active; a.title = "说明"
    for i, (k, v) in enumerate(GUIDE, 1):
        c = a.cell(i, 1, k); c.font = Font(name=F, bold=True, size=14 if i == 1 else 10); c.alignment = wrap
        d = a.cell(i, 2, v); d.font = body; d.alignment = wrap
    a.column_dimensions["A"].width = 22; a.column_dimensions["B"].width = 110
    for i in range(2, len(GUIDE) + 1):
        a.row_dimensions[i].height = 15 * (GUIDE[i - 1][1].count("\n") + 1 + len(GUIDE[i - 1][1]) // 95)

    s = wb.create_sheet("标注")
    s["A1"] = "标注人："; s["A1"].font = Font(name=F, bold=True, size=11)
    s["B1"] = rater; s["B1"].fill = yellow; s["B1"].font = Font(name=F, size=11)
    s["D1"] = "日期："; s["D1"].font = Font(name=F, bold=True, size=11); s["E1"].fill = yellow
    cols = ["序号", "题号", "类别", "数据库", "Question（题目）", "Evidence（提示）", "两种数法（中文说明）", "要你判断的", "你的判断", "备注", "标准答案 SQL（备查）"]
    widths = [5, 7, 12, 14, 38, 30, 52, 30, 14, 22, 40]
    for j, (c, w) in enumerate(zip(cols, widths), 1):
        x = s.cell(3, j, c); x.font = head; x.fill = hfill; x.alignment = wrap; x.border = box
        s.column_dimensions[x.column_letter].width = w
    dv = DataValidation(type="list", formula1='"' + ",".join(CHOICES) + '"', allow_blank=True)
    s.add_data_validation(dv)
    for i, (cls, qid, e, how, q) in enumerate(items, 1):
        r = i + 3
        vals = [i, qid, cls, e["db_id"], e["question"], e.get("evidence", ""), how, q, "", "", e["SQL"]]
        for j, v in enumerate(vals, 1):
            x = s.cell(r, j, v); x.font = mono if j == 11 else body; x.alignment = wrap; x.border = box
            if j == 11:
                x.font = Font(name="Courier New", size=8, color="808080")
        s.cell(r, 9).fill = yellow; s.cell(r, 10).fill = yellow
        dv.add(s.cell(r, 9))
        s.row_dimensions[r].height = max(75, 15 * (how.count("\n") + 2 + len(how) // 50))
    s.freeze_panes = "E4"

    p = wb.create_sheet("进度")
    p["A1"] = "已填"; p["B1"] = f'=COUNTA(标注!I4:I{len(items) + 3})'
    p["A2"] = "总数"; p["B2"] = len(items)
    for k, c in enumerate(CHOICES, 3):
        p.cell(k, 1, c); p.cell(k, 2, f'=COUNTIF(标注!I4:I{len(items) + 3},"{c}")')
    wb.active = 0
    wb.save(path)


for n, who in ((1, "Jiacheng Ding"), (2, "Cong Guo")):
    build(os.path.join(HERE, f"annotation_rater{n}.xlsx"), who)
print(len(items), "items;", sum(i[0] == "连表后重复计数" for i in items), "fan-outs,", sum(i[0] == "第一名是空值" for i in items), "NULL tops")
