import pathlib
import re
import sys


sys.stdout.reconfigure(encoding="utf-8")

RAW_PATH = pathlib.Path(r"D:\vqa\references_ocr_raw.txt")
OUT_PATH = pathlib.Path(r"D:\vqa\参考文献_格式化.txt")

MANUAL_REFS = {
    1: "国务院. 教育强国建设规划纲要（2024-2035年）[EB/OL]. 2025.",
    2: "教育部等九部门关于加快推进教育数字化的意见[J]. 中国教育信息化, 2025, 31(04): 3-8.",
    3: "袁振国. 教育数字化转型：转什么，怎么转[J]. 华东师范大学学报（教育科学版）, 2023, 41(03): 1-11.",
    4: "祝智庭, 胡姣. 教育数字化转型的实践逻辑与发展机遇[J]. 电化教育研究, 2022, 43(01): 5-15.",
    5: "Flanders N A. Teacher influence, pupil attitudes, and achievement: Ned a. flanders: number 12[M]. US Department of Health, Education, and Welfare, Office of Education, 1965.",
    6: "崔允漷. 论课堂观察LICC范式：一种专业的听评课[J]. 教育研究, 2012, 33(05): 79-83.",
    7: "刘清堂, 何皓怡, 吴林静, 等. 基于人工智能的课堂教学行为分析方法及其应用[J]. 中国电化教育, 2019(09): 13-21.",
    8: "杨晓哲. 基于人工智能的课堂分析架构：一种智能的课堂教学研究[J]. 全球教育展望, 2021, 50(12): 55-65.",
    9: "Jia L, Sun H, Jiang J, et al. Enhanced speaker-turn aware hierarchical model for automated classroom dialogue act classification[J]. Expert Systems with Applications, 2025: 129047.",
    10: "Attia A, Demszky D, Liu J, et al. From weak labels to strong results: Utilizing 5,000 hours of noisy classroom transcripts with minimal accurate data[C]. Interspeech 2025. ISCA, 2025: 3678-3682.",
    11: "Shapiro B R, Metts E C, Zhao E. The interaction geography slicer: Designing exploratory spatial data visualization tools for teachers' reflective practice[C]. CHI '25: Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. New York, NY, USA: Association for Computing Machinery, 2025.",
    12: "杨晓哲, 王晴晴, 王若昕. 生成式人工智能的有限能力与教育变革[J]. 全球教育展望, 2023, 52(06): 3-12.",
    13: "杨晓哲, 王晴晴. 以ChatGPT为代表的人工智能对教育的影响[J]. 中国信息技术教育, 2023(08): 4-8.",
    14: "杨宗凯, 王俊, 吴砥, 等. ChatGPT/生成式人工智能对教育的影响探析及应对策略[J]. 华东师范大学学报（教育科学版）, 2023, 41(07): 26-35.",
    15: "卢宇, 余京蕾, 陈鹏鹤, 等. 生成式人工智能的教育应用与展望：以ChatGPT系统为例[J]. 中国远程教育, 2023, 43(04): 24-31+51.",
    26: "杨健, 孙淼, 张丽芬. 低资源语言自动语音识别中的数据处理与数据增强综述[J]. 计算机科学, 2025, 52(08): 86-99.",
    34: "Zhu S, Liu X, Li Y, et al. Improving grammatical error correction with dynamic linguistic knowledge fusion[C]. Natural Language Processing and Chinese Computing: 14th National CCF Conference, NLPCC 2025, Urumqi, China, August 7-9, 2025, Proceedings, Part II. Berlin, Heidelberg: Springer-Verlag, 2025: 222-235.",
    39: "Liu X, Li Y, Yu Z. Chinese grammatical error correction via large language model guided optimization training[C]. Sun M, Liang J, Han X, et al. Chinese Computational Linguistics. Singapore: Springer Nature Singapore, 2025: 522-539.",
    47: "Li S, Chen C, Kwok C, et al. Investigating asr error correction with large language model and multilingual 1-best hypotheses[C]. 2024: 1315-1319.",
    48: "方海光, 高辰柱, 陈佳. 改进型弗兰德斯互动分析系统及其应用[J]. 中国电化教育, 2012(10): 109-113.",
    50: "杨晓哲, 刘昕, 王若昕. 中小学课堂教学时间分配与互动模式研究：基于1008节课堂视频的智能分析[J]. 教育发展研究, 2024, 43(Z2): 59-66.",
    73: "Uchiyama S, Umemura K, Morita Y. Large language model-based system to provide immediate feedback to students in flipped classroom preparation learning[J]. arXiv preprint arXiv:2307.11388, 2023.",
    85: "Yu Z, Xie M, Gao J, et al. From raw video to pedagogical insights: A unified framework for student behavior analysis[C]. Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, February 20-27, 2024, Vancouver, Canada. 2024: 23241-23249.",
    126: "Schlotterbeck D, Jimenez A, Araya R, et al. \"teacher, can you say it again?\" improving automatic speech recognition performance over classroom environments with limited data[C]. Rodrigo M M, Matsuda N, Cristea A I, et al. Artificial Intelligence in Education. Cham: Springer International Publishing, 2022: 269-280.",
}


def is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith("--- page"):
        return True
    if s == "华东师范大学博士学位论文" or s == "参考文献":
        return True
    if re.fullmatch(r"\d{3}", s):
        return True
    return False


def normalize(text: str) -> str:
    replacements = {
        "鈥?": "'",
        "鈫?": "->",
        "–": "-",
        "—": "-",
        "－": "-",
        "−": "-",
        "．": ".",
        "，": ", ",
        "：": ": ",
        "；": "; ",
        "（": "(",
        "）": ")",
        "、": ", ",
        "[川]": "[J]",
        "[叮]": "[J]",
        "[刀]": "[J]",
        "[]]": "[J]",
        "[刁": "[J",
        "0oo": "000",
        "Ass0ciates": "Associates",
        "fo1": "for",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"([A-Za-z])- +([A-Za-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.:;?\)])", r"\1", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"\[ +", "[", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_refs() -> dict[int, str]:
    refs: dict[int, list[str]] = {}
    current = None

    for raw_line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if is_noise(raw_line):
            continue
        line = raw_line.strip()
        match = re.match(r"^\[(\d{1,3})\]\s*(.*)$", line)
        if match:
            current = int(match.group(1))
            refs.setdefault(current, [])
            rest = match.group(2).strip()
            if rest:
                refs[current].append(rest)
            continue
        if current is not None:
            refs[current].append(line)

    parsed = {num: normalize(" ".join(parts)) for num, parts in refs.items()}
    parsed.update(MANUAL_REFS)
    return parsed


def main() -> None:
    refs = parse_refs()
    missing = [num for num in range(1, 166) if num not in refs or not refs[num]]
    if missing:
        raise SystemExit(f"Missing references: {missing}")

    lines = [f"[{num}] {normalize(refs[num])}" for num in range(1, 166)]
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"references: {len(lines)}")


if __name__ == "__main__":
    main()
