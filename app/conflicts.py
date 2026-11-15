"""已发布位点的编辑冲突核查。

在归一化后的已发布位点（SNV / 纯插入 / 纯缺失）上，找出无法直接同时套用到
同一条参考编辑路径的位点对。冲突只定义在**实际编辑的参考碱基**上：

- SNV：替换 POS 处 1 个参考碱基；
- 缺失：删除锚点之外的 len(REF)-1 个参考碱基（锚点本身保留，不算被删）；
- 插入：在断点（锚点碱基之后）插入序列，本身不改动参考碱基，
  但断点两侧参考碱基都被删除时，插入位置无法定位。

五类冲突（且仅有这五类）：

1. ``snv_same_position``            同位点不同 SNV；
2. ``deletion_overlap``             两处缺失的实际删除碱基重叠；
3. ``snv_in_deletion``              SNV 落入缺失的实际删除范围；
4. ``insertion_same_breakpoint``    同一断点的不同插入；
5. ``insertion_breakpoint_deleted`` 插入断点两侧参考碱基均被删除。

只报告实际冲突对：冲突不可传递（甲-乙、乙-丙各自冲突不推出甲-丙冲突），
仅共用 VCF 锚点碱基（如 SNV 落在缺失的锚点上、缺失共用锚点）不算冲突。
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

# 冲突类型（API 返回值）
SNV_SAME_POSITION = "snv_same_position"
DELETION_OVERLAP = "deletion_overlap"
SNV_IN_DELETION = "snv_in_deletion"
INSERTION_SAME_BREAKPOINT = "insertion_same_breakpoint"
INSERTION_BREAKPOINT_DELETED = "insertion_breakpoint_deleted"


@dataclass(frozen=True)
class Edit:
    """一个已发布位点对应的实际参考编辑。

    属性按类别取用：
    - SNV：``edit_pos`` 为被替换的 1-based 参考位置；
    - 插入：``breakpoint`` 为断点左侧参考位置，插入发生在 breakpoint 与
      breakpoint+1 之间（右锚点边界情形下 breakpoint = pos - 1）；
    - 缺失：``del_start``/``del_end`` 为实际被删除的 1-based 参考区间
      （闭区间，不含锚点碱基）。
    """

    kind: str  # 'snv' | 'ins' | 'del'
    pos: int
    ref: str
    alt: str
    lines: tuple[int, ...]
    edit_pos: int = 0
    breakpoint: int = 0
    del_start: int = 0
    del_end: int = 0

    def locus(self) -> dict:
        """API 输出用的位点表示（含归并的原始 VCF 行号）。"""
        return {"pos": self.pos, "ref": self.ref, "alt": self.alt, "lines": list(self.lines)}


def classify(pos: int, ref: str, alt: str, lines: Iterable[int]) -> Edit:
    """把归一化位点分类为 SNV / 插入 / 缺失，并算出实际编辑的参考坐标。

    归一化保证：SNV 为单碱基 REF/ALT；纯插入 len(REF)==1；纯缺失 len(ALT)==1。
    锚点在左是常态；染色体最左端的变异为右锚点（ALT/REF 以锚点碱基结尾）。
    """
    lines_t = tuple(lines)
    if len(ref) == 1 and len(alt) == 1:
        return Edit("snv", pos, ref, alt, lines_t, edit_pos=pos)
    if len(alt) > len(ref):
        # 纯插入：左锚点时断点在 pos 之后；右锚点（染色体左端）时断点在 pos 之前
        breakpoint = pos if alt.startswith(ref) else pos - 1
        return Edit("ins", pos, ref, alt, lines_t, breakpoint=breakpoint)
    # 纯缺失：实际删除区间为锚点之外的 len(ref)-1 个碱基
    if ref.startswith(alt):  # 左锚点：锚点在 pos，删除 pos+1 .. pos+len(ref)-1
        start, end = pos + 1, pos + len(ref) - 1
    else:  # 右锚点（染色体左端）：锚点在末尾，删除 pos .. pos+len(ref)-2
        start, end = pos, pos + len(ref) - 2
    return Edit("del", pos, ref, alt, lines_t, del_start=start, del_end=end)


def _pair(type_: str, a: Edit, b: Edit) -> dict:
    return {"type": type_, "a": a.locus(), "b": b.locus()}


def _sort_key(pair: dict) -> tuple:
    a, b = pair["a"], pair["b"]
    return (a["pos"], b["pos"], pair["type"], a["ref"], a["alt"], b["ref"], b["alt"])


def find_conflicts(variants: list[dict]) -> list[dict]:
    """在一组已发布归一化位点中找出全部冲突位点对。

    参数:
        variants: [{"pos", "ref", "alt", "lines"}, ...]（已发布快照内容）。

    返回:
        [{"type", "a", "b"}, ...]，按 (a.pos, b.pos, type) 排序，确定可复现。
        每对只按上述五类规则判定，不做传递闭包。
    """
    edits = [classify(v["pos"], v["ref"], v["alt"], v["lines"]) for v in variants]
    snvs = [e for e in edits if e.kind == "snv"]
    inss = [e for e in edits if e.kind == "ins"]
    dels = sorted(
        (e for e in edits if e.kind == "del"),
        key=lambda e: (e.del_start, e.del_end, e.pos),
    )
    pairs: list[dict] = []

    # 1) 同位点不同 SNV（相同 ALT 的已在归一化时归并，组内必然 ALT 不同）
    by_pos: dict[int, list[Edit]] = {}
    for s in snvs:
        by_pos.setdefault(s.edit_pos, []).append(s)
    for group in by_pos.values():
        for x, y in combinations(sorted(group, key=lambda e: e.alt), 2):
            pairs.append(_pair(SNV_SAME_POSITION, x, y))

    # 2) 实际删除碱基重叠（相邻不重叠、仅共用锚点均不算）
    for i, d1 in enumerate(dels):
        for d2 in dels[i + 1 :]:
            if d2.del_start > d1.del_end:  # 按 del_start 排序，之后的更不会重叠
                break
            pairs.append(_pair(DELETION_OVERLAP, d1, d2))

    # 点查询：覆盖某 1-based 参考位置的全部缺失（实际删除区间，不含锚点）
    starts = [d.del_start for d in dels]

    def covering(p: int) -> list[Edit]:
        return [d for d in dels[: bisect_right(starts, p)] if d.del_end >= p]

    # 3) SNV 落入缺失的实际删除范围（落在锚点上不算）
    for s in snvs:
        for d in covering(s.edit_pos):
            pairs.append(_pair(SNV_IN_DELETION, s, d))

    # 4) 同一断点的不同插入（相同插入序列的已在归一化时归并）
    by_breakpoint: dict[int, list[Edit]] = {}
    for ins in inss:
        by_breakpoint.setdefault(ins.breakpoint, []).append(ins)
    for group in by_breakpoint.values():
        for x, y in combinations(sorted(group, key=lambda e: e.alt), 2):
            pairs.append(_pair(INSERTION_SAME_BREAKPOINT, x, y))

    # 5) 插入断点两侧参考碱基均被删除（一侧被删可与插入组合，不冲突）
    for ins in inss:
        left = covering(ins.breakpoint)
        right = covering(ins.breakpoint + 1)
        if left and right:
            for d in dict.fromkeys(left + right):  # 覆盖两侧的同一缺失只记一次
                pairs.append(_pair(INSERTION_BREAKPOINT_DELETED, ins, d))

    pairs.sort(key=_sort_key)
    return pairs
