"""编辑冲突核查：分类/检测单元测试 + API 端到端测试（含边界与并发快照）。"""
import threading

import pytest
from fastapi.testclient import TestClient

from app.conflicts import (
    DELETION_OVERLAP,
    INSERTION_BREAKPOINT_DELETED,
    INSERTION_SAME_BREAKPOINT,
    SNV_IN_DELETION,
    SNV_SAME_POSITION,
    classify,
    find_conflicts,
)

from test_api import SAMPLES, create_dataset, publish, upload


def V(pos, ref, alt, lines=()):
    return {"pos": pos, "ref": ref, "alt": alt, "lines": list(lines)}


def pair_keys(conflicts):
    """忽略行号的冲突对集合，便于断言。"""
    return {
        (
            c["type"],
            (c["a"]["pos"], c["a"]["ref"], c["a"]["alt"]),
            (c["b"]["pos"], c["b"]["ref"], c["b"]["alt"]),
        )
        for c in conflicts
    }


# ---------- classify：归一化位点 -> 实际编辑 ----------


def test_classify_snv():
    e = classify(17, "A", "G", [8])
    assert e.kind == "snv" and e.edit_pos == 17


def test_classify_left_anchored_insertion():
    e = classify(20, "G", "GA", [])
    assert e.kind == "ins" and e.breakpoint == 20  # 断点在锚点 20 之后


def test_classify_right_anchored_insertion_at_chromosome_start():
    """染色体最左端的右锚点插入（ALT 以锚点结尾）：断点在 POS 之前。"""
    e = classify(1, "G", "AG", [])
    assert e.kind == "ins" and e.breakpoint == 0


def test_classify_left_anchored_deletion():
    e = classify(20, "GAC", "G", [])
    assert e.kind == "del" and (e.del_start, e.del_end) == (21, 22)  # 锚点 20 保留


def test_classify_right_anchored_deletion_at_chromosome_start():
    """染色体最左端的右锚点缺失：锚点在末尾，删除前面的碱基。"""
    e = classify(1, "GC", "C", [])
    assert e.kind == "del" and (e.del_start, e.del_end) == (1, 1)  # 锚点是第 2 位 C


# ---------- find_conflicts：五类规则与边界 ----------


def test_same_position_different_snvs_all_pairs():
    confs = find_conflicts([V(17, "A", "G"), V(17, "A", "T"), V(17, "A", "C")])
    assert pair_keys(confs) == {
        (SNV_SAME_POSITION, (17, "A", "C"), (17, "A", "G")),
        (SNV_SAME_POSITION, (17, "A", "C"), (17, "A", "T")),
        (SNV_SAME_POSITION, (17, "A", "G"), (17, "A", "T")),
    }


def test_deletion_overlap_and_adjacent_boundary():
    d1 = V(20, "GAC", "G")   # 删 21-22
    d2 = V(21, "ACG", "A")   # 删 22-23（与 d1 重叠 22）
    d3 = V(22, "CGT", "C")   # 删 23-24（与 d2 重叠 23，与 d1 不重叠）
    d4 = V(24, "TT", "T")    # 删 25（与 d3 相邻不重叠）
    confs = find_conflicts([d1, d2, d3, d4])
    assert pair_keys(confs) == {
        (DELETION_OVERLAP, (20, "GAC", "G"), (21, "ACG", "A")),
        (DELETION_OVERLAP, (21, "ACG", "A"), (22, "CGT", "C")),
        # d1-d3 不传递；d3-d4 相邻不算；d1/d2 与 d4 不重叠
    }


def test_nested_deletion_overlaps():
    confs = find_conflicts([V(20, "GACG", "G"), V(21, "AC", "A")])  # 21-23 与 22
    assert pair_keys(confs) == {(DELETION_OVERLAP, (20, "GACG", "G"), (21, "AC", "A"))}


def test_snv_in_deletion_but_not_on_anchor():
    d = V(20, "GAC", "G")  # 锚点 20，删 21-22
    confs = find_conflicts([d, V(20, "G", "A"), V(21, "A", "T"), V(22, "C", "T"), V(23, "G", "A")])
    # 20 是锚点、23 在范围外：不冲突；21、22 落入删除范围：冲突
    assert pair_keys(confs) == {
        (SNV_IN_DELETION, (21, "A", "T"), (20, "GAC", "G")),
        (SNV_IN_DELETION, (22, "C", "T"), (20, "GAC", "G")),
    }


def test_same_breakpoint_insertions_and_anchor_side_boundary():
    confs = find_conflicts([
        V(30, "T", "TA"),   # 断点 30
        V(30, "T", "TC"),   # 断点 30，不同序列 -> 冲突
        V(1, "G", "AG"),    # 右锚点，断点 0
        V(1, "G", "GA"),    # 左锚点，断点 1（与断点 0 不同 -> 不冲突）
    ])
    assert pair_keys(confs) == {(INSERTION_SAME_BREAKPOINT, (30, "T", "TA"), (30, "T", "TC"))}


def test_insertion_breakpoint_both_flanks_deleted_by_two_deletions():
    ins = V(35, "A", "AG")        # 断点 35，侧翼 35/36
    dl = V(34, "TA", "T")         # 删 35
    dr = V(35, "AC", "A")         # 删 36
    confs = find_conflicts([ins, dl, dr])
    assert pair_keys(confs) == {
        (INSERTION_BREAKPOINT_DELETED, (35, "A", "AG"), (34, "TA", "T")),
        (INSERTION_BREAKPOINT_DELETED, (35, "A", "AG"), (35, "AC", "A")),
    }


def test_insertion_breakpoint_both_flanks_deleted_by_single_deletion():
    ins = V(10, "A", "AG")        # 断点 10，侧翼 10/11
    d = V(9, "TAT", "T")          # 删 10-11（同一缺失覆盖两侧）
    confs = find_conflicts([ins, d])
    assert pair_keys(confs) == {(INSERTION_BREAKPOINT_DELETED, (10, "A", "AG"), (9, "TAT", "T"))}


def test_insertion_one_flank_deleted_is_not_conflict():
    ins = V(35, "A", "AG")        # 断点 35，侧翼 35/36
    confs = find_conflicts([ins, V(34, "TA", "T")])  # 仅 35 被删
    assert confs == []


def test_insertion_breakpoint_zero_never_conflicts():
    """染色体起点前的断点（右锚点插入）没有左侧参考碱基，两侧均删不成立。"""
    confs = find_conflicts([V(1, "G", "AG"), V(1, "GC", "C")])  # 断点 0；缺失删 1
    assert confs == []


def test_shared_anchor_only_is_not_conflict():
    """缺失锚点被另一缺失的删除区覆盖、但删除区本身不重叠：不算冲突。"""
    d1 = V(20, "GAC", "G")  # 删 21-22
    d2 = V(22, "CGT", "C")  # 锚点 22 落在 d1 删除区内，但自身删 23-24
    assert find_conflicts([d1, d2]) == []


def test_no_transitive_conflict():
    """甲-乙、乙-丙冲突不推出甲-丙冲突（仅报告实际对）。"""
    confs = find_conflicts([V(20, "GAC", "G"), V(21, "ACG", "A"), V(22, "CGT", "C")])
    keys = pair_keys(confs)
    assert len(keys) == 2
    assert (DELETION_OVERLAP, (20, "GAC", "G"), (22, "CGT", "C")) not in keys


def test_empty_and_single_variant():
    assert find_conflicts([]) == []
    assert find_conflicts([V(17, "A", "G")]) == []


def test_lines_carried_into_pairs():
    confs = find_conflicts([V(17, "A", "G", [5, 6]), V(17, "A", "T", [7])])
    assert confs[0]["a"]["lines"] == [5, 6]
    assert confs[0]["b"]["lines"] == [7]


# ---------- API 端到端 ----------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def get_conflicts(client, dataset_id):
    return client.get(f"/datasets/{dataset_id}/conflicts")


def test_conflicts_require_published_dataset(client):
    assert get_conflicts(client, 999).status_code == 404  # 不存在
    did = create_dataset(client)
    r = get_conflicts(client, did)
    assert r.status_code == 409 and "尚未发布" in r.json()["detail"]
    # 已上传未发布（暂存任务）同样拒绝：核查只读已发布数据
    job = upload(client, did, "conflicts.vcf").json()
    assert job["status"] == "ready"
    assert get_conflicts(client, did).status_code == 409


def test_conflicts_empty_published_dataset(client):
    """已发布但无位点（VCF 只有头部）：可以核查，结果为空。"""
    did = create_dataset(client)
    r = client.post(
        f"/datasets/{did}/uploads",
        files={
            "fasta": ("chr1.fa", (SAMPLES / "chr1.fa").read_bytes(), "text/plain"),
            "vcf": ("empty.vcf", b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n", "text/plain"),
        },
    )
    assert r.status_code == 201 and r.json()["status"] == "ready"
    assert publish(client, did, r.json()["id"], 0).status_code == 200
    body = get_conflicts(client, did).json()
    assert body["version"] == 1 and body["variant_count"] == 0
    assert body["conflict_count"] == 0 and body["conflicts"] == []


def test_conflicts_full_boundary_sample(client):
    """边界样例：17 行归并 15 个位点，恰好 9 对冲突，含全部五类。"""
    did = create_dataset(client)
    job = upload(client, did, "conflicts.vcf").json()
    assert job["status"] == "ready" and job["total_records"] == 17
    assert len(job["variants"]) == 15
    assert publish(client, did, job["id"], 0).status_code == 200

    r = get_conflicts(client, did)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 1 and body["chrom"] == "chr1"
    assert body["variant_count"] == 15 and body["conflict_count"] == 9

    assert pair_keys(body["conflicts"]) == {
        # 同位点不同 SNV（A>G 由第 5、6 行归并）
        (SNV_SAME_POSITION, (17, "A", "G"), (17, "A", "T")),
        # 删除碱基重叠；del1-del3 不传递
        (DELETION_OVERLAP, (20, "GAC", "G"), (21, "ACG", "A")),
        (DELETION_OVERLAP, (21, "ACG", "A"), (22, "CGT", "C")),
        # SNV 落入删除范围；锚点上的 SNV(20 G>A) 不冲突
        (SNV_IN_DELETION, (22, "C", "T"), (20, "GAC", "G")),
        (SNV_IN_DELETION, (22, "C", "T"), (21, "ACG", "A")),
        (SNV_IN_DELETION, (24, "T", "A"), (22, "CGT", "C")),
        # 同断点不同插入（T>TA 由第 15、16 行归并）
        (INSERTION_SAME_BREAKPOINT, (30, "T", "TA"), (30, "T", "TC")),
        # 插入断点两侧均被删除（一侧被删的 20 G>GA、36 C>CG 不冲突）
        (INSERTION_BREAKPOINT_DELETED, (35, "A", "AG"), (34, "TA", "T")),
        (INSERTION_BREAKPOINT_DELETED, (35, "A", "AG"), (35, "AC", "A")),
    }

    # 双方原始 VCF 行号随冲突对返回（含归并行号）
    by_type = {}
    for c in body["conflicts"]:
        by_type.setdefault(c["type"], []).append(c)
    snv_pair = by_type[SNV_SAME_POSITION][0]
    assert snv_pair["a"]["lines"] == [5, 6] and snv_pair["b"]["lines"] == [7]
    ins_pair = by_type[INSERTION_SAME_BREAKPOINT][0]
    assert ins_pair["a"]["lines"] == [15, 16] and ins_pair["b"]["lines"] == [17]


def test_conflicts_follow_latest_published_version(client):
    """核查结果反映最新已发布版本；暂存中的新任务不影响核查。"""
    did = create_dataset(client)
    j1 = upload(client, did, "conflicts.vcf").json()
    assert publish(client, did, j1["id"], 0).status_code == 200
    assert get_conflicts(client, did).json()["conflict_count"] == 9

    j2 = upload(client, did, "batch1.vcf").json()  # 暂存，未发布
    body = get_conflicts(client, did).json()
    assert body["version"] == 1 and body["conflict_count"] == 9  # 仍是 v1 快照

    assert publish(client, did, j2["id"], 1).status_code == 200
    body = get_conflicts(client, did).json()
    assert body["version"] == 2 and body["variant_count"] == 5
    assert body["conflict_count"] == 0 and body["conflicts"] == []


def test_conflicts_snapshot_consistent_during_concurrent_publish(client):
    """并发发布期间：每次核查的版本号与全部位点对来自同一已提交快照。"""
    did = create_dataset(client)
    j1 = upload(client, did, "batch1.vcf").json()["id"]
    assert publish(client, did, j1, 0).status_code == 200  # v1：5 位点 0 冲突
    j2 = upload(client, did, "conflicts.vcf").json()["id"]  # 发布后 v2：15 位点 9 冲突

    barrier = threading.Barrier(2)
    publish_status = []

    def pub():
        barrier.wait(timeout=10)
        publish_status.append(publish(client, did, j2, 1).status_code)

    t = threading.Thread(target=pub)
    t.start()
    barrier.wait(timeout=10)
    seen = set()
    for _ in range(50):
        r = get_conflicts(client, did)
        assert r.status_code == 200
        body = r.json()
        seen.add((body["version"], body["variant_count"], body["conflict_count"]))
    t.join(timeout=30)

    assert publish_status == [200]
    # 只允许出现两个一致快照状态之一，绝不出现版本与位点对的混合
    assert seen <= {(1, 5, 0), (2, 15, 9)}
    assert get_conflicts(client, did).json()["version"] == 2
