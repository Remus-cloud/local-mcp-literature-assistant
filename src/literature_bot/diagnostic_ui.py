"""Experimental read-only Streamlit diagnostics, separate from core checking."""
import streamlit as st

from .verification.presentation import diagnostic_view, export_trial, ISSUE_LABELS


def render_diagnostics(message: dict, key: str) -> None:
    if not st.session_state.get("show_diagnostics", False):
        return
    if "verification" not in message:
        return
    with st.expander("实验性科学诊断 · 只读试用", expanded=False):
        st.warning("尚未通过独立人工验收。提示可能误报或漏报，不修改回答，不认证科学真伪。")
        for run in message.get("search_runs", []):
            if run.get("reranking_applied"):
                st.caption(f"本次实际使用 SPECTER2：对 {run.get('candidate_count')} 篇 arXiv 候选重排；正文仍用 BM25。")
            else:
                st.caption(f"论文搜索：{run.get('retrieval_mode', 'unknown')}；本次未报告 SPECTER2 重排。")
        try:
            view = diagnostic_view(message["content"], message["verification"])
        except (KeyError, TypeError, ValueError, AttributeError):
            st.info("诊断数据格式不兼容，未展示结论。原回答保留。")
            return
        if view["status"] != "diagnostics_recorded":
            st.info(f"本次没有可展示的完整诊断：{view['status']}。未检查不等于通过。")
        else:
            st.caption(f"已记录 {view['declared_claims']} 条声明结论；"
                       f"MiniCheck 实际评分 {view['scored_claims']} 条。其他结论不算通过。")
            if view["claims"]:
                selected = st.selectbox(
                    "选择要核对的结论", range(len(view["claims"])),
                    format_func=lambda index: view["claims"][index]["id"] + " · " + view["claims"][index]["text"][:65],
                    key=f"{key}_claim",
                )
                if type(selected) is not int or not 0 <= selected < len(view["claims"]):
                    st.info("请选择有效结论。")
                    return
                claim = view["claims"][selected]
                st.text(claim["text"])
                if claim["issues"]:
                    for issue in claim["issues"]:
                        st.markdown("- " + issue)
                else:
                    st.info("已执行的检查未发现问题；不代表该结论已经验证正确。")
                st.caption(claim["semantic"])
                for evidence in claim["evidence"]:
                    st.caption(f"{evidence['record_id']} · {evidence['arxiv_id']} · PDF 第 {evidence['page_number']} 页")
                    st.text(evidence["text"])
                if st.checkbox("查看本条检查明细", key=f"{key}_details"):
                    st.json(claim["details"])
            for event in view["events"]:
                st.info(ISSUE_LABELS.get(event["category"], "来源比较需人工复核"))
                for quote in event.get("quotes", []):
                    st.text(quote)
                st.caption("不同条件：" + ", ".join(event.get("different_conditions", []))
                           + "；缺失条件：" + ", ".join(event.get("missing_conditions", [])))
        st.caption("下载文件含问题、回答和论文片段；仅主动点击才导出，请自行妥善保管。")
        st.download_button("下载本次诊断记录", export_trial(message),
                           file_name="literature_trial.json", mime="application/json", key=f"{key}_export")
        st.download_button("下载人工盲评材料（不含诊断）", export_trial(message, blind=True),
                           file_name="literature_blind_review.json", mime="application/json", key=f"{key}_blind")
