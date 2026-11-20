            row_types = [v.strip() for v in row["Violations"].split(";") if v.strip()]
            if chosen and not any(v in chosen for v in row_types):
                continue
            if search and search.lower() not in row["Vehicle ID"].lower() and search.lower() not in row["Number Plate"].lower():
                continue
            cards.append((row["Vehicle ID"], row))

        if not cards:
            st.info("No vehicles match the current filters.")

        cols = st.columns(2)
        for i, (vid, item) in enumerate(cards):
            unread = str(item["Number Plate"]).startswith("Unreadable")
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**Vehicle {vid}**")
                    crop = best_plate_image(run, vid)
                    if crop:
                        st.image(str(crop), use_container_width=True)
                    st.markdown(
                        f'<span class="plate {"unread" if unread else ""}">'
                        f'{item["Number Plate"]}</span>',
                        unsafe_allow_html=True
                    )
                    ocr_confidence = item["OCR Confidence"]
                    ocr_confidence_text = f"{ocr_confidence:.0%}" if ocr_confidence is not None else "—"
                    st.caption(f'OCR {ocr_confidence_text} · {item["OCR Status"]}')
                    st.markdown(
                        "".join(f'<span class="badge">{v.strip()}</span>'
                                for v in item["Violations"].split(";") if v.strip()),
                        unsafe_allow_html=True
                    )
                    st.caption(
                        f'Frames {item["First Frame"]}–{item["Last Frame"]}'
                    )

    folder = Path(run["folder"])
    ev_files = sorted((folder / "evidence").glob("*.jpg"))
    pl_files = sorted((folder / "plate_crops").glob("*.jpg"))

    if ev_files:
        with st.expander(f"Violation evidence images ({len(ev_files)})"):
            cols = st.columns(3)
            for i, p in enumerate(ev_files):
                with cols[i % 3]:
                    st.image(str(p), caption=p.name, use_container_width=True)

    if pl_files:
        with st.expander(f"Number-plate crops ({len(pl_files)})"):
            cols = st.columns(4)
            for i, p in enumerate(pl_files):
                with cols[i % 4]:
                    st.image(str(p), caption=p.name, use_container_width=True)

    with st.expander("OCR readings"):
        if run["ocr_results"]:
            st.dataframe(pd.DataFrame(run["ocr_results"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No OCR predictions available.")

    with st.expander("Evidence details"):
        if run.get("evidence_rows"):
            st.dataframe(pd.DataFrame(run["evidence_rows"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No violation evidence records were created.")

    with st.expander("All detections"):
        st.dataframe(pd.DataFrame(run["records"]),
                     use_container_width=True, hide_index=True)


# ==========================================================
# EXPORT
# ==========================================================

st.write("")

if run:
    report_path = Path(run["report"])
    with open(report_path, "rb") as f:
        report_bytes = f.read()

    is_xlsx = report_path.suffix == ".xlsx"
    st.download_button(
        "↓  EXPORT EXCEL REPORT" if is_xlsx else "↓  EXPORT CSV REPORT",
        data=report_bytes,
        file_name=report_path.name,
        mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
              if is_xlsx else "text/csv"),
        use_container_width=True
    )
else:
    st.download_button(
        "↓  EXPORT EXCEL REPORT",
        data=b"",
        file_name="yolocop_report.xlsx",
        use_container_width=True,
        disabled=True
    )


# ==========================================================
# ROAD SAFETY MESSAGE  (centered, end of page)
# ==========================================================

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="pledge-stage">
  <div class="pledge">
    <div class="pledge-line">Aapke aur sabki suraksha ke liye</div>
    <div class="pledge-line">Traffic niyamon ka kripya paalan kijiye</div>
    <div class="pledge-because">KYONKI</div>
    <div class="pledge-big">"Road kisi ke baap ka nahi hai"</div>
    <div class="pledge-by">— Akshay Kumar aka Khiladi</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
