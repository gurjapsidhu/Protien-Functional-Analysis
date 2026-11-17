                    "Model Score (%)": float(score) * 100,
                })

            chart_data = pd.DataFrame(chart_rows)

            if not chart_data.empty:
                st.bar_chart(
                    chart_data.set_index("GO Term"),
                    color="#00D9C8",
                )

            # Detailed result cards
            st.markdown("### Ranked Molecular Functions")

            report_rows = []

            for rank, (go_id, score) in enumerate(
                top_predictions,
                start=1,
            ):

                score = float(score)
                go_info = get_go_info(go_id)

                name = go_info["name"]
                namespace = go_info["namespace"]
                definition = go_info["definition"]

                report_rows.append({
                    "Rank": rank,
                    "GO ID": go_id,
                    "Function": name,
                    "Namespace": namespace,
                    "Model Score (%)": round(score * 100, 2),
                    "Definition": definition,
                })

                with st.container(border=True):

                    card_left, card_right = st.columns([4, 1])

                    with card_left:
                        st.markdown(f"### {rank}. {name}")
                        st.caption(f"{go_id} · {namespace}")

                    with card_right:
                        st.metric(
                            "Model Score",
                            f"{score * 100:.1f}%",
                        )

                    st.progress(max(0.0, min(1.0, score)))

                    st.write(definition)

                    st.markdown(
                        f"[View this Gene Ontology term]"
                        f"(https://amigo.geneontology.org/amigo/term/{go_id})"
                    )

            # Export results
            st.divider()
            st.markdown("### Export Analysis")

            report_df = pd.DataFrame(report_rows)

            download_col, fasta_col = st.columns(2)

            with download_col:
                st.download_button(
                    "DOWNLOAD PREDICTION REPORT",
                    data=report_df.to_csv(index=False).encode("utf-8"),
                    file_name="protein_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with fasta_col:
                fasta_data = f">Protein_Sequence\n{sequence}\n"

                st.download_button(
                    "DOWNLOAD FASTA",
                    data=fasta_data,
                    file_name="protein_sequence.fasta",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption(
                "Check that the model's expected feature columns match "
                "the amino_acids list stored in protein_model.joblib."
            )


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown(
    """
    <div style="
        margin-top: 55px;
        padding-top: 18px;
        border-top: 1px solid rgba(100,160,190,.18);
        text-align: center;
        color: #71869A;
        font-size: 11px;
        letter-spacing: .5px;">
        PROTEIN FUNCTION ANALYSIS · MACHINE LEARNING · GENE ONTOLOGY
    </div>
    """,
    unsafe_allow_html=True,
)
