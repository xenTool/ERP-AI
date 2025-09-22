
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

# region Imports (Projektmodule)
# Fachlogik in Core-Modulen gekapselt (I/O, Features, Regeln, Metriken, Visuals, Modelle, AutoML)
from core import io as io_core
from core import features, rules, metrics, visuals
from core import forecast_prophet, forecast_lstm, cluster_kmeans, automl
# endregion

# region Konfiguration & Verzeichnisse
# Basisverzeichnis der App und Standard-Ausgabeordner anlegen
APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# endregion

# region Streamlit Sitebar erzeugen
# Seitentitel/Viewport; Title in der Hauptfläche
st.set_page_config(page_title="ERP AI App", layout="wide")
st.title("📊 ERP AI App — Forecast & Segmentation")

# --- Sidebar: Steuerungselemente / Parameter
with st.sidebar:
    st.header("⚙️ Einstellungen")
    analysis_goal = st.selectbox("Analyseziel", ["Prognose", "Segmentierung"])

    st.markdown("### 📥 Daten-Upload")
    uploaded = st.file_uploader("CSV hochladen", type=["csv"])

    st.markdown("### 🔧 Parameter")
    automl_on = st.toggle("AutoML aktivieren (optional)", value=False)

    # Prognose-spezifische Parameter
    if analysis_goal == "Prognose":
        # Auswahl, wie die Zeitreihe resampled werden soll:
        # "D" = Daily (Tageswerte), "W" = Weekly (Wochenwerte).
        # Default ist Tagesebene (index=0), weil Verkaufsdaten oft täglich vorliegen.
        freq = st.selectbox("Resampling-Frequenz", ["D", "W"], index=0)

        # Prognosehorizont = Anzahl zukünftiger Tage, für die vorhergesagt wird.
        # Wertebereich 1–365 Tage erlaubt (1 Geschäftsjahr).
        # Default = 30 Tage, da das ein typisches Business-Szenario (1 Periode) ist.
        horizon = st.number_input("Prognosehorizont (Tage)", 1, 365, 30)

        # Fenstergröße für LSTM (Sequenzlänge der Input-Daten).
        # Wertebereich: 5–180 (mindestens eine Woche, maximal ~6 Monate).
        # Default = 30 → typisches Setup, 1 Monat zurückschauen, um Muster zu erkennen.
        lstm_window = st.number_input("LSTM Fenstergröße", 5, 180, 30)

        # Anzahl Trainings-Epochen für LSTM.
        # Wertebereich: 1–200 (sehr kurze bis mittellange Trainingsläufe).
        # Default = 30 → pragmatischer Wert, der in kleinen Datasets ausreicht
        # ohne zu lange Trainingszeiten zu verursachen.
        lstm_epochs = st.number_input("LSTM Epochen", 1, 200, 30)

        # Schalter, ob Ausreißer in der Zeitreihe per IQR-Methode "geclippt" werden sollen.
        # (IQR = Interquartilsabstand, robuste Methode, um Ausreißer zu erkennen).
        # Default = False, weil man Rohdaten zuerst sehen sollte;
        # Nutzer kann es aber aktivieren, wenn die Prognose durch Ausreißer verzerrt wird.
        iqr_on = st.toggle("Ausreißer (IQR) clippen", value=False)
    else:
        # Auto-k Auswahl für KMeans.
        # Wenn True: bestes k wird automatisch mit Silhouette ausgewählt.
        # Default = True, um Einsteiger:innen das manuelle Tuning zu ersparen.
        auto_k = st.toggle("k automatisch wählen (Silhouette)", value=True)

        # Falls auto_k ausgeschaltet ist: manuelle Eingabe von k.
        # Wertebereich: 2–12 Cluster (2 Minimum sinnvoll, 12 als typisches Oberlimit).
        # Default = 3 Cluster, praktikable Segmentanzahl im Marketing
        k = st.number_input("Anzahl Cluster (k)", 2, 12, 3, disabled=auto_k)

    st.markdown("### 💾 Export")

    # Schalter: Soll zusätzlich zur CSV-Ausgabe auch Parquet gespeichert werden?
    # Vorteil: Parquet ist binär, komprimiert und schneller für große Datenmengen.
    # Default = False, weil CSV leichter weitergegeben werden kann.
    allow_parquet = st.toggle("Parquet im App-Verzeichnis speichern", value=False)

# Kurze Info in der Hauptfläche
st.write("### Datenvorschau & Schema")

# endregion


# region Datenanalyse
if uploaded:
    try:
        # Datei einlesen und Schema erkennen
        raw_df = pd.read_csv(uploaded)
        schema_info = io_core.infer_schema(raw_df)
        st.write("Erkanntes Schema:", schema_info)

        # Ziel -> Zeitreihen-Prognose
        if analysis_goal == "Prognose":

            st.write("#### Spalten-Mapping (id, date, qty)")
            options = raw_df.columns.tolist() # einzelne Spalten der CSV als Auswahl zur Verfügung stellen

            # Vorschlag abrufen / Versuchen Spalten automatisch zuzuordnen
            sales_suggest = io_core.suggest_mapping_sales(raw_df)

            # Hilfsfunktion um Spalte dem passenden Index zuzuordnen
            def _def_idx(name: str, fallback_first: bool = True) -> int:
                col = sales_suggest.get(name)
                if col in options:
                    return options.index(col)
                return 0 if fallback_first else 0

            #Auswahlboxen erstellen und versuchen Spalte automatisch vorzubelegen
            id_col = st.selectbox("id-Spalte", options=options, index=_def_idx("id"))
            date_col = st.selectbox("date-Spalte", options=options, index=_def_idx("date"))
            qty_col = st.selectbox("qty/Absatz-Spalte", options=options, index=_def_idx("qty"))

            # Normalisierte Sales-Tabelle laden (einheitliche Spaltennamen)
            df = io_core.load_sales_df(raw_df, mapping={
                "id": id_col,
                "date": date_col,
                "qty": qty_col,
            })
            # erste 10 Zeilen anzeigen
            st.dataframe(df.head(10), use_container_width=True)

            # Auswahlbox anzeigen, um Zielobjekt (Artikel) auszuwählen
            options_ids = df["id"].astype(str).unique().tolist()
            target_id = st.selectbox("Zielobjekt (id)", options=options_ids)

            # Zeitreihe bauen, optional Ausreißer clippen, Zeitfeatures anreichern
            ts_df = features.build_time_series(
                df[df["id"] == target_id].copy(),
                id_col="id", date_col="date", y_col="qty", freq=freq
            )
            if iqr_on:
                ts_df = features.iqr_clip(ts_df, col="qty", k=1.5)
                st.caption("IQR-Clipping aktiv (k=1.5).")
            ts_df = features.add_time_features(ts_df)

            st.write("#### Feature Engineering (Zeitreihe)")
            st.dataframe(ts_df.head(10), use_container_width=True)

            # Regelbasierte Modellauswahl (Vor-Empfehlung)
            stats = rules.compute_ts_stats(ts_df, y_col="qty")
            model_rec, reason = rules.recommend_model(stats)
            st.info(f"Modell-Empfehlung: **{model_rec.upper()}** — {reason}")

            # Möglichkeit, Empfehlung zu überschreiben
            model_choice = st.selectbox(
                "Modellwahl (Empfehlung vorausgewählt)",
                ["prophet", "lstm"],
                index=0 if model_rec == "prophet" else 1
            )

            # region Training & Prognose mit AutoML-Fallback
            st.write("### Training & Prognose")
            with st.spinner("Trainiere Modell und erstelle Prognose…"):
                if automl_on:
                    st.caption("AutoML aktiv: teste Prophet & LSTM mit kleinen, sinnvollen Settings …")

                    # kleines, schnelles Suchraster
                    search_space = {
                        "prophet": [
                            {"holidays": True},
                            {"holidays": False},
                        ],
                        "lstm": [
                            {"window": max(12, int(lstm_window)), "epochs": min(25, int(lstm_epochs))},
                            {"window": max(24, int(lstm_window)), "epochs": min(35, int(lstm_epochs))},
                        ]
                    }

                    try:
                        chosen, best_params, fc_best, m_best, leaderboard = automl.select_and_tune_forecast(
                            ts_df, horizon=int(horizon),
                            search_space=search_space,
                            holidays_path=APP_DIR / "assets/holidays_de.csv"
                        )
                        st.success(f"AutoML: **{chosen.upper()}** gewählt (RMSE={m_best['rmse']:.3f}).")
                        with st.expander("🔎 AutoML Leaderboard"):
                            st.dataframe(leaderboard, use_container_width=True)
                        fc_df, m = fc_best, m_best
                    except Exception as e:
                        # AutoML Exception abfangen → Empfehlung verwenden
                        st.warning(f"AutoML fehlgeschlagen ({e}) – nutze Empfehlung {model_rec.upper()}.")
                        if model_rec == "prophet":
                            fc_df, m = forecast_prophet.fit_predict(ts_df, horizon=int(horizon),
                                                                    holidays_path=APP_DIR / "assets/holidays_de.csv")
                        else:
                            # LSTM kann an TF-Import scheitern → Prophet-Fallback
                            try:
                                fc_df, m = forecast_lstm.fit_predict(ts_df, horizon=int(horizon),
                                                                     window=int(lstm_window), epochs=int(lstm_epochs))
                            except ImportError as ie:
                                st.warning(f"LSTM nicht verfügbar ({ie}) – Fallback Prophet.")
                                fc_df, m = forecast_prophet.fit_predict(ts_df, horizon=int(horizon),
                                                                        holidays_path=APP_DIR / "assets/holidays_de.csv")
                # Manuelle Modellwahl ohne AutoML
                else:
                    if model_choice == "prophet":
                        fc_df, m = forecast_prophet.fit_predict(ts_df, horizon=int(horizon),
                                                                holidays_path=APP_DIR / "assets/holidays_de.csv")
                    else:
                        try:
                            fc_df, m = forecast_lstm.fit_predict(ts_df, horizon=int(horizon),
                                                                 window=int(lstm_window), epochs=int(lstm_epochs))
                        except ImportError as e:
                            st.warning(f"TensorFlow nicht verfügbar (\n{e}\n) — Fallback auf Prophet.")
                            fc_df, m = forecast_prophet.fit_predict(ts_df, horizon=int(horizon),
                                                                    holidays_path=APP_DIR / "assets/holidays_de.csv")
            # endregion

            # region Visualisierung & Metriken
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = visuals.plot_forecast(ts_df, fc_df, y_col="qty")
                st.pyplot(fig, use_container_width=True)
            with col2:
                st.metric("RMSE", f"{m['rmse']:.3f}")
                st.metric("NRMSE", f"{m['nrmse']:.3f}")
            # endregion

            # region Downloads (CSV, PNG)
            st.write("### Downloads")
            fname_csv = OUTPUT_DIR / f"forecast_{target_id}_{pd.Timestamp.now().date()}.csv"
            fc_df.to_csv(fname_csv, index=False)
            st.download_button("📥 Prognose als CSV", data=fc_df.to_csv(index=False), file_name=fname_csv.name)

            png_bytes, png_name = visuals.save_current_fig_png(fig, out_path=OUTPUT_DIR / f"report_{target_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
            st.download_button("🖼️ Report-Snapshot (PNG)", data=png_bytes, file_name=png_name, mime="image/png")
            # endregion
        #endregion

        # region Workflow: Kundensegmentierung (k-means)
        else:
            # Vorschläge für Kunden-/Bestellspalten
            cust_suggest = io_core.suggest_mapping_customers(raw_df)

            st.write("#### Spalten-Mapping (customer_id, order_id, order_date, net_amount, returned)")
            options = raw_df.columns.tolist()


            # region Helper: Index-Ermittlung für Selectboxe
            def _idx(colname: str | None) -> int:
                return options.index(colname) if (colname in options) else 0
            # endregion

            # --- Mapping-UI
            customer_id = st.selectbox("customer_id", options=options, index=_idx(cust_suggest.get("customer_id")))
            order_id = st.selectbox("order_id", options=options, index=_idx(cust_suggest.get("order_id")))
            order_date = st.selectbox("order_date", options=options, index=_idx(cust_suggest.get("order_date")))
            net_amount = st.selectbox("net_amount", options=options, index=_idx(cust_suggest.get("net_amount")))

            # returned ist optional: Dropdown mit None + Optionen, Default auf Vorschlag oder None
            ret_options = [None] + options
            ret_suggest = cust_suggest.get("returned")
            ret_index = ret_options.index(ret_suggest) if ret_suggest in ret_options else 0
            returned = st.selectbox("returned (optional)", options=ret_options, index=ret_index)

            # Normalisierte Kundentabelle laden
            df = io_core.load_customer_df(raw_df, mapping={
                "customer_id": customer_id,
                "order_id": order_id,
                "order_date": order_date,
                "net_amount": net_amount,
                "returned": returned,
            })
            st.dataframe(df.head(10), use_container_width=True)

            #  Kundenfeatures bauen (KPIs)
            agg = features.build_customer_features(df,
                                                   customer_col="customer_id",
                                                   order_date_col="order_date",
                                                   amount_col="net_amount",
                                                   returned_col="returned")
            st.write("#### Aggregierte Kundenfeatures")
            st.dataframe(agg.head(10), use_container_width=True)

            # k-Auswahl (automatisch per Silhouette oder manuell)
            if auto_k:
                best_k, tbl = automl.select_k_for_clustering(agg, k_min=2, k_max=10)
                st.info(f"Optimiertes k (Silhouette): **{best_k}**")
                k_to_use = best_k

                with st.expander("k-Leaderboard (Silhouette & Inertia)"):
                    if not tbl.empty:
                        # Kurven + Tabelle
                        figk = visuals.plot_k_curves(tbl)
                        st.pyplot(figk, use_container_width=True)

                        # Tabelle anzeigen
                        st.dataframe(tbl, use_container_width=True)

                        # CSV-Download
                        csv_bytes = tbl.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 k-Leaderboard als CSV",
                            data=csv_bytes,
                            file_name=f"k_leaderboard_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )

                        # kurzer Hinweis, falls Unterschiede klein sind
                        try:
                            top = tbl.sort_values(["silhouette", "k"], ascending=[False, True]).head(2)
                            if len(top) == 2 and abs(top.iloc[0]["silhouette"] - top.iloc[1]["silhouette"]) < 0.02:
                                st.caption(
                                    "ℹ️ Die besten k liegen sehr nah beieinander – fachliche Entscheidung kann sinnvoll sein.")
                        except Exception:
                            pass
            else:
                k_to_use = int(k)

            # Clustering durchführen
            labels, met, centers, plot_data, profile = cluster_kmeans.segment(agg, k=int(k_to_use))

            # Visualiserung & Kennzahlen
            st.write("### Ergebnisse")
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = visuals.plot_clusters_2d(plot_data)
                st.pyplot(fig, use_container_width=True)
            with c2:
                st.metric("Silhouette", f"{met['silhouette']:.3f}")
                st.write("Clustergrößen:")
                st.write(met["sizes"])

            # Exporte erstellen
            out_assign = agg.copy()
            out_assign["cluster"] = labels
            fname_csv = OUTPUT_DIR / f"clusters_{pd.Timestamp.now().date()}.csv"
            out_assign.to_csv(fname_csv, index=False)
            st.download_button("📥 Clusterzuordnung als CSV", data=out_assign.to_csv(index=False), file_name=fname_csv.name)

            st.write("#### Cluster-Profile (Mittelwerte/Median/Count)")
            st.dataframe(profile, use_container_width=True)
        # endregion

        # region Optional: Rohdaten-Snapshot als Parquet exportieren
        if allow_parquet:
            snap = OUTPUT_DIR / f"snapshot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            try:
                raw_df.to_parquet(snap, index=False)
                st.success(f"Parquet gespeichert: {snap.name}")
            except Exception as e:
                st.warning(f"Parquet nicht gespeichert ({e})")
        # endregion

    # region Fehlerbehandlung
    except Exception as e:
        st.error(f"Fehler beim Verarbeiten der Datei: {e}")
    # endregion

# Keine Datei ausgewählt - Hinweis anzegien
else:
    # Solange keine Datei ausgewählt ist - Info anzeigen
    st.info("Bitte CSV hochladen, um zu starten. Beispiele liegen im Ordner `examples/`.")
#endregion