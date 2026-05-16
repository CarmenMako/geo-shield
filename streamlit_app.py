"""
GEO-SHIELD Professional Auditor
Prémiová geodetická aplikace pro analýzu map a katastrálních dokumentů.
Powered by Google Gemini 2.5 Flash.
"""

import os
import io
import textwrap
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import pypdf
from PIL import Image
from google import genai
from google.genai import types

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle,
)
from reportlab.pdfgen import canvas as rl_canvas


SYSTEM_INSTRUCTION = (
    "Jsi špičkový geodetický auditor. "
    "ZÁKAZ FABULACE: Pokud z vložených podkladů (obrázek nebo PDF) nelze s 100 procentní jistotou "
    "vyčíst konkrétní údaj jako je přesné číslo parcely, vlastnictví nebo typ budovy, NIKDY si ho "
    "nevymýšlej ani neodhaduj. "
    "Striktně rozlišuj ČÍSLO POPISNÉ budovy a ČÍSLO PARCELY. "
    "Pokud znáš jen adresu, uveď, že parcela musí být dohledána v katastru nemovitostí. "
    "U všech neověřených právních a katastrálních detailů generuj varování: "
    "Tento údaj nebyl ověřen v oficiální databázi ČÚZK a je nutné jej předložit k revizi."
)

REPORT_TYPES = {
    "Kompletní geodetický audit": (
        "Proveď KOMPLETNÍ GEODETICKÝ AUDIT všech dostupných podkladů. "
        "Zahrň: identifikaci parcel a jejich čísel, vlastnické vztahy (pokud jsou čitelné), "
        "typ a využití ploch, hranice pozemků, případné věcné břemeno nebo zástavní právo viditelné "
        "z dokumentu, soulad skutečného stavu se zákresy, a doporučení pro geodetický průzkum v terénu. "
        "Výstup strukturuj do sekcí s nadpisy."
    ),
    "Rychlá kontrola parcel": (
        "Proveď RYCHLOU KONTROLU PARCEL. Zaměř se výhradně na: čísla parcel, jejich výměry "
        "(pokud jsou uvedeny), katastrální území, druh pozemku. "
        "Výstup musí být stručný – maximálně odrážkový seznam klíčových zjištění."
    ),
    "Právní prověření pro investory": (
        "Proveď PRÁVNÍ PROVĚŘENÍ Z POHLEDU INVESTORA. Identifikuj: vlastnické vztahy a případné "
        "spoluvlastnictví, existenci zástavních práv nebo věcných břemen (pouze pokud jsou čitelné "
        "z dokumentu), územně-plánovací informace, rizikové faktory bránící investici, "
        "a doporučení pro due diligence. Každé riziko jasně označ jako RIZIKO: ..."
    ),
}

MODEL_ID = "gemini-3.1-flash-lite-preview"

BRAND_DARK = colors.HexColor("#1a2b4a")
BRAND_ACCENT = colors.HexColor("#2e6da4")
BRAND_WARNING = colors.HexColor("#c0392b")
BRAND_LIGHT = colors.HexColor("#f0f4f8")


def get_api_key() -> str | None:
    """Načte API klíč z Streamlit secrets nebo z prostředí."""
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key and key.strip():
            return key.strip()
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip() or None


def extract_pdf_text(uploaded_file) -> str:
    """Extrahuje veškerý text z PDF souboru."""
    reader = pypdf.PdfReader(uploaded_file)
    pages_text = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(f"--- Strana {i} ---\n{text.strip()}")
    return "\n\n".join(pages_text)


def image_to_bytes(uploaded_file) -> bytes:
    """Převede nahraný soubor na bytes pro Gemini."""
    return uploaded_file.getvalue()


def run_gemini_analysis(
    api_key: str,
    report_type: str,
    address_context: str,
    pil_image: Image.Image | None,
    pdf_text: str | None,
) -> str:
    """Odešle požadavek do Gemini a vrátí textovou odpověď."""
    client = genai.Client(api_key=api_key)
    report_instruction = REPORT_TYPES[report_type]

    parts: list = []

    if address_context.strip():
        parts.append(
            types.Part.from_text(
                text=f"Kontext / adresa lokality: {address_context.strip()}"
            )
        )

    if pdf_text:
        parts.append(
            types.Part.from_text(
                text=f"Obsah přiloženého PDF dokumentu (extrahovaný text):\n\n{pdf_text}"
            )
        )

    if pil_image is not None:
        parts.append(pil_image)

    parts.append(
        types.Part.from_text(
            text=(
                f"Typ požadované analýzy: {report_type}\n\n"
                f"Instrukce pro analýzu: {report_instruction}"
            )
        )
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        max_output_tokens=8192,
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=parts,
        config=config,
    )
    return response.text


def _draw_header_footer(canvas_obj, doc, report_type: str, address_context: str):
    """Vykreslí hlavičku a patičku na každé stránce PDF."""
    canvas_obj.saveState()
    page_w, page_h = A4

    canvas_obj.setFillColor(BRAND_DARK)
    canvas_obj.rect(0, page_h - 28 * mm, page_w, 28 * mm, fill=1, stroke=0)

    canvas_obj.setFillColor(BRAND_ACCENT)
    canvas_obj.rect(0, page_h - 30 * mm, page_w, 2 * mm, fill=1, stroke=0)

    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont("Helvetica-Bold", 15)
    canvas_obj.drawString(15 * mm, page_h - 13 * mm, "GEO-SHIELD Professional Auditor")

    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.HexColor("#a8c4e0"))
    canvas_obj.drawRightString(
        page_w - 15 * mm,
        page_h - 10 * mm,
        f"Vygenerováno: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
    )
    canvas_obj.drawRightString(
        page_w - 15 * mm,
        page_h - 16 * mm,
        f"Typ: {report_type}",
    )

    canvas_obj.setFillColor(BRAND_DARK)
    canvas_obj.rect(0, 0, page_w, 14 * mm, fill=1, stroke=0)

    canvas_obj.setFillColor(colors.HexColor("#a8c4e0"))
    canvas_obj.setFont("Helvetica", 7)
    canvas_obj.drawString(
        15 * mm,
        5 * mm,
        "GEO-SHIELD · Výstup AI není právně závazným dokumentem · Ověřte v ČÚZK: nahlizenidokn.cuzk.cz",
    )
    canvas_obj.drawRightString(
        page_w - 15 * mm,
        5 * mm,
        f"Strana {doc.page}",
    )

    canvas_obj.restoreState()


def _draw_cuzk_stamp(canvas_obj, doc):
    """Vykreslí šikmé razítko NEOVĚŘENO ČÚZK na každou stránku."""
    canvas_obj.saveState()
    page_w, page_h = A4

    canvas_obj.translate(page_w / 2, page_h / 2)
    canvas_obj.rotate(38)

    stamp_text = "NEOVĚŘENO ČÚZK"
    font_size = 48
    canvas_obj.setFont("Helvetica-Bold", font_size)
    text_w = canvas_obj.stringWidth(stamp_text, "Helvetica-Bold", font_size)

    canvas_obj.setStrokeColor(BRAND_WARNING)
    canvas_obj.setLineWidth(2.5)
    canvas_obj.setFillAlpha(0)
    canvas_obj.setStrokeAlpha(0.18)
    padding = 8
    canvas_obj.roundRect(
        -text_w / 2 - padding,
        -font_size / 2 - padding / 2,
        text_w + padding * 2,
        font_size + padding,
        5,
        fill=0,
        stroke=1,
    )

    canvas_obj.setFillColor(BRAND_WARNING)
    canvas_obj.setFillAlpha(0.12)
    canvas_obj.setFont("Helvetica-Bold", font_size)
    canvas_obj.drawCentredString(0, 0, stamp_text)

    canvas_obj.restoreState()


def generate_pdf(
    report_type: str,
    address_context: str,
    analysis_text: str,
) -> bytes:
    """Sestaví a vrátí PDF report jako bytes."""
    buffer = io.BytesIO()

    def on_page(canvas_obj, doc):
        _draw_cuzk_stamp(canvas_obj, doc)
        _draw_header_footer(canvas_obj, doc, report_type, address_context)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=35 * mm,
        bottomMargin=22 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        title=f"GEO-SHIELD – {report_type}",
        author="GEO-SHIELD Professional Auditor",
    )

    styles = getSampleStyleSheet()

    style_meta_label = ParagraphStyle(
        "MetaLabel",
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=BRAND_ACCENT,
        spaceAfter=1,
    )
    style_meta_value = ParagraphStyle(
        "MetaValue",
        fontName="Helvetica",
        fontSize=9,
        textColor=BRAND_DARK,
        spaceAfter=0,
    )
    style_heading = ParagraphStyle(
        "GeoHeading",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=BRAND_DARK,
        spaceBefore=8,
        spaceAfter=3,
        borderPad=(0, 0, 2, 0),
    )
    style_body = ParagraphStyle(
        "GeoBody",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=colors.HexColor("#222222"),
        leading=14,
        spaceAfter=4,
    )
    style_warning = ParagraphStyle(
        "GeoWarning",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        textColor=BRAND_WARNING,
        leading=12,
        spaceAfter=3,
        leftIndent=6,
        borderPad=4,
    )
    style_stamp_box = ParagraphStyle(
        "StampBox",
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=BRAND_WARNING,
        alignment=TA_CENTER,
        spaceAfter=0,
    )

    story = []

    meta_data = [
        ["Typ analýzy:", report_type],
        ["Datum:", datetime.now().strftime("%d. %m. %Y, %H:%M")],
        ["Lokalita / adresa:", address_context.strip() if address_context.strip() else "—"],
        ["Model AI:", MODEL_ID],
    ]
    meta_table_data = [
        [
            Paragraph(label, style_meta_label),
            Paragraph(value, style_meta_value),
        ]
        for label, value in meta_data
    ]

    meta_table = Table(
        meta_table_data,
        colWidths=[42 * mm, None],
        hAlign="LEFT",
    )
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, BRAND_ACCENT),
            ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#c8d8ea")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 6 * mm))

    warning_box_data = [[
        Paragraph(
            "⚠  UPOZORNĚNÍ: Tento report byl vygenerován umělou inteligencí a nebyl ověřen "
            "v oficiální databázi ČÚZK. Všechna katastrální data vyžadují ověření na "
            "nahlizenidokn.cuzk.cz před jakýmkoliv právním nebo investičním využitím.",
            style_stamp_box,
        )
    ]]
    warning_table = Table(warning_box_data, hAlign="CENTER", colWidths=[175 * mm])
    warning_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 1.2, BRAND_WARNING),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf3f2")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(warning_table)
    story.append(Spacer(1, 6 * mm))

    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_ACCENT, spaceAfter=4))
    story.append(Paragraph("Výsledek analýzy", style_heading))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#c8d8ea"), spaceBefore=2, spaceAfter=5))

    for line in analysis_text.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 2 * mm))
            continue

        line_escaped = (
            stripped
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        if stripped.startswith("## ") or stripped.startswith("# "):
            heading_text = stripped.lstrip("#").strip()
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(heading_text, style_heading))
        elif stripped.startswith("**") and stripped.endswith("**"):
            bold_text = stripped[2:-2]
            story.append(Paragraph(f"<b>{bold_text}</b>", style_body))
        elif "ČÚZK" in stripped or "varování" in stripped.lower() or stripped.startswith("⚠") or "RIZIKO:" in stripped:
            story.append(Paragraph(line_escaped, style_warning))
        elif stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("• "):
            bullet_text = stripped[2:].strip()
            story.append(Paragraph(f"• &nbsp; {bullet_text}", style_body))
        else:
            story.append(Paragraph(line_escaped, style_body))

    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BRAND_ACCENT))
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "Dokument byl vygenerován automaticky aplikací GEO-SHIELD Professional Auditor. "
            "Není náhradou za odborný geodetický posudek ani právně závazným dokumentem.",
            ParagraphStyle(
                "Disclaimer",
                fontName="Helvetica-Oblique",
                fontSize=7.5,
                textColor=colors.HexColor("#888888"),
                alignment=TA_CENTER,
            ),
        )
    )

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buffer.getvalue()


def clipboard_button(text: str) -> None:
    """Vykreslí tlačítko pro zkopírování textu do schránky přes JavaScript."""
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    html_code = f"""
    <button
        onclick="(function(){{
            navigator.clipboard.writeText(`{safe_text}`)
                .then(function(){{
                    var btn = document.getElementById('copy-btn');
                    btn.innerText = '✓ Zkopírováno';
                    btn.style.background = '#2ecc71';
                    setTimeout(function(){{
                        btn.innerText = '📋 Kopírovat do schránky';
                        btn.style.background = '#4a90d9';
                    }}, 2000);
                }})
                .catch(function(){{ alert('Kopírování selhalo – použijte Ctrl+C.'); }});
        }})();"
        id="copy-btn"
        style="
            background: #4a90d9;
            color: white;
            border: none;
            padding: 0.45rem 1.1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            cursor: pointer;
        "
    >📋 Kopírovat do schránky</button>
    """
    components.html(html_code, height=50)


def count_words(text: str) -> int:
    return len(text.split()) if text.strip() else 0


APP_PASSWORD = "Carmen2026"


def check_password() -> bool:
    """Zobrazí přihlašovací bránu a vrátí True, pokud uživatel zadal správné heslo."""
    if st.session_state.get("authenticated"):
        return True

    st.title("🛡️ GEO-SHIELD Professional Auditor")
    st.divider()
    st.subheader("🔒 Přihlášení")
    password = st.text_input(
        "Zadejte přístupové heslo",
        type="password",
        placeholder="Přístupové heslo…",
    )
    if st.button("Vstoupit", type="primary"):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Nesprávné heslo. Přístup odepřen.")
    return False


def main():
    st.set_page_config(
        page_title="GEO-SHIELD Professional Auditor",
        page_icon="🛡️",
        layout="wide",
    )

    if not check_password():
        st.stop()

    st.title("🛡️ GEO-SHIELD Professional Auditor")
    st.caption(
        "Prémiový nástroj pro geodetickou analýzu map a katastrálních dokumentů "
        "· Powered by Google Gemini 2.5 Flash"
    )

    st.divider()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "❌ Gemini API klíč nebyl nalezen. "
            "Nastavte proměnnou prostředí `GEMINI_API_KEY` nebo doplňte hodnotu v souboru `.env`."
        )
        st.stop()

    report_type = st.selectbox(
        "📋 Typ reportu",
        options=list(REPORT_TYPES.keys()),
        help="Zvolte hloubku a zaměření geodetické analýzy.",
    )

    address_context = st.text_area(
        "📍 Kontext nebo adresa lokality",
        placeholder=(
            "Zadejte adresu, katastrální území, číslo LV nebo jiný upřesňující kontext "
            "(např. 'Praha 10, Vinohrady, ulice Mánesova 12'). Nepovinné."
        ),
        height=80,
    )

    uploaded_file = st.file_uploader(
        "📎 Nahrajte podklad k analýze",
        type=["jpg", "jpeg", "png", "pdf"],
        help="Podporované formáty: JPG, PNG (screenshot mapy) nebo PDF (výpis z katastru, GP apod.).",
    )

    if uploaded_file is not None:
        file_ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
        if file_ext in ("jpg", "jpeg", "png"):
            st.image(uploaded_file, caption=f"Náhled: {uploaded_file.name}")
        else:
            st.info(f"📄 PDF soubor nahrán: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} kB)")

    run_btn = st.button(
        "🚀 Spustit analýzu podkladů",
        type="primary",
        help="Spustí analýzu pomocí Gemini 2.5 Flash.",
    )

    if "report_type_used" not in st.session_state:
        st.session_state.report_type_used = ""
    if "address_context_used" not in st.session_state:
        st.session_state.address_context_used = ""

    if run_btn:
        if uploaded_file is None and not address_context.strip():
            st.warning("⚠️ Nahrajte soubor nebo zadejte alespoň kontext / adresu lokality.")
            st.stop()

        pil_image = None
        pdf_text = None

        if uploaded_file is not None:
            file_ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
            if file_ext == "pdf":
                with st.spinner("Extrahuji text z PDF…"):
                    uploaded_file.seek(0)
                    pdf_text = extract_pdf_text(uploaded_file)
                    if not pdf_text.strip():
                        st.warning(
                            "⚠️ Z PDF se nepodařilo extrahovat žádný text. "
                            "Zkuste nahrát skenovaný dokument jako PNG/JPG."
                        )
            else:
                uploaded_file.seek(0)
                pil_image = Image.open(uploaded_file)

        with st.spinner("⏳ Analyzuji podklady pomocí Gemini 2.5 Flash…"):
            try:
                result = run_gemini_analysis(
                    api_key=api_key,
                    report_type=report_type,
                    address_context=address_context,
                    pil_image=pil_image,
                    pdf_text=pdf_text,
                )
                st.session_state['result'] = result
                st.session_state.report_type_used = report_type
                st.session_state.address_context_used = address_context
                st.success("✅ Analýza dokončena.")
            except Exception as e:
                st.error(f"Chyba při analýze: {str(e)}")
                st.stop()

    st.subheader("📝 Výsledek analýzy")
    edited_text = st.text_area(
        label="Analýza (plně editovatelná – upravte text před uložením)",
        value=st.session_state.get('result', ''),
        height=450,
        key="edited_output",
        placeholder="Výsledek analýzy se zobrazí zde po spuštění…",
    )

    char_count = len(edited_text)
    word_count = count_words(edited_text)
    st.caption(f"📊 Počet znaků: **{char_count:,}** · Počet slov: **{word_count:,}**")

    if edited_text.strip():
        col_copy, col_dl_txt, col_dl_md, col_dl_pdf, col_spacer = st.columns([2, 2, 2, 2, 4])

        with col_copy:
            clipboard_button(edited_text)

        with col_dl_txt:
            st.download_button(
                label="⬇️ Stáhnout .txt",
                data=edited_text.encode("utf-8"),
                file_name="geo_shield_report.txt",
                mime="text/plain",
            )

        with col_dl_md:
            st.download_button(
                label="⬇️ Stáhnout .md",
                data=edited_text.encode("utf-8"),
                file_name="geo_shield_report.md",
                mime="text/markdown",
            )

        with col_dl_pdf:
            with st.spinner("Připravuji PDF…"):
                pdf_bytes = generate_pdf(
                    report_type=st.session_state.report_type_used or report_type,
                    address_context=st.session_state.address_context_used or address_context,
                    analysis_text=edited_text,
                )
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                label="⬇️ Stáhnout PDF",
                data=pdf_bytes,
                file_name=f"geo_shield_report_{ts}.pdf",
                mime="application/pdf",
            )

    st.divider()
    st.caption(
        "GEO-SHIELD Professional Auditor · Všechna katastrální data musí být ověřena "
        "v oficiální databázi ČÚZK (nahlizenidokn.cuzk.cz) · "
        "Výstup AI není právně závazným dokumentem."
    )


if __name__ == "__main__":
    main()
