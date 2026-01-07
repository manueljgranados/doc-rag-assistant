from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def _extract_text_fallback(resp: Any) -> str:
    # Fallback por si output_text no está disponible en alguna versión
    try:
        if hasattr(resp, "output_text") and resp.output_text:
            return str(resp.output_text)
    except Exception:
        pass

    # Intento de parseo del objeto "output"
    try:
        out = getattr(resp, "output", None) or []
        parts: list[str] = []
        for item in out:
            if getattr(item, "type", None) == "message":
                content = getattr(item, "content", None) or []
                for c in content:
                    if getattr(c, "type", None) == "output_text":
                        parts.append(getattr(c, "text", ""))
        return "\n".join([p for p in parts if p]).strip()
    except Exception:
        return ""


class OpenAIAnswerer:
    def __init__(self, model: str):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en el entorno.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, question: str, context_blocks: list[str], prompt_style: str = "about") -> str:
        base = (
            "Responda en español de España y con tono formal. "
            "Use únicamente la información del contexto. "
            "Si falta información, indíquelo. "
            "Incluya referencias a las citas tal como aparecen (entre corchetes)."
        )

        if prompt_style == "objectives_conclusions":
            style = (
                "Estructure la respuesta en dos apartados:\n"
                "1) Objetivos (lista con viñetas)\n"
                "2) Conclusiones (lista con viñetas)\n"
                "Cada viñeta debe incluir al menos una cita."
            )
        else:
            style = "Responda directamente a la pregunta. Incluya citas en las frases relevantes."

        instructions = f"{base}\n{style}"

        context = "\n\n".join(context_blocks)

        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}",
        )
        text = _extract_text_fallback(resp)
        return text.strip() or "No se pudo generar respuesta con OpenAI."
