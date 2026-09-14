"""
Client MCP — version Groq (gratuit, rapide, sans historique de facturation).

Ce script :
1. Lance mcp_server/server.py comme sous-processus (transport stdio).
2. Récupère la liste des tools exposés par le serveur MCP.
3. Envoie tes questions à l'API Groq (modèle Llama 3.3 70B) en lui donnant
   accès à ces tools au format OpenAI standard (function calling).
4. Exécute les appels de tools demandés et renvoie les résultats au modèle,
   jusqu'à obtenir une réponse finale en langage naturel.

Prérequis :
    pip install mcp groq
    export GROQ_API_KEY="ta_clé_gsk_..."

Lancement :
    python mcp_client/client.py
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack

from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "openai/gpt-oss-120b"

SERVER_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mcp_server", "server.py")
)

SYSTEM_PROMPT = (
    "Tu es l'assistant MLOps du pipeline de prédiction de churn client. "
    "Tu as accès à des outils pour consulter le monitoring de l'API, "
    "les dernières prédictions et le rapport de Data Drift (Evidently). "
    "Réponds en français, de façon claire et synthétique, et appuie-toi "
    "toujours sur les données renvoyées par les outils plutôt que de "
    "supposer des chiffres. Si le drift est détecté sur des colonnes "
    "importantes, signale-le clairement et donne un avis sur l'utilité "
    "d'un réentraînement du modèle."
)


class MCPAgentClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.client = Groq()  # lit GROQ_API_KEY dans l'environnement

    async def connect(self):
        server_params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

        tools_response = await self.session.list_tools()
        names = [t.name for t in tools_response.tools]
        print(f"Connecté au serveur MCP. Tools disponibles : {names}")

    async def _build_groq_tools(self) -> list:
        tools_response = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_response.tools
        ]

    async def ask(self, question: str) -> str:
        groq_tools = await self._build_groq_tools()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        response = self.client.chat.completions.create(
            model=MODEL, messages=messages, tools=groq_tools
        )

        while True:
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or ""

            # Ajoute la réponse du modèle (avec ses demandes d'appel de tools)
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                print(f"  -> Appel de l'outil : {tc.function.name}({args})")
                result = await self.session.call_tool(tc.function.name, args)
                text_result = "".join(getattr(block, "text", "") for block in result.content)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": text_result,
                    }
                )

            response = self.client.chat.completions.create(
                model=MODEL, messages=messages, tools=groq_tools
            )

    async def close(self):
        await self.exit_stack.aclose()


async def main():
    client = MCPAgentClient()
    try:
        await client.connect()
        print("\nAgent MCP prêt. Pose tes questions (ex: 'y a-t-il du drift ?').")
        print("Tape 'quit' pour sortir.\n")

        while True:
            question = input("Toi: ").strip()
            if question.lower() in ("quit", "exit"):
                break
            if not question:
                continue

            answer = await client.ask(question)
            print(f"\nAgent: {answer}\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())