import os
from dotenv import load_dotenv

# --- LangChain agent import ---
try:
    from .agent import get_agent
except ImportError:
    from agent import get_agent  # if you run inside the folder

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Initialize the agent once at startup
_agent = get_agent()
# If get_agent returns (agent, tool) tuple, keep the first:
if isinstance(_agent, tuple):
    _agent = _agent[0]

def handle_message(text: str) -> str:
    """Use the LangChain agent to handle any incoming Telegram text."""
    try:
        return _agent.run(text)
    except Exception as e:
        return f"Oops, something went wrong: {e}"

# --- Telegram bot (long polling) ---
from telegram.ext import Application, CommandHandler, MessageHandler, filters

async def cmd_start(update, context):
    await update.message.reply_text(
        "Hi! Send me a flight number and date, e.g., BA284 on 2025-09-18"
    )

async def on_text(update, context):
    text = update.message.text or ""
    answer = handle_message(text)
    await update.message.reply_text(answer)

def main():
    if not TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in .env")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()

if __name__ == "__main__":
    main()

