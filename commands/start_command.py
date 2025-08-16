from telegram import Update
from telegram.ext import ContextTypes
from keyboards.main_menu import get_main_menu

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /start command and sends the main menu with inline keyboard options.
    """
    user = update.effective_user
    welcome_text = (
        f"👋 Hello {user.first_name or 'Trader'}!\n\n"
        "Welcome to your market analysis bot.\n"
        "Please choose a category to begin:"
    )

    await update.message.reply_text(
        text=welcome_text,
        reply_markup=get_main_menu()
    )
