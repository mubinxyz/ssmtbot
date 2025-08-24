# commands/start_command.py

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from models.user import get_or_create_user


def get_main_menu() -> InlineKeyboardMarkup:
    """
    Returns the main menu inline keyboard with market categories and an Alerts entry.
    
    """
    
    buttons = [
        [InlineKeyboardButton("📊 FOREX CURRENCIES", callback_data="menu_forex")],
        [InlineKeyboardButton("📈 FUTURES (STOCKS)", callback_data="menu_futures")],
        [InlineKeyboardButton("💰 CRYPTO CURRENCY (USD)", callback_data="menu_crypto_usd")],
        [InlineKeyboardButton("💵 CRYPTO CURRENCY (USDT) (coming soon)", callback_data="menu_crypto_usdt")],
        [InlineKeyboardButton("🪙 METALS", callback_data="menu_metals")],
        [InlineKeyboardButton("⚡ Energy", callback_data="menu_energy")],
        # Alerts row appended at the end:
        [InlineKeyboardButton("🔔 Alerts", callback_data="menu_alerts")],
    ]
    return InlineKeyboardMarkup(buttons)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /start command and sends the main menu with inline keyboard options.
    """
    user_data = get_or_create_user(
        chat_id=update.message.chat.id,
        username=update.message.from_user.username,
        first_name=update.message.from_user.first_name,
        last_name=update.message.from_user.last_name
    )
    
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


async def back_to_main_handler(update, context) -> None:
    """
    Handles the back to main menu callback query.
    """
    query = update.callback_query
    await query.answer()
    
    user = update.effective_user
    welcome_text = (
        f"👋 Hello {user.first_name or 'Trader'}!\n\n"
        "Welcome to your market analysis bot.\n"
        "Please choose a category to begin:"
    )
    
    await query.edit_message_text(
        text=welcome_text,
        reply_markup=get_main_menu()
    )