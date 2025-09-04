# commands/start_command.py

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from models.user import get_or_create_user, is_user_registered, save_unregistered_user

def get_main_menu() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("📊 FOREX CURRENCIES", callback_data="menu_forex")],
        [InlineKeyboardButton("📈 FUTURES (STOCKS)", callback_data="menu_futures")],
        [InlineKeyboardButton("💰 CRYPTO CURRENCY (USD)", callback_data="menu_crypto_usd")],
        [InlineKeyboardButton("💵 CRYPTO CURRENCY (USDT) (coming soon)", callback_data="menu_crypto_usdt")],
        [InlineKeyboardButton("🪙 METALS", callback_data="menu_metals")],
        [InlineKeyboardButton("⚡ Energy", callback_data="menu_energy")],
        [InlineKeyboardButton("🔔 Alerts", callback_data="menu_alerts")],
    ]
    return InlineKeyboardMarkup(buttons)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat.id
    user = update.effective_user

    # Check if user is registered
    if not is_user_registered(chat_id):
        # Send “not registered” message
        await update.message.reply_text(
            "❌ Sorry, you are not registered to use this bot.\n"
            "Please contact @mubinxyz for purchasing access.\n\n"
            "❌ متأسفیم، شما برای استفاده از این ربات ثبت‌نام نکرده‌اید.\n"
            "لطفاً برای خرید و دریافت دسترسی با @mubinxyz تماس بگیرید."
        )
 

        # Optionally save the chat_id for later registration
        save_unregistered_user(chat_id, user.username, user.first_name, user.last_name)
        return

    # Registered users proceed as usual
    user_data = get_or_create_user(
        chat_id=chat_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name
    )

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