import logging
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
)
from config import BOT_TOKEN, LOG_LEVEL

# Command handlers
from commands.start_command import start_command

# Callback handlers
from handlers.forex_handler import forex_handler
from handlers.forex_dxy_eu_gu_handler import forex_dxy_eu_gu_handler
from handlers.forex_dxy_eu_gu_15min_handler import forex_dxy_eu_gu_15min_handler
from handlers.forex_dxy_eu_gu_15min_alert_handler import forex_dxy_eu_gu_15min_alert_handler
from handlers.back_to_main_handler import back_to_main_handler
from handlers.back_to_forex_handler import back_to_forex_handler
from handlers.back_to_timeframe_handler import back_to_timeframe_handler
from handlers.forex_dxy_eu_gu_15min_chart_handler import forex_dxy_eu_gu_15min_chart_handler
# You can add chart handler and back navigation handlers here later

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=LOG_LEVEL
)

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # Register command
    app.add_handler(CommandHandler("start", start_command))

    # Register callback query handlers
    app.add_handler(CallbackQueryHandler(forex_handler, pattern="^menu_forex$"))
    app.add_handler(CallbackQueryHandler(forex_dxy_eu_gu_handler, pattern="^forex_dxy_eu_gu$"))
    app.add_handler(CallbackQueryHandler(forex_dxy_eu_gu_15min_handler, pattern="^forex_dxy_eu_gu_15min_ssmt$"))
    app.add_handler(CallbackQueryHandler(forex_dxy_eu_gu_15min_alert_handler, pattern="^forex_dxy_eu_gu_15min_activate_alert$"))
    app.add_handler(CallbackQueryHandler(back_to_main_handler, pattern="^back_to_main$"))
    app.add_handler(CallbackQueryHandler(back_to_forex_handler, pattern="^forex_dxy_eu_gu_back$"))
    app.add_handler(CallbackQueryHandler(back_to_timeframe_handler, pattern="^forex_dxy_eu_gu_back_to_timeframes$"))
    app.add_handler(CallbackQueryHandler(forex_dxy_eu_gu_15min_chart_handler, pattern="^forex_dxy_eu_gu_15min_view_charts$"))

    # Start polling
    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
