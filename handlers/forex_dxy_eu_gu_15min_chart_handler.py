# handlers/forex_dxy_eu_gu_15min_chart_handler.py


from telegram import Update
from telegram.ext import ContextTypes
from services.chart_service import generate_chart

async def forex_dxy_eu_gu_15min_chart_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles 'View Charts' for 15min SSMT Dxy, EU, GU.
    Generates and sends charts for each symbol.
    """
    query = update.callback_query
    await query.answer()

    symbols = ["USDX", "EURUSD", "GBPUSD"]  # Adjust based on your normalization logic
    timeframe = 15

    await query.edit_message_text("📊 Generating charts...")

    chart_buffers = await generate_chart(symbols, timeframe)

    if not chart_buffers:
        await query.message.reply_text("⚠️ No chart data available.")
        return

    for buf in chart_buffers:
        await query.message.reply_photo(photo=buf)
