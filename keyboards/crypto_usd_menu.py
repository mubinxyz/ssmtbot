# keyboards/crypto_usd_menu.py


from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_crypto_usd_menu():
    buttons = [
        [InlineKeyboardButton("BTCUSD / ETHUSD / XRPUSD", callback_data="btc_eth_xrp")],
        [InlineKeyboardButton("BTCUSD / ETHUSD / TOTAL (market cap)", callback_data="btc_eth_total")],
        [InlineKeyboardButton("BTCUSD / XRPUSD / DOGEUSD", callback_data="btc_xrp_doge")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(buttons)

