# keyboards/metals_menu.py

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_metals_menu():
    buttons = [
        [InlineKeyboardButton("DXY / XAU / XAG / AUD", callback_data="dxy_xau_xag_aud")],
        [InlineKeyboardButton("XAU / XAG / AUD", callback_data="xau_xag_aud")],
        [InlineKeyboardButton("DXY / XAU / AUD", callback_data="dxy_xau_aud")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(buttons)
