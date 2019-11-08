import neural
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Settings
TOKEN = '1023230889:AAEAuFDVPXMNpOd9o7ZWE7AjEEJAEDzx2LA'


def start(update, context):
    update.message.reply_text(
        "Hello, give me a picture, please.")


def image(update, context):
    update.effective_message.photo[-1].get_file().download('./img/image.jpg')
    update.message.reply_text(neural.main())


def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, image))
    updater.start_polling()

    updater.idle()



main()
