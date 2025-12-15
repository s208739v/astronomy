import discord
from discord.ext import tasks
import os
import glob
import asyncio
import time

# --- 設定 ---
TOKEN = ""  # ここにBotトークンを入れる（またはファイルから読み込む）
#CHANNEL_ID = 1364943999651025050 # 通知を送るチャンネルID
CHANNEL_ID = 1365888853562363954

# 監視対象のフォルダ（検知スクリプトの保存先と同じにする）
# 例: "./detected_meteors"
WATCH_DIR = r"F:\SharpCap Captures\2025-12-13\meteor"

# 監視する拡張子
TARGET_EXT = "*.mp4" # 動画を送る場合
#TARGET_EXT = "*.png" # 画像を送る場合はこちら

# --- トークン読み込み関数 (既存のコードに合わせています) ---
def read_token():
    global TOKEN
    try:
        with open('botid.txt', 'r', encoding='utf-8') as f:
            TOKEN = f.read().strip()
    except FileNotFoundError:
        pass

if not TOKEN:
    read_token()

# --- Bot本体 ---
read_token()
intents = discord.Intents.default()
bot = discord.Bot(intents=intents)

# 既に送信済みのファイルを記録するセット（重複送信防止）
processed_files = set()

@bot.event
async def on_ready():
    print(f'🤖 ログイン完了: {bot.user}')
    print(f'👀 監視対象フォルダ: {os.path.abspath(WATCH_DIR)}')
    
    # 起動時に、現在フォルダにあるファイルは「送信済み」としてマークする
    # (これをしないと、Bot起動時に過去の流星が一気に連投されてしまいます)
    initial_scan()
    
    # 監視ループ開始
    monitor_folder.start()

def initial_scan():
    """起動時に既存ファイルをリストアップして無視リストに入れる"""
    # recursive=True でサブフォルダも全検索
    search_pattern = os.path.join(WATCH_DIR, "**", TARGET_EXT)
    files = glob.glob(search_pattern, recursive=True)
    
    count = 0
    for f in files:
        processed_files.add(f)
        count += 1
    print(f"📁 既存ファイル {count} 件をスキップリストに登録しました。")

@tasks.loop(seconds=5)  # 5秒ごとにチェック
async def monitor_folder():
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        return

    # 指定フォルダ以下の全ファイルを再スキャン
    search_pattern = os.path.join(WATCH_DIR, "**", TARGET_EXT)
    current_files = glob.glob(search_pattern, recursive=True)
    
    # 更新日時順に並べる（古い順に送信するため）
    current_files.sort(key=os.path.getmtime)

    for file_path in current_files:
        # まだ処理していないファイルが見つかったら
        if file_path not in processed_files:
            
            # ファイル書き込み中の可能性を考慮してサイズチェック
            if os.path.getsize(file_path) == 0:
                continue # まだ書き込み中（0バイト）なら今回はスキップ

            print(f"✨ 新規検出: {file_path}")
            
            try:
                # 送信処理
                # ファイル名から日時などが推測できるならメッセージに入れると良い
                file_name = os.path.basename(file_path)
                msg = f"💫 **流星を検知しました！**\nファイル: `{file_name}`"
                time.sleep(2)   # ファイル安定化のため少し待機
                await channel.send(msg, file=discord.File(file_path))
                print(f"✅ 送信成功: {file_name}")
                
                # 送信成功したら「処理済み」に追加
                processed_files.add(file_path)
                
            except Exception as e:
                print(f"❌ 送信エラー: {e}")
                # エラーが出た場合、processed_filesに追加しないことで次回のループで再送を試みる
                # (ただし無限ループ防止のため、特定のエラーなら追加してしまうのも手です)

# トークンがあるか確認して実行
if TOKEN:
    bot.run(TOKEN)
else:
    print("❌ エラー: トークンが設定されていません。")