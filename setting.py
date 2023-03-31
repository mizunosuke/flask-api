import os

if os.getenv('GAE_APPLICATION', None):
# 本番環境
DEBUG = False
ALLOWED_HOSTS = ['<本番環境ドメイン>']
else:
# 開発環境
DEBUG = True
ALLOWED_HOSTS = ['*']