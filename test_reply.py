#!/usr/bin/env python3
"""测试脚本：对指定BV号视频生成总结并回复到评论区"""
import json, sys, os

# 设置配置
with open('config.json') as f:
    _CFG = json.load(f)

os.environ["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# 代理
import socket as _socket
_proxy_host = _CFG.get("proxy", {}).get("host", "127.0.0.1")
_proxy_port = _CFG.get("proxy", {}).get("port", 7890)
try:
    s = _socket.create_connection((_proxy_host, _proxy_port), timeout=1)
    s.close()
    os.environ["http_proxy"] = f"http://{_proxy_host}:{_proxy_port}"
    os.environ["https_proxy"] = f"http://{_proxy_host}:{_proxy_port}"
except:
    pass

# 导入脚本模块但不执行 main
# 通过修改 __name__ 来阻止 main() 执行
import importlib
import types

bili = types.ModuleType('bili')
bili.__file__ = 'bili_monitor.py'

# 先设好 __name__ 
code = open('bili_monitor.py').read()
# 将 if __name__ == '__main__' 改为 if False
code = code.replace("if __name__ == '__main__':", "if False:")

exec(compile(code, 'bili_monitor.py', 'exec'), bili.__dict__)

# 现在可以用 bili.xxx 调用函数了
BV = "BV14qLK6KE7w"
print(f"视频: {BV}")
print("="*50)

# 1. 获取视频信息
info = bili.get_video_info(BV)
if info:
    print(f"标题: {info.get('title')}")
    print(f"时长: {info.get('duration')}秒")

# 2. 处理视频（下载+截帧+ASR+总结）
print("\n开始处理视频...")
duration, summary = bili.process_video(BV, notify_callback=bili.notify_qq)
ad_result = bili.load_ad_result(BV)
print(f"\n{'='*50}")
print(f"时长: {duration}")
print(f"总结: {summary}")
print(f"广告: {ad_result.get('reply_prefix', '未识别到插入广告')}")
print(f"{'='*50}")

# 3. 查找该视频的评论区，找到一条主评论回复
# 先获取视频的 oid（就是 aid）
if info:
    aid = info.get('aid')
    print(f"\n视频 aid: {aid}")
    
    # 获取评论区顶级评论
    comments = bili.fetch_top_level_comments(str(aid), 1)
    if comments:
        first_comment = comments[0]
        rpid = str(first_comment.get('rpid'))
        user = first_comment.get('member', {}).get('uname', '未知用户')
        content = first_comment.get('content', {}).get('message', '')
        print(f"目标评论: rpid={rpid}, 用户={user}")
        print(f"评论内容: {content[:80]}")
        
        # 4. 回复总结到这条评论下
        print(f"\n发送回复...")
        success, resp = bili.send_reply_with_audit_check(
            str(aid), rpid, rpid, summary, 1,
            notify_callback=bili.notify_qq
        )
        if success:
            print(f"✅ 回复成功!")
            state = resp.get('data', {}).get('state', '?')
            print(f"审核状态: state={state}")
        else:
            print(f"❌ 回复失败: {resp}")
    else:
        print("该视频暂无评论")
        # 直接用 reply_comment 需要有评论才能回复，没有评论就发顶级评论
        print("尝试发送顶级评论...")
        # B站 reply/add 发顶级评论: root=0, parent=0
        resp = bili.reply_comment(str(aid), "0", "0", summary, 1)
        print(f"结果: {json.dumps(resp, ensure_ascii=False)[:300]}")
