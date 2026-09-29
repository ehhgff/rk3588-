#!/usr/bin/env python3
"""
RK3588医疗RAG HTTP服务
通过ADB反向代理提供API服务
"""

import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid_medical_rag_v2 import HybridMedicalRAGv2

# 全局RAG实例
rag = None

class RAGHandler(BaseHTTPRequestHandler):
    """RAG请求处理器"""
    
    def log_message(self, format, *args):
        """自定义日志"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {args[0]}")
    
    def do_GET(self):
        """处理GET请求"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = '''
            <html>
            <head><title>RK3588医疗RAG服务</title></head>
            <body>
                <h1>RK3588医疗RAG服务</h1>
                <p>状态: 运行中</p>
                <p>API端点: POST /query</p>
                <form action="/query" method="post">
                    <input type="text" name="query" placeholder="输入医疗问题" size="50"/>
                    <input type="submit" value="查询"/>
                </form>
            </body>
            </html>
            '''
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'status': 'ok',
                'device': 'RK3588',
                'acceleration': 'RKNN NPU',
                'data_count': len(rag.dialogues) if rag else 0,
                'embedding_dim': rag.embedding_dim if rag else 0
            }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """处理POST请求"""
        if self.path == '/query':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            # 解析查询
            try:
                # 尝试JSON格式
                data = json.loads(post_data)
                query = data.get('query', '')
            except:
                # 尝试表单格式
                params = {}
                for param in post_data.split('&'):
                    if '=' in param:
                        k, v = param.split('=', 1)
                        params[k] = v
                query = params.get('query', '')
            
            if not query:
                self.send_response(400)
                self.end_headers()
                return
            
            # 执行查询
            start = time.time()
            results, predicted = rag.search(query, top_k=3, final_k=3, use_intent=True)
            elapsed = (time.time() - start) * 1000
            
            # 构建响应
            response = {
                'query': query,
                'predicted_department': predicted,
                'latency_ms': round(elapsed, 2),
                'results': []
            }
            
            for dialogue, score, source in results:
                response['results'].append({
                    'department': dialogue.department,
                    'title': dialogue.title,
                    'similarity': round(score, 4),
                    'source': source
                })
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode())
            
        else:
            self.send_response(404)
            self.end_headers()


def main():
    """主函数"""
    global rag
    
    print("=" * 70)
    print("RK3588医疗RAG HTTP服务")
    print("=" * 70)
    
    # 加载RAG
    print("\n加载RAG索引...")
    rag = HybridMedicalRAGv2(
        embedding_dim=768,
        index_path="/userdata/medical_rag/sbert_768_final"
    )
    
    if not rag.load():
        print("✗ 索引加载失败")
        return
    
    print(f"✓ 加载完成")
    print(f"  数据量: {len(rag.dialogues)}")
    print(f"  维度: {rag.embedding_dim}")
    
    # 启动HTTP服务
    port = 8081
    server = HTTPServer(('127.0.0.1', port), RAGHandler)
    
    print(f"\n✓ HTTP服务已启动")
    print(f"  地址: http://127.0.0.1:{port}")
    print(f"  通过ADB反向代理访问: http://localhost:{port}")
    print("\n按Ctrl+C停止服务")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n服务已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
