#!/usr/bin/env python3
"""
RK3588医疗RAG演示 - 过滤增强版
修复：
1. 反义词冲突过滤（高血压vs低血压）
2. 人群过滤（普通患者vs孕妇/儿童等）
3. 结果医学合理性验证
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '/userdata/medical_rag')

# 设置离线模式
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class FilteredDemoRAG:
    """过滤增强版RAG - 修复高血压/低血压混淆问题"""
    
    # 科室关键词
    DEPT_KEYWORDS = {
        '内科': ['高血压', '糖尿病', '血糖', '感冒', '发烧', '咳嗽', '胃痛', '胃疼', '胃炎', '胃溃疡', '心脏病', '血脂', '胆固醇', '甲亢', '低血压', '低血糖', '贫血', '中风', '脑梗', '心梗', '冠心病', '哮喘', '肺炎', '支气管炎', '肝硬化', '肝炎', '肾炎', '尿毒症', '甲减', '甲状腺'],
        '儿科': ['孩子', '小孩', '宝宝', '婴儿', '儿童', '幼儿', '退烧', '疫苗', '生长发育', '乳牙', '奶粉', '辅食', '新生儿', '早产儿'],
        '妇产科': ['孕妇', '怀孕', '孕期', '产妇', '分娩', '产后', '月经', '痛经', '妇科检查', '产检', '胎动', '预产期', '顺产', '剖腹产', '宫外孕', '流产', '不孕', '避孕', '月经不调', '白带异常'],
        '皮肤科': ['湿疹', '痤疮', '痘痘', '皮肤过敏', '皮炎', '瘙痒', '皮疹', '白癜风', '疱疹', '银屑病', '牛皮癣', '荨麻疹', '黄褐斑', '雀斑', '疤痕', '脱发', '白发'],
        '外科': ['骨折', '手术', '腰疼', '腿麻', '腰椎', '关节', '扭伤', '跌打', '痔疮', '阑尾炎', '胆结石', '肾结石', '肿瘤', '癌症', '切除', '缝合'],
        '心理科': ['失眠', '焦虑', '抑郁', '睡不着', '心理咨询', '精神', '压力', '情绪', '睡眠障碍', '强迫症', '恐惧症', '自闭症', '多动症', '精神分裂'],
        '五官科': ['眼睛', '眼科', '视力', '近视', '干眼症', '耳朵', '鼻子', '牙齿', '喉咙', '远视', '散光', '白内障', '青光眼', '耳鸣', '耳聋', '鼻炎', '鼻窦炎', '牙痛', '牙龈', '咽炎', '扁桃体炎'],
        '中医科': ['中药', '调理', '气虚', '血虚', '针灸', '拔罐', '上火', '体质', '湿气', '肝火', '肾虚', '脾虚', '气血不足', '经络', '穴位'],
        '男科': ['阳痿', '早泄', '前列腺', '肾虚', '精子', '性功能障碍', '包皮过长', '睾丸', '精索静脉曲张'],
        '传染病科': ['乙肝', '丙肝', '艾滋病', 'HIV', '梅毒', '结核', '流感', '新冠', '肺炎', '肝炎', '病毒', '细菌', '传染', '疫苗'],
        '肿瘤科': ['肿瘤', '癌症', '化疗', '放疗', '恶性', '良性', '转移', '复发', '靶向治疗', '免疫治疗'],
    }
    
    # 关键医学术语（带反义词）
    MEDICAL_KEYWORDS = {
        '高血压': {
            'synonyms': ['高血压', '血压高', '降压', '降血压', '控制血压', '血压控制', '高血压病', '原发性高血压', '高压高', '低压高'],
            'antonyms': ['低血压', '血压低'],
            'dept': '内科',
            'category': '心血管疾病'
        },
        '低血压': {
            'synonyms': ['低血压', '血压低', '升压', '血压偏低', '血压下降', '体位性低血压'],
            'antonyms': ['高血压', '血压高'],
            'dept': '内科',
            'category': '心血管疾病'
        },
        '糖尿病': {
            'synonyms': ['糖尿病', '血糖高', '高血糖', '控糖', '降糖', '胰岛素', '二型糖尿病', '2型糖尿病', '血糖控制', '糖耐量异常'],
            'antonyms': ['低血糖', '血糖低'],
            'dept': '内科',
            'category': '代谢疾病'
        },
        '低血糖': {
            'synonyms': ['低血糖', '血糖低', '升糖', '血糖偏低', '血糖下降', '低血糖反应'],
            'antonyms': ['高血糖', '糖尿病', '血糖高'],
            'dept': '内科',
            'category': '代谢疾病'
        },
        '失眠': {
            'synonyms': ['失眠', '睡不着', '睡眠障碍', '入睡困难', '早醒', '睡眠质量差', '难以入睡', '睡眠浅', '多梦'],
            'antonyms': ['嗜睡', '昏睡'],
            'dept': '心理科',
            'category': '睡眠障碍'
        },
        '皮肤过敏': {
            'synonyms': ['皮肤过敏', '皮炎', '皮疹', '瘙痒', '皮肤痒', '过敏', '荨麻疹', '湿疹', '风团'],
            'dept': '皮肤科',
            'category': '皮肤疾病'
        },
        '痤疮': {
            'synonyms': ['痤疮', '痘痘', '青春痘', '粉刺', '黑头', '白头', '闭口', '暗疮'],
            'dept': '皮肤科',
            'category': '皮肤疾病'
        },
        '儿童发烧': {
            'synonyms': ['儿童发烧', '孩子发烧', '宝宝发烧', '婴儿发烧', '小儿发烧', '儿童发热', '孩子发热', '发热', '发烧'],
            'dept': '儿科',
            'category': '儿科疾病'
        },
        '孕妇': {
            'synonyms': ['孕妇', '怀孕', '孕期', '产妇', '准妈妈', '妊娠期', '妊娠', '产检'],
            'dept': '妇产科',
            'category': '妇产科疾病'
        },
        '胃痛': {
            'synonyms': ['胃痛', '胃疼', '胃炎', '胃溃疡', '胃酸', '胃胀', '消化不良', '胃不适', '胃脘痛'],
            'dept': '内科',
            'category': '消化系统疾病'
        },
        '感冒': {
            'synonyms': ['感冒', '上呼吸道感染', '流感', '着凉', '风寒', '风热', '鼻塞', '流鼻涕', '打喷嚏', '发烧', '咳嗽'],
            'dept': '内科',
            'category': '呼吸系统疾病'
        },
        '心脏病': {
            'synonyms': ['心脏病', '冠心病', '心绞痛', '心梗', '心肌梗死', '心悸', '心律不齐', '房颤', '胸闷', '胸痛'],
            'dept': '内科',
            'category': '心血管疾病'
        },
    }
    
    # 特殊人群关键词（用于过滤）
    SPECIAL_POPULATIONS = {
        '孕妇': ['孕妇', '怀孕', '孕期', '产妇', '准妈妈', '妊娠期', '妊娠', '产检', '待产'],
        '儿童': ['儿童', '孩子', '小孩', '宝宝', '婴儿', '幼儿', '新生儿', '早产儿', '小儿'],
        '老年人': ['老人', '老年', '高龄', '岁数大', '年迈'],
    }
    
    def __init__(self):
        self.dialogues = []
        self.titles = []
        self.questions = []
        self.answers = []
        self.departments = []
        self.vector_index = None
        self.bm25 = None
        self.tokenized_corpus = []
        self.encoder = None
        self.encoder_loaded = False
        
    def predict_department(self, query: str) -> str:
        """预测查询意图科室"""
        query = query.lower()
        scores = {}
        
        # 先检查关键医学术语
        for term, info in self.MEDICAL_KEYWORDS.items():
            for syn in info['synonyms']:
                if syn in query:
                    dept = info['dept']
                    scores[dept] = scores.get(dept, 0) + 100
                    break
        
        # 检查科室关键词
        for dept, keywords in self.DEPT_KEYWORDS.items():
            for kw in keywords:
                if kw in query:
                    scores[dept] = scores.get(dept, 0) + 10 * len(kw)
        
        return max(scores, key=scores.get) if scores else None
    
    def detect_medical_terms(self, query_text: str):
        """检测查询中的关键医学术语"""
        detected = []
        query_lower = query_text.lower()
        
        for term, info in self.MEDICAL_KEYWORDS.items():
            matched_syn = None
            for syn in info['synonyms']:
                if syn in query_lower:
                    matched_syn = syn
                    break
            
            if matched_syn:
                detected.append({
                    'term': term,
                    'matched': matched_syn,
                    'synonyms': info['synonyms'],
                    'dept': info['dept'],
                    'antonyms': info.get('antonyms', [])
                })
        
        return detected
    
    def detect_population(self, text: str) -> list:
        """检测文本中的特殊人群"""
        populations = []
        text_lower = text.lower()
        
        for pop_name, keywords in self.SPECIAL_POPULATIONS.items():
            for kw in keywords:
                if kw in text_lower:
                    populations.append(pop_name)
                    break
        
        return populations
    
    def check_antonym_conflict(self, query_terms: list, result_text: str) -> bool:
        """检查反义词冲突"""
        result_lower = result_text.lower()
        
        for term_info in query_terms:
            antonyms = term_info.get('antonyms', [])
            for antonym in antonyms:
                if antonym in result_lower:
                    return True
        
        return False
    
    def load(self):
        """加载资源"""
        print("=" * 70)
        print("加载医疗RAG系统（过滤增强版）")
        print("=" * 70)
        
        # 加载数据
        print("\n[1/4] 加载医疗数据...")
        with open("/userdata/medical_rag/sbert_768_final.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.dialogues = data['dialogues']
        for d in self.dialogues:
            self.titles.append(d.get('title', ''))
            self.questions.append(d.get('question', ''))
            self.answers.append(d.get('answer', ''))
            self.departments.append(d.get('department', '未知'))
        
        print(f"  ✓ {len(self.dialogues)} 条对话")
        
        # 加载索引
        print("\n[2/4] 加载FAISS索引...")
        import faiss
        self.vector_index = faiss.read_index("/userdata/medical_rag/sbert_768_final_vector.faiss")
        print(f"  ✓ {self.vector_index.ntotal} 向量")
        
        print("\n[3/4] 加载BM25索引...")
        import pickle
        with open("/userdata/medical_rag/sbert_768_final_bm25.pkl", 'rb') as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data['bm25']
        self.tokenized_corpus = bm25_data['tokenized_corpus']
        print(f"  ✓ BM25加载完成")
        
        # 加载编码器
        print("\n[4/4] 加载SBERT编码器...")
        self._load_encoder()
        
        return True
    
    def _load_encoder(self):
        """加载编码器"""
        try:
            from sentence_transformers import SentenceTransformer
            
            cache_dir = "/root/.cache/huggingface/hub"
            model_path = f"{cache_dir}/models--shibing624--text2vec-base-chinese/snapshots"
            
            if Path(model_path).exists():
                snapshots = list(Path(model_path).glob("*"))
                if snapshots:
                    local_path = str(snapshots[0])
                    self.encoder = SentenceTransformer(local_path, device='cpu')
                else:
                    self.encoder = SentenceTransformer(
                        "shibing624/text2vec-base-chinese",
                        device='cpu',
                        cache_folder=cache_dir
                    )
            else:
                self.encoder = SentenceTransformer(
                    "shibing624/text2vec-base-chinese",
                    device='cpu',
                    cache_folder=cache_dir
                )
            
            self.encoder_loaded = True
            print("  ✓ SBERT编码器就绪")
        except Exception as e:
            print(f"  ✗ 编码器加载失败: {e}")
    
    def encode_query(self, query_text: str):
        """编码查询"""
        if not self.encoder_loaded:
            raise RuntimeError("编码器未加载")
        
        start = time.time()
        embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        encode_time = (time.time() - start) * 1000
        
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype('float32')
        
        import faiss
        faiss.normalize_L2(embedding)
        
        return embedding, encode_time
    
    def search_with_filter(self, query_text: str, k: int = 3):
        """
        带过滤的搜索（修复版）
        - 反义词冲突过滤
        - 人群匹配过滤
        """
        # 预测科室
        predicted_dept = self.predict_department(query_text)
        
        # 检测关键医学术语
        detected_terms = self.detect_medical_terms(query_text)
        
        # 检测查询人群
        query_populations = self.detect_population(query_text)
        
        if detected_terms:
            print(f"  检测到术语: {[t['term'] for t in detected_terms]}")
        if query_populations:
            print(f"  查询人群: {query_populations}")
        
        # 编码查询
        query_vector, encode_time = self.encode_query(query_text)
        
        # 向量搜索（获取更多候选用于过滤）
        t0 = time.time()
        vec_scores, vec_indices = self.vector_index.search(query_vector, k*6)
        vec_time = (time.time() - t0) * 1000
        
        # BM25搜索
        import jieba
        tokens = list(jieba.cut(query_text.lower()))
        tokens = [t for t in tokens if t.strip()]
        
        t0 = time.time()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_time = (time.time() - t0) * 1000
        
        bm25_top_k = np.argsort(bm25_scores)[-k*6:][::-1]
        
        # RRF融合
        combined = {}
        
        for rank, idx in enumerate(vec_indices[0]):
            if 0 <= idx < len(self.dialogues):
                combined[idx] = combined.get(idx, 0) + 0.6 / (rank + 1)
        
        for rank, idx in enumerate(bm25_top_k):
            if 0 <= idx < len(self.dialogues):
                combined[idx] = combined.get(idx, 0) + 0.4 / (rank + 1)
        
        # 意图增强
        if predicted_dept:
            for idx in list(combined.keys()):
                if self.departments[idx] == predicted_dept:
                    combined[idx] *= 1.2
        
        # 关键词增强 + 严格过滤
        filtered_results = []
        excluded_count = 0
        
        for idx in list(combined.keys()):
            boost = 1.0
            title_lower = self.titles[idx].lower()
            question_lower = self.questions[idx].lower()
            result_text = title_lower + " " + question_lower
            
            # 检查反义词冲突（严格过滤！）
            if self.check_antonym_conflict(detected_terms, result_text):
                print(f"  ✗ 排除反义词冲突: {self.titles[idx][:40]}...")
                excluded_count += 1
                del combined[idx]  # 完全删除，不保留
                continue
            
            # 人群匹配检查
            result_populations = self.detect_population(result_text)
            if query_populations and result_populations:
                # 查询有特定人群，结果也有特定人群，检查是否匹配
                if not any(pop in result_populations for pop in query_populations):
                    # 人群不匹配
                    print(f"  ⚠ 人群不匹配: {self.titles[idx][:40]}...")
                    combined[idx] *= 0.3
            elif not query_populations and result_populations:
                # 查询是普通患者，结果是特殊人群
                print(f"  ✗ 排除特殊人群结果: {self.titles[idx][:40]}...")
                excluded_count += 1
                del combined[idx]  # 完全删除
                continue
            
            # 同义词匹配提升
            for term_info in detected_terms:
                synonyms = term_info['synonyms']
                if any(syn in title_lower or syn in question_lower for syn in synonyms):
                    boost *= 1.5
            
            combined[idx] *= boost
            
            # 检查匹配的关键词
            matched_keywords = []
            for term_info in detected_terms:
                synonyms = term_info['synonyms']
                if any(syn in title_lower or syn in question_lower for syn in synonyms):
                    matched_keywords.append(term_info['term'])
            
            filtered_results.append({
                'idx': idx,
                'score': combined[idx],
                'keywords': matched_keywords
            })
        
        if excluded_count > 0:
            print(f"  共排除 {excluded_count} 个冲突结果")
        
        # 排序取Top-K
        filtered_results.sort(key=lambda x: -x['score'])
        top_results = filtered_results[:k]
        
        # 组装结果
        results = []
        for r in top_results:
            idx = r['idx']
            doc_vector = self.vector_index.reconstruct(int(idx))
            cosine_sim = float(np.dot(query_vector[0], doc_vector))
            cosine_sim = max(0.0, min(1.0, cosine_sim))
            
            results.append({
                'index': idx,
                'department': self.departments[idx],
                'title': self.titles[idx],
                'question': self.questions[idx],
                'answer': self.answers[idx][:100] + '...' if len(self.answers[idx]) > 100 else self.answers[idx],
                'similarity': cosine_sim,
                'matched_keywords': r['keywords'],
                'rrf_score': r['score'],
            })
        
        total_time = encode_time + vec_time + bm25_time
        
        return results, predicted_dept, detected_terms, {
            'encode': encode_time,
            'vector': vec_time,
            'bm25': bm25_time,
            'total': total_time
        }
    
    def run_demo(self):
        """运行演示"""
        test_queries = [
            ("高血压吃什么药", "内科"),
            ("感冒发烧怎么办", "内科"),
            ("胃痛怎么缓解", "内科"),
            ("糖尿病饮食注意事项", "内科"),
            ("失眠如何治疗", "心理科"),
            ("皮肤过敏怎么处理", "皮肤科"),
            ("儿童发烧怎么办", "儿科"),
            ("孕妇注意事项", "妇产科"),
        ]
        
        print("\n" + "=" * 70)
        print("医疗RAG演示（过滤增强版）")
        print(f"编码器: {'SBERT' if self.encoder_loaded else '不可用'}")
        print("=" * 70)
        
        correct = 0
        total_times = {'encode': 0, 'vector': 0, 'bm25': 0, 'total': 0}
        
        for query, expected in test_queries:
            print(f"\n{'='*70}")
            print(f"查询: {query} (期望: {expected})")
            print('='*70)
            
            results, predicted, detected_terms, times = self.search_with_filter(query, k=3)
            
            for key in times:
                total_times[key] += times[key]
            
            print(f"预测科室: {predicted}")
            print(f"耗时: 编码{times['encode']:.1f}ms + 向量{times['vector']:.1f}ms + BM25{times['bm25']:.1f}ms = 总计{times['total']:.1f}ms\n")
            
            if results:
                top1 = results[0]
                is_correct = (top1['department'] == expected)
                if is_correct:
                    correct += 1
                
                marker = "✓" if is_correct else "✗"
                keywords_str = f" [关键词:{','.join(top1['matched_keywords'])}]" if top1['matched_keywords'] else ""
                print(f"Top-1 [{marker}] {top1['department']} | "
                      f"相似度:{top1['similarity']:.3f}{keywords_str}")
                print(f"      {top1['title'][:50]}...")
                
                for i, r in enumerate(results, 1):
                    kw_str = f" [{','.join(r['matched_keywords'])}]" if r['matched_keywords'] else ""
                    print(f"  [{i}] {r['department']:8} | sim:{r['similarity']:.3f}{kw_str}")
                    print(f"      {r['title'][:45]}...")
        
        # 统计
        print("\n" + "=" * 70)
        print("统计结果")
        print("=" * 70)
        n = len(test_queries)
        print(f"科室准确率: {correct}/{n} = {correct/n*100:.1f}%")
        print(f"\n平均耗时:")
        print(f"  编码: {total_times['encode']/n:.1f}ms")
        print(f"  向量检索: {total_times['vector']/n:.1f}ms")
        print(f"  BM25检索: {total_times['bm25']/n:.1f}ms")
        print(f"  总计: {total_times['total']/n:.1f}ms")
        print(f"QPS: {n/(total_times['total']/1000):.1f}")


def main():
    print("=" * 70)
    print("RK3588医疗RAG演示（过滤增强版）")
    print("=" * 70)
    
    rag = FilteredDemoRAG()
    if not rag.load():
        return 1
    
    rag.run_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
