#!/usr/bin/env python3
"""
优化版 RAG 服务器 - 使用 RKNN + 混合搜索 + 科室过滤
提供 Unix Socket 接口
"""

import os
import sys
import json
import time
import socket
import threading
import pickle
import numpy as np
from collections import defaultdict

# 添加医疗RAG目录到路径
sys.path.insert(0, '/userdata/medical_rag_full')

# 导入 RAGServiceONNX 类（避免执行 __main__）
# 通过 exec 导入特定的类定义
exec(open('/userdata/medical_rag_full/rag_service_onnx_optimized.py').read().split('if __name__')[0])

# 现在 RAGServiceONNX 类已经定义

# 科室关键词映射（纯口语版 - 仅用于日常对话分诊）
DEPT_KEYWORDS = {
    '内科': [
        '内科', '感冒', '发烧', '发热', '咳嗽', '咳痰', '嗓子疼', '流鼻涕', '鼻塞', '头疼', '头晕', 
        '肚子疼', '胃疼', '胃胀', '胃酸', '烧心', '反酸', '恶心', '想吐', '干呕', '呕吐', '拉肚子', 
        '腹泻', '肚子胀', '不消化', '没胃口', '吃不下', '便秘', '大便干', '几天没拉', 
        '胸闷', '心慌', '心跳快', '心里不舒服', '喘不上气', '气短', '气不够用', '憋得慌', 
        '血压高', '高血压', '血压低', '血糖高', '糖尿病', '血糖低', '头晕眼花', '眼前发黑', 
        '浑身没劲', '乏力', '累', '没精神', '发虚', '出虚汗', '盗汗', '水肿', '腿肿', '脚肿', '脸肿', 
        '眼皮肿', '尿少', '尿多', '尿频', '尿急', '尿痛', '腰酸', '腰疼', '小便有泡沫', '小便发红', 
        '贫血', '脸色白', '嘴唇白', '痛风', '脚趾头疼', '关节红肿', '尿酸高', '血脂高', '脂肪肝', 
        '甲状腺', '甲亢', '甲减', '脖子粗', '爱出汗', '怕热', '怕冷'
    ],

    '外科': [
        '外科', '摔了', '碰了', '磕了', '扭了', '伤了', '破皮', '流血', '伤口', '口子', '划口子', 
        '缝针', '拆线', '换药', '包扎', '狗咬了', '猫抓了', '被钉子扎了', '烧伤', '烫伤', '起泡', 
        '跌打损伤', '崴脚', '脚扭了', '腰扭了', '闪了腰', '落枕', '脖子不能动', 
        '疙瘩', '包块', '肿块', '硬块', '囊肿', '粉瘤', '脂肪瘤', '身上长了个东西', 
        '痔疮', '便血', '大便带血', '屁股疼', '肛门疼', '肛门痒', '肛门口有肉球', 
        '疝气', '小肠气', '肚子鼓包', '阑尾炎', '肚子右下方疼', '胆囊炎', '胆结石', '后背疼', 
        '肾结石', '腰疼得打滚', '小便疼', '尿不出来', '排尿费劲', '尿线细', 
        '静脉曲张', '小腿青筋', '蚯蚓腿', '腿发胀', '腿沉', '包皮', '包皮长', 
        '甲状腺结节', '脖子有疙瘩', '乳腺增生', '乳房胀痛', '乳房有块', '腋下有疙瘩'
    ],

    '儿科': [
        '儿科', '小孩', '孩子', '宝宝', '婴儿', '幼儿', '儿童', '新生儿', '月子里的娃', 
        '发烧', '发热', '身上烫', '咳嗽', '流鼻涕', '鼻塞', '鼻子不通气', '打喷嚏', 
        '呕吐', '吐奶', '溢奶', '拉肚子', '拉稀', '拉水', '大便有沫', '大便发绿', '便秘', '大便干', 
        '不吃饭', '厌食', '积食', '肚子胀', '肚子疼', '哭闹', '夜里哭', '不睡觉', '睡不踏实', 
        '出疹子', '起疙瘩', '身上红点', '湿疹', '口水疹', '红屁股', '尿布疹', 
        '打疫苗', '预防针', '水痘', '手足口', '腮腺炎', '长痄腮', 
        '不长个', '发育慢', '走路晚', '说话晚', '尿床', '尿裤子'
    ],

    '妇产科': [
        '妇科', '产科', '妇产科', '怀孕', '有喜', '大肚子', '孕妇', '产妇', '坐月子', 
        '没来月经', '月经推迟', '月经不调', '例假不准', '大姨妈', '来事了', 
        '痛经', '肚子疼', '腰酸', '月经量少', '月经量多', '经期长', '下面流血', 
        '白带', '白带多', '白带黄', '白带有味', '下面痒', '外阴痒', '同房出血', 
        '备孕', '要孩子', '怀不上', '不孕', '流产', '小产', '药流', '人流', '刮宫', 
        '早孕反应', '孕吐', '吐得厉害', '胎动', '肚子发紧', '宫缩', '见红', '破水', 
        '顺产', '剖腹产', '剖宫产', '侧切', '撕裂', '恶露', '奶水不足', '堵奶', '乳腺炎', 
        '更年期', '心烦', '出汗', '潮热', '失眠', '子宫肌瘤', '卵巢囊肿', '盆腔炎'
    ],

    '皮肤科': [
        '皮肤科', '皮肤', '起疙瘩', '长包', '长痘', '痘痘', '粉刺', '痤疮', '闭口', 
        '痒', '刺挠', '痒得难受', '皮肤痒', '身上痒', '头皮痒', 
        '过敏', '起疹子', '红疹', '风团', '风疙瘩', '荨麻疹', '起泛片', 
        '湿疹', '皮炎', '脱皮', '掉皮', '干裂', '手裂口子', 
        '癣', '脚气', '烂脚丫', '灰指甲', '指甲变厚', '指甲发黄', 
        '脱发', '掉头发', '斑秃', '鬼剃头', '头皮屑多', 
        '长瘊子', '刺瘊', '扁平疣', '鸡眼', '脚上长硬疙瘩', 
        '痣', '痦子', '黑点', '老年斑', '雀斑', '黄褐斑', '蝴蝶斑', 
        '带状疱疹', '缠腰龙', '转腰龙', '蛇盘疮', '单纯疱疹', '嘴上起泡', 
        '白癜风', '白斑', '白块', '银屑病', '牛皮癣'
    ],

    '眼科': [
        '眼科', '眼睛', '眼', '视力', '看不清', '模糊', '眼花', 
        '近视', '近视眼', '远视', '老花', '老花眼', '散光', 
        '眼干', '眼涩', '眼磨', '眼里有东西', '沙子磨眼', 
        '流泪', '迎风流泪', '眼屎多', '眼眵多', '眼睛黏糊', 
        '眼红', '白眼珠红', '红眼病', '眼睛痒', '怕光', '见光睁不开眼', 
        '麦粒肿', '针眼', '眼皮长疙瘩', '霰粒肿', '眼皮肿', '眼袋', 
        '白内障', '青光眼', '眼压高', '飞蚊症', '眼前有黑影', '眼前有黑点', 
        '重影', '看东西双影', '斜眼', '斗鸡眼', '对眼', '弱视', '眨眼睛', '眼皮跳'
    ],

    '耳鼻喉科': [
        '耳鼻喉', '五官科', '耳朵', '鼻子', '喉咙', '嗓子', 
        '耳朵疼', '耳朵痒', '耳朵堵', '耳朵闷', '耳背', '听不清', '耳朵聋', 
        '耳鸣', '耳朵响', '耳朵嗡嗡响', '耳朵流脓', '耳朵流水', '中耳炎', 
        '掏耳朵', '耳屎堵了', '耳朵里长东西', '晕', '头晕', '天旋地转', '晕车', 
        '鼻子堵', '鼻子不通气', '流鼻涕', '打喷嚏', '鼻炎', '过敏性鼻炎', 
        '鼻子干', '鼻子痒', '流鼻血', '鼻出血', '擤鼻涕带血', '闻不到味', 
        '嗓子疼', '喉咙痛', '咽炎', '咽喉炎', '扁桃体发炎', '扁桃体肿大', 
        '嗓子干', '嗓子痒', '想咳嗽', '清嗓子', '有痰', '咳不出咽不下', 
        '嗓子有东西', '异物感', '声音哑', '说不出话', '失声', '打呼噜', '打鼾', 
        '睡觉憋气', '呼吸暂停', '腺样体肥大', '小孩张嘴呼吸'
    ],

    '口腔科': [
        '口腔科', '牙科', '牙齿', '嘴', 
        '牙疼', '牙痛', '蛀牙', '虫牙', '牙洞', '塞牙', 
        '牙酸', '怕冷怕热', '吃凉的疼', '吃热的疼', 
        '牙龈出血', '刷牙出血', '牙龈肿', '牙床子肿', '牙花子肿', 
        '牙松了', '牙齿松动', '长智齿', '尽头牙疼', '拔牙', '补牙', '镶牙', '洗牙', 
        '牙结石', '牙黄', '口臭', '嘴里有味', 
        '口腔溃疡', '嘴里起泡', '烂嘴', '舌头起泡', '舌头疼', 
        '嘴唇干', '裂口子', '嘴角烂', '地图舌', '舌苔厚', '吃东西没味', 
        '挂钩疼', '腮帮子疼', '张嘴响', '下巴掉了', '嘴张不开'
    ],

    '精神科': [
        '精神科', '抑郁', '抑郁症', '焦虑', '焦虑症', '想不开', '情绪低落', 
        '高兴不起来', '没兴趣', '不想活', '想死', '有轻生念头', '自残', '伤害自己', 
        '睡不着', '失眠', '睡不好', '整宿不睡', '早醒', '入睡困难', 
        '疑心重', '多疑', '觉得有人害我', '觉得被跟踪', '幻觉', '幻听', '听见有人说话', 
        '妄想', '胡思乱想', '停不下来', '强迫', '反复洗手', '反复检查', 
        '害怕', '恐惧', '不敢出门', '怕见人', '社恐', '心慌手抖', '紧张', 
        '发脾气', '暴躁', '易怒', '控制不住情绪', '精神分裂', '行为怪异', '胡言乱语'
    ],

    '心理科': [
        '心理科', '心理咨询', '心理', '心里堵得慌', '郁闷', '烦躁', '憋屈', 
        '压力大', '心累', '想不通', '纠结', '心事重', '心情不好', 
        '感情问题', '婚姻问题', '夫妻吵架', '婆媳矛盾', '亲子关系', '孩子不听话', 
        '工作压力', '学习压力', '厌学', '不想上学', '考试紧张', 
        '没自信', '自卑', '内向', '不爱说话', '怕见生人', '社交困难', 
        '失恋', '离婚', '丧亲', '心里难受', '想哭', '觉得没意思'
    ],

    '中医科': [
        '中医科', '中医', '中药', '号脉', '把脉', '看舌苔', 
        '调理', '调理身体', '补一补', '去去火', '上火', '火大', '有火', 
        '体虚', '身子虚', '气虚', '血虚', '气血不足', '阳虚', '阴虚', 
        '肾虚', '肝火旺', '湿气重', '有湿气', '寒气大', '胃寒', '宫寒', 
        '气血不通', '血瘀', '经络不通', '身上疼', '酸懒', '不舒服', 
        '针灸', '扎针', '拔罐', '拔火罐', '刮痧', '推拿', '按摩', '艾灸', '烤电'
    ],

    '骨科': [
        '骨科', '骨头', '骨伤', 
        '脖子疼', '颈椎', '颈椎病', '肩膀疼', '肩周炎', '五十肩', '胳膊抬不起来', 
        '腰疼', '腰椎', '腰间盘', '腰突', '坐骨神经痛', '屁股疼', '腿麻', 
        '背疼', '后背疼', '脊柱', '驼背', '侧弯', 
        '膝盖疼', '腿疼', '关节疼', '关节炎', '关节肿', '有积液', 
        '骨质增生', '骨刺', '骨质疏松', '骨头脆', 
        '骨折', '骨裂', '摔骨折', '扭伤', '拉伤', '肌肉拉伤', '筋疼', 
        '脚后跟疼', '足跟痛', '脚底板疼', '拇外翻', '大脚骨', '脚趾变形'
    ],

    '肿瘤科': [
        '肿瘤科', '肿瘤', '癌', '癌症', '长瘤子', 
        '肿块', '疙瘩', '结节', '占位', '阴影', 
        '化疗', '放疗', '烤电', '掉头发', '吃不下饭', '癌痛', 
        '肺癌', '胃癌', '肝癌', '肠癌', '乳腺癌', '宫颈癌', '淋巴瘤', 
        '良性', '恶性', '转移', '扩散', '复发', '晚期'
    ],

    '神经科': [
        '神经科', '神经内科', 
        '头疼', '头痛', '偏头痛', '头晕', '头昏', '迷糊', '不清醒', 
        '癫痫', '羊角风', '抽风', '抽了', '口吐白沫', 
        '帕金森', '手抖', '头晃', '身子颤', '拿东西抖', 
        '面瘫', '口眼歪斜', '嘴歪眼斜', '面神经炎', 
        '脑梗', '脑血栓', '脑出血', '中风', '半身不遂', '偏瘫', 
        '手脚麻', '胳膊麻', '腿麻', '半边身子麻', '没知觉', 
        '记性差', '爱忘事', '脑子糊涂', '痴呆', '老年痴呆', 
        '三叉神经痛', '脸疼', '牙床子疼（非牙齿）', '神经痛', '坐骨神经痛'
    ],

    '心血管科': [
        '心血管科', '心脏科', 
        '心脏病', '心脏不好', '冠心病', '心绞痛', '心梗', '心肌梗死', 
        '胸痛', '胸口疼', '胸闷', '心口窝疼', '心慌', '心悸', '心里咯噔一下', 
        '心律不齐', '早搏', '房颤', '心跳快', '心跳慢', '心脏乱跳', 
        '心衰', '心力衰竭', '气不够用', '喘', '躺不平', 
        '高血压', '血压高', '高血脂', '血脂稠', '动脉硬化', '血管堵了', 
        '做过支架', '搭桥', '安了起搏器'
    ],

    '消化科': [
        '消化科', '消化内科', '胃肠科', 
        '胃疼', '胃痛', '胃胀', '胃酸', '烧心', '反酸', '打嗝', '嗳气', 
        '恶心', '干呕', '呕吐', '没胃口', '不消化', '积食', 
        '肚子疼', '腹痛', '腹胀', '肚子咕噜叫', '拉肚子', '腹泻', '拉水', '拉稀', 
        '便秘', '大便干', '大便费力', '好几天不拉', '大便不规律', 
        '胃炎', '胃溃疡', '肠炎', '结肠炎', '痔疮出血', 
        '脂肪肝', '酒精肝', '肝硬化', '肝炎', '黄疸', '眼黄皮肤黄', 
        '胰腺炎', '肚子剧痛', '胆结石', '胆囊炎'
    ],

    '呼吸科': [
        '呼吸科', '呼吸内科', 
        '感冒', '发烧', '发热', '咳嗽', '咳痰', '痰多', '黄痰', '白痰', '干咳', 
        '嗓子痒', '咽炎', '支气管炎', '气管不好', '老慢支', 
        '哮喘', '喘不上气', '喘鸣', '气短', '呼吸困难', '憋气', 
        '肺炎', '肺气肿', '慢阻肺', '肺大泡', 
        '胸痛', '胸口疼', '咳血', '痰中带血', '打呼噜', '睡觉憋气'
    ],

    '传染病科': [
        '传染科', '感染科', 
        '发烧', '发热', '发冷', '打摆子', 
        '肝炎', '乙肝', '大三阳', '小三阳', '丙肝', '黄疸', 
        '结核', '肺痨', '咳嗽', '咳血', '盗汗', '低烧', 
        '流感', '新冠', '阳性', '病毒感染', '细菌感染', 
        '拉肚子', '痢疾', '拉脓拉血', '肠炎', 
        '出疹子', '水痘', '麻疹', '风疹', '手足口病', '腮腺炎', '猩红热', 
        '被狗咬了', '狂犬病', '破伤风', '艾滋病'
    ],

    '整形美容': [
        '整形', '美容', '整容', 
        '双眼皮', '剌双眼皮', '开眼角', '隆鼻', '垫鼻子', 
        '除皱', '去皱', '打除皱针', '瘦脸针', '玻尿酸', '填充', 
        '抽脂', '吸脂', '减肥', '隆胸', '丰胸', '假体', 
        '去疤', '疤痕', '疤', '祛斑', '去痣', '点痦子', 
        '植发', '种头发', '发际线', '脱毛', '腋臭', '狐臭', '去腋臭'
    ]
}

class RAGServiceDeptOptimized(RAGServiceONNX):
    """科室分类优化的RAG服务 - 继承原始服务"""
    
    def __init__(self, data_dir=None):
        # 先调用父类初始化
        super().__init__(data_dir)
        
        # 加载科室索引
        self.dept_indices = None
        self.dept_doc_maps = None
        self._load_dept_index()
        
        # 预加载 tokenizer，避免首次查询延迟
        self._preload_tokenizer_for_search()
    
    def _load_dept_index(self):
        """加载科室分类索引"""
        index_path = f"{self.data_dir}/dept_vector_index.pkl"
        
        if os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.dept_indices = data['dept_indices']
                    self.dept_doc_maps = data['dept_doc_maps']
                print(f"[RAG-科室优化] 加载 {len(self.dept_indices)} 个科室索引", file=sys.stderr)
            except Exception as e:
                print(f"[RAG-科室优化] 加载科室索引失败: {e}", file=sys.stderr)
                self.dept_indices = None
    
    def _preload_tokenizer_for_search(self):
        """预加载 tokenizer 用于关键词搜索（复用已有 tokenizer）"""
        try:
            start = time.time()
            # 使用已有的 tokenizer 进行分词预热
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # 预热 tokenizer
                _ = self.tokenizer.encode("预热分词器")
                elapsed = (time.time() - start) * 1000
                print(f"[RAG-科室优化] Tokenizer 预加载完成，耗时: {elapsed:.1f}ms", file=sys.stderr)
            else:
                print(f"[RAG-科室优化] Tokenizer 未就绪，将使用 jieba", file=sys.stderr)
        except Exception as e:
            print(f"[RAG-科室优化] Tokenizer 预加载失败: {e}，将使用 jieba", file=sys.stderr)
    
    def _keyword_search(self, query, top_k=200):
        """关键词搜索 - 使用已有 tokenizer 或 jieba"""
        from collections import defaultdict
        
        # 优先使用 jieba 进行分词，避免 tokenizer 问题
        import jieba
        query_words = list(jieba.cut(query.lower()))
        
        query_keywords = [w for w in query_words if len(w) >= 2]
        
        print(f"[RAG-关键词搜索] 查询关键词: {query_keywords}", file=sys.stderr)
        
        doc_scores = defaultdict(int)
        for word in query_keywords:
            for doc_id in self.keyword_index.get(word, set()):
                doc_scores[doc_id] += 1
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        print(f"[RAG-关键词搜索] 找到 {len(sorted_docs)} 个文档", file=sys.stderr)
        
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]
    
    def _detect_departments(self, query):
        """从查询中检测可能的科室"""
        query_lower = query.lower()
        matched_depts = []
        
        for dept, keywords in DEPT_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    matched_depts.append(dept)
                    break
        
        return matched_depts if matched_depts else None
    
    def _hybrid_search_with_dept(self, query_text, k=3, client_depts=None):
        """混合搜索 - 科室优化版"""
        import time
        total_start = time.time()
        
        # Step 1: 检测可能的科室
        t1 = time.time()
        matched_depts = client_depts if client_depts else self._detect_departments(query_text)
        t1_elapsed = (time.time() - t1) * 1000
        if matched_depts:
            print(f"[RAG-科室优化] 检测科室: {matched_depts}", file=sys.stderr)
        
        # Step 2: 关键词搜索召回候选集（优化：减少数量加快检索）
        t2 = time.time()
        top_k = 50 if matched_depts else 50  # 增加候选数量，确保能找到目标科室的文档
        candidate_ids = self._keyword_search(query_text, top_k=top_k)
        t2_elapsed = (time.time() - t2) * 1000
        
        if not candidate_ids:
            # 退化为纯向量搜索
            t3 = time.time()
            query_vector = self.encode_query(query_text)
            t3_elapsed = (time.time() - t3) * 1000
            print(f"[RAG-耗时统计] 科室检测: {t1_elapsed:.1f}ms, 关键词搜索: {t2_elapsed:.1f}ms, 向量编码: {t3_elapsed:.1f}ms", file=sys.stderr)
            if query_vector is None:
                return None
            vec_scores, vec_indices = self.vector_index.search(query_vector, k)
            results = []
            for rank, idx in enumerate(vec_indices[0]):
                if 0 <= idx < len(self.dialogues):
                    cosine_sim = float(vec_scores[0][rank])
                    cosine_sim = max(0.0, min(1.0, cosine_sim))
                    results.append({
                        'department': self.departments[idx],
                        'title': self.titles[idx],
                        'question': self.questions[idx],
                        'answer': self.answers[idx],
                        'similarity': cosine_sim
                    })
            return results[:k]
        
        # Step 3: 编码查询
        t3 = time.time()
        query_vector = self.encode_query(query_text)
        t3_elapsed = (time.time() - t3) * 1000
        if query_vector is None:
            return None
        
        # Step 4: 按科室分组，优先搜索匹配的科室
        dept_candidates = defaultdict(list)
        for doc_id in candidate_ids:
            dept = self.departments[doc_id]
            dept_candidates[dept].append(doc_id)
        
        # 打印候选文档的科室分布
        print(f"[RAG-科室优化] 候选文档科室分布: {dict(dept_candidates)}", file=sys.stderr)
        
        all_results = []
        
        # 优先搜索匹配的科室
        search_depts = matched_depts if matched_depts else list(dept_candidates.keys())
        
        print(f"[RAG-科室优化] 搜索顺序: {search_depts}", file=sys.stderr)
        
        for dept in search_depts:
            if dept not in dept_candidates:
                continue
            
            docs_in_dept = dept_candidates[dept]
            
            # 使用科室子索引进行精排
            if self.dept_indices and dept in self.dept_indices:
                dept_index = self.dept_indices[dept]
                dept_doc_map = self.dept_doc_maps[dept]
                
                # 找出候选在子索引中的位置
                valid_indices = []
                valid_doc_ids = []
                for doc_id in docs_in_dept:
                    try:
                        idx_in_sub = dept_doc_map.index(doc_id)
                        valid_indices.append(idx_in_sub)
                        valid_doc_ids.append(doc_id)
                    except ValueError:
                        continue
                
                if valid_indices:
                    # 提取候选向量并计算相似度
                    try:
                        candidate_vectors = np.array([dept_index.reconstruct(int(i)) for i in valid_indices])
                        similarities = np.dot(candidate_vectors, query_vector.T).flatten()
                        
                        # 收集结果
                        for i, sim in enumerate(similarities):
                            all_results.append((valid_doc_ids[i], float(sim), dept))
                    except Exception as e:
                        print(f"[RAG-科室优化] 科室精排失败 {dept}: {e}", file=sys.stderr)
            else:
                # 回退：使用完整索引提取向量
                for doc_id in docs_in_dept:
                    try:
                        vec = self.vector_index.reconstruct(int(doc_id))
                        sim = float(np.dot(vec, query_vector.T))
                        all_results.append((doc_id, sim, dept))
                    except:
                        continue
        
        # 按相似度排序（目标科室优先）
        t4 = time.time()
        # 为目标科室的结果添加权重
        weighted_results = []
        dept_results = {}
        
        # 首先收集目标科室的结果
        for doc_id, sim, dept in all_results:
            if matched_depts and dept in matched_depts:
                if dept not in dept_results:
                    dept_results[dept] = []
                dept_results[dept].append((doc_id, sim, dept))
        
        # 然后添加非目标科室的结果
        for doc_id, sim, dept in all_results:
            if not (matched_depts and dept in matched_depts):
                if dept not in dept_results:
                    dept_results[dept] = []
                dept_results[dept].append((doc_id, sim, dept))
        
        # 构建最终结果列表，优先目标科室
        for dept in list(dept_results.keys()):
            # 目标科室的结果权重增加 0.5（进一步增加权重）
            weight = 0.5 if matched_depts and dept in matched_depts else 0
            for doc_id, sim, dept in dept_results[dept]:
                weighted_sim = sim + weight
                weighted_results.append((doc_id, weighted_sim, dept))
        
        # 按加权相似度排序
        weighted_results.sort(key=lambda x: -x[1])
        t4_elapsed = (time.time() - t4) * 1000
        
        # 取top-k
        results = []
        for doc_id, sim, dept in weighted_results[:k]:
            # 添加调试信息
            try:
                # 尝试将 doc_id 转换为整数
                int_doc_id = int(doc_id)
                if 0 <= int_doc_id < len(self.departments):
                    department = self.departments[int_doc_id]
                    title = self.titles[int_doc_id]
                    question = self.questions[int_doc_id]
                    answer = self.answers[int_doc_id]
                else:
                    # 如果 doc_id 超出范围，使用默认值
                    department = dept
                    title = ""
                    question = ""
                    answer = ""
            except (ValueError, IndexError) as e:
                # 如果 doc_id 不是整数或超出范围，使用默认值
                print(f"[RAG-调试] doc_id 处理错误: {e}, doc_id: {doc_id}, 类型: {type(doc_id)}", file=sys.stderr)
                # 尝试使用原始的 doc_id 作为索引（如果是字符串类型的数字）
                try:
                    # 尝试将 doc_id 作为字符串索引
                    if isinstance(doc_id, str) and doc_id.isdigit():
                        int_doc_id = int(doc_id)
                        if 0 <= int_doc_id < len(self.departments):
                            department = self.departments[int_doc_id]
                            title = self.titles[int_doc_id]
                            question = self.questions[int_doc_id]
                            answer = self.answers[int_doc_id]
                        else:
                            department = dept
                            title = ""
                            question = ""
                            answer = ""
                    else:
                        # 尝试使用 department 作为索引，查找该科室的第一个文档
                        print(f"[RAG-调试] 尝试使用科室 {dept} 查找文档", file=sys.stderr)
                        # 遍历所有文档，找到第一个属于该科室的文档
                        for i, dept_name in enumerate(self.departments):
                            if dept_name == dept:
                                department = dept_name
                                title = self.titles[i]
                                question = self.questions[i]
                                answer = self.answers[i]
                                print(f"[RAG-调试] 找到科室 {dept} 的文档: {question[:20]}...", file=sys.stderr)
                                break
                        else:
                            # 如果没有找到该科室的文档，使用默认值
                            department = dept
                            title = ""
                            question = ""
                            answer = ""
                except Exception as e2:
                    print(f"[RAG-调试] 二次处理错误: {e2}", file=sys.stderr)
                    department = dept
                    title = ""
                    question = ""
                    answer = ""
            
            results.append({
                'department': department,
                'title': title,
                'question': question,
                'answer': answer,
                'similarity': sim
            })
        
        total_elapsed = (time.time() - total_start) * 1000
        print(f"[RAG-耗时统计] 科室检测: {t1_elapsed:.1f}ms, 关键词搜索: {t2_elapsed:.1f}ms, 向量编码: {t3_elapsed:.1f}ms, 排序: {t4_elapsed:.1f}ms, 总计: {total_elapsed:.1f}ms", file=sys.stderr)
        
        return results
    
    def search(self, query_text, k=2, client_depts=None):
        """搜索 - 使用科室优化版混合搜索"""
        if not self.loaded:
            return None
        
        # 如果启用混合搜索且科室索引已加载，使用科室优化版
        if self._use_hybrid_search and self.keyword_index is not None and self.dept_indices is not None:
            return self._hybrid_search_with_dept(query_text, k, client_depts)
        
        # 回退到父类的混合搜索
        return super().search(query_text, k)


class RAGOptimizedServer:
    """优化版 RAG 服务器"""
    
    def __init__(self, socket_path='/tmp/rag_optimized.sock'):
        self.socket_path = socket_path
        self.rag_service = None
        self.running = False
        
    def start(self):
        """启动服务器"""
        print("[RAG优化服务器] 启动中...")
        
        # 加载 RAG 服务（使用科室优化版）
        start_time = time.time()
        self.rag_service = RAGServiceDeptOptimized('/userdata/medical_rag_full')
        load_time = (time.time() - start_time) * 1000
        
        if not self.rag_service.loaded:
            print("[RAG优化服务器] 服务加载失败!")
            return False
        
        print(f"[RAG优化服务器] 服务加载完成，耗时: {load_time:.1f}ms")
        print(f"[RAG优化服务器] 数据量: {len(self.rag_service.dialogues)} 条")
        
        # 清理旧 socket
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        # 创建 socket
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.socket_path)
        self.server.listen(5)
        os.chmod(self.socket_path, 0o777)
        self.running = True
        
        print(f"[RAG优化服务器] 监听: {self.socket_path}")
        
        # 主循环
        try:
            while self.running:
                conn, _ = self.server.accept()
                threading.Thread(target=self._handle_client, args=(conn,)).start()
        except KeyboardInterrupt:
            print("\n[RAG优化服务器] 停止中...")
        finally:
            self.stop()
    
    def _handle_client(self, conn):
        """处理客户端请求"""
        try:
            conn.settimeout(30)
            data = conn.recv(8192).decode('utf-8')
            
            if not data:
                return
            
            request = json.loads(data)
            action = request.get('action', '')
            
            if action == 'ping':
                response = {'status': 'ok', 'message': 'pong'}
            
            elif action == 'search':
                query = request.get('query', '')
                k = request.get('k', 2)
                departments = request.get('departments', None)
                
                start_time = time.time()
                results = self.rag_service.search(query, k=k, client_depts=departments)
                search_time = (time.time() - start_time) * 1000
                
                response = {
                    'status': 'ok',
                    'data': {
                        'results': results if results else [],
                        'time_ms': search_time,
                        'hybrid_mode': True,
                        'dept_optimized': self.rag_service.dept_indices is not None
                    }
                }
            
            elif action == 'prompt':
                query = request.get('query', '')
                k = request.get('k', 2)
                
                results = self.rag_service.search(query, k=k)
                prompt = self.rag_service.format_prompt(query, results)
                
                response = {
                    'status': 'ok',
                    'data': {
                        'query': query,
                        'prompt': prompt,
                        'results': results
                    }
                }
            
            else:
                response = {'status': 'error', 'message': f'未知操作: {action}'}
            
            conn.send(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            error_response = {'status': 'error', 'message': str(e)}
            try:
                conn.send(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            conn.close()
    
    def stop(self):
        """停止服务器"""
        self.running = False
        try:
            self.server.close()
        except:
            pass
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        print("[RAG优化服务器] 已停止")


if __name__ == '__main__':
    server = RAGOptimizedServer()
    server.start()
