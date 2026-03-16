import os
import sys

# --- 针对 D 盘路径的 GPU 补丁自动修复 ---
if os.name == 'nt':
    # 获取当前 venv 的 site-packages 路径
    venv_path = os.path.join(os.getcwd(), 'venv', 'Lib', 'site-packages')
    nvidia_dir = os.path.join(venv_path, 'nvidia')
    
    if os.path.exists(nvidia_dir):
        # 遍历所有可能的 dll 目录
        for root, dirs, files in os.walk(nvidia_dir):
            if 'bin' in dirs:
                bin_path = os.path.join(root, 'bin')
                os.add_dll_directory(bin_path)
                print(f"✅ 已手动加载 GPU 依赖库: {bin_path}")
# ---------------------------------------

from moviepy import VideoFileClip
from faster_whisper import WhisperModel
import jieba
from collections import Counter
import cv2
from fpdf import FPDF
from datetime import datetime
import jieba.analyse 
from tqdm import tqdm
from openai import OpenAI
import json

#配置
VIDEO_DIR = "videos"
AUDIO_DIR = "audios"
SCREENSHOT_DIR = "screenshots"
RESULT_DIR = "results"
VIDEO_PATH = os.path.join(VIDEO_DIR, "yhkt.mp4")  
AUDIO_PATH = os.path.join(AUDIO_DIR, "audio.wav")  
KEYWORDS = ["考试","会出","期末","考过","必考","出题","会考","出过"]  
MODEL_SIZE = "small"
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf" 
STOP_WORDS = ["我们", "这个", "那个", "然后", "就是", "这样", "好了", "同学们", 
              "大家", "这里", "一个", "一些", "很多", "现在", "那么", "所以","同学们","老师","课程","内容","知识点","思想","理论","问题"]

def extract_audio(video_path,output_audio):
    video=VideoFileClip(video_path)
    audio=video.audio
    video.audio.write_audiofile(output_audio,fps=16000,nbytes=2,codec='pcm_s16le')
    video.close()

def save_ppt_screenshot(v_path, time_s, out_name):
    """
    使用 MoviePy 在指定时间点截取一帧
    """
    try:
        # 确保保存图片的文件夹存在
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        
        # 加载视频并截图
        with VideoFileClip(v_path) as clip:
            # 这里的 time_s 是秒，clip.save_frame 会自动定位
            # 稍微往后偏 0.5 秒，避开可能存在的转场黑屏
            target_t = min(time_s + 0.5, clip.duration - 0.1)
            clip.save_frame(out_name, t=target_t)
            
        print(f"📸 截图成功: {out_name}")
        return True
    except Exception as e:
        print(f"❌ 截图失败: {e}")
        return False

def transcribe_audio(model_obj, audio_path):
    segments_generator, info = model_obj.transcribe(audio_path, beam_size=5, language="zh")
    segments_list = []
    print(f"--- 音频总时长: {info.duration:.1f} 秒 ---")
    with tqdm(total=round(info.duration, 1), unit="s", desc="语音识别") as pbar:
        last_pos = 0
        for s in segments_generator:
            segments_list.append({
                'start': s.start,
                'end': s.end,
                'text': s.text
            })
            tqdm.write(f"[{s.start:.1f}s -> {s.end:.1f}s]: {s.text}")
            increment = s.end - last_pos
            if increment > 0:
                pbar.update(increment)
                last_pos = s.end
                
    return segments_list

def find_keyword_hits(segments, keywords):
    hit_times = []
    last_hit_time = -999
    for s in segments:
        text = s['text']
        start_time = s['start']
        found = any(kw in text for kw in keywords)
        if found:
            if start_time - last_hit_time > 120:  
                hit_times.append(start_time)
                last_hit_time = start_time
                print(f"🚩 命中关键词，时间点：{start_time:.1f}s，内容：{text}")
    return hit_times

def analyze_hits_and_screenshot(hit_times, all_segments, v_path, save_dir):
    results = []
    for t in hit_times:
        start_range, end_range = max(0, t - 60), t + 60
        
        # 1. 获取这个窗口内的所有 segment 对象（包含时间戳）
        window_segments = [s for s in all_segments if start_range <= s['start'] <= end_range]
        window_text = "".join([s['text'] for s in window_segments])

        # 2. 请求 DeepSeek 进行“视觉定位”
        print(f"🧠 DeepSeek 正在思考哪一秒的 PPT 最重要...")
        ai_data = get_llm_advanced_analysis(window_segments,t)
        
        # 3. 【核心进阶】使用 AI 推荐的时间点截图
        best_t = ai_data.get('best_time', t) # 如果 AI 没给，就用默认时间 t
        
        img_name = f"ppt_{int(best_t)}s.jpg"
        img_path = os.path.join(save_dir, img_name)
        
        # 执行截图
        save_ppt_screenshot(v_path, best_t, img_path)
        
        results.append({
            'hit_time': best_t,
            'top_words': ai_data['keywords'],
            'summary': ai_data['summary'],
            'img_path': img_path
        })
    return results

# 1. 确保这个类定义在外面
class MyCoursePDF(FPDF):
    def footer(self):
        # 设置距离底部 1.5 厘米
        self.set_y(-15)
        self.set_font("Hans", size=10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")

# 2. 修正后的函数（注意看 pdf.output 的位置）
def create_final_pdf(analysis_results, course_name, filename):
    pdf = MyCoursePDF()
    # 注册字体 (确保 FONT_PATH 在全局已定义)
    pdf.add_font("Hans", "", FONT_PATH)
    pdf.add_font("Hans", "B", FONT_PATH) 

    # --- 1. 封面页 ---
    pdf.add_page()
    pdf.set_font("Hans", size=30)
    pdf.ln(60)
    pdf.cell(0, 20, txt=course_name, ln=True, align='C')
    pdf.set_font("Hans", size=18)
    pdf.cell(0, 20, txt="AI 智能复习讲义", ln=True, align='C')
    pdf.ln(40)
    pdf.set_font("Hans", size=12)
    pdf.cell(0, 10, txt=f"生成日期：{datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')

    # --- 2. 内容页 (循环开始) ---
    for item in analysis_results:
        pdf.add_page()
        
        # 标题栏
        t = item['hit_time']
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("Hans", size=15)
        pdf.cell(0, 12, txt=f"重点片段：{int(t // 60)}分{int(t % 60)}秒", ln=True, fill=True)
        pdf.ln(5)

        # 核心词汇
        pdf.set_font("Hans", size=12)
        pdf.set_text_color(31, 73, 125)
        pdf.multi_cell(0, 10, txt="核心词汇：" + " | ".join(item['top_words']))
        pdf.ln(2)

        # 插入截图
        img_path = item['img_path']
        if os.path.exists(img_path):
            pdf.image(img_path, x=15, w=180)
            # 手动向下移动游标，避免之后的文字覆盖在图片上方 (180宽时的16:9比例高度约101)
            pdf.set_y(pdf.get_y() + 105)
            pdf.ln(5)

        # AI 摘要 (黄色背景)
        pdf.set_fill_color(252, 243, 207)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Hans", size=12)
        pdf.cell(0, 8, txt="【AI 考点总结】：", ln=True)
        pdf.set_font("Hans", size=11)
        pdf.multi_cell(0, 8, txt=item['summary'], fill=True)
        # ❌ 这里绝对不能有 pdf.output()

    # --- 3. 封包保存 (关键：必须在 for 循环完全结束后执行) ---
    # 请确保这一行代码的缩进和上面的 for 对齐，而不是缩进在 for 里面
    pdf.output(filename)
    print(f"✨ 最终讲义已成功保存至: {filename}")
# 1. 初始化客户端 (确保 API Key 已填)
client = OpenAI(
    api_key="YOUR_API_KEY_HERE", 
    base_url="https://api.deepseek.com",
    timeout=60.0 # 长视频分析较慢，设置60秒超时
)

def get_llm_advanced_analysis(window_segments, default_time):
    """
    window_segments: 当前时间窗口内的 segments 列表
    default_time: 原始关键词命中的时间点（作为保底）
    """
    # --- 1. 构造带时间戳的上下文文字 ---
    # 格式如：[120.5s] 同学们看这个公式...
    context_with_time = ""
    for s in window_segments:
        context_with_time += f"[{s['start']:.1f}s] {s['text']}\n"

    # --- 2. 编写针对性的 Prompt ---
    prompt = f"""
    你是一个专业的学术助教。我将为你提供一段课堂录音转述，每句话前带有 [时间s] 格式的时间戳。
    请完成以下任务：
    1. 提取 3-5 个核心专业名词作为关键词。
    2. 将这段内容总结为 100 字以内的重点摘要，说明这段内容为何重要。
    3. **视觉定位**：在这段对话中，哪一秒最适合截取 PPT 画面作为复习参考？
       请分析哪一刻老师正在讲解核心定义、展示公式或图表。
       注意：请直接从提供的 [时间s] 标签中选择一个最准确的绝对时间。

    对话记录：
    {context_with_time}

    请严格按以下 JSON 格式回复，不要包含任何额外文字：
    {{
        "keywords": ["专业词1", "专业词2"],
        "summary": "考点摘要...",
        "best_time": 123.4
    }}
    """

    try:
        # --- 3. 调用 DeepSeek API ---
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严谨的学术助手，擅长分析课堂重点。"},
                {"role": "user", "content": prompt},
            ],
            response_format={ 'type': 'json_object' }, # 强制返回 JSON
            stream=False
        )

        # --- 4. 解析结果 ---
        content_str = response.choices[0].message.content
        res_dict = json.loads(content_str)
        
        # 提取字段，如果 AI 漏掉字段则使用保底值
        keywords = res_dict.get("keywords", ["重点内容"])
        summary = res_dict.get("summary", "未能生成摘要")
        
        # 获取 AI 建议的时间点
        ai_time = res_dict.get("best_time", default_time)
        
        # 逻辑校验：如果 AI 编造了一个完全不在窗口内的时间，则退回到默认时间
        if not (window_segments[0]['start'] - 5 <= ai_time <= window_segments[-1]['end'] + 5):
            print(f"⚠️ AI 建议的时间 {ai_time}s 超出窗口，使用默认值 {default_time}s")
            ai_time = default_time
            
        return {
            "keywords": keywords,
            "summary": summary,
            "best_time": float(ai_time)
        }

    except Exception as e:
        print(f"❌ DeepSeek 分析失败: {e}")
        # 如果 API 挂了，返回基础数据防止整个程序崩溃
        return {
            "keywords": ["识别失败"],
            "summary": "由于网络或 API 原因，未能生成 AI 摘要。",
            "best_time": default_time
        }
# ================= 主函数 =================
if __name__ == "__main__":
    # 1. 自动创建基础文件夹
    for d in [VIDEO_DIR, AUDIO_DIR, SCREENSHOT_DIR, RESULT_DIR]:
        os.makedirs(d, exist_ok=True)
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.mkv', '.avi','.ts'))]
    if not video_files:
        print(f"❌ '{VIDEO_DIR}' 文件夹里没有视频文件！")
        import sys
        sys.exit()
    print("--- 发现以下视频 ---")
    for i, f in enumerate(video_files):
        print(f"{i+1}. {f}")
    choice = int(input("请输入要处理的视频编号: ")) - 1
    target_video_name = video_files[choice]
    base_name = os.path.splitext(target_video_name)[0]
    VIDEO_PATH = os.path.join(VIDEO_DIR, target_video_name)
    AUDIO_PATH = os.path.join(AUDIO_DIR, f"{base_name}.wav")
    current_screenshot_dir = os.path.join(SCREENSHOT_DIR, base_name)
    os.makedirs(current_screenshot_dir, exist_ok=True)
    PDF_PATH = os.path.join(RESULT_DIR, f"{base_name}_复习讲义.pdf")
    if not os.path.exists(AUDIO_PATH):
        print(f"--- 提取音频中: {VIDEO_PATH} ---")
        extract_audio(VIDEO_PATH, AUDIO_PATH)
    else:
        print(f"--- 发现已有音频 {AUDIO_PATH}，跳过提取 ---")
    
    print(f"--- 正在加载模型识别视频: {target_video_name} ---")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    segments = transcribe_audio(model, AUDIO_PATH)
    hits = find_keyword_hits(segments, KEYWORDS)
    
    if hits:
        final_data = analyze_hits_and_screenshot(hits, segments, VIDEO_PATH, current_screenshot_dir)
        create_final_pdf(final_data, f"{base_name} 课堂笔记", PDF_PATH)
        print(f"✨ 处理完成！PDF 已生成到: {PDF_PATH}")
    else:
        print("💡 没发现重点。")