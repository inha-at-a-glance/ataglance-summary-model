import requests
import boto3
import json
import time
from datetime import datetime
import math
import torch
import pymysql
from transformers import BartForConditionalGeneration, BartConfig, PreTrainedTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity


class ClovaSpeechClient:
    def __init__(self, CLOVA_INVOKE_URL, CLOVA_SECRET_KEY):
        self.invoke_url = CLOVA_INVOKE_URL
        self.secret = CLOVA_SECRET_KEY

    def req_url(self, url, completion, callback=None, userdata=None, forbiddens=None, boostings=None, wordAlignment=True, fullText=True, diarization=None, sed=None):
        print("clova speech request: ", url)
        request_body = {
            'url': url,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))


class S3VideoProcessor:
    def __init__(self, secrets_path='config.json'):
        with open(secrets_path, 'r', encoding='utf-8') as file:
            self.secrets = json.load(file)
        
        self.initialize_models()

        self.stt_client = ClovaSpeechClient(
            self.secrets["clova"]['invoke_url'], 
            self.secrets["clova"]['secret_key']
        )

    def initialize_models(self):
        # BART 모델 초기화 로직
        config_path = self.secrets["aws"]["model_paths"]["kobart_config"]
        config = BartConfig.from_json_file(config_path)
        model_path = self.secrets["aws"]["model_paths"]["kobart_model"]

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.secrets["model"]["tokenizer"])
        self.model = BartForConditionalGeneration.from_pretrained(model_path, config=config)
        self.model.eval()

    def get_processed_ids(self):
        """
        RDS에서 처리된 news_id 가져오기
        """
        connection = pymysql.connect(
            host=self.secrets['rds']['host'],
            user=self.secrets['rds']['user'],
            password=self.secrets['rds']['password'],
            database=self.secrets['rds']['database']
        )
        processed_ids = set()

        try:
            with connection.cursor() as cursor:
                query = "SELECT news_id FROM TextProc"
                cursor.execute(query)
                results = cursor.fetchall()
                processed_ids = {row[0] for row in results}
        except Exception as e:
            print(f"RDS 조회 중 오류 발생: {e}")
        finally:
            connection.close()

        return processed_ids

    def process_bucket(self, table_name='News', sleep_interval=60):
        """
        RDS에서 새 news_id를 확인하고 처리되지 않은 경우만 처리
        """
        while True:
            try:
                processed_ids = self.get_processed_ids()

                connection = pymysql.connect(
                    host=self.secrets['rds']['host'],
                    user=self.secrets['rds']['user'],
                    password=self.secrets['rds']['password'],
                    database=self.secrets['rds']['database']
                )

                try:
                    with connection.cursor() as cursor:
                        query = f"""
                        SELECT news_id FROM {table_name}
                        WHERE news_id NOT IN (SELECT news_id FROM TextProc)
                        """
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        start = datetime.now()

                        for row in rows:
                            news_id = row[0]
                            self.process_video(news_id)
                        print(datetime.now()-start)
                
                except Exception as e:
                    print(f"RDS 처리 중 오류 발생: {e}")
                finally:
                    connection.close()

                # 일정 시간 대기
                time.sleep(sleep_interval)

            except Exception as e:
                print(f"버킷 모니터링 중 오류 발생: {e}")
                time.sleep(sleep_interval)

    def process_video(self, news_id):
        """
        뉴스 ID를 사용해 비디오 처리 및 요약
        """
        print(f"Processing video for news_id: {news_id}")
        video_url = f'https://ataglance-bucket.s3.amazonaws.com/{news_id}/{news_id}.mp4'
        self.news_id = news_id
        raw_news = self.stt_client.req_url(url=video_url, completion='sync').json()
        print(raw_news['message'])

        full_text = raw_news['text']

        filter_keywords = ["기자", "뉴스", "앵커"]
        sentences = [text for text in full_text.split('.') if not any(keyword in text for keyword in filter_keywords)]

        num_sentences = len(sentences)
        split_size = math.floor(num_sentences / 4)

        conversation_segments = [ ". ".join(sentences[i:i + split_size]).strip() + '.' 
                                 for i in range(0, num_sentences, split_size)]

        each_sent_encode = [
            {'text': segment["text"], 'start': segment['start'], 'end': segment['end']} 
            for segment in raw_news["segments"] 
            if segment["speaker"]["label"] != '1' # 최초 발화자는 앵커로 판단
        ]

        # 요약 
        seg_summary = []
        for seg in conversation_segments:
            if len(seg) < 10:
                seg_summary.append(seg)
                continue
            
            raw_input_ids = self.tokenizer.encode(seg)
            input_ids = [self.tokenizer.bos_token_id] + raw_input_ids + [self.tokenizer.eos_token_id]

            summary_ids = self.model.generate(
                torch.tensor([input_ids]), 
                num_beams=4, 
                max_length=50, 
                eos_token_id=1, 
                no_repeat_ngram_size=2,
                repetition_penalty=1.4
            )
            summary_text = self.tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            seg_summary.append(summary_text)

        # 코사인 유사도 기준 타임스탬프 저장
        time_stamp = []
        for line in seg_summary:
            cur_time = []
            if len(line) < 5: continue
            
            embedding1 = self.get_bert_embedding(line)
            for origin in each_sent_encode:
                embedding2 = self.get_bert_embedding(origin['text'])
                cur_sim = cosine_similarity(embedding1, embedding2)
                cur_time.append({'time':[origin['start'], origin['end']], 'sim':cur_sim})

            time_stamp.append(max(cur_time, key=lambda x: x['sim'], default=None))

        time_stamp = [entry['time'] for entry in time_stamp]
        print(time_stamp)

        self.save_to_database(news_id,seg_summary[:len(time_stamp)], time_stamp)

    def get_bert_embedding(self, sentence):
        inputs = self.bert_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        embedding = outputs.encoder_last_hidden_state.mean(dim=1).numpy()
        return embedding

    def save_to_database(self, news_id, summary, timestamps):
        """
        요약 결과 데이터베이스 저장
        """
        connection = pymysql.connect(
            host=self.secrets['rds']['host'],
            user=self.secrets['rds']['user'],
            password=self.secrets['rds']['password'],
            database=self.secrets['rds']['database']
        )

        try:
            with connection.cursor() as cursor:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(now)
                time_stamp_json = json.dumps({'time': timestamps}, ensure_ascii=False, indent=4)
                text_json = json.dumps({'text': summary}, ensure_ascii=False, indent=4)
                
                insert_query = """
                INSERT INTO TextProc (news_id, summary_txt, created_at, updated_at, time_stamp)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (news_id, text_json, now, now, time_stamp_json))
                
                text_id = cursor.lastrowid
                videoproc_query = """
                INSERT INTO VideoProc (news_id, text_id)
                VALUES (%s, %s)
                """
                videoproc_data = (news_id, text_id)
                cursor.execute(videoproc_query, videoproc_data)
                connection.commit()

        except Exception as e:
            connection.rollback()
            print(f"데이터베이스 저장 중 오류: {e}")

        finally:
            connection.close()

# 메인 실행
if __name__ == '__main__':
    processor = S3VideoProcessor()
    processor.process_bucket()
