from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os
import easyocr
import re
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# City Code Dictionary
cityCode_dict = {
    "A": "Banten", "AA": "Magelang", "AB": "Yogyakarta", "AD": "Solo", "AE": "Madiun", "AG": "Kediri",
    "B": "Jakarta", "BA": "Padang", "BB": "Medan", "BD": "Bengkulu", "BE": "Lampung",
    "BG": "Sumsel", "BH": "Jambi", "BK": "Siantar", "BL": "Aceh", "BM": "Riau",
    "BN": "Babel", "BP": "Batam", "D": "Bandung", "DE": "Maluku", "DG": "Malut",
    "DH": "Kupang", "DK": "Bali", "DM": "Gorontalo", "DR": "Lombok", "E": "Cirebon",
    "F": "Bogor", "G": "Pekalongan", "H": "Semarang", "K": "Pati", "L": "Surabaya",
    "M": "Madura", "N": "Malang", "P": "Jember", "PA": "Papua", "PB": "Papua Barat",
    "R": "Banyumas", "S": "Bojonegoro", "T": "Purwakarta", "W": "Sidoarjo", "Z": "Garut"
}

class HybridALPR:
    def __init__(self, plate_model_path):
        # YOLO di GPU, Preprocessing di CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cpu_cores = mp.cpu_count()
        
        print(f" YOLO Device: {self.device.upper()}")
        print(f" CPU Cores: {self.cpu_cores}")
        print(f" Hybrid Mode: GPU/CPU Load Balancing")
        
        self.plate_detector = YOLO(plate_model_path)
        self.plate_detector.to(self.device)
        
        # EasyOCR dengan GPU
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    def preprocess_single(self, plate_crop):
        """Preprocessing optimized - Upscaling 6x"""
        h, w = plate_crop.shape[:2]
        scale = 3
        
        # Upscaling 6x dengan INTER_CUBIC
        upscaled = cv2.resize(plate_crop, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        # Sharpening untuk karakter jelas
        kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharp)
        
        # Denoising optimal
        denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
        
        # CLAHE untuk kontras tinggi
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def preprocess_plate(self, plate_crop):
        """Preprocessing 4 variants optimal - Balance akurasi & resource"""
        # Base preprocessing
        enhanced = self.preprocess_single(plate_crop)
        results = [enhanced]
        
        # Variant 2: Adaptive Threshold (TERBAIK untuk plat kotor)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        results.append(binary)
        
        # Variant 3: Otsu (TERBAIK untuk plat bersih)
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(otsu)
        
        # Variant 4: Inverted (PENTING untuk plat gelap)
        inverted = cv2.bitwise_not(enhanced)
        results.append(inverted)
        
        return results
    
    def ocr_single(self, proc_img):
        """OCR single image (untuk parallel processing)"""
        return self.reader.readtext(
            proc_img, 
            detail=1, 
            paragraph=False,
            batch_size=1,
            contrast_ths=0.05,
            adjust_contrast=0.3,
            text_threshold=0.4,
            low_text=0.2,
            link_threshold=0.2,
            width_ths=0.3,
            height_ths=0.2
        )
    
    def parse_plate(self, ocr_results_list):
        """Parse ultra akurat dengan multiple strategy"""
        all_candidates = []
        
        for ocr_results in ocr_results_list:
            if not ocr_results:
                continue
            
            # Filter baris atas
            if len(ocr_results) > 1:
                y_coords = [(bbox[0][1] + bbox[2][1]) / 2 for bbox, _, _ in ocr_results]
                y_median = np.median(y_coords)
                y_std = np.std(y_coords)
                threshold = y_median - y_std * 0.5
                filtered = [(bbox, text, conf) for (bbox, text, conf), y in zip(ocr_results, y_coords) if y <= threshold]
                if filtered:
                    ocr_results = filtered
            
            ocr_results.sort(key=lambda x: x[0][0][0])
            
            segments = []
            for bbox, text, conf in ocr_results:
                if conf > 0.15:
                    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if clean:
                        segments.append({'text': clean, 'conf': conf})
            
            if not segments:
                continue
            
            all_text = ''.join([s['text'] for s in segments])
            total_conf = sum([s['conf'] for s in segments]) / len(segments)
            
            if len(all_text) < 5:
                continue
            
            corrections = {
                'prefix': {
                    'T': 'B', '7': 'L', '1': 'I', '0': 'O', '9': 'O', 
                    'J': 'I', 'N': 'H', '8': 'B', '5': 'S', '6': 'G',
                    'Q': 'O', '2': 'Z'
                },
                'number': {
                    'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 
                    'T': '7', 'B': '8', 'S': '5', 'G': '6', 'Z': '2',
                    'o': '0', 'l': '1', 'i': '1', 'C': '0'
                },
                'suffix': {
                    '0': 'O', '1': 'I', '7': 'T', '8': 'B', '9': 'O', 
                    '5': 'S', '6': 'G', '2': 'Z', '3': 'B'
                }
            }
            
            for min_prefix in [1, 2]:
                for max_suffix in [1, 2, 3]:
                    prefix = ""
                    for c in all_text:
                        if c.isalpha() and len(prefix) < 2:
                            prefix += corrections['prefix'].get(c, c)
                        else:
                            break
                    
                    if len(prefix) < min_prefix:
                        continue
                    
                    suffix = ""
                    for c in reversed(all_text):
                        if c.isalpha() and len(suffix) < max_suffix:
                            suffix = corrections['suffix'].get(c, c) + suffix
                        else:
                            break
                    
                    if len(suffix) < 1:
                        continue
                    
                    number = all_text[len(prefix):len(all_text)-len(suffix)]
                    number = ''.join([corrections['number'].get(c, c) for c in number])
                    
                    if (prefix and number and suffix and 
                        number.isdigit() and 
                        suffix.isalpha() and
                        1 <= len(number) <= 4 and
                        prefix in cityCode_dict):
                        
                        all_candidates.append({
                            'prefix': prefix,
                            'number': number,
                            'suffix': suffix,
                            'conf': total_conf,
                            'length': len(all_text)
                        })
        
        if not all_candidates:
            return None, None, None
        
        all_candidates.sort(key=lambda x: (x['conf'], x['length']), reverse=True)
        best = all_candidates[0]
        
        return best['prefix'], best['number'], best['suffix']
    
    def check_odd_even(self, number):
        if number and number.isdigit():
            last_digit = int(number[-1])
            return "GANJIL" if last_digit % 2 == 1 else "GENAP"
        return "UNKNOWN"
    
    def check_violation(self, number):
        today = datetime.now().weekday()
        if today < 5:
            last_digit = int(number[-1]) if number and number.isdigit() else -1
            if today in [0, 2, 4] and last_digit % 2 == 0:
                return True, "GENAP di hari GANJIL"
            elif today in [1, 3] and last_digit % 2 == 1:
                return True, "GANJIL di hari GENAP"
        return False, ""
    
    def process_image(self, image):
        plates_info = []
        
        # YOLO Detection (GPU/CPU)
        results = self.plate_detector(image, conf=0.15, device=self.device, verbose=False)
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    h_img, w_img = image.shape[:2]
                    pad = 15
                    x1_pad = max(0, x1 - pad)
                    x2_pad = min(w_img, x2 + pad)
                    y1_pad = max(0, y1 - pad)
                    y2_pad = min(h_img, y2 + pad)
                    
                    plate_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    if plate_crop.size == 0:
                        continue
                    
                    # Preprocessing (CPU)
                    processed_images = self.preprocess_plate(plate_crop)
                    
                    # OCR SEQUENTIAL dengan memory cleanup
                    ocr_results_list = []
                    for idx, proc_img in enumerate(processed_images):
                        ocr_results = self.reader.readtext(
                            proc_img, 
                            detail=1, 
                            paragraph=False,
                            batch_size=1,
                            contrast_ths=0.05,
                            adjust_contrast=0.3,
                            text_threshold=0.4,
                            low_text=0.2,
                            link_threshold=0.2,
                            width_ths=0.3,
                            height_ths=0.2
                        )
                        ocr_results_list.append(ocr_results)
                        
                        # Clear CUDA cache setelah setiap varian
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                    
                    # Clear CUDA cache setelah semua OCR selesai
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Parse
                    result = self.parse_plate(ocr_results_list)
                    prefix, number, suffix = result
                    
                    if prefix and number and suffix:
                        full_plate = f"{prefix} {number} {suffix}"
                        city = cityCode_dict.get(prefix, "Unknown")
                        odd_even = self.check_odd_even(number)
                        is_violation, violation_msg = self.check_violation(number)
                        
                        plates_info.append({
                            'bbox': (x1, y1, x2, y2),
                            'plate': full_plate,
                            'city': city,
                            'odd_even': odd_even,
                            'violation': is_violation,
                            'violation_msg': violation_msg
                        })
                    
                    # Clear memory setelah setiap plat
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
        
        # Clear memory setelah semua plat di image ini
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return plates_info
    
    def draw_results(self, image, plates_info):
        output = image.copy()
        
        for info in plates_info:
            x1, y1, x2, y2 = info['bbox']
            
            color = (0, 0, 255) if info['violation'] else (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            texts = [
                f"{info['plate']}",
                f"{info['city']} ({info['odd_even']})"
            ]
            
            if info['violation']:
                texts.append(f"VIOLATION: {info['violation_msg']}")
            
            y_offset = y1 - 15
            for text in reversed(texts):
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(output, (x1, y_offset - th - 5), (x1 + tw + 10, y_offset + 5), color, -1)
                cv2.putText(output, text, (x1 + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset -= (th + 10)
        
        return output

def main():
    print("\n🇮🇩 ALPR INDONESIA - HYBRID CPU+GPU VERSION")
    print("="*60)
    print("⚡ Upscaling: 6x (High Accuracy)")
    print("⚡ Variants: 4 (Efficient)")
    print("⚡ YOLO: GPU (CUDA)")
    print("⚡ OCR: CPU")
    print("⚡ Memory: Auto-cleanup")
    print("="*60)
    
    plate_model = 'models/indonesia_plates_all5/weights/best.pt'
    
    if not os.path.exists(plate_model):
        print("❌ Model tidak ditemukan!")
        return
    
    alpr = HybridALPR(plate_model)
    
    input_folder = "test_images"
    output_folder = f"hasil_alpr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_folder, exist_ok=True)
    
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"📸 Processing {len(files)} images...")
    print(f"📁 Output: {output_folder}\n")
    
    total_plates = 0
    total_violations = 0
    processed_count = 0
    
    for filename in files:
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        processed_count += 1
        print(f"🔍 [{processed_count}/{len(files)}] {filename}", end='')
        
        plates_info = alpr.process_image(image)
        
        # Clear memory setelah setiap gambar
        if alpr.device == 'cuda':
            torch.cuda.empty_cache()
        
        if plates_info:
            print()
            for info in plates_info:
                total_plates += 1
                print(f"   ✅ {info['plate']} - {info['city']} ({info['odd_even']})")
                
                if info['violation']:
                    total_violations += 1
                    print(f"      ⚠️ {info['violation_msg']}")
            
            result_image = alpr.draw_results(image, plates_info)
            cv2.imwrite(os.path.join(output_folder, filename), result_image)
        else:
            print(" ❌")
    
    print(f"\n{'='*60}")
    print(f"📊 Total Gambar: {processed_count}")
    print(f"🚗 Total Plat: {total_plates}")
    print(f"⚠️  Total Pelanggaran: {total_violations}")
    print(f"🎯 Akurasi: {(total_plates/processed_count*100):.1f}%" if processed_count > 0 else "")
    print(f"📁 Hasil: {output_folder}")
    print("="*60)

if __name__ == "__main__":
    main()
